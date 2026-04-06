/**
 * @file renderer.cpp
 * @brief OpenGL renderer with marching cubes, Phong lighting, and tank wireframe.
 */

#include "sloshing/renderer.h"
#include "sloshing/parallel.h"
#include "marching_cubes_tables.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>
#include <iostream>
#include <mutex>

namespace sloshing {

// ============================================================================
// Shader sources (embedded — OpenGL 4.1 / GLSL 410)
// ============================================================================

static const char* surface_vert_src = R"glsl(
#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNormalMatrix;

out vec3 vWorldPos;
out vec3 vNormal;

void main() {
    vWorldPos = vec3(uModel * vec4(aPos, 1.0));
    vNormal = normalize(uNormalMatrix * aNormal);
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)glsl";

static const char* surface_frag_src = R"glsl(
#version 410 core
in vec3 vWorldPos;
in vec3 vNormal;

uniform vec3 uCameraPos;
uniform vec3 uLightDir;
uniform vec3 uWaterColor;

out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uCameraPos - vWorldPos);
    vec3 H = normalize(L + V);

    // Phong lighting
    float ambient = 0.15;
    float diffuse = max(dot(N, L), 0.0) * 0.6;
    float specular = pow(max(dot(N, H), 0.0), 64.0) * 0.4;

    // Fresnel-like effect for water
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    vec3 skyColor = vec3(0.6, 0.8, 1.0);

    vec3 color = uWaterColor * (ambient + diffuse) + vec3(specular);
    color = mix(color, skyColor, fresnel * 0.3);

    FragColor = vec4(color, 1.0);
}
)glsl";

static const char* wireframe_vert_src = R"glsl(
#version 410 core
layout(location = 0) in vec3 aPos;

uniform mat4 uMVP;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)glsl";

static const char* wireframe_frag_src = R"glsl(
#version 410 core
uniform vec3 uColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(uColor, 1.0);
}
)glsl";

// ============================================================================
// Marching Cubes surface extraction
// ============================================================================

/// @brief Edge vertex positions (two vertex indices per edge).
static const int edgeVertices[12][2] = {
    {0,1}, {1,2}, {2,3}, {3,0},
    {4,5}, {5,6}, {6,7}, {7,4},
    {0,4}, {1,5}, {2,6}, {3,7}
};

/// @brief Cube corner offsets in (i,j,k) space.
static const int cornerOffsets[8][3] = {
    {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
    {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
};

/// @brief Linearly interpolate between two positions at the zero-crossing.
static float vertexInterpT(float v1, float v2) {
    if (std::abs(v1 - v2) < 1e-10f) return 0.5f;
    return std::clamp(-v1 / (v2 - v1), 0.0f, 1.0f);
}

/// @brief Compute the gradient of phi at grid point (ci,cj,ck) using central differences.
///
/// At solid-wall boundaries (x- and z-walls, floor) the wall-normal component is
/// set to zero — the reflection BC makes phi symmetric across the wall, so the
/// gradient normal to the wall vanishes.  This gives correct 90° contact-angle
/// normals and eliminates the dark shading band at wall edges.
static glm::vec3 phiGradient(const Array3D<double>& phi, int ci, int cj, int ck,
                               float fdx, float fdy, float fdz) {
    int ni = phi.ni(), nj = phi.nj(), nk = phi.nk();

    float gx, gy, gz;

    // x: zero at left/right walls (reflection BC), central elsewhere.
    if (ci <= 1 || ci >= ni - 2)
        gx = 0.0f;
    else
        gx = static_cast<float>(phi(ci+1, cj, ck) - phi(ci-1, cj, ck)) / (2.0f * fdx);

    // y: one-sided at boundaries, central elsewhere.
    if (cj == 0)         gy = static_cast<float>(phi(ci, 1, ck) - phi(ci, 0, ck)) / fdy;
    else if (cj >= nj-1) gy = static_cast<float>(phi(ci, nj-1, ck) - phi(ci, nj-2, ck)) / fdy;
    else                 gy = static_cast<float>(phi(ci, cj+1, ck) - phi(ci, cj-1, ck)) / (2.0f * fdy);

    // z: zero at front/back walls (reflection BC), central elsewhere.
    if (ck <= 1 || ck >= nk - 2)
        gz = 0.0f;
    else
        gz = static_cast<float>(phi(ci, cj, ck+1) - phi(ci, cj, ck-1)) / (2.0f * fdz);

    return {gx, gy, gz};
}

std::vector<Vertex> marchingCubes(const Array3D<double>& phi,
                                   double dx, double dy, double dz) {
    std::vector<Vertex> vertices;
    int ni = phi.ni(), nj = phi.nj(), nk = phi.nk();
    float fdx = static_cast<float>(dx);
    float fdy = static_cast<float>(dy);
    float fdz = static_cast<float>(dz);
    // Skip boundary cubes: solid walls at i=0/ni-1, k=0/nk-1.
    // Extend one row down (j=0) and keep top (j=nj-1) open.
    int i0 = 1, i1 = ni - 2;
    int j0 = 0, j1 = nj - 1;
    int k0 = 1, k1 = nk - 2;

    // One vector per k-slice — each k is processed by exactly one thread
    int k_range = k1 - k0;
    std::vector<std::vector<Vertex>> slice_verts(k_range);

    sloshing::parallel_for(k0, k1, [&](int k) {
        auto& local_verts = slice_verts[k - k0];

        for (int j = j0; j < j1; ++j) {
            for (int i = i0; i < i1; ++i) {
                float val[8];
                glm::vec3 pos_arr[8];
                glm::vec3 grad_arr[8];
                for (int c = 0; c < 8; ++c) {
                    int ci = i + cornerOffsets[c][0];
                    int cj = j + cornerOffsets[c][1];
                    int ck = k + cornerOffsets[c][2];
                    // Extrapolate phi at floor (cj=0) to match wall cap gphi,
                    // avoiding solid-cell BC artefacts.
                    val[c] = (cj == 0)
                        ? static_cast<float>(2.0 * phi(ci, 1, ck) - phi(ci, 2, ck))
                        : static_cast<float>(phi(ci, cj, ck));

                    // Snap outermost fluid-cell corners to wall/floor/top faces
                    // so the MC surface meets the wall caps seamlessly.
                    float px = (ci == 1)      ? fdx
                             : (ci == ni - 2) ? (ni - 1) * fdx
                             :                  (ci + 0.5f) * fdx;
                    float py = (cj <= 0)      ? 0.0f
                             : (cj >= nj - 1) ? nj * fdy
                             :                  (cj + 0.5f) * fdy;
                    float pz = (ck == 1)      ? fdz
                             : (ck == nk - 2) ? (nk - 1) * fdz
                             :                  (ck + 0.5f) * fdz;
                    pos_arr[c] = glm::vec3(px, py, pz);

                    grad_arr[c] = phiGradient(phi, ci,
                                              std::max(cj, 1), ck,
                                              fdx, fdy, fdz);
                }

                int cubeIndex = 0;
                for (int c = 0; c < 8; ++c)
                    if (val[c] < 0.0f) cubeIndex |= (1 << c);

                if (mc_tables::edgeTable[cubeIndex] == 0) continue;

                glm::vec3 edgeVerts[12];
                glm::vec3 edgeNormals[12];
                for (int e = 0; e < 12; ++e) {
                    if (mc_tables::edgeTable[cubeIndex] & (1 << e)) {
                        int v0 = edgeVertices[e][0];
                        int v1 = edgeVertices[e][1];
                        float t = vertexInterpT(val[v0], val[v1]);
                        edgeVerts[e] = pos_arr[v0] + t * (pos_arr[v1] - pos_arr[v0]);
                        glm::vec3 n = grad_arr[v0] + t * (grad_arr[v1] - grad_arr[v0]);
                        float len = glm::length(n);
                        edgeNormals[e] = (len > 1e-8f) ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
                    }
                }

                for (int t = 0; mc_tables::triTable[cubeIndex][t] != -1; t += 3) {
                    int e0 = mc_tables::triTable[cubeIndex][t];
                    int e1 = mc_tables::triTable[cubeIndex][t + 1];
                    int e2 = mc_tables::triTable[cubeIndex][t + 2];

                    local_verts.push_back({edgeVerts[e0], edgeNormals[e0]});
                    local_verts.push_back({edgeVerts[e1], edgeNormals[e1]});
                    local_verts.push_back({edgeVerts[e2], edgeNormals[e2]});
                }
            }
        }
    });

    // Concatenate slice results in order
    size_t total_size = 0;
    for (auto& sv : slice_verts) total_size += sv.size();
    vertices.reserve(total_size);
    for (auto& sv : slice_verts) {
        vertices.insert(vertices.end(), sv.begin(), sv.end());
    }

    return vertices;
}

// ============================================================================
// Renderer implementation
// ============================================================================

GLuint Renderer::createShaderProgram(const char* vert_src, const char* frag_src) {
    auto compile = [](GLenum type, const char* src) -> GLuint {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);
        GLint success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char log[512];
            glGetShaderInfoLog(shader, 512, nullptr, log);
            std::cerr << "Shader compile error: " << log << std::endl;
        }
        return shader;
    };

    GLuint vert = compile(GL_VERTEX_SHADER, vert_src);
    GLuint frag = compile(GL_FRAGMENT_SHADER, frag_src);

    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(program, 512, nullptr, log);
        std::cerr << "Shader link error: " << log << std::endl;
    }

    glDeleteShader(vert);
    glDeleteShader(frag);
    return program;
}

bool Renderer::init(int width, int height, const std::string& title) {
    width_ = width;
    height_ = height;

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    // Request OpenGL 4.1 Core (macOS maximum)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4); // MSAA

    window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window_) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1); // V-sync

    // Load OpenGL functions
    int version = gladLoadGL(glfwGetProcAddress);
    if (!version) {
        std::cerr << "Failed to load OpenGL" << std::endl;
        return false;
    }

    // OpenGL state
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    // Blending disabled — fully opaque water rendering.
    glClearColor(0.85f, 0.9f, 0.95f, 1.0f);

    // Compile shaders
    surface_shader_ = createShaderProgram(surface_vert_src, surface_frag_src);
    wireframe_shader_ = createShaderProgram(wireframe_vert_src, wireframe_frag_src);

    // Create surface VAO/VBO (initially empty)
    glGenVertexArrays(1, &surface_vao_);
    glGenBuffers(1, &surface_vbo_);

    // Create tank wireframe
    glGenVertexArrays(1, &tank_vao_);
    glGenBuffers(1, &tank_vbo_);

    // Create wall cap mesh VAO/VBO
    glGenVertexArrays(1, &wall_vao_);
    glGenBuffers(1, &wall_vbo_);

    updateCameraPosition();
    return true;
}

void Renderer::shutdown() {
    if (surface_vao_) glDeleteVertexArrays(1, &surface_vao_);
    if (surface_vbo_) glDeleteBuffers(1, &surface_vbo_);
    if (wall_vao_) glDeleteVertexArrays(1, &wall_vao_);
    if (wall_vbo_) glDeleteBuffers(1, &wall_vbo_);
    if (tank_vao_) glDeleteVertexArrays(1, &tank_vao_);
    if (tank_vbo_) glDeleteBuffers(1, &tank_vbo_);
    if (surface_shader_) glDeleteProgram(surface_shader_);
    if (wireframe_shader_) glDeleteProgram(wireframe_shader_);
    if (window_) glfwDestroyWindow(window_);
    glfwTerminate();
}

bool Renderer::shouldClose() const {
    return glfwWindowShouldClose(window_);
}

void Renderer::beginFrame() {
    glfwGetFramebufferSize(window_, &width_, &height_);
    glViewport(0, 0, width_, height_);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Renderer::endFrame() {
    glfwSwapBuffers(window_);
    glfwPollEvents();
}

void Renderer::updateCameraPosition() {
    camera_eye_.x = camera_target_.x + camera_distance_ * std::cos(camera_phi_) * std::sin(camera_theta_);
    camera_eye_.y = camera_target_.y + camera_distance_ * std::sin(camera_phi_);
    camera_eye_.z = camera_target_.z + camera_distance_ * std::cos(camera_phi_) * std::cos(camera_theta_);
}

void Renderer::setCamera(const glm::vec3& eye, const glm::vec3& target, const glm::vec3& up) {
    camera_eye_ = eye;
    camera_target_ = target;
    camera_up_ = up;
}

void Renderer::orbitCamera(float dtheta, float dphi) {
    camera_theta_ += dtheta;
    camera_phi_ = std::clamp(camera_phi_ + dphi, -1.4f, 1.4f);
    updateCameraPosition();
}

void Renderer::zoomCamera(float amount) {
    camera_distance_ = std::clamp(camera_distance_ - amount, 0.5f, 10.0f);
    updateCameraPosition();
}

void Renderer::encompass() {
    // Reset orbit angles to a nice 3/4 view
    camera_theta_ = 0.8f;
    camera_phi_ = 0.5f;
    // Compute distance to fit the tank diagonal in the FOV
    float diag = std::sqrt(tank_lx_ * tank_lx_ + tank_ly_ * tank_ly_ + tank_lz_ * tank_lz_);
    float fov_rad = glm::radians(45.0f);
    camera_distance_ = diag / (2.0f * std::tan(fov_rad * 0.5f)) * 0.9f;
    camera_distance_ = std::clamp(camera_distance_, 0.5f, 10.0f);
    updateCameraPosition();
}

void Renderer::createTankMesh(float x0, float y0, float z0, float x1, float y1, float z1) {
    // 12 edges of a box = 24 vertices (line segments)
    float vertices[] = {
        // Bottom face
        x0,y0,z0, x1,y0,z0,   x1,y0,z0, x1,y0,z1,   x1,y0,z1, x0,y0,z1,   x0,y0,z1, x0,y0,z0,
        // Top face
        x0,y1,z0, x1,y1,z0,   x1,y1,z0, x1,y1,z1,   x1,y1,z1, x0,y1,z1,   x0,y1,z1, x0,y1,z0,
        // Vertical edges
        x0,y0,z0, x0,y1,z0,   x1,y0,z0, x1,y1,z0,   x1,y0,z1, x1,y1,z1,   x0,y0,z1, x0,y1,z1
    };

    glBindVertexArray(tank_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, tank_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    tank_vertex_count_ = 24;
}

void Renderer::uploadSurfaceMesh(const std::vector<Vertex>& vertices) {
    glBindVertexArray(surface_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, surface_vbo_);
    glBufferData(GL_ARRAY_BUFFER,
                 vertices.size() * sizeof(Vertex),
                 vertices.data(),
                 GL_DYNAMIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, position)));
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, normal)));
    glEnableVertexAttribArray(1);

    surface_vertex_count_ = static_cast<int>(vertices.size());
}

/// @brief Fill one marching-squares cell of the water region (phi < 0) on a wall plane.
///
/// Corners are ordered CCW when viewed from outside: c0=bottom-left, c1=bottom-right,
/// c2=top-right, c3=top-left. Emits fan-triangulated polygons into @p out.
static void fillMSCell(float p0, float p1, float p2, float p3,
                        glm::vec3 c0, glm::vec3 c1, glm::vec3 c2, glm::vec3 c3,
                        glm::vec3 n, std::vector<Vertex>& out) {
    int cfg = (p0 < 0 ? 1 : 0) | (p1 < 0 ? 2 : 0) | (p2 < 0 ? 4 : 0) | (p3 < 0 ? 8 : 0);
    if (cfg == 0) return;

    // Walk CCW around the cell collecting polygon vertices for the phi<0 region.
    // At each corner: add it if inside. At each edge: add the interpolated zero-crossing.
    float pv[4] = {p0, p1, p2, p3};
    glm::vec3 cv[4] = {c0, c1, c2, c3};
    std::vector<glm::vec3> poly;
    poly.reserve(6);
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) & 3;
        if (pv[i] < 0)
            poly.push_back(cv[i]);
        if ((pv[i] < 0) != (pv[j] < 0)) {
            float t = pv[i] / (pv[i] - pv[j]);
            poly.push_back(glm::mix(cv[i], cv[j], t));
        }
    }

    // Fan-triangulate the polygon from vertex 0.
    for (int i = 1; i + 1 < (int)poly.size(); ++i) {
        out.push_back({poly[0], n});
        out.push_back({poly[i], n});
        out.push_back({poly[i + 1], n});
    }
}

/// @brief Generate smooth wall-cap geometry for all 6 tank faces using marching squares.
///
/// Corner positions use the same coordinate mapping as marching cubes (cell centres
/// with boundary snapping) so that the zero-crossing contour on the wall is identical
/// to the MC isosurface edge, eliminating the visible seam.
static std::vector<Vertex> generateWallCaps(const MACGrid& grid, const Array3D<double>& phi) {
    std::vector<Vertex> verts;
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();
    float dx = static_cast<float>(grid.dx());
    float dy = static_cast<float>(grid.dy());
    float dz = static_cast<float>(grid.dz());
    float lx = static_cast<float>(grid.lx());
    float ly = static_cast<float>(grid.ly());

    // Position mapping matching MC corner snapping for seamless join.
    // MC snaps: ci=1→dx, ci=ni-2→(ni-1)*dx; ck=1→dz, ck=nk-2→(nk-1)*dz.
    // y uses cell centres without snapping; boundary rows snap to floor/ceiling.
    auto pos_x = [&](int ci) -> float {
        if (ci <= 1) return dx;
        if (ci >= ni - 2) return (ni - 1) * dx;
        return (ci + 0.5f) * dx;
    };
    auto pos_y = [&](int cj) -> float {
        if (cj <= 0) return 0.0f;       // floor boundary
        if (cj >= nj - 1) return ly;    // top: matches MC snap
        return (cj + 0.5f) * dy;        // cell centres (no floor snap)
    };
    auto pos_z = [&](int ck) -> float {
        if (ck <= 1) return dz;
        if (ck >= nk - 2) return (nk - 1) * dz;
        return (ck + 0.5f) * dz;
    };

    auto gphi = [&](int i, int j, int k) -> float {
        if (i < 0 || i >= ni || j < 0 || j >= nj || k < 0 || k >= nk)
            return 1.0f;
        if (j == 0) // extrapolate below first fluid cell (avoids solid-cell BC artefact)
            return static_cast<float>(2.0 * phi(i, 1, k) - phi(i, 2, k));
        return static_cast<float>(phi(i, j, k));
    };

    // Left wall face (x=dx): phi from i=1.
    {
        glm::vec3 n(-1, 0, 0);
        for (int k = 1; k < nk - 2; ++k)
            for (int j = 0; j < nj - 1; ++j)
                fillMSCell(gphi(1,j,k), gphi(1,j+1,k), gphi(1,j+1,k+1), gphi(1,j,k+1),
                    {dx,pos_y(j),pos_z(k)},{dx,pos_y(j+1),pos_z(k)},
                    {dx,pos_y(j+1),pos_z(k+1)},{dx,pos_y(j),pos_z(k+1)},
                    n, verts);
    }

    // Right wall face (x=lx-dx): phi from i=ni-2.
    {
        glm::vec3 n(1, 0, 0);
        float xr = lx - dx;
        for (int k = 1; k < nk - 2; ++k)
            for (int j = 0; j < nj - 1; ++j)
                fillMSCell(gphi(ni-2,j,k), gphi(ni-2,j+1,k), gphi(ni-2,j+1,k+1), gphi(ni-2,j,k+1),
                    {xr,pos_y(j),pos_z(k)},{xr,pos_y(j+1),pos_z(k)},
                    {xr,pos_y(j+1),pos_z(k+1)},{xr,pos_y(j),pos_z(k+1)},
                    n, verts);
    }

    // Front wall face (z=dz): phi from k=1.
    {
        glm::vec3 n(0, 0, -1);
        for (int j = 0; j < nj - 1; ++j)
            for (int i = 1; i < ni - 2; ++i)
                fillMSCell(gphi(i,j,1), gphi(i+1,j,1), gphi(i+1,j+1,1), gphi(i,j+1,1),
                    {pos_x(i),pos_y(j),dz},{pos_x(i+1),pos_y(j),dz},
                    {pos_x(i+1),pos_y(j+1),dz},{pos_x(i),pos_y(j+1),dz},
                    n, verts);
    }

    // Back wall face (z=lz-dz): phi from k=nk-2.
    {
        glm::vec3 n(0, 0, 1);
        float zb = (nk - 1) * dz;
        for (int j = 0; j < nj - 1; ++j)
            for (int i = 1; i < ni - 2; ++i)
                fillMSCell(gphi(i,j,nk-2), gphi(i+1,j,nk-2), gphi(i+1,j+1,nk-2), gphi(i,j+1,nk-2),
                    {pos_x(i),pos_y(j),zb},{pos_x(i+1),pos_y(j),zb},
                    {pos_x(i+1),pos_y(j+1),zb},{pos_x(i),pos_y(j+1),zb},
                    n, verts);
    }

    // Floor face (y=0): phi from j=1.  Normal faces up (toward camera).
    {
        glm::vec3 n(0, 1, 0);
        for (int k = 1; k < nk - 2; ++k)
            for (int i = 1; i < ni - 2; ++i)
                fillMSCell(gphi(i,1,k), gphi(i+1,1,k), gphi(i+1,1,k+1), gphi(i,1,k+1),
                    {pos_x(i),0,pos_z(k)},{pos_x(i+1),0,pos_z(k)},
                    {pos_x(i+1),0,pos_z(k+1)},{pos_x(i),0,pos_z(k+1)},
                    n, verts);
    }

    // No top safety fill: the MC y-snapping (cj=nj-1 → ly) handles the top
    // boundary.  A horizontal fill here would show through MC gaps as a grey
    // artefact due to the flat normal mismatch.

    return verts;
}

/// @brief Upload wall cap vertices to GPU.
static void uploadWallCaps(GLuint vao, GLuint vbo, const std::vector<Vertex>& verts) {
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, position)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, normal)));
    glEnableVertexAttribArray(1);
}

void Renderer::render(const MACGrid& grid) {
    float lx = static_cast<float>(grid.lx());
    float ly = static_cast<float>(grid.ly());
    float lz = static_cast<float>(grid.lz());

    // Recreate tank wireframe if dimensions changed.
    // Drawn at inner fluid domain: inset by 1 solid cell in x and z to match the caps.
    if (lx != tank_lx_ || ly != tank_ly_ || lz != tank_lz_ || tank_vertex_count_ == 0) {
        float wdx = static_cast<float>(grid.dx());
        float wdz = static_cast<float>(grid.dz());
        createTankMesh(wdx, 0.0f, wdz, lx - wdx, ly, lz - wdz);
        tank_lx_ = lx; tank_ly_ = ly; tank_lz_ = lz;
    }

    // Update camera target to tank center
    camera_target_ = glm::vec3(lx * 0.5f, ly * 0.5f, lz * 0.5f);
    updateCameraPosition();

    // Matrices
    float aspect = static_cast<float>(width_) / std::max(height_, 1);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.01f, 100.0f);
    glm::mat4 view = glm::lookAt(camera_eye_, camera_target_, camera_up_);
    // Model matrix: translate to offset, then rotate around tank center
    glm::vec3 center(lx * 0.5f, ly * 0.5f, lz * 0.5f);
    glm::mat4 model = glm::translate(glm::mat4(1.0f), tank_offset_);
    model = glm::translate(model, center);
    model = glm::rotate(model, tank_tilt_x_, glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, tank_tilt_z_, glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::translate(model, -center);
    glm::mat4 mvp = projection * view * model;
    glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(model)));

    // Temporal + spatial smoothing of phi to reduce surface flickering.
    int ni = grid.phi.ni(), nj = grid.phi.nj(), nk = grid.phi.nk();
    if (phi_smoothed_.ni() != ni ||
        phi_smoothed_.nj() != nj ||
        phi_smoothed_.nk() != nk) {
        phi_smoothed_ = grid.phi; // first frame or grid resize: seed with current
    } else {
        // Temporal EMA: 30% current, 70% previous (heavier smoothing).
        for (int k = 0; k < nk; ++k)
            for (int j = 0; j < nj; ++j)
                for (int i = 0; i < ni; ++i)
                    phi_smoothed_(i, j, k) =
                        0.3 * grid.phi(i, j, k) + 0.7 * phi_smoothed_(i, j, k);
    }

    // Extract surface mesh using smoothed phi
    auto surface_verts = marchingCubes(phi_smoothed_, grid.dx(), grid.dy(), grid.dz());
    uploadSurfaceMesh(surface_verts);

    // Generate and upload wall cap mesh (water volume visible through tank walls)
    auto wall_verts = generateWallCaps(grid, phi_smoothed_);
    uploadWallCaps(wall_vao_, wall_vbo_, wall_verts);
    wall_vertex_count_ = static_cast<int>(wall_verts.size());

    // --- Render fluid surface + wall caps (same shader, same water color) ---
    glUseProgram(surface_shader_);
    glUniformMatrix4fv(glGetUniformLocation(surface_shader_, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniformMatrix4fv(glGetUniformLocation(surface_shader_, "uModel"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix3fv(glGetUniformLocation(surface_shader_, "uNormalMatrix"), 1, GL_FALSE, glm::value_ptr(normalMatrix));
    glUniform3fv(glGetUniformLocation(surface_shader_, "uCameraPos"), 1, glm::value_ptr(camera_eye_));
    glUniform3f(glGetUniformLocation(surface_shader_, "uLightDir"), 0.5f, 1.0f, 0.3f);
    glUniform3f(glGetUniformLocation(surface_shader_, "uWaterColor"), 0.1f, 0.4f, 0.8f);
    if (surface_vertex_count_ > 0) {
        glBindVertexArray(surface_vao_);
        glDrawArrays(GL_TRIANGLES, 0, surface_vertex_count_);
    }
    if (wall_vertex_count_ > 0) {
        // Push wall caps slightly behind MC surface in depth to avoid
        // Z-fighting on co-planar wall-face geometry.
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0f, 1.0f);
        glBindVertexArray(wall_vao_);
        glDrawArrays(GL_TRIANGLES, 0, wall_vertex_count_);
        glDisable(GL_POLYGON_OFFSET_FILL);
    }

    // --- Render tank wireframe ---
    glUseProgram(wireframe_shader_);
    glUniformMatrix4fv(glGetUniformLocation(wireframe_shader_, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform3f(glGetUniformLocation(wireframe_shader_, "uColor"), 0.3f, 0.3f, 0.3f);

    glBindVertexArray(tank_vao_);
    glLineWidth(2.0f);
    glDrawArrays(GL_LINES, 0, tank_vertex_count_);
}

void Renderer::renderOverlay(double time, double dt, double volume_error,
                              int pressure_iters, int fps) {
    // OpenGL 4.1 doesn't have built-in text rendering.
    // We'll use the window title as a simple overlay.
    char title[256];
    snprintf(title, sizeof(title),
             "Sloshing 3D | t=%.3f dt=%.5f | Vol err=%.2e | P-iters=%d | FPS=%d",
             time, dt, volume_error, pressure_iters, fps);
    glfwSetWindowTitle(window_, title);
}

} // namespace sloshing
