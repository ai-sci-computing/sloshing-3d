/**
 * @file renderer.h
 * @brief OpenGL renderer with marching cubes surface extraction.
 *
 * Renders the fluid surface by extracting the zero-isosurface of the level set
 * using marching cubes, plus a wireframe tank and basic Phong lighting.
 */
#pragma once

#include "sloshing/grid.h"
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>

namespace sloshing {

/**
 * @brief Vertex with position and normal for the fluid surface mesh.
 */
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
};

/**
 * @brief Extracts the zero-isosurface of the level set using marching cubes.
 *
 * @param phi Level set field.
 * @param dx,dy,dz Grid spacing.
 * @return Vector of triangles (every 3 vertices = 1 triangle).
 */
std::vector<Vertex> marchingCubes(const Array3D<double>& phi,
                                   double dx, double dy, double dz);

/**
 * @brief OpenGL renderer for the sloshing tank simulation.
 */
class Renderer {
public:
    /**
     * @brief Initialize the renderer and create the window.
     * @param width Window width in pixels.
     * @param height Window height in pixels.
     * @param title Window title.
     * @return true if initialization succeeded.
     */
    bool init(int width = 1280, int height = 720, const std::string& title = "Sloshing Tank 3D");

    /// @brief Clean up OpenGL resources and close window.
    void shutdown();

    /// @brief Check if the window should close.
    bool shouldClose() const;

    /// @brief Begin a new frame (clear, set up matrices).
    void beginFrame();

    /**
     * @brief Render the fluid surface and tank.
     * @param grid The MAC grid (reads phi for surface extraction).
     */
    void render(const MACGrid& grid);

    /// @brief End the frame (swap buffers, poll events).
    void endFrame();

    /// @brief Get the GLFW window handle (for input handling).
    GLFWwindow* window() const { return window_; }

    /// @name Camera control
    /// @{
    void setCamera(const glm::vec3& eye, const glm::vec3& target, const glm::vec3& up);
    void orbitCamera(float dtheta, float dphi);
    void zoomCamera(float amount);
    void encompass();  ///< Reset camera to fit the full tank in view
    float cameraTheta() const { return camera_theta_; }
    /// @}

    /// @brief Set the tank's visual displacement (applied to model matrix).
    void setTankOffset(const glm::vec3& offset) { tank_offset_ = offset; }

    /// @brief Set the tank's tilt angles (radians) for visual rotation.
    void setTankTilt(float tilt_x, float tilt_z) { tank_tilt_x_ = tilt_x; tank_tilt_z_ = tilt_z; }

    /// @brief Render overlay text with simulation info.
    void renderOverlay(double time, double dt, double volume_error,
                       int pressure_iters, int fps);

private:
    GLFWwindow* window_ = nullptr;
    int width_ = 1280, height_ = 720;

    // Shader programs
    GLuint surface_shader_ = 0;    ///< Phong-lit surface shader
    GLuint wireframe_shader_ = 0;  ///< Simple wireframe shader

    // Fluid surface mesh
    GLuint surface_vao_ = 0, surface_vbo_ = 0;
    int surface_vertex_count_ = 0;

    // Tank wireframe
    GLuint tank_vao_ = 0, tank_vbo_ = 0;
    int tank_vertex_count_ = 0;

    // Wall cap mesh (water volume fills at boundary faces)
    GLuint wall_vao_ = 0, wall_vbo_ = 0;
    int wall_vertex_count_ = 0;

    // Temporally-smoothed phi for flicker-free marching cubes
    Array3D<double> phi_smoothed_;

    // Camera
    glm::vec3 camera_eye_ = {1.5f, 1.0f, 1.5f};
    glm::vec3 camera_target_ = {0.5f, 0.25f, 0.5f};
    glm::vec3 camera_up_ = {0.0f, 1.0f, 0.0f};
    float camera_distance_ = 2.0f;
    float camera_theta_ = 0.8f;  ///< Azimuthal angle
    float camera_phi_ = 0.5f;    ///< Polar angle

    // Tank dimensions (set when rendering)
    float tank_lx_ = 1.0f, tank_ly_ = 0.5f, tank_lz_ = 1.0f;

    // Tank visual offset (for shake feedback)
    glm::vec3 tank_offset_{0.0f};

    // Tank tilt angles (radians)
    float tank_tilt_x_ = 0.0f;
    float tank_tilt_z_ = 0.0f;

    /// @brief Compile and link a shader program.
    GLuint createShaderProgram(const char* vert_src, const char* frag_src);

    /// @brief Create the tank wireframe VAO from (x0,y0,z0) to (x1,y1,z1).
    void createTankMesh(float x0, float y0, float z0, float x1, float y1, float z1);

    /// @brief Upload fluid surface mesh to GPU.
    void uploadSurfaceMesh(const std::vector<Vertex>& vertices);

    /// @brief Update camera eye position from orbit parameters.
    void updateCameraPosition();
};

} // namespace sloshing
