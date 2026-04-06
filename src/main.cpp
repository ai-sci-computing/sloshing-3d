/**
 * @file main.cpp
 * @brief Entry point for the Sloshing Tank 3D application.
 *
 * Trackpad-friendly controls:
 *   - Drag:          Grab and shake the tank
 *   - Option+drag:   Tilt the tank (gravity shifts, water sloshes)
 *   - Scroll:        Zoom (vertical) + tilt (horizontal)
 *   - F:             Fit/encompass view (reset tilt + camera)
 *   - R:             Reset simulation
 *   - ESC:           Quit
 */

#include "sloshing/simulation.h"
#include "sloshing/renderer.h"
#include <iostream>
#include <chrono>
#include <cmath>

using namespace sloshing;

struct AppState {
    Simulation* sim = nullptr;
    Renderer* renderer = nullptr;

    // Mouse state
    bool left_button_down = false;
    bool alt_held = false;
    double mouse_x = 0, mouse_y = 0;

    // Accumulated mouse delta for shaking (consumed each physics update)
    double accum_dx = 0, accum_dy = 0;

    // Tank shake state (translation)
    glm::dvec3 tank_pos{0.0};
    glm::dvec3 tank_vel{0.0};
    glm::dvec3 smooth_mouse_vel{0.0};

    // Tank tilt state (rotation, radians)
    // tilt_x = rotation around x-axis (tips tank forward/backward)
    // tilt_z = rotation around z-axis (tips tank left/right)
    double tilt_x = 0.0;
    double tilt_z = 0.0;

    // Shake tuning
    double pixels_to_meters = 0.003;
    double return_stiffness = 20.0;
    double return_damping = 8.0;
    double max_displacement = 0.20;
    double drag_smoothing = 0.08;

    // Tilt tuning
    double max_tilt = 0.35;    ///< Max tilt angle (~20 degrees)
    double tilt_sensitivity = 0.003; ///< Pixels to radians

    // FPS
    double fps_accumulator = 0.0;
    int fps_frames = 0;
    int fps = 0;
};

static AppState g_app;

static void mouseButtonCallback(GLFWwindow* /*window*/, int button, int action, int /*mods*/) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_app.left_button_down = (action == GLFW_PRESS);
        if (action == GLFW_PRESS) {
            g_app.accum_dx = 0;
            g_app.accum_dy = 0;
            g_app.smooth_mouse_vel = glm::dvec3(0.0);
        }
    }
}

static void cursorPosCallback(GLFWwindow* /*window*/, double xpos, double ypos) {
    double dx = xpos - g_app.mouse_x;
    double dy = ypos - g_app.mouse_y;
    g_app.mouse_x = xpos;
    g_app.mouse_y = ypos;

    if (g_app.left_button_down) {
        if (g_app.alt_held) {
            // Option+drag → tilt the tank
            g_app.tilt_z -= dx * g_app.tilt_sensitivity;
            g_app.tilt_x += dy * g_app.tilt_sensitivity;
            g_app.tilt_x = std::clamp(g_app.tilt_x, -g_app.max_tilt, g_app.max_tilt);
            g_app.tilt_z = std::clamp(g_app.tilt_z, -g_app.max_tilt, g_app.max_tilt);
        } else {
            // Normal drag → accumulate for tank shaking
            g_app.accum_dx += dx;
            g_app.accum_dy += dy;
        }
    }
}

static void scrollCallback(GLFWwindow* /*window*/, double xoffset, double yoffset) {
    // Vertical scroll = zoom
    g_app.renderer->zoomCamera(static_cast<float>(yoffset) * 0.1f);
    // Horizontal scroll = tilt tank
    if (std::abs(xoffset) > 0.01) {
        g_app.tilt_z -= xoffset * 0.015;
        g_app.tilt_z = std::clamp(g_app.tilt_z, -g_app.max_tilt, g_app.max_tilt);
    }
}

static void keyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);

    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        SimulationConfig config;
        config.ni = g_app.sim->grid().ni();
        config.nj = g_app.sim->grid().nj();
        config.nk = g_app.sim->grid().nk();
        *g_app.sim = Simulation(config);
        g_app.tank_pos = glm::dvec3(0.0);
        g_app.tank_vel = glm::dvec3(0.0);
        g_app.smooth_mouse_vel = glm::dvec3(0.0);
        g_app.tilt_x = 0.0;
        g_app.tilt_z = 0.0;
        std::cout << "Simulation reset." << std::endl;
    }

    if (key == GLFW_KEY_F && action == GLFW_PRESS) {
        g_app.tilt_x = 0.0;
        g_app.tilt_z = 0.0;
        g_app.renderer->encompass();
    }

    if (key == GLFW_KEY_LEFT_ALT || key == GLFW_KEY_RIGHT_ALT)
        g_app.alt_held = (action != GLFW_RELEASE);
}

/// @brief Compute gravity direction in the tilted tank frame.
/// When the tank is tilted, gravity has horizontal components that
/// push the water sideways — exactly like tilting a real tank.
static glm::dvec3 tiltedGravityAccel(double gravity) {
    // Small-angle: sin(tilt) ≈ tilt, cos(tilt) ≈ 1
    // But we use exact trig for larger tilts.
    // Gravity in lab frame: (0, -g, 0)
    // Tilted tank: rotate gravity vector by -tilt_x around x, -tilt_z around z.
    // The horizontal component of gravity in the tank frame creates a pseudo-force.
    double sx = std::sin(g_app.tilt_x);
    double cx = std::cos(g_app.tilt_x);
    double sz = std::sin(g_app.tilt_z);
    double cz = std::cos(g_app.tilt_z);

    // Gravity projected into tilted frame:
    // After rotation, the "down" direction in tank coords is:
    //   gx = g * sin(tilt_z)   (tilting around z pushes water in x)
    //   gz = -g * sin(tilt_x)  (tilting around x pushes water in z)
    // These appear as horizontal accelerations in the tank frame.
    return glm::dvec3(
        gravity * sz,       // x-component from z-tilt
        0.0,                // vertical handled by simulation's own gravity
        -gravity * sx       // z-component from x-tilt
    );
}

/// @brief Update tank shake physics. Returns acceleration for the simulation.
static glm::dvec3 updateTankPhysics(double dt) {
    if (dt <= 0.0 || dt > 0.1) return glm::dvec3(0.0);

    glm::dvec3 accel{0.0};

    if (g_app.left_button_down && !g_app.alt_held) {
        double dx = g_app.accum_dx;
        double dy = g_app.accum_dy;
        g_app.accum_dx = 0;
        g_app.accum_dy = 0;

        float theta = g_app.renderer->cameraTheta();
        double cos_t = std::cos(theta);
        double sin_t = std::sin(theta);

        glm::dvec3 mouse_vel{0.0};
        mouse_vel.x = (dx * cos_t + dy * sin_t) * g_app.pixels_to_meters / dt;
        mouse_vel.z = (-dx * sin_t + dy * cos_t) * g_app.pixels_to_meters / dt;

        double alpha = 1.0 - std::exp(-dt / g_app.drag_smoothing);
        g_app.smooth_mouse_vel += alpha * (mouse_vel - g_app.smooth_mouse_vel);

        accel = (g_app.smooth_mouse_vel - g_app.tank_vel) / std::max(dt, 0.005);

        double accel_mag = glm::length(accel);
        if (accel_mag > 15.0)
            accel *= 15.0 / accel_mag;
    } else {
        // Return to center
        accel = -g_app.return_stiffness * g_app.tank_pos
                - g_app.return_damping * g_app.tank_vel;

        double accel_mag = glm::length(accel);
        if (accel_mag > 15.0)
            accel *= 15.0 / accel_mag;

        if (glm::length(g_app.tank_pos) < 0.0005 && glm::length(g_app.tank_vel) < 0.005) {
            g_app.tank_pos = glm::dvec3(0.0);
            g_app.tank_vel = glm::dvec3(0.0);
            // Still return tilt acceleration even when shake is settled
            return tiltedGravityAccel(g_app.sim->grid().ly() > 0 ? 9.81 : 0.0);
        }
    }

    g_app.tank_vel += accel * dt;
    g_app.tank_pos += g_app.tank_vel * dt;

    double disp = glm::length(g_app.tank_pos);
    if (disp > g_app.max_displacement) {
        g_app.tank_pos *= g_app.max_displacement / disp;
        glm::dvec3 dir = glm::normalize(g_app.tank_pos);
        double vel_along = glm::dot(g_app.tank_vel, dir);
        if (vel_along > 0.0)
            g_app.tank_vel -= vel_along * dir;
    }

    // Combine shake acceleration + tilt pseudo-gravity
    return accel + tiltedGravityAccel(9.81);
}

int main(int argc, char* argv[]) {
    SimulationConfig config;
    config.ni = 32;
    config.nj = 16;
    config.nk = 32;

    if (argc >= 4) {
        config.ni = std::atoi(argv[1]);
        config.nj = std::atoi(argv[2]);
        config.nk = std::atoi(argv[3]);
    }
    std::cout << "Grid: " << config.ni << " x " << config.nj
              << " x " << config.nk << std::endl;

    Simulation sim(config);
    g_app.sim = &sim;
    std::cout << "Initial fluid volume: " << sim.totalVolume() << " m^3" << std::endl;

    Renderer renderer;
    g_app.renderer = &renderer;

    if (!renderer.init(1280, 720, "Sloshing Tank 3D")) {
        std::cerr << "Failed to initialize renderer" << std::endl;
        return 1;
    }

    GLFWwindow* window = renderer.window();
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);

    std::cout << "\nControls:" << std::endl;
    std::cout << "  Drag:          Grab and shake the tank" << std::endl;
    std::cout << "  Option+drag:   Tilt the tank" << std::endl;
    std::cout << "  Scroll:        Zoom (vertical) / tilt (horizontal)" << std::endl;
    std::cout << "  F:             Fit view (reset tilt)" << std::endl;
    std::cout << "  R:             Reset simulation" << std::endl;
    std::cout << "  ESC:           Quit\n" << std::endl;

    auto last_time = std::chrono::high_resolution_clock::now();

    while (!renderer.shouldClose()) {
        auto now = std::chrono::high_resolution_clock::now();
        double wall_dt = std::chrono::duration<double>(now - last_time).count();
        last_time = now;

        g_app.fps_accumulator += wall_dt;
        g_app.fps_frames++;
        if (g_app.fps_accumulator >= 1.0) {
            g_app.fps = g_app.fps_frames;
            g_app.fps_frames = 0;
            g_app.fps_accumulator -= 1.0;
        }

        glm::dvec3 tank_accel = updateTankPhysics(wall_dt);

        // Set visual transform: translation + tilt rotation
        renderer.setTankOffset(glm::vec3(g_app.tank_pos));
        renderer.setTankTilt(static_cast<float>(g_app.tilt_x),
                             static_cast<float>(g_app.tilt_z));

        double sim_dt = std::min(wall_dt, 1.0 / 30.0);
        sim.advance(sim_dt, tank_accel);

        renderer.beginFrame();
        renderer.render(sim.grid());
        renderer.renderOverlay(sim.time(), sim.lastDt(),
                               sim.volumeError(), sim.pressureIterations(), g_app.fps);
        renderer.endFrame();
    }

    renderer.shutdown();
    return 0;
}
