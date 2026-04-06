/**
 * @file simulation.h
 * @brief Main simulation driver: time integration and non-inertial frame forces.
 *
 * Orchestrates the full simulation pipeline per timestep:
 * 1. Apply body forces (gravity + non-inertial pseudo-forces from tank motion)
 * 2. Advect velocity, level set, and VOF
 * 3. Pressure projection (enforce incompressibility)
 * 4. CLSVOF coupling (volume correction + redistancing)
 * 5. Boundary conditions
 *
 * The tank motion is specified as an acceleration in the lab frame.
 * In the non-inertial (tank) frame, this appears as a pseudo-force
 * equal and opposite to the tank acceleration.
 */
#pragma once

#include "sloshing/grid.h"
#include "sloshing/pressure_solver.h"
#include "sloshing/clsvof.h"
#include <glm/glm.hpp>

namespace sloshing {

/**
 * @brief Configuration for the simulation.
 */
struct SimulationConfig {
    // Grid
    int ni = 64;            ///< Cells in x
    int nj = 32;            ///< Cells in y
    int nk = 64;            ///< Cells in z
    double lx = 1.0;        ///< Tank length (m)
    double ly = 0.5;        ///< Tank height (m)
    double lz = 1.0;        ///< Tank depth (m)
    double water_level = 0.25; ///< Initial water level (m)

    // Physics
    double gravity = 9.81;  ///< Gravitational acceleration (m/s²)
    double density_liquid = 1000.0;
    double density_air = 1.0;
    double max_tank_accel = 20.0; ///< Max tank acceleration (m/s²), for CFL safety

    // Time stepping
    double cfl_number = 0.35;    ///< CFL number for adaptive dt
    double max_dt = 0.01;        ///< Maximum timestep (s)
    double min_dt = 1e-5;        ///< Minimum timestep (s)

    // Solver
    PressureSolverConfig pressure_config;
    CLSVOFConfig clsvof_config;
    int velocity_extension_layers = 3;  ///< Layers of velocity extrapolation into air
};

/**
 * @brief Main simulation state and driver.
 */
class Simulation {
public:
    explicit Simulation(const SimulationConfig& config = {});

    /**
     * @brief Advance the simulation by one timestep.
     *
     * Uses adaptive timestepping based on CFL condition.
     * The tank acceleration is provided in the lab frame.
     *
     * @param tank_acceleration Acceleration of the tank in lab frame (m/s²).
     * @return The timestep size used.
     */
    double step(const glm::dvec3& tank_acceleration = glm::dvec3(0.0));

    /**
     * @brief Advance the simulation by a target wall-clock time.
     *
     * Takes as many sub-steps as needed to advance by target_dt,
     * using adaptive CFL-based timestepping.
     *
     * @param target_dt Target time to advance (s).
     * @param tank_acceleration Acceleration of the tank.
     * @return Number of sub-steps taken.
     */
    int advance(double target_dt, const glm::dvec3& tank_acceleration = glm::dvec3(0.0));

    /// @name Accessors
    /// @{
    MACGrid& grid() { return grid_; }
    const MACGrid& grid() const { return grid_; }
    double time() const { return time_; }
    double lastDt() const { return last_dt_; }
    double totalVolume() const { return total_volume_; }
    double volumeError() const;
    int pressureIterations() const { return pressure_iters_; }
    /// @}

private:
    SimulationConfig config_;
    MACGrid grid_;
    PressureSolver pressure_solver_;
    double time_ = 0.0;
    double last_dt_ = 0.001;
    double initial_volume_ = 0.0;
    double total_volume_ = 0.0;
    int pressure_iters_ = 0;

    /// @brief Compute adaptive timestep from CFL condition.
    double computeDt() const;

    /// @brief Apply gravity and non-inertial pseudo-forces to velocity.
    void applyBodyForces(double dt, const glm::dvec3& tank_acceleration);
};

} // namespace sloshing
