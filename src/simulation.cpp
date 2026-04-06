/**
 * @file simulation.cpp
 * @brief Simulation driver: time stepping, body forces, pipeline orchestration.
 */

#include "sloshing/simulation.h"
#include "sloshing/advection.h"
#include "sloshing/clsvof.h"
#include "sloshing/parallel.h"
#include <cmath>
#include <algorithm>

namespace sloshing {

Simulation::Simulation(const SimulationConfig& config)
    : config_(config)
    , grid_(config.ni, config.nj, config.nk,
            config.lx, config.ly, config.lz)
    , pressure_solver_(config.pressure_config)
{
    // Initialize with a flat water surface
    initializeFlatSurface(grid_, config.water_level);
    initial_volume_ = computeTotalVolume(grid_);
    total_volume_ = initial_volume_;

    // Initialize pressure to hydrostatic equilibrium: p = rho * g * (water_level - y)
    // Starting from zero pressure causes a large residual on the first solve,
    // exciting spurious free-surface oscillations.
    double rho = config.pressure_config.density_liquid;
    for (int k = 0; k < grid_.nk(); ++k)
        for (int j = 0; j < grid_.nj(); ++j)
            for (int i = 0; i < grid_.ni(); ++i) {
                if (grid_.cell_type(i, j, k) == CellType::Fluid) {
                    double y_center = (j + 0.5) * grid_.dy();
                    grid_.pressure(i, j, k) = rho * config.gravity * (config.water_level - y_center);
                }
            }
}

double Simulation::volumeError() const {
    if (initial_volume_ < 1e-12) return 0.0;
    return (total_volume_ - initial_volume_) / initial_volume_;
}

double Simulation::computeDt() const {
    double dx = grid_.dx(), dy = grid_.dy(), dz = grid_.dz();
    double max_vel = 1e-10; // Avoid division by zero

    // Find maximum velocity magnitude (single pass over all components)
    int ni = grid_.ni(), nj = grid_.nj(), nk = grid_.nk();
    max_vel = parallel_reduce(0, nk, max_vel,
        [&](int k, double& local) {
            for (int j = 0; j < nj; ++j) {
                for (int i = 0; i <= ni; ++i)
                    local = std::max(local, std::abs(grid_.u(i, j, k)));
                for (int i = 0; i < ni; ++i)
                    local = std::max(local, std::abs(grid_.w(i, j, k)));
            }
            for (int j = 0; j <= nj; ++j)
                for (int i = 0; i < ni; ++i)
                    local = std::max(local, std::abs(grid_.v(i, j, k)));
            // w has nk+1 faces; handle the last k-face
            if (k == nk - 1)
                for (int j = 0; j < nj; ++j)
                    for (int i = 0; i < ni; ++i)
                        local = std::max(local, std::abs(grid_.w(i, j, nk)));
        },
        [](double a, double b) { return std::max(a, b); });

    double min_dx = std::min({dx, dy, dz});
    double dt = config_.cfl_number * min_dx / max_vel;

    // Also limit by gravity CFL: dt < sqrt(2 * dx / g)
    double dt_gravity = std::sqrt(2.0 * min_dx / config_.gravity);
    dt = std::min(dt, dt_gravity);

    return std::clamp(dt, config_.min_dt, config_.max_dt);
}

void Simulation::applyBodyForces(double dt, const glm::dvec3& tank_acceleration) {
    // In the non-inertial (tank) frame, the effective gravity is:
    //   g_eff = (0, -g, 0) - tank_acceleration
    //
    // The tank_acceleration is in the lab frame. The pseudo-force on the fluid
    // is -tank_acceleration (Newton's second law in non-inertial frame).

    glm::dvec3 tank_accel = tank_acceleration;
    // Clamp for CFL safety
    double accel_mag = glm::length(tank_accel);
    if (accel_mag > config_.max_tank_accel) {
        tank_accel *= config_.max_tank_accel / accel_mag;
    }

    double fx = -tank_accel.x;
    double fy = -config_.gravity - tank_accel.y;
    double fz = -tank_accel.z;

    // Apply body forces only to faces adjacent to at least one fluid cell.
    // Applying gravity to pure-air faces creates unphysical velocities that
    // persist through the pressure solve and feed back into the interface
    // via velocity extension, causing spurious currents in still water.

    int ni = grid_.ni(), nj = grid_.nj(), nk = grid_.nk();

    // Apply to u-velocities (x-force at x-faces)
    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j < nj; ++j)
            for (int i = 0; i <= ni; ++i) {
                bool left_fluid  = (i > 0  && grid_.cell_type(i - 1, j, k) == CellType::Fluid);
                bool right_fluid = (i < ni && grid_.cell_type(i, j, k)     == CellType::Fluid);
                if (left_fluid || right_fluid)
                    grid_.u(i, j, k) += dt * fx;
            }
    });

    // Apply to v-velocities (y-force at y-faces)
    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j <= nj; ++j)
            for (int i = 0; i < ni; ++i) {
                bool below_fluid = (j > 0  && grid_.cell_type(i, j - 1, k) == CellType::Fluid);
                bool above_fluid = (j < nj && grid_.cell_type(i, j, k)     == CellType::Fluid);
                if (below_fluid || above_fluid)
                    grid_.v(i, j, k) += dt * fy;
            }
    });

    // Apply to w-velocities (z-force at z-faces)
    parallel_for(0, nk + 1, [&](int k) {
        for (int j = 0; j < nj; ++j)
            for (int i = 0; i < ni; ++i) {
                bool back_fluid  = (k > 0  && grid_.cell_type(i, j, k - 1) == CellType::Fluid);
                bool front_fluid = (k < nk && grid_.cell_type(i, j, k)     == CellType::Fluid);
                if (back_fluid || front_fluid)
                    grid_.w(i, j, k) += dt * fz;
            }
    });
}

double Simulation::step(const glm::dvec3& tank_acceleration) {
    double dt = computeDt();
    last_dt_ = dt;

    // --- Step 1: Apply body forces (gravity + pseudo-forces) ---
    applyBodyForces(dt, tank_acceleration);

    // --- Step 2: Enforce boundary conditions ---
    grid_.enforceBoundaryConditions();

    // --- Step 3: Pressure projection (make velocity divergence-free) ---
    pressure_iters_ = pressure_solver_.solve(grid_, dt);

    // --- Step 4: Enforce BCs again after projection ---
    grid_.enforceBoundaryConditions();

    // --- Step 4b: Extend fluid velocities into air region ---
    // After projection, air cells still carry unphysical body-force velocities.
    // Extrapolate the corrected fluid velocities outward so that semi-Lagrangian
    // backtraces near the interface see a smooth, physical velocity field.
    extendVelocityIntoAir(grid_, config_.velocity_extension_layers);

    // --- Step 5: Advect velocity ---
    advectVelocity(grid_, dt);
    grid_.enforceBoundaryConditions();

    // --- Step 6: Advect level set and VOF ---
    Array3D<double> phi_new(grid_.ni(), grid_.nj(), grid_.nk());
    Array3D<double> vof_new(grid_.ni(), grid_.nj(), grid_.nk());

    advectLevelSet(grid_, phi_new, dt);
    advectVOF(grid_, vof_new, dt);

    grid_.phi.swap(phi_new);
    grid_.vof.swap(vof_new);

    // --- Step 7: CLSVOF coupling ---
    clsvofCoupling(grid_, config_.clsvof_config);

    // --- Step 8: Update volume tracking ---
    total_volume_ = computeTotalVolume(grid_);
    time_ += dt;

    return dt;
}

int Simulation::advance(double target_dt, const glm::dvec3& tank_acceleration) {
    double remaining = target_dt;
    int steps = 0;

    while (remaining > config_.min_dt * 0.5) {
        double dt = step(tank_acceleration);
        remaining -= dt;
        ++steps;

        // Safety: don't take more than 100 sub-steps per frame
        if (steps >= 100) break;
    }

    return steps;
}

} // namespace sloshing
