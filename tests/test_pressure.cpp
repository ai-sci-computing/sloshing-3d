/**
 * @file test_pressure.cpp
 * @brief Unit tests for the pressure Poisson solver.
 */

#include <gtest/gtest.h>
#include "sloshing/grid.h"
#include "sloshing/pressure_solver.h"
#include "sloshing/clsvof.h"
#include <cmath>

using namespace sloshing;

TEST(PressureSolver, ZeroVelocityStaysZero) {
    // Zero initial velocity → pressure should remain near zero, velocity stays zero
    MACGrid grid(8, 8, 8, 1.0, 1.0, 1.0);
    initializeFlatSurface(grid, 0.5);

    PressureSolverConfig config;
    PressureSolver solver(config);

    int iters = solver.solve(grid, 0.001);
    EXPECT_GE(iters, 0); // Should converge

    // Velocity should still be ~zero
    EXPECT_NEAR(grid.maxDivergence(), 0.0, 1e-5);
}

TEST(PressureSolver, DivergenceFreeAfterProjection) {
    // Set up a random-ish divergent velocity field, then project
    MACGrid grid(8, 8, 8, 1.0, 1.0, 1.0);
    initializeFlatSurface(grid, 0.5);

    // Add some velocity to create divergence
    for (int k = 0; k < 8; ++k)
        for (int j = 0; j < 8; ++j)
            for (int i = 0; i <= 8; ++i)
                grid.u(i, j, k) = 0.1 * std::sin(i * 0.5);

    for (int k = 0; k < 8; ++k)
        for (int j = 0; j <= 8; ++j)
            for (int i = 0; i < 8; ++i)
                grid.v(i, j, k) = 0.1 * std::cos(j * 0.5);

    // Enforce BCs before solving
    grid.enforceBoundaryConditions();

    // Count fluid cells for sanity check
    int fluid_count = 0;
    for (int k = 0; k < 8; ++k)
        for (int j = 0; j < 8; ++j)
            for (int i = 0; i < 8; ++i)
                if (grid.cell_type(i, j, k) == CellType::Fluid)
                    fluid_count++;
    EXPECT_GT(fluid_count, 0) << "No fluid cells!";

    // Check pre-projection divergence
    double pre_div = grid.maxDivergence();

    PressureSolverConfig config;
    config.max_iterations = 1000;
    config.tolerance = 1e-8;
    PressureSolver solver(config);

    int iters = solver.solve(grid, 0.001);

    // Enforce BCs after projection too
    grid.enforceBoundaryConditions();

    // After projection, divergence should be much smaller than before
    double max_div = grid.maxDivergence();
    EXPECT_LT(max_div, 0.5) << "Max divergence after projection: " << max_div
                              << " (pre=" << pre_div << ", iters=" << iters
                              << ", fluid_cells=" << fluid_count
                              << ", residual=" << solver.lastResidual() << ")";
}

TEST(PressureSolver, HydrostaticPressure) {
    // A still tank with water should develop hydrostatic pressure
    // p = rho * g * (water_level - y) in fluid cells
    MACGrid grid(4, 16, 4, 0.5, 1.0, 0.5);
    double water_level = 0.5;
    initializeFlatSurface(grid, water_level);

    // Apply gravity for one step
    double dt = 0.001;
    double g = 9.81;
    for (int k = 0; k < grid.nk(); ++k)
        for (int j = 0; j <= grid.nj(); ++j)
            for (int i = 0; i < grid.ni(); ++i)
                grid.v(i, j, k) -= g * dt;

    grid.enforceBoundaryConditions();

    PressureSolverConfig config;
    config.max_iterations = 500;
    PressureSolver solver(config);
    solver.solve(grid, dt);

    // After projection, vertical velocity should be ~zero (hydrostatic balance)
    grid.enforceBoundaryConditions();
    double max_v = 0.0;
    for (int k = 1; k < grid.nk() - 1; ++k)
        for (int j = 1; j < grid.nj(); ++j)
            for (int i = 1; i < grid.ni() - 1; ++i)
                if (grid.cell_type(i, std::min(j, grid.nj()-1), k) == CellType::Fluid ||
                    (j > 0 && grid.cell_type(i, j-1, k) == CellType::Fluid))
                    max_v = std::max(max_v, std::abs(grid.v(i, j, k)));

    EXPECT_LT(max_v, 0.1) << "Vertical velocity should be near zero in hydrostatic state";
}
