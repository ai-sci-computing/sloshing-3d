/**
 * @file test_clsvof.cpp
 * @brief Unit tests for CLSVOF coupling, level set, and VOF.
 */

#include <gtest/gtest.h>
#include "sloshing/grid.h"
#include "sloshing/clsvof.h"
#include "sloshing/advection.h"
#include "sloshing/simulation.h"
#include <cmath>

using namespace sloshing;

TEST(CLSVOF, InitFlatSurface) {
    MACGrid grid(8, 8, 8, 1.0, 1.0, 1.0);
    double water_level = 0.5;
    initializeFlatSurface(grid, water_level);

    // Below water: phi < 0, vof = 1
    EXPECT_LT(grid.phi(4, 0, 4), 0.0);
    EXPECT_DOUBLE_EQ(grid.vof(4, 0, 4), 1.0);

    // Above water: phi > 0, vof = 0
    EXPECT_GT(grid.phi(4, 7, 4), 0.0);
    EXPECT_DOUBLE_EQ(grid.vof(4, 7, 4), 0.0);

    // Interface cell: 0 < vof < 1
    // Water level at 0.5, cell j=3 spans [0.375, 0.5], j=4 spans [0.5, 0.625]
    // j=3: fully submerged (top at 0.5 = water_level), vof = 1
    // j=4: partially (bottom at 0.5 = water_level), vof = 0
    // Actually: j=3 center at 0.4375, j=4 center at 0.5625
    // j=3: bottom=0.375, top=0.5 → vof = (0.5-0.375)/0.125 = 1.0
    // j=4: bottom=0.5, top=0.625 → vof = 0.0
    // Hmm, with 8 cells over 1.0, dy = 0.125
    // j=3: [0.375, 0.5] → fully below → vof = 1
    // j=4: [0.5, 0.625] → fully above → vof = 0
    // The interface is exactly on a cell boundary.
    // Let's check phi sign at the interface
    EXPECT_LE(grid.phi(4, 3, 4), 0.0); // Below or at water level
    EXPECT_GE(grid.phi(4, 4, 4), 0.0); // At or above water level
}

TEST(CLSVOF, VolumeConservationStatic) {
    // A still tank should perfectly conserve volume
    MACGrid grid(8, 8, 8, 1.0, 1.0, 1.0);
    initializeFlatSurface(grid, 0.5);

    double initial_vol = computeTotalVolume(grid);
    EXPECT_GT(initial_vol, 0.0);

    // Expected volume: 1.0 * 0.5 * 1.0 = 0.5 m³
    EXPECT_NEAR(initial_vol, 0.5, 0.02);
}

TEST(CLSVOF, Reinitialization) {
    // Start with a distorted level set, reinitialize, check it's closer to SDF
    MACGrid grid(16, 16, 16, 1.0, 1.0, 1.0);
    double dx = grid.dx();

    // Set phi = y - 0.5 (which is already a signed distance)
    for (int k = 0; k < 16; ++k)
        for (int j = 0; j < 16; ++j)
            for (int i = 0; i < 16; ++i)
                grid.phi(i, j, k) = (j + 0.5) * dx - 0.5;

    // Distort it: phi *= 2 (no longer unit gradient)
    for (int k = 0; k < 16; ++k)
        for (int j = 0; j < 16; ++j)
            for (int i = 0; i < 16; ++i)
                grid.phi(i, j, k) *= 2.0;

    // Reinitialize (need enough iterations to restore |grad phi| = 1)
    reinitializeLevelSet(grid.phi, dx, dx, dx, 30, 0.3);

    // Check that |grad phi| ≈ 1 away from boundaries
    int count = 0;
    double avg_grad_error = 0.0;
    for (int k = 2; k < 14; ++k)
        for (int j = 2; j < 14; ++j)
            for (int i = 2; i < 14; ++i) {
                double dpx = (grid.phi(i+1,j,k) - grid.phi(i-1,j,k)) / (2*dx);
                double dpy = (grid.phi(i,j+1,k) - grid.phi(i,j-1,k)) / (2*dx);
                double dpz = (grid.phi(i,j,k+1) - grid.phi(i,j,k-1)) / (2*dx);
                double grad_mag = std::sqrt(dpx*dpx + dpy*dpy + dpz*dpz);
                avg_grad_error += std::abs(grad_mag - 1.0);
                count++;
            }
    avg_grad_error /= count;

    // After reinitialization, average gradient magnitude should be close to 1
    EXPECT_LT(avg_grad_error, 0.2)
        << "Average |grad phi| - 1 error: " << avg_grad_error;
}

TEST(CLSVOF, PLICVolumeSymmetry) {
    // Volume below plane with normal (0,1,0) at d=0.5 should be 0.5
    glm::dvec3 normal(0.0, 1.0, 0.0);
    double d = plicFindPlaneConstant(normal, 0.5);
    double vol = plicVolumeBelowPlane(normal, d);
    EXPECT_NEAR(vol, 0.5, 1e-6);
}

TEST(CLSVOF, PLICVolumeExtremes) {
    glm::dvec3 normal(0.0, 1.0, 0.0);

    // Full cell
    double d_full = plicFindPlaneConstant(normal, 1.0);
    double vol_full = plicVolumeBelowPlane(normal, d_full);
    EXPECT_NEAR(vol_full, 1.0, 1e-6);

    // Empty cell
    double d_empty = plicFindPlaneConstant(normal, 0.0);
    double vol_empty = plicVolumeBelowPlane(normal, d_empty);
    EXPECT_NEAR(vol_empty, 0.0, 1e-4);
}

TEST(CLSVOF, PLICVolumeDiagonalNormal) {
    // Diagonal normal: should still work
    glm::dvec3 normal = glm::normalize(glm::dvec3(1.0, 1.0, 1.0));
    for (double target = 0.1; target <= 0.9; target += 0.1) {
        double d = plicFindPlaneConstant(normal, target);
        double vol = plicVolumeBelowPlane(normal, d);
        EXPECT_NEAR(vol, target, 1e-3)
            << "Target volume " << target << ", got " << vol;
    }
}

TEST(Simulation, VolumeConservationOverTime) {
    // Run a few timesteps and check volume doesn't drift badly
    SimulationConfig config;
    config.ni = 8;
    config.nj = 8;
    config.nk = 8;
    config.lx = 1.0;
    config.ly = 1.0;
    config.lz = 1.0;
    config.water_level = 0.5;

    Simulation sim(config);
    double initial_vol = sim.totalVolume();

    // Run 10 timesteps with no external force
    for (int i = 0; i < 10; ++i) {
        sim.step(glm::dvec3(0.0));
    }

    double vol_error = std::abs(sim.volumeError());
    // On a very coarse 8^3 grid, 10% volume error is acceptable
    EXPECT_LT(vol_error, 0.10)
        << "Volume error after 10 steps: " << vol_error * 100 << "%";
}
