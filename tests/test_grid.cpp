/**
 * @file test_grid.cpp
 * @brief Unit tests for the MAC grid and Array3D.
 */

#include <gtest/gtest.h>
#include "sloshing/grid.h"
#include <cmath>

using namespace sloshing;

// ============================================================================
// Array3D tests
// ============================================================================

TEST(Array3D, ConstructionAndAccess) {
    Array3D<double> arr(4, 5, 6, 1.0);
    EXPECT_EQ(arr.ni(), 4);
    EXPECT_EQ(arr.nj(), 5);
    EXPECT_EQ(arr.nk(), 6);
    EXPECT_EQ(arr.size(), 120u);
    EXPECT_DOUBLE_EQ(arr(0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(arr(3, 4, 5), 1.0);
}

TEST(Array3D, WriteThenRead) {
    Array3D<double> arr(3, 3, 3, 0.0);
    arr(1, 1, 1) = 42.0;
    EXPECT_DOUBLE_EQ(arr(1, 1, 1), 42.0);
    EXPECT_DOUBLE_EQ(arr(0, 0, 0), 0.0);
}

TEST(Array3D, Fill) {
    Array3D<double> arr(2, 2, 2, 0.0);
    arr.fill(3.14);
    for (int k = 0; k < 2; ++k)
        for (int j = 0; j < 2; ++j)
            for (int i = 0; i < 2; ++i)
                EXPECT_DOUBLE_EQ(arr(i, j, k), 3.14);
}

TEST(Array3D, Swap) {
    Array3D<double> a(2, 2, 2, 1.0);
    Array3D<double> b(3, 3, 3, 2.0);
    a.swap(b);
    EXPECT_EQ(a.ni(), 3);
    EXPECT_EQ(b.ni(), 2);
    EXPECT_DOUBLE_EQ(a(0, 0, 0), 2.0);
    EXPECT_DOUBLE_EQ(b(0, 0, 0), 1.0);
}

// ============================================================================
// MACGrid tests
// ============================================================================

TEST(MACGrid, Construction) {
    MACGrid grid(10, 5, 8, 1.0, 0.5, 0.8);
    EXPECT_EQ(grid.ni(), 10);
    EXPECT_EQ(grid.nj(), 5);
    EXPECT_EQ(grid.nk(), 8);
    EXPECT_DOUBLE_EQ(grid.dx(), 0.1);
    EXPECT_DOUBLE_EQ(grid.dy(), 0.1);
    EXPECT_DOUBLE_EQ(grid.dz(), 0.1);
}

TEST(MACGrid, InvalidDimensions) {
    EXPECT_THROW(MACGrid(0, 5, 5, 1.0, 1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(MACGrid(5, 5, 5, -1.0, 1.0, 1.0), std::invalid_argument);
}

TEST(MACGrid, StaggeredSizes) {
    MACGrid grid(4, 3, 5, 1.0, 0.75, 1.25);
    // u: (ni+1) x nj x nk
    EXPECT_EQ(grid.u.ni(), 5);
    EXPECT_EQ(grid.u.nj(), 3);
    EXPECT_EQ(grid.u.nk(), 5);
    // v: ni x (nj+1) x nk
    EXPECT_EQ(grid.v.ni(), 4);
    EXPECT_EQ(grid.v.nj(), 4);
    EXPECT_EQ(grid.v.nk(), 5);
    // w: ni x nj x (nk+1)
    EXPECT_EQ(grid.w.ni(), 4);
    EXPECT_EQ(grid.w.nj(), 3);
    EXPECT_EQ(grid.w.nk(), 6);
}

TEST(MACGrid, CellCenter) {
    MACGrid grid(10, 10, 10, 1.0, 1.0, 1.0);
    auto center = grid.cellCenter(0, 0, 0);
    EXPECT_DOUBLE_EQ(center.x, 0.05);
    EXPECT_DOUBLE_EQ(center.y, 0.05);
    EXPECT_DOUBLE_EQ(center.z, 0.05);

    auto center2 = grid.cellCenter(9, 9, 9);
    EXPECT_DOUBLE_EQ(center2.x, 0.95);
    EXPECT_DOUBLE_EQ(center2.y, 0.95);
    EXPECT_DOUBLE_EQ(center2.z, 0.95);
}

TEST(MACGrid, DivergenceOfUniformFlow) {
    // Uniform flow in x should have zero divergence
    MACGrid grid(8, 8, 8, 1.0, 1.0, 1.0);
    // Set uniform u = 1.0
    for (int k = 0; k < 8; ++k)
        for (int j = 0; j < 8; ++j)
            for (int i = 0; i <= 8; ++i)
                grid.u(i, j, k) = 1.0;

    grid.phi.fill(-1.0); // All fluid
    grid.classifyCells();

    for (int k = 1; k < 7; ++k)
        for (int j = 1; j < 7; ++j)
            for (int i = 1; i < 7; ++i)
                EXPECT_NEAR(grid.divergence(i, j, k), 0.0, 1e-12);
}

TEST(MACGrid, DivergenceOfExpandingFlow) {
    // u = x, v = y, w = z → div = 3
    MACGrid grid(8, 8, 8, 1.0, 1.0, 1.0);
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();

    for (int k = 0; k < 8; ++k)
        for (int j = 0; j < 8; ++j)
            for (int i = 0; i <= 8; ++i)
                grid.u(i, j, k) = i * dx;

    for (int k = 0; k < 8; ++k)
        for (int j = 0; j <= 8; ++j)
            for (int i = 0; i < 8; ++i)
                grid.v(i, j, k) = j * dy;

    for (int k = 0; k <= 8; ++k)
        for (int j = 0; j < 8; ++j)
            for (int i = 0; i < 8; ++i)
                grid.w(i, j, k) = k * dz;

    grid.phi.fill(-1.0);
    grid.classifyCells();

    for (int k = 1; k < 7; ++k)
        for (int j = 1; j < 7; ++j)
            for (int i = 1; i < 7; ++i)
                EXPECT_NEAR(grid.divergence(i, j, k), 3.0, 1e-10);
}

TEST(MACGrid, InterpolateVelocityConstant) {
    // Constant velocity field: u=1, v=2, w=3
    MACGrid grid(4, 4, 4, 1.0, 1.0, 1.0);
    for (size_t idx = 0; idx < grid.u.size(); ++idx) grid.u.data()[idx] = 1.0;
    for (size_t idx = 0; idx < grid.v.size(); ++idx) grid.v.data()[idx] = 2.0;
    for (size_t idx = 0; idx < grid.w.size(); ++idx) grid.w.data()[idx] = 3.0;

    auto vel = grid.interpolateVelocity({0.5, 0.5, 0.5});
    EXPECT_NEAR(vel.x, 1.0, 1e-10);
    EXPECT_NEAR(vel.y, 2.0, 1e-10);
    EXPECT_NEAR(vel.z, 3.0, 1e-10);
}
