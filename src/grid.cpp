/**
 * @file grid.cpp
 * @brief Implementation of the MAC staggered grid.
 */

#include "sloshing/grid.h"
#include "sloshing/parallel.h"
#include <cmath>
#include <algorithm>

namespace sloshing {

MACGrid::MACGrid(int ni, int nj, int nk, double lx, double ly, double lz)
    : ni_(ni), nj_(nj), nk_(nk)
    , lx_(lx), ly_(ly), lz_(lz)
    , dx_(lx / ni), dy_(ly / nj), dz_(lz / nk)
    , u(ni + 1, nj, nk, 0.0)
    , v(ni, nj + 1, nk, 0.0)
    , w(ni, nj, nk + 1, 0.0)
    , pressure(ni, nj, nk, 0.0)
    , phi(ni, nj, nk, 0.0)
    , vof(ni, nj, nk, 0.0)
    , cell_type(ni, nj, nk, CellType::Air)
{
    if (ni <= 0 || nj <= 0 || nk <= 0)
        throw std::invalid_argument("Grid dimensions must be positive");
    if (lx <= 0.0 || ly <= 0.0 || lz <= 0.0)
        throw std::invalid_argument("Domain lengths must be positive");
}

void MACGrid::classifyCells() {
    parallel_for(0, nk_, [&](int k) {
        for (int j = 0; j < nj_; ++j) {
            for (int i = 0; i < ni_; ++i) {
                if (i == 0 || i == ni_ - 1 ||
                    j == 0 ||
                    k == 0 || k == nk_ - 1) {
                    cell_type(i, j, k) = CellType::Solid;
                } else if (phi(i, j, k) <= 0.0) {
                    cell_type(i, j, k) = CellType::Fluid;
                } else {
                    cell_type(i, j, k) = CellType::Air;
                }
            }
        }
    });
    // Top boundary (j == nj-1) is open (air), not solid — free surface can reach it
}

void MACGrid::enforceBoundaryConditions() {
    // No-slip at solid walls: velocity components on solid faces are zero.
    // X-faces: u = 0 at i=0 and i=ni
    for (int k = 0; k < nk_; ++k)
        for (int j = 0; j < nj_; ++j) {
            u(0, j, k) = 0.0;
            u(ni_, j, k) = 0.0;
        }

    // Y-faces: v = 0 at j=0 (floor). Top is open.
    for (int k = 0; k < nk_; ++k)
        for (int i = 0; i < ni_; ++i) {
            v(i, 0, k) = 0.0;
        }

    // Z-faces: w = 0 at k=0 and k=nk
    for (int j = 0; j < nj_; ++j)
        for (int i = 0; i < ni_; ++i) {
            w(i, j, 0) = 0.0;
            w(i, j, nk_) = 0.0;
        }

    // Free-slip on walls: tangential velocity component at wall is mirrored
    // For left/right walls (i=0, i=ni-1): v and w are tangential
    for (int k = 0; k < nk_; ++k) {
        for (int j = 0; j < nj_ + 1; ++j) {
            // Left wall: reflect v
            if (ni_ > 1) {
                v(0, j, k) = v(1, j, k);
                v(ni_ - 1, j, k) = v(ni_ - 2, j, k);
            }
        }
        for (int j = 0; j < nj_; ++j) {
            if (ni_ > 1) {
                w(0, j, k) = w(1, j, k);
                w(ni_ - 1, j, k) = w(ni_ - 2, j, k);
            }
        }
    }

    // Front/back walls (k=0, k=nk-1): u and v are tangential
    for (int j = 0; j < nj_; ++j) {
        for (int i = 0; i < ni_ + 1; ++i) {
            if (nk_ > 1) {
                u(i, j, 0) = u(i, j, 1);
                u(i, j, nk_ - 1) = u(i, j, nk_ - 2);
            }
        }
    }
    for (int j = 0; j < nj_ + 1; ++j) {
        for (int i = 0; i < ni_; ++i) {
            if (nk_ > 1) {
                v(i, j, 0) = v(i, j, 1);
                v(i, j, nk_ - 1) = v(i, j, nk_ - 2);
            }
        }
    }

    // Floor (j=0): u and w are tangential — free-slip
    for (int k = 0; k < nk_; ++k) {
        for (int i = 0; i < ni_ + 1; ++i) {
            if (nj_ > 1) u(i, 0, k) = u(i, 1, k);
        }
    }
    for (int k = 0; k < nk_ + 1; ++k) {
        for (int i = 0; i < ni_; ++i) {
            if (nj_ > 1) w(i, 0, k) = w(i, 1, k);
        }
    }
}

double MACGrid::divergence(int i, int j, int k) const {
    return (u(i + 1, j, k) - u(i, j, k)) / dx_
         + (v(i, j + 1, k) - v(i, j, k)) / dy_
         + (w(i, j, k + 1) - w(i, j, k)) / dz_;
}

double MACGrid::maxDivergence() const {
    return parallel_reduce(1, nk_ - 1, 0.0,
        [&](int k, double& local) {
            for (int j = 1; j < nj_ - 1; ++j)
                for (int i = 1; i < ni_ - 1; ++i)
                    if (cell_type(i, j, k) == CellType::Fluid)
                        local = std::max(local, std::abs(divergence(i, j, k)));
        },
        [](double a, double b) { return std::max(a, b); });
}

double MACGrid::interpolateComponent(const Array3D<double>& field,
                                      const glm::dvec3& pos,
                                      const glm::dvec3& offset) const {
    // Convert position to grid coordinates relative to the field's stagger offset
    double gx = (pos.x / dx_) - offset.x;
    double gy = (pos.y / dy_) - offset.y;
    double gz = (pos.z / dz_) - offset.z;

    // Clamp to valid range
    int ni = field.ni(), nj = field.nj(), nk = field.nk();
    gx = std::clamp(gx, 0.0, static_cast<double>(ni - 1));
    gy = std::clamp(gy, 0.0, static_cast<double>(nj - 1));
    gz = std::clamp(gz, 0.0, static_cast<double>(nk - 1));

    int i0 = std::min(static_cast<int>(gx), ni - 2);
    int j0 = std::min(static_cast<int>(gy), nj - 2);
    int k0 = std::min(static_cast<int>(gz), nk - 2);

    double fx = gx - i0;
    double fy = gy - j0;
    double fz = gz - k0;

    // Trilinear interpolation
    return (1 - fx) * (1 - fy) * (1 - fz) * field(i0, j0, k0)
         + fx       * (1 - fy) * (1 - fz) * field(i0 + 1, j0, k0)
         + (1 - fx) * fy       * (1 - fz) * field(i0, j0 + 1, k0)
         + fx       * fy       * (1 - fz) * field(i0 + 1, j0 + 1, k0)
         + (1 - fx) * (1 - fy) * fz       * field(i0, j0, k0 + 1)
         + fx       * (1 - fy) * fz       * field(i0 + 1, j0, k0 + 1)
         + (1 - fx) * fy       * fz       * field(i0, j0 + 1, k0 + 1)
         + fx       * fy       * fz       * field(i0 + 1, j0 + 1, k0 + 1);
}

glm::dvec3 MACGrid::interpolateVelocity(const glm::dvec3& pos) const {
    // u lives at (i, j+0.5, k+0.5) → offset (0, 0.5, 0.5)
    double iu = interpolateComponent(u, pos, {0.0, 0.5, 0.5});
    // v lives at (i+0.5, j, k+0.5) → offset (0.5, 0, 0.5)
    double iv = interpolateComponent(v, pos, {0.5, 0.0, 0.5});
    // w lives at (i+0.5, j+0.5, k) → offset (0.5, 0.5, 0)
    double iw = interpolateComponent(w, pos, {0.5, 0.5, 0.0});
    return {iu, iv, iw};
}

double MACGrid::interpolateScalar(const Array3D<double>& field,
                                   const glm::dvec3& pos) const {
    // Cell-centered fields live at (i+0.5, j+0.5, k+0.5) → offset (0.5, 0.5, 0.5)
    return interpolateComponent(field, pos, {0.5, 0.5, 0.5});
}

} // namespace sloshing
