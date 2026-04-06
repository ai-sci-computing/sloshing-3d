/**
 * @file clsvof.cpp
 * @brief CLSVOF coupling: VOF correction, level set reinitialization.
 */

#include "sloshing/clsvof.h"
#include "sloshing/advection.h"
#include "sloshing/fluid_utils.h"
#include "sloshing/parallel.h"
#include <cmath>
#include <algorithm>

namespace sloshing {

void initializeFlatSurface(MACGrid& grid, double water_level) {
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();
    double dy = grid.dy();

    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                double y_center = (j + 0.5) * dy;
                grid.phi(i, j, k) = y_center - water_level;

                double y_bottom = j * dy;
                double y_top = (j + 1) * dy;

                if (y_top <= water_level) {
                    grid.vof(i, j, k) = 1.0;
                } else if (y_bottom >= water_level) {
                    grid.vof(i, j, k) = 0.0;
                } else {
                    grid.vof(i, j, k) = (water_level - y_bottom) / dy;
                }
            }
        }
    });

    grid.classifyCells();
}

double computeTotalVolume(const MACGrid& grid) {
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();
    double cell_volume = grid.dx() * grid.dy() * grid.dz();
    return parallel_reduce(0, nk, 0.0,
        [&](int k, double& local) {
            for (int j = 0; j < nj; ++j)
                for (int i = 0; i < ni; ++i)
                    local += grid.vof(i, j, k) * cell_volume;
        },
        [](double a, double b) { return a + b; });
}

// ============================================================================
// Level set reinitialization (PDE-based, Sussman et al. 1994)
// ============================================================================

/// @brief Sign function with smoothing near zero.
static double smoothSign(double phi, double dx) {
    return phi / std::sqrt(phi * phi + dx * dx);
}

/// @brief Godunov upwind scheme for |grad phi| in the reinitialization equation.
static double godunovGradMag(double phi_c,
                              double phi_xm, double phi_xp,
                              double phi_ym, double phi_yp,
                              double phi_zm, double phi_zp,
                              double dx, double dy, double dz,
                              double sign_phi0) {
    // One-sided differences
    double dxm = (phi_c - phi_xm) / dx;
    double dxp = (phi_xp - phi_c) / dx;
    double dym = (phi_c - phi_ym) / dy;
    double dyp = (phi_yp - phi_c) / dy;
    double dzm = (phi_c - phi_zm) / dz;
    double dzp = (phi_zp - phi_c) / dz;

    double grad_sq = 0.0;

    if (sign_phi0 > 0) {
        // Use Godunov's scheme for expanding fronts
        double ax = std::max(dxm, 0.0), bx = std::min(dxp, 0.0);
        double ay = std::max(dym, 0.0), by = std::min(dyp, 0.0);
        double az = std::max(dzm, 0.0), bz = std::min(dzp, 0.0);
        grad_sq = std::max(ax * ax, bx * bx)
                + std::max(ay * ay, by * by)
                + std::max(az * az, bz * bz);
    } else {
        double ax = std::min(dxm, 0.0), bx = std::max(dxp, 0.0);
        double ay = std::min(dym, 0.0), by = std::max(dyp, 0.0);
        double az = std::min(dzm, 0.0), bz = std::max(dzp, 0.0);
        grad_sq = std::max(ax * ax, bx * bx)
                + std::max(ay * ay, by * by)
                + std::max(az * az, bz * bz);
    }

    return std::sqrt(grad_sq);
}

void reinitializeLevelSet(Array3D<double>& phi,
                           double dx, double dy, double dz,
                           int iterations, double dt_factor) {
    int ni = phi.ni(), nj = phi.nj(), nk = phi.nk();
    double min_dx = std::min({dx, dy, dz});
    double dtau = dt_factor * min_dx; // Pseudo-timestep

    // Freeze band: cells within this distance of the zero-level set are
    // not modified. This prevents the reinitialization PDE from drifting
    // the interface position while still smoothing phi farther away.
    double freeze_band = 0.75 * min_dx;

    // Store initial phi for sign function and freeze check
    Array3D<double> phi0 = phi;
    Array3D<double> phi_new(ni, nj, nk, 0.0);

    for (int iter = 0; iter < iterations; ++iter) {
        parallel_for(0, nk, [&](int k) {
            for (int j = 0; j < nj; ++j) {
                for (int i = 0; i < ni; ++i) {
                    if (std::abs(phi0(i, j, k)) < freeze_band) {
                        phi_new(i, j, k) = phi(i, j, k);
                        continue;
                    }

                    double s = smoothSign(phi0(i, j, k), min_dx);

                    double pxm = (i > 0)      ? phi(i - 1, j, k) : phi(i, j, k);
                    double pxp = (i < ni - 1)  ? phi(i + 1, j, k) : phi(i, j, k);
                    double pym = (j > 0)      ? phi(i, j - 1, k) : phi(i, j, k);
                    double pyp = (j < nj - 1)  ? phi(i, j + 1, k) : phi(i, j, k);
                    double pzm = (k > 0)      ? phi(i, j, k - 1) : phi(i, j, k);
                    double pzp = (k < nk - 1)  ? phi(i, j, k + 1) : phi(i, j, k);

                    double grad_mag = godunovGradMag(
                        phi(i, j, k), pxm, pxp, pym, pyp, pzm, pzp,
                        dx, dy, dz, s);

                    phi_new(i, j, k) = phi(i, j, k) - dtau * s * (grad_mag - 1.0);
                }
            }
        });

        phi.swap(phi_new);
    }
}

// ============================================================================
// VOF → Level Set correction
// ============================================================================

void correctLevelSetWithVOF(MACGrid& grid, double blend) {
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    double h = std::min({dx, dy, dz});

    // Work from a snapshot of phi so that normal computation is not
    // contaminated by already-modified neighbors (read-write hazard).
    Array3D<double> phi_snapshot = grid.phi;

    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                if (grid.cell_type(i, j, k) == CellType::Solid)
                    continue;

                double f = grid.vof(i, j, k);

                if (f <= 1e-10) {
                    if (grid.phi(i, j, k) <= 0.0)
                        grid.phi(i, j, k) = 0.5 * h;
                    continue;
                }
                if (f >= 1.0 - 1e-10) {
                    if (grid.phi(i, j, k) >= 0.0)
                        grid.phi(i, j, k) = -0.5 * h;
                    continue;
                }

                glm::dvec3 normal = computeNormal(phi_snapshot, i, j, k, dx, dy, dz);
                double d = plicFindPlaneConstant(normal, f);

                double phi_at_center = glm::dot(normal, glm::dvec3(0.5, 0.5, 0.5)) - d;
                double phi_new_val = phi_at_center * h;

                double phi_old = grid.phi(i, j, k);
                double discrepancy = std::abs(phi_new_val - phi_old);
                if (discrepancy > 0.1 * h) {
                    grid.phi(i, j, k) = (1.0 - blend) * phi_old + blend * phi_new_val;
                }
            }
        }
    });
}

// ============================================================================
// LS→VOF correction: reconstruct VOF from level set
// ============================================================================

void correctVOFWithLevelSet(MACGrid& grid) {
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    double h = std::min({dx, dy, dz});

    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                if (grid.cell_type(i, j, k) == CellType::Solid)
                    continue;

                double phi_val = grid.phi(i, j, k);

                if (phi_val < -h) {
                    grid.vof(i, j, k) = 1.0;
                    continue;
                }
                if (phi_val > h) {
                    grid.vof(i, j, k) = 0.0;
                    continue;
                }

                glm::dvec3 normal = computeNormal(grid.phi, i, j, k, dx, dy, dz);
                double d = glm::dot(normal, glm::dvec3(0.5, 0.5, 0.5)) - phi_val / h;
                double f = plicVolumeBelowPlane(normal, d);
                grid.vof(i, j, k) = std::clamp(f, 0.0, 1.0);
            }
        }
    });
}

// ============================================================================
// Air-region phi cleanup
// ============================================================================

/// @brief Clamp phi to positive in pure-air cells far from the interface.
/// This eliminates floating debris (small negative phi pockets in air)
/// that cause marching cubes to generate stray triangles.
/// Only acts on cells that are clearly deep in the air (no fluid or partial
/// cells within a 2-cell neighborhood) to avoid perturbing near-interface phi.
static void cleanAirPhi(MACGrid& grid) {
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();
    double h = std::min({grid.dx(), grid.dy(), grid.dz()});

    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                if (grid.vof(i, j, k) > 1e-6) continue;
                if (grid.phi(i, j, k) >= 0.0) continue;

                bool near_fluid = false;
                for (int dk = -2; dk <= 2 && !near_fluid; ++dk)
                    for (int dj_off = -2; dj_off <= 2 && !near_fluid; ++dj_off)
                        for (int di = -2; di <= 2 && !near_fluid; ++di) {
                            int ii = i+di, jj = j+dj_off, kk = k+dk;
                            if (ii < 0 || ii >= ni || jj < 0 || jj >= nj || kk < 0 || kk >= nk)
                                continue;
                            if (grid.vof(ii, jj, kk) > 0.01)
                                near_fluid = true;
                        }

                if (!near_fluid) {
                    grid.phi(i, j, k) = h;
                }
            }
        }
    });
}

// ============================================================================
// Mild Laplacian smoothing near the interface
// ============================================================================

/// @brief Apply one pass of Laplacian smoothing to phi in a narrow band
/// around the zero level set. This damps high-frequency bumps without
/// moving the interface significantly.
static void smoothPhiNearInterface(Array3D<double>& phi,
                                     double dx, double dy, double dz,
                                     double band_width, double alpha) {
    int ni = phi.ni(), nj = phi.nj(), nk = phi.nk();
    Array3D<double> phi_smooth = phi;

    parallel_for(1, nk - 1, [&](int k) {
        for (int j = 1; j < nj - 1; ++j) {
            for (int i = 1; i < ni - 1; ++i) {
                if (std::abs(phi(i, j, k)) > band_width) continue;

                double lap = (phi(i+1,j,k) + phi(i-1,j,k) - 2.0*phi(i,j,k)) / (dx*dx)
                           + (phi(i,j+1,k) + phi(i,j-1,k) - 2.0*phi(i,j,k)) / (dy*dy)
                           + (phi(i,j,k+1) + phi(i,j,k-1) - 2.0*phi(i,j,k)) / (dz*dz);

                double min_dx2 = std::min({dx*dx, dy*dy, dz*dz});
                phi_smooth(i, j, k) = phi(i, j, k) + alpha * min_dx2 * lap;
            }
        }
    });

    phi.swap(phi_smooth);
}

// ============================================================================
// Full CLSVOF coupling step
// ============================================================================

void clsvofCoupling(MACGrid& grid, const CLSVOFConfig& config) {
    // Step 1: Volume correction (direction depends on mode)
    if (config.volume_correction) {
        switch (config.correction_mode) {
            case CorrectionMode::VOF_to_LS:
                correctLevelSetWithVOF(grid, config.correction_blend);
                break;
            case CorrectionMode::LS_to_VOF:
                // LS→VOF: reconstruct VOF from the (less diffusive) level set.
                // Applied after reinitialization below so phi is a clean SDF.
                break;
            case CorrectionMode::None:
                break;
        }
    }

    // Step 1b: Smooth high-frequency noise from VOF→LS correction
    if (config.volume_correction && config.correction_mode == CorrectionMode::VOF_to_LS) {
        double h = std::min({grid.dx(), grid.dy(), grid.dz()});
        smoothPhiNearInterface(grid.phi, grid.dx(), grid.dy(), grid.dz(),
                                1.5 * h, 0.05);
    }

    // Step 2: Clean up spurious negative phi in pure-air regions
    cleanAirPhi(grid);

    // Step 3: Reinitialize level set to signed distance function
    reinitializeLevelSet(grid.phi,
                          grid.dx(), grid.dy(), grid.dz(),
                          config.reinit_iterations,
                          config.reinit_dt_factor);

    // Step 4: LS→VOF correction (after reinit so phi is a proper SDF)
    if (config.volume_correction && config.correction_mode == CorrectionMode::LS_to_VOF) {
        correctVOFWithLevelSet(grid);
    }

    // Step 5: Update cell classification
    grid.classifyCells();
}

} // namespace sloshing
