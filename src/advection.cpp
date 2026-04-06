/**
 * @file advection.cpp
 * @brief Implementation of advection schemes: semi-Lagrangian and PLIC-based VOF.
 */

#include "sloshing/advection.h"
#include "sloshing/fluid_utils.h"
#include "sloshing/parallel.h"
#include <cmath>
#include <algorithm>
#include <array>
#include <numeric>

namespace sloshing {

// ============================================================================
// Interface normal computation
// ============================================================================

glm::dvec3 computeNormal(const Array3D<double>& phi,
                          int i, int j, int k,
                          double dx, double dy, double dz) {
    int ni = phi.ni(), nj = phi.nj(), nk = phi.nk();

    // Central differences with one-sided fallback at boundaries
    double dpdx, dpdy, dpdz;

    if (i == 0)          dpdx = (phi(1, j, k) - phi(0, j, k)) / dx;
    else if (i == ni - 1) dpdx = (phi(ni - 1, j, k) - phi(ni - 2, j, k)) / dx;
    else                  dpdx = (phi(i + 1, j, k) - phi(i - 1, j, k)) / (2.0 * dx);

    if (j == 0)          dpdy = (phi(i, 1, k) - phi(i, 0, k)) / dy;
    else if (j == nj - 1) dpdy = (phi(i, nj - 1, k) - phi(i, nj - 2, k)) / dy;
    else                  dpdy = (phi(i, j + 1, k) - phi(i, j - 1, k)) / (2.0 * dy);

    if (k == 0)          dpdz = (phi(i, j, 1) - phi(i, j, 0)) / dz;
    else if (k == nk - 1) dpdz = (phi(i, j, nk - 1) - phi(i, j, nk - 2)) / dz;
    else                  dpdz = (phi(i, j, k + 1) - phi(i, j, k - 1)) / (2.0 * dz);

    glm::dvec3 n(dpdx, dpdy, dpdz);
    double len = glm::length(n);
    if (len < 1e-12) return glm::dvec3(0.0, 1.0, 0.0);
    return n / len;
}

// ============================================================================
// PLIC volume computation (Scardovelli & Zaleski, JCP 2000)
// ============================================================================

/**
 * @brief Volume of unit cube below plane n1*x + n2*y + n3*z = alpha.
 *
 * Assumes n1 <= n2 <= n3 > 0 and alpha in [0, n1+n2+n3].
 * Uses the analytical piecewise formula from Scardovelli & Zaleski (2000).
 */
static double plicVolumeForwardSorted(double n1, double n2, double n3, double alpha) {
    if (alpha <= 0.0) return 0.0;
    double nsum = n1 + n2 + n3;
    if (alpha >= nsum) return 1.0;

    double denom = 6.0 * n1 * n2 * n3;
    if (denom < 1e-30) {
        // Degenerate: one or two components are ~0
        // Handle 2D-like cases
        if (n1 < 1e-15 && n2 < 1e-15) {
            // Effectively 1D: n3 * z = alpha
            return std::clamp(alpha / n3, 0.0, 1.0);
        }
        if (n1 < 1e-15) {
            // 2D case: n2*y + n3*z = alpha
            double nsum2 = n2 + n3;
            if (alpha >= nsum2) return 1.0;
            if (alpha < n2) return alpha * alpha / (2.0 * n2 * n3);
            else return (2.0 * alpha - n2) / (2.0 * n3);
        }
        return 0.5;
    }

    double vol;
    double n12 = n1 + n2;

    if (alpha < n1) {
        vol = alpha * alpha * alpha / denom;
    } else if (alpha < n2) {
        vol = (3.0 * alpha * (alpha - n1) + n1 * n1) / (6.0 * n2 * n3);
    } else if (alpha < std::min(n12, n3)) {
        // Both n1 and n2 crossed
        double a1 = alpha - n1;
        double a2 = alpha - n2;
        vol = (alpha * alpha * alpha
             - n1 * n1 * n1
             - n2 * n2 * n2) / denom;
        // Correct formula:
        vol = (alpha * alpha * (3.0 * n12 - alpha)
             + n1 * n1 * (n1 - 3.0 * alpha)
             + n2 * n2 * (n2 - 3.0 * alpha)) / denom;
    } else if (alpha < n3) {
        // n1+n2 <= alpha < n3 case (n3 is the dominant component).
        // The inclusion-exclusion formula simplifies to a linear function:
        // V = [alpha^3 - (alpha-n1)^3 - (alpha-n2)^3 + (alpha-n1-n2)^3] / (6*n1*n2*n3)
        //   = 3*n1*n2*(2*alpha - n1 - n2) / (6*n1*n2*n3)
        //   = (2*alpha - n1 - n2) / (2*n3)
        vol = (2.0 * alpha - n1 - n2) / (2.0 * n3);
    } else {
        // alpha >= n3
        if (alpha < n12) {
            // n3 <= alpha < n12
            vol = (alpha * alpha * (3.0 * n12 - alpha)
                 + n1 * n1 * (n1 - 3.0 * alpha)
                 + n2 * n2 * (n2 - 3.0 * alpha)
                 - (alpha - n3) * (alpha - n3) * (alpha - n3)) / denom;
        } else {
            // alpha >= max(n12, n3)
            double b1 = nsum - alpha;
            if (b1 <= 0.0) return 1.0;
            vol = 1.0 - b1 * b1 * b1 / denom;
        }
    }

    return std::clamp(vol, 0.0, 1.0);
}

double plicVolumeBelowPlane(const glm::dvec3& normal, double d) {
    double nx = std::abs(normal.x);
    double ny = std::abs(normal.y);
    double nz = std::abs(normal.z);

    // Adjust d for the signs: plane n·x = d in the unit cube [0,1]³
    // When a component of n is negative, reflect: x_i → 1 - x_i
    // This adds |n_i| to d for each negative component
    double alpha = d;
    if (normal.x < 0) alpha += nx;
    if (normal.y < 0) alpha += ny;
    if (normal.z < 0) alpha += nz;

    // Sort components: n1 <= n2 <= n3
    std::array<double, 3> n = {nx, ny, nz};
    std::sort(n.begin(), n.end());

    return plicVolumeForwardSorted(n[0], n[1], n[2], alpha);
}

double plicFindPlaneConstant(const glm::dvec3& normal, double vof_fraction) {
    if (vof_fraction <= 1e-10) return -10.0;
    if (vof_fraction >= 1.0 - 1e-10) {
        return std::abs(normal.x) + std::abs(normal.y) + std::abs(normal.z) + 10.0;
    }

    // Bisection to find d such that volume(d) = vof_fraction
    double nsum = std::abs(normal.x) + std::abs(normal.y) + std::abs(normal.z);
    double lo = -nsum - 1.0;
    double hi = 2.0 * nsum + 1.0;

    for (int iter = 0; iter < 60; ++iter) {
        double mid = 0.5 * (lo + hi);
        double vol = plicVolumeBelowPlane(normal, mid);
        if (vol < vof_fraction) lo = mid;
        else hi = mid;
        if (hi - lo < 1e-14) break;
    }
    return 0.5 * (lo + hi);
}

// ============================================================================
// Semi-Lagrangian level set advection with BFECC error correction
// ============================================================================

/// @brief Core semi-Lagrangian advection of a scalar field (no error correction).
static void semiLagrangianAdvect(const MACGrid& grid, const Array3D<double>& field_in,
                                   Array3D<double>& field_out, double dt) {
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();

    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                glm::dvec3 pos = grid.cellCenter(i, j, k);

                // RK2 backtrace
                glm::dvec3 vel = grid.interpolateVelocity(pos);
                glm::dvec3 mid = pos - 0.5 * dt * vel;
                mid.x = std::clamp(mid.x, 0.0, grid.lx());
                mid.y = std::clamp(mid.y, 0.0, grid.ly());
                mid.z = std::clamp(mid.z, 0.0, grid.lz());

                glm::dvec3 vel_mid = grid.interpolateVelocity(mid);
                glm::dvec3 departure = pos - dt * vel_mid;
                departure.x = std::clamp(departure.x, 0.0, grid.lx());
                departure.y = std::clamp(departure.y, 0.0, grid.ly());
                departure.z = std::clamp(departure.z, 0.0, grid.lz());

                field_out(i, j, k) = grid.interpolateScalar(field_in, departure);
            }
        }
    });
}

void advectLevelSet(const MACGrid& grid, Array3D<double>& phi_new, double dt) {
    // Standard semi-Lagrangian advection (no error correction).
    // BFECC for LS causes large frequency errors by over-sharpening the interface.
    semiLagrangianAdvect(grid, grid.phi, phi_new, dt);
}

// ============================================================================
// Trilinear interpolation helper for staggered velocity components
// ============================================================================

/// @brief Helper to trilinearly interpolate a staggered field at a point.
static double interpStaggered(const Array3D<double>& field,
                               double gx, double gy, double gz) {
    int ni = field.ni(), nj = field.nj(), nk = field.nk();

    gx = std::clamp(gx, 0.0, static_cast<double>(ni - 1));
    gy = std::clamp(gy, 0.0, static_cast<double>(nj - 1));
    gz = std::clamp(gz, 0.0, static_cast<double>(nk - 1));

    int i0 = std::clamp(static_cast<int>(gx), 0, ni - 2);
    int j0 = std::clamp(static_cast<int>(gy), 0, nj - 2);
    int k0 = std::clamp(static_cast<int>(gz), 0, nk - 2);

    double fx = std::clamp(gx - i0, 0.0, 1.0);
    double fy = std::clamp(gy - j0, 0.0, 1.0);
    double fz = std::clamp(gz - k0, 0.0, 1.0);

    return (1 - fx) * (1 - fy) * (1 - fz) * field(i0, j0, k0)
         + fx       * (1 - fy) * (1 - fz) * field(i0 + 1, j0, k0)
         + (1 - fx) * fy       * (1 - fz) * field(i0, j0 + 1, k0)
         + fx       * fy       * (1 - fz) * field(i0 + 1, j0 + 1, k0)
         + (1 - fx) * (1 - fy) * fz       * field(i0, j0, k0 + 1)
         + fx       * (1 - fy) * fz       * field(i0 + 1, j0, k0 + 1)
         + (1 - fx) * fy       * fz       * field(i0, j0 + 1, k0 + 1)
         + fx       * fy       * fz       * field(i0 + 1, j0 + 1, k0 + 1);
}

// ============================================================================
// Semi-Lagrangian velocity advection
// ============================================================================

void advectVelocity(MACGrid& grid, double dt) {
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();

    Array3D<double> u_old = grid.u;
    Array3D<double> v_old = grid.v;
    Array3D<double> w_old = grid.w;

    // Interpolate velocity from snapshot arrays (not the live grid) to avoid
    // data races during parallel advection — the live grid.u/v/w are being
    // overwritten concurrently by other threads.
    auto interpVelSnapshot = [&](const glm::dvec3& pos) -> glm::dvec3 {
        double iu = interpStaggered(u_old,
            pos.x / dx, pos.y / dy - 0.5, pos.z / dz - 0.5);
        double iv = interpStaggered(v_old,
            pos.x / dx - 0.5, pos.y / dy, pos.z / dz - 0.5);
        double iw = interpStaggered(w_old,
            pos.x / dx - 0.5, pos.y / dy - 0.5, pos.z / dz);
        return {iu, iv, iw};
    };

    // RK2 backtrace + clamp
    auto backtrace = [&](glm::dvec3 pos) -> glm::dvec3 {
        glm::dvec3 vel = interpVelSnapshot(pos);
        glm::dvec3 mid = pos - 0.5 * dt * vel;
        mid.x = std::clamp(mid.x, 0.0, grid.lx());
        mid.y = std::clamp(mid.y, 0.0, grid.ly());
        mid.z = std::clamp(mid.z, 0.0, grid.lz());
        glm::dvec3 vel_mid = interpVelSnapshot(mid);
        glm::dvec3 dep = pos - dt * vel_mid;
        dep.x = std::clamp(dep.x, 0.0, grid.lx());
        dep.y = std::clamp(dep.y, 0.0, grid.ly());
        dep.z = std::clamp(dep.z, 0.0, grid.lz());
        return dep;
    };

    // Semi-Lagrangian advect of a staggered component field
    auto slAdvectComponent = [&](const Array3D<double>& src, Array3D<double>& dst,
                                  double ox, double oy, double oz) {
        int fni = dst.ni(), fnj = dst.nj(), fnk = dst.nk();
        parallel_for(0, fnk, [&](int k) {
            for (int j = 0; j < fnj; ++j)
                for (int i = 0; i < fni; ++i) {
                    double px = (i + ox) * dx;
                    double py = (j + oy) * dy;
                    double pz = (k + oz) * dz;
                    glm::dvec3 dep = backtrace({px, py, pz});
                    dst(i, j, k) = interpStaggered(src,
                        dep.x / dx - ox, dep.y / dy - oy, dep.z / dz - oz);
                }
        });
    };

    slAdvectComponent(u_old, grid.u, 0.0, 0.5, 0.5);
    slAdvectComponent(v_old, grid.v, 0.5, 0.0, 0.5);
    slAdvectComponent(w_old, grid.w, 0.5, 0.5, 0.0);
}

// ============================================================================
// VOF advection with upwind directional splitting
// ============================================================================

/// @brief Superbee flux limiter: psi(r) = max(0, min(2r, 1), min(r, 2)).
/// Gives the most anti-diffusive TVD limiter — sharp interfaces with no
/// oscillations. r is the ratio of consecutive gradients (upwind/local).
static double superbee(double r) {
    if (r <= 0.0) return 0.0;
    return std::max(std::min(2.0 * r, 1.0), std::min(r, 2.0));
}

/// @brief Compute face flux with TVD high-resolution scheme.
/// Uses Superbee limiter for 2nd-order accuracy at interfaces while
/// maintaining TVD boundedness. Falls back to 1st-order upwind where
/// the upwind-upwind cell is out of bounds.
///
/// @param vel       Face velocity (positive = flow from lo to hi)
/// @param vof_uu    VOF in the upwind-upwind cell (for gradient ratio)
/// @param vof_donor VOF in the donor (upwind) cell
/// @param vof_acc   VOF in the acceptor (downwind) cell
/// @param dt_over_dx  dt / dx for this sweep direction
/// @param have_uu   True if vof_uu is valid (not at boundary)
/// @return Net flux from lo to hi (positive = rightward)
static double tvdFlux(double vel, double vof_uu, double vof_donor,
                       double vof_acc, double dt_over_dx, bool have_uu) {
    double C = std::abs(vel) * dt_over_dx; // Courant number

    // 1st-order upwind flux
    double flux_lo = C * vof_donor;

    // 2nd-order correction with limiter
    double flux_hi = flux_lo;
    if (have_uu) {
        double dlocal = vof_acc - vof_donor;
        double dupwind = vof_donor - vof_uu;
        if (std::abs(dlocal) > 1e-14) {
            double r = dupwind / dlocal;
            double psi = superbee(r);
            // Lax-Wendroff flux for the high-order part
            double flux_lw = C * (vof_donor + 0.5 * (1.0 - C) * dlocal);
            flux_hi = flux_lo + psi * (flux_lw - flux_lo);
        }
    }

    // Boundedness: can't drain donor below 0 or fill acceptor above 1
    double flux = std::clamp(flux_hi, 0.0, std::min(vof_donor, 1.0 - vof_acc));

    return (vel > 0.0) ? flux : -flux;
}

void advectVOF(const MACGrid& grid, Array3D<double>& vof_new, double dt) {
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();

    // Conservative flux-form VOF advection with directional splitting.
    // Each face flux is computed once and applied symmetrically to both cells.
    // Flux limiting ensures boundedness without mass-destroying clamps.
    // Solid-wall faces have zero flux (no-penetration).

    Array3D<double> vof_tmp1(ni, nj, nk, 0.0);
    Array3D<double> vof_tmp2(ni, nj, nk, 0.0);

    // Copy current VOF
    for (size_t idx = 0; idx < vof_new.size(); ++idx)
        vof_tmp1.data()[idx] = grid.vof.data()[idx];

    // --- X-sweep: flux-form with TVD ---
    for (size_t idx = 0; idx < vof_tmp1.size(); ++idx)
        vof_tmp2.data()[idx] = vof_tmp1.data()[idx];

    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j < nj; ++j)
            for (int i = 0; i < ni - 1; ++i) {
                bool solid_left = (grid.cell_type(i, j, k) == CellType::Solid);
                bool solid_right = (grid.cell_type(i + 1, j, k) == CellType::Solid);
                if (solid_left || solid_right) continue;

                double vel = grid.u(i + 1, j, k);
                if (std::abs(vel) < 1e-14) continue;

                double vof_donor, vof_acc, vof_uu;
                bool have_uu;
                if (vel > 0.0) {
                    vof_donor = vof_tmp1(i, j, k);
                    vof_acc = vof_tmp1(i + 1, j, k);
                    have_uu = (i > 0 && grid.cell_type(i - 1, j, k) != CellType::Solid);
                    vof_uu = have_uu ? vof_tmp1(i - 1, j, k) : 0.0;
                } else {
                    vof_donor = vof_tmp1(i + 1, j, k);
                    vof_acc = vof_tmp1(i, j, k);
                    have_uu = (i + 2 < ni && grid.cell_type(i + 2, j, k) != CellType::Solid);
                    vof_uu = have_uu ? vof_tmp1(i + 2, j, k) : 0.0;
                }

                double flux = tvdFlux(vel, vof_uu, vof_donor, vof_acc,
                                        dt / dx, have_uu);
                vof_tmp2(i, j, k)     -= flux;
                vof_tmp2(i + 1, j, k) += flux;
            }
    });

    // --- Y-sweep: flux-form with TVD ---
    for (size_t idx = 0; idx < vof_tmp1.size(); ++idx)
        vof_tmp1.data()[idx] = vof_tmp2.data()[idx];

    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j < nj - 1; ++j)
            for (int i = 0; i < ni; ++i) {
                bool solid_below = (grid.cell_type(i, j, k) == CellType::Solid);
                bool solid_above = (grid.cell_type(i, j + 1, k) == CellType::Solid);
                if (solid_below || solid_above) continue;

                double vel = grid.v(i, j + 1, k);
                if (std::abs(vel) < 1e-14) continue;

                double vof_donor, vof_acc, vof_uu;
                bool have_uu;
                if (vel > 0.0) {
                    vof_donor = vof_tmp2(i, j, k);
                    vof_acc = vof_tmp2(i, j + 1, k);
                    have_uu = (j > 0 && grid.cell_type(i, j - 1, k) != CellType::Solid);
                    vof_uu = have_uu ? vof_tmp2(i, j - 1, k) : 0.0;
                } else {
                    vof_donor = vof_tmp2(i, j + 1, k);
                    vof_acc = vof_tmp2(i, j, k);
                    have_uu = (j + 2 < nj && grid.cell_type(i, j + 2, k) != CellType::Solid);
                    vof_uu = have_uu ? vof_tmp2(i, j + 2, k) : 0.0;
                }

                double flux = tvdFlux(vel, vof_uu, vof_donor, vof_acc,
                                        dt / dy, have_uu);
                vof_tmp1(i, j, k)     -= flux;
                vof_tmp1(i, j + 1, k) += flux;
            }
    });

    // --- Z-sweep: flux-form with TVD (parallelize over j to avoid k-coupling) ---
    for (size_t idx = 0; idx < vof_new.size(); ++idx)
        vof_new.data()[idx] = vof_tmp1.data()[idx];

    parallel_for(0, nj, [&](int j) {
        for (int i = 0; i < ni; ++i)
            for (int k = 0; k < nk - 1; ++k) {
                bool solid_back = (grid.cell_type(i, j, k) == CellType::Solid);
                bool solid_front = (grid.cell_type(i, j, k + 1) == CellType::Solid);
                if (solid_back || solid_front) continue;

                double vel = grid.w(i, j, k + 1);
                if (std::abs(vel) < 1e-14) continue;

                double vof_donor, vof_acc, vof_uu;
                bool have_uu;
                if (vel > 0.0) {
                    vof_donor = vof_tmp1(i, j, k);
                    vof_acc = vof_tmp1(i, j, k + 1);
                    have_uu = (k > 0 && grid.cell_type(i, j, k - 1) != CellType::Solid);
                    vof_uu = have_uu ? vof_tmp1(i, j, k - 1) : 0.0;
                } else {
                    vof_donor = vof_tmp1(i, j, k + 1);
                    vof_acc = vof_tmp1(i, j, k);
                    have_uu = (k + 2 < nk && grid.cell_type(i, j, k + 2) != CellType::Solid);
                    vof_uu = have_uu ? vof_tmp1(i, j, k + 2) : 0.0;
                }

                double flux = tvdFlux(vel, vof_uu, vof_donor, vof_acc,
                                        dt / dz, have_uu);
                vof_new(i, j, k)     -= flux;
                vof_new(i, j, k + 1) += flux;
            }
    });
}

// ============================================================================
// Velocity extension into air
// ============================================================================

/// @brief Helper: extend a single velocity component outward from valid faces.
static void extendComponent(Array3D<double>& field, Array3D<int>& valid, int layers) {
    int ni = field.ni(), nj = field.nj(), nk = field.nk();

    // Zero out invalid faces before extending
    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j < nj; ++j)
            for (int i = 0; i < ni; ++i)
                if (!valid(i, j, k))
                    field(i, j, k) = 0.0;
    });

    // Layer-by-layer propagation from valid faces into invalid neighbors.
    // Reuse two buffers instead of allocating per layer.
    Array3D<double> field_buf(ni, nj, nk, 0.0);
    Array3D<int> valid_buf(ni, nj, nk, 0);

    for (int layer = 0; layer < layers; ++layer) {
        // Copy current state into buffers
        parallel_for(0, nk, [&](int k) {
            for (int j = 0; j < nj; ++j)
                for (int i = 0; i < ni; ++i) {
                    field_buf(i, j, k) = field(i, j, k);
                    valid_buf(i, j, k) = valid(i, j, k);
                }
        });

        parallel_for(0, nk, [&](int k) {
            for (int j = 0; j < nj; ++j)
                for (int i = 0; i < ni; ++i) {
                    if (valid(i, j, k)) continue;
                    double sum = 0.0;
                    int count = 0;
                    if (i > 0      && valid(i-1, j, k)) { sum += field(i-1, j, k); ++count; }
                    if (i < ni - 1 && valid(i+1, j, k)) { sum += field(i+1, j, k); ++count; }
                    if (j > 0      && valid(i, j-1, k)) { sum += field(i, j-1, k); ++count; }
                    if (j < nj - 1 && valid(i, j+1, k)) { sum += field(i, j+1, k); ++count; }
                    if (k > 0      && valid(i, j, k-1)) { sum += field(i, j, k-1); ++count; }
                    if (k < nk - 1 && valid(i, j, k+1)) { sum += field(i, j, k+1); ++count; }
                    if (count > 0) {
                        field_buf(i, j, k) = sum / count;
                        valid_buf(i, j, k) = 1;
                    }
                }
        });

        field.swap(field_buf);
        valid.swap(valid_buf);
    }

    // Zero any remaining invalid faces
    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j < nj; ++j)
            for (int i = 0; i < ni; ++i)
                if (!valid(i, j, k))
                    field(i, j, k) = 0.0;
    });
}

void extendVelocityIntoAir(MACGrid& grid, int layers) {
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();

    // --- u-component: valid if at least one adjacent cell is fluid ---
    Array3D<int> u_valid(ni + 1, nj, nk, 0);
    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j < nj; ++j)
            for (int i = 0; i <= ni; ++i) {
                bool left  = (i > 0  && grid.cell_type(i - 1, j, k) == CellType::Fluid);
                bool right = (i < ni && grid.cell_type(i, j, k)     == CellType::Fluid);
                u_valid(i, j, k) = (left || right) ? 1 : 0;
            }
    });
    extendComponent(grid.u, u_valid, layers);

    // --- v-component ---
    Array3D<int> v_valid(ni, nj + 1, nk, 0);
    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j <= nj; ++j)
            for (int i = 0; i < ni; ++i) {
                bool below = (j > 0  && grid.cell_type(i, j - 1, k) == CellType::Fluid);
                bool above = (j < nj && grid.cell_type(i, j, k)     == CellType::Fluid);
                v_valid(i, j, k) = (below || above) ? 1 : 0;
            }
    });
    extendComponent(grid.v, v_valid, layers);

    // --- w-component ---
    Array3D<int> w_valid(ni, nj, nk + 1, 0);
    parallel_for(0, nk + 1, [&](int k) {
        for (int j = 0; j < nj; ++j)
            for (int i = 0; i < ni; ++i) {
                bool back  = (k > 0  && grid.cell_type(i, j, k - 1) == CellType::Fluid);
                bool front = (k < nk && grid.cell_type(i, j, k)     == CellType::Fluid);
                w_valid(i, j, k) = (back || front) ? 1 : 0;
            }
    });
    extendComponent(grid.w, w_valid, layers);
}

} // namespace sloshing
