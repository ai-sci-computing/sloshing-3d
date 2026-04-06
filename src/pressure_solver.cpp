/**
 * @file pressure_solver.cpp
 * @brief PCG pressure Poisson solver with standard Laplacian (no ghost fluid).
 */

#include "sloshing/pressure_solver.h"
#include "sloshing/parallel.h"
#include <cmath>
#include <algorithm>

namespace sloshing {

PressureSolver::PressureSolver(const PressureSolverConfig& config)
    : config_(config) {}

void PressureSolver::ensureSize(const MACGrid& grid) {
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();
    if (rhs_.ni() != ni || rhs_.nj() != nj || rhs_.nk() != nk) {
        rhs_ = Array3D<double>(ni, nj, nk, 0.0);
        r_ = Array3D<double>(ni, nj, nk, 0.0);
        z_ = Array3D<double>(ni, nj, nk, 0.0);
        s_ = Array3D<double>(ni, nj, nk, 0.0);
        precon_ = Array3D<double>(ni, nj, nk, 0.0);
        Ap_ = Array3D<double>(ni, nj, nk, 0.0);
        Adiag_ = Array3D<double>(ni, nj, nk, 0.0);
        Ax_ = Array3D<double>(ni, nj, nk, 0.0);
        Ay_ = Array3D<double>(ni, nj, nk, 0.0);
        Az_ = Array3D<double>(ni, nj, nk, 0.0);
    }
}

void PressureSolver::buildLaplacian(const MACGrid& grid, double dt) {
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    double scale = dt / config_.density_liquid;

    Adiag_.fill(0.0);
    Ax_.fill(0.0);
    Ay_.fill(0.0);
    Az_.fill(0.0);

    // Standard Laplacian: fluid-air faces use the same 1/dx² coefficient as
    // fluid-fluid faces (no ghost fluid theta). Air cells are Dirichlet p=0,
    // so there is no off-diagonal entry — only a diagonal contribution.
    // This is consistent with the velocity projection (which also uses 1/dx),
    // ensuring L = div∘grad and thus truly divergence-free velocity after
    // projection. The trade-off is that the pressure BC is placed at the
    // cell-center rather than at the sub-grid interface, but this avoids
    // the L≠div∘grad inconsistency that causes spurious currents.

    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                if (grid.cell_type(i, j, k) != CellType::Fluid) continue;

                // X+ neighbor
                if (i + 1 < ni) {
                    if (grid.cell_type(i + 1, j, k) == CellType::Fluid) {
                        double coeff = scale / (dx * dx);
                        Adiag_(i, j, k) += coeff;
                        Ax_(i, j, k) = -coeff;
                    } else if (grid.cell_type(i + 1, j, k) == CellType::Air) {
                        double coeff = scale / (dx * dx);
                        Adiag_(i, j, k) += coeff;
                    }
                }

                // X- neighbor
                if (i - 1 >= 0) {
                    if (grid.cell_type(i - 1, j, k) == CellType::Fluid) {
                        double coeff = scale / (dx * dx);
                        Adiag_(i, j, k) += coeff;
                    } else if (grid.cell_type(i - 1, j, k) == CellType::Air) {
                        double coeff = scale / (dx * dx);
                        Adiag_(i, j, k) += coeff;
                    }
                }

                // Y+ neighbor
                if (j + 1 < nj) {
                    if (grid.cell_type(i, j + 1, k) == CellType::Fluid) {
                        double coeff = scale / (dy * dy);
                        Adiag_(i, j, k) += coeff;
                        Ay_(i, j, k) = -coeff;
                    } else if (grid.cell_type(i, j + 1, k) == CellType::Air) {
                        double coeff = scale / (dy * dy);
                        Adiag_(i, j, k) += coeff;
                    }
                }

                // Y- neighbor
                if (j - 1 >= 0) {
                    if (grid.cell_type(i, j - 1, k) == CellType::Fluid) {
                        double coeff = scale / (dy * dy);
                        Adiag_(i, j, k) += coeff;
                    } else if (grid.cell_type(i, j - 1, k) == CellType::Air) {
                        double coeff = scale / (dy * dy);
                        Adiag_(i, j, k) += coeff;
                    }
                }

                // Z+ neighbor
                if (k + 1 < nk) {
                    if (grid.cell_type(i, j, k + 1) == CellType::Fluid) {
                        double coeff = scale / (dz * dz);
                        Adiag_(i, j, k) += coeff;
                        Az_(i, j, k) = -coeff;
                    } else if (grid.cell_type(i, j, k + 1) == CellType::Air) {
                        double coeff = scale / (dz * dz);
                        Adiag_(i, j, k) += coeff;
                    }
                }

                // Z- neighbor
                if (k - 1 >= 0) {
                    if (grid.cell_type(i, j, k - 1) == CellType::Fluid) {
                        double coeff = scale / (dz * dz);
                        Adiag_(i, j, k) += coeff;
                    } else if (grid.cell_type(i, j, k - 1) == CellType::Air) {
                        double coeff = scale / (dz * dz);
                        Adiag_(i, j, k) += coeff;
                    }
                }
            }
        }
    });
}

void PressureSolver::buildRHS(const MACGrid& grid, double dt) {
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();

    rhs_.fill(0.0);
    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j < nj; ++j)
            for (int i = 0; i < ni; ++i)
                if (grid.cell_type(i, j, k) == CellType::Fluid)
                    rhs_(i, j, k) = -grid.divergence(i, j, k);
    });
}

void PressureSolver::applyLaplacian(const Array3D<double>& x, Array3D<double>& result,
                                     const MACGrid& grid) const {
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();
    result.fill(0.0);

    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                if (grid.cell_type(i, j, k) != CellType::Fluid) continue;

                double val = Adiag_(i, j, k) * x(i, j, k);

                if (i + 1 < ni && grid.cell_type(i + 1, j, k) == CellType::Fluid)
                    val += Ax_(i, j, k) * x(i + 1, j, k);
                if (i - 1 >= 0 && grid.cell_type(i - 1, j, k) == CellType::Fluid)
                    val += Ax_(i - 1, j, k) * x(i - 1, j, k);

                if (j + 1 < nj && grid.cell_type(i, j + 1, k) == CellType::Fluid)
                    val += Ay_(i, j, k) * x(i, j + 1, k);
                if (j - 1 >= 0 && grid.cell_type(i, j - 1, k) == CellType::Fluid)
                    val += Ay_(i, j - 1, k) * x(i, j - 1, k);

                if (k + 1 < nk && grid.cell_type(i, j, k + 1) == CellType::Fluid)
                    val += Az_(i, j, k) * x(i, j, k + 1);
                if (k - 1 >= 0 && grid.cell_type(i, j, k - 1) == CellType::Fluid)
                    val += Az_(i, j, k - 1) * x(i, j, k - 1);

                result(i, j, k) = val;
            }
        }
    });
}

void PressureSolver::buildPreconditioner(const MACGrid& grid) {
    // Modified Incomplete Cholesky (MIC(0)) preconditioner.
    // Sequential due to wavefront dependency, but converges in ~3x fewer
    // iterations than Jacobi, which more than compensates.
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();
    double tau = 0.97; // MIC tuning parameter
    double safety = 0.25;

    precon_.fill(0.0);

    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                if (grid.cell_type(i, j, k) != CellType::Fluid) continue;

                double e = Adiag_(i, j, k);

                if (i > 0 && grid.cell_type(i - 1, j, k) == CellType::Fluid) {
                    double ax = Ax_(i - 1, j, k) * precon_(i - 1, j, k);
                    e -= ax * ax;
                    if (tau > 0.0) {
                        double ay_im1 = (j + 1 < nj && grid.cell_type(i - 1, j + 1, k) == CellType::Fluid)
                                       ? Ay_(i - 1, j, k) : 0.0;
                        double az_im1 = (k + 1 < nk && grid.cell_type(i - 1, j, k + 1) == CellType::Fluid)
                                       ? Az_(i - 1, j, k) : 0.0;
                        e -= tau * ax * (ay_im1 + az_im1) * precon_(i - 1, j, k);
                    }
                }
                if (j > 0 && grid.cell_type(i, j - 1, k) == CellType::Fluid) {
                    double ay = Ay_(i, j - 1, k) * precon_(i, j - 1, k);
                    e -= ay * ay;
                    if (tau > 0.0) {
                        double ax_jm1 = (i + 1 < ni && grid.cell_type(i + 1, j - 1, k) == CellType::Fluid)
                                       ? Ax_(i, j - 1, k) : 0.0;
                        double az_jm1 = (k + 1 < nk && grid.cell_type(i, j - 1, k + 1) == CellType::Fluid)
                                       ? Az_(i, j - 1, k) : 0.0;
                        e -= tau * ay * (ax_jm1 + az_jm1) * precon_(i, j - 1, k);
                    }
                }
                if (k > 0 && grid.cell_type(i, j, k - 1) == CellType::Fluid) {
                    double az = Az_(i, j, k - 1) * precon_(i, j, k - 1);
                    e -= az * az;
                    if (tau > 0.0) {
                        double ax_km1 = (i + 1 < ni && grid.cell_type(i + 1, j, k - 1) == CellType::Fluid)
                                       ? Ax_(i, j, k - 1) : 0.0;
                        double ay_km1 = (j + 1 < nj && grid.cell_type(i, j + 1, k - 1) == CellType::Fluid)
                                       ? Ay_(i, j, k - 1) : 0.0;
                        e -= tau * az * (ax_km1 + ay_km1) * precon_(i, j, k - 1);
                    }
                }

                if (e < safety * Adiag_(i, j, k))
                    e = Adiag_(i, j, k);

                precon_(i, j, k) = (e > 0.0) ? 1.0 / std::sqrt(e) : 0.0;
            }
        }
    }
}

void PressureSolver::applyPreconditioner(const Array3D<double>& r, Array3D<double>& z,
                                          const MACGrid& grid) const {
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();
    z.fill(0.0);

    // Forward substitution: L * q = r
    // Reuse Ap_ as scratch for q (not in use during preconditioner apply)
    Array3D<double>& q = const_cast<Array3D<double>&>(Ap_);
    q.fill(0.0);

    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                if (grid.cell_type(i, j, k) != CellType::Fluid) continue;

                double t = r(i, j, k);

                if (i > 0 && grid.cell_type(i - 1, j, k) == CellType::Fluid)
                    t -= Ax_(i - 1, j, k) * precon_(i - 1, j, k) * q(i - 1, j, k);
                if (j > 0 && grid.cell_type(i, j - 1, k) == CellType::Fluid)
                    t -= Ay_(i, j - 1, k) * precon_(i, j - 1, k) * q(i, j - 1, k);
                if (k > 0 && grid.cell_type(i, j, k - 1) == CellType::Fluid)
                    t -= Az_(i, j, k - 1) * precon_(i, j, k - 1) * q(i, j, k - 1);

                q(i, j, k) = t * precon_(i, j, k);
            }
        }
    }

    // Backward substitution: L^T * z = q
    for (int k = nk - 1; k >= 0; --k) {
        for (int j = nj - 1; j >= 0; --j) {
            for (int i = ni - 1; i >= 0; --i) {
                if (grid.cell_type(i, j, k) != CellType::Fluid) continue;

                double t = q(i, j, k);

                if (i + 1 < ni && grid.cell_type(i + 1, j, k) == CellType::Fluid)
                    t -= Ax_(i, j, k) * precon_(i, j, k) * z(i + 1, j, k);
                if (j + 1 < nj && grid.cell_type(i, j + 1, k) == CellType::Fluid)
                    t -= Ay_(i, j, k) * precon_(i, j, k) * z(i, j + 1, k);
                if (k + 1 < nk && grid.cell_type(i, j, k + 1) == CellType::Fluid)
                    t -= Az_(i, j, k) * precon_(i, j, k) * z(i, j, k + 1);

                z(i, j, k) = t * precon_(i, j, k);
            }
        }
    }
}

double PressureSolver::dot(const Array3D<double>& a, const Array3D<double>& b,
                            const MACGrid& grid) const {
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();
    return parallel_reduce(0, nk, 0.0,
        [&](int k, double& local) {
            for (int j = 0; j < nj; ++j)
                for (int i = 0; i < ni; ++i)
                    if (grid.cell_type(i, j, k) == CellType::Fluid)
                        local += a(i, j, k) * b(i, j, k);
        },
        [](double a_val, double b_val) { return a_val + b_val; });
}

void PressureSolver::projectVelocity(MACGrid& grid, double dt) {
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    double scale = dt / config_.density_liquid;

    // Standard velocity projection using cell-center pressure difference
    // divided by cell spacing. Consistent with the standard Laplacian
    // (no ghost fluid theta) — both use 1/dx, so the divergence removed
    // by the Laplacian exactly matches the divergence removed here.

    // Update u-velocities
    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 1; i < ni; ++i) {
                bool left_fluid = grid.cell_type(i - 1, j, k) == CellType::Fluid;
                bool right_fluid = grid.cell_type(i, j, k) == CellType::Fluid;
                if (left_fluid || right_fluid) {
                    if (grid.cell_type(i - 1, j, k) == CellType::Solid ||
                        grid.cell_type(i, j, k) == CellType::Solid) {
                        grid.u(i, j, k) = 0.0;
                    } else {
                        double pl = left_fluid ? grid.pressure(i - 1, j, k) : 0.0;
                        double pr = right_fluid ? grid.pressure(i, j, k) : 0.0;
                        grid.u(i, j, k) -= scale * (pr - pl) / dx;
                    }
                }
            }
        }
    });

    // Update v-velocities
    parallel_for(0, nk, [&](int k) {
        for (int j = 1; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                bool below_fluid = grid.cell_type(i, j - 1, k) == CellType::Fluid;
                bool above_fluid = grid.cell_type(i, j, k) == CellType::Fluid;
                if (below_fluid || above_fluid) {
                    if (grid.cell_type(i, j - 1, k) == CellType::Solid ||
                        grid.cell_type(i, j, k) == CellType::Solid) {
                        grid.v(i, j, k) = 0.0;
                    } else {
                        double pb = below_fluid ? grid.pressure(i, j - 1, k) : 0.0;
                        double pt = above_fluid ? grid.pressure(i, j, k) : 0.0;
                        grid.v(i, j, k) -= scale * (pt - pb) / dy;
                    }
                }
            }
        }
    });

    // Update w-velocities
    parallel_for(1, nk, [&](int k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                bool back_fluid = grid.cell_type(i, j, k - 1) == CellType::Fluid;
                bool front_fluid = grid.cell_type(i, j, k) == CellType::Fluid;
                if (back_fluid || front_fluid) {
                    if (grid.cell_type(i, j, k - 1) == CellType::Solid ||
                        grid.cell_type(i, j, k) == CellType::Solid) {
                        grid.w(i, j, k) = 0.0;
                    } else {
                        double pb = back_fluid ? grid.pressure(i, j, k - 1) : 0.0;
                        double pf = front_fluid ? grid.pressure(i, j, k) : 0.0;
                        grid.w(i, j, k) -= scale * (pf - pb) / dz;
                    }
                }
            }
        }
    });
}

int PressureSolver::solve(MACGrid& grid, double dt) {
    ensureSize(grid);

    // Zero out velocity at solid-fluid faces before computing RHS.
    // This ensures the divergence computation is consistent with the
    // Laplacian's Neumann BC assumption at walls.
    int ni = grid.ni(), nj = grid.nj(), nk = grid.nk();
    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j < nj; ++j)
            for (int i = 0; i <= ni; ++i) {
                bool left_solid = (i == 0) || grid.cell_type(i - 1, j, k) == CellType::Solid;
                bool right_solid = (i == ni) || grid.cell_type(i, j, k) == CellType::Solid;
                if (left_solid || right_solid) grid.u(i, j, k) = 0.0;
            }
    });
    parallel_for(0, nk, [&](int k) {
        for (int j = 0; j <= nj; ++j)
            for (int i = 0; i < ni; ++i) {
                bool below_solid = (j == 0) || grid.cell_type(i, j - 1, k) == CellType::Solid;
                bool above_solid = (j == nj) || grid.cell_type(i, j, k) == CellType::Solid;
                if (below_solid || above_solid) grid.v(i, j, k) = 0.0;
            }
    });
    parallel_for(0, nk + 1, [&](int k) {
        for (int j = 0; j < nj; ++j)
            for (int i = 0; i < ni; ++i) {
                bool back_solid = (k == 0) || grid.cell_type(i, j, k - 1) == CellType::Solid;
                bool front_solid = (k == nk) || grid.cell_type(i, j, k) == CellType::Solid;
                if (back_solid || front_solid) grid.w(i, j, k) = 0.0;
            }
    });

    // Step 1: Build the linear system
    buildLaplacian(grid, dt);
    buildRHS(grid, dt);
    buildPreconditioner(grid);

    // Initialize: r = rhs - A*p (start with p=0 or previous pressure as initial guess)
    // We keep pressure from previous step as initial guess
    applyLaplacian(grid.pressure, Ap_, grid);

    parallel_for(0, grid.nk(), [&](int k) {
        for (int j = 0; j < grid.nj(); ++j)
            for (int i = 0; i < grid.ni(); ++i)
                r_(i, j, k) = rhs_(i, j, k) - Ap_(i, j, k);
    });

    // Check if already solved
    double rhs_norm = std::sqrt(dot(rhs_, rhs_, grid));
    if (rhs_norm < 1e-12) {
        last_residual_ = 0.0;
        projectVelocity(grid, dt);
        return 0;
    }

    applyPreconditioner(r_, z_, grid);

    // s = z
    int total = static_cast<int>(s_.size());
    parallel_for(0, total, [&](int idx) {
        s_.data()[idx] = z_.data()[idx];
    });

    double rz = dot(r_, z_, grid);

    // PCG iteration
    int iter;
    for (iter = 0; iter < config_.max_iterations; ++iter) {
        // Ap = A * s
        applyLaplacian(s_, Ap_, grid);
        double sAp = dot(s_, Ap_, grid);

        if (std::abs(sAp) < 1e-30) break; // Breakdown

        double alpha = rz / sAp;

        // p += alpha * s
        // r -= alpha * Ap
        parallel_for(0, grid.nk(), [&](int k) {
            for (int j = 0; j < grid.nj(); ++j)
                for (int i = 0; i < grid.ni(); ++i)
                    if (grid.cell_type(i, j, k) == CellType::Fluid) {
                        grid.pressure(i, j, k) += alpha * s_(i, j, k);
                        r_(i, j, k) -= alpha * Ap_(i, j, k);
                    }
        });

        // Check convergence
        double r_norm = std::sqrt(dot(r_, r_, grid));
        last_residual_ = r_norm / rhs_norm;
        if (last_residual_ < config_.tolerance) {
            ++iter;
            break;
        }

        // z = M^{-1} r
        applyPreconditioner(r_, z_, grid);

        double rz_new = dot(r_, z_, grid);
        double beta = rz_new / (rz + 1e-30);
        rz = rz_new;

        // s = z + beta * s
        parallel_for(0, grid.nk(), [&](int k) {
            for (int j = 0; j < grid.nj(); ++j)
                for (int i = 0; i < grid.ni(); ++i)
                    s_(i, j, k) = z_(i, j, k) + beta * s_(i, j, k);
        });
    }

    // Step 4: Project velocity
    projectVelocity(grid, dt);

    return (last_residual_ < config_.tolerance) ? iter : -iter;
}

} // namespace sloshing
