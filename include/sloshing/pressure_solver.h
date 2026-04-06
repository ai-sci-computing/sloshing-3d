/**
 * @file pressure_solver.h
 * @brief Preconditioned Conjugate Gradient solver for the pressure Poisson equation.
 *
 * Solves the variable-coefficient Poisson equation arising from the incompressibility
 * constraint ∇·u = 0. Uses the ghost fluid method at the liquid-air interface to
 * handle the pressure jump condition (p_air = 0 at the free surface).
 *
 * The linear system is: A * p = rhs, where
 *   rhs_ij = -(1/dt) * div(u*)   (divergence of the intermediate velocity)
 *   A is the discrete Laplacian with variable coefficients at the free surface.
 *
 * Preconditioning: Modified Incomplete Cholesky (MIC(0)) for fast convergence.
 */
#pragma once

#include "sloshing/grid.h"

namespace sloshing {

/**
 * @brief Configuration for the pressure solver.
 */
struct PressureSolverConfig {
    int max_iterations = 500;      ///< Maximum PCG iterations
    double tolerance = 1e-6;       ///< Convergence tolerance (relative residual)
    double density_liquid = 1000.0; ///< Liquid density (kg/m³)
    double density_air = 1.0;       ///< Air density (kg/m³)
};

/**
 * @brief Solves the pressure Poisson equation using PCG with ghost fluid BCs.
 */
class PressureSolver {
public:
    explicit PressureSolver(const PressureSolverConfig& config = {});

    /**
     * @brief Solve for pressure and project velocity to be divergence-free.
     *
     * Steps:
     * 1. Compute RHS from velocity divergence
     * 2. Build variable-coefficient Laplacian (ghost fluid at interface)
     * 3. Solve via PCG with MIC(0) preconditioner
     * 4. Subtract pressure gradient from velocity
     *
     * @param grid The MAC grid (modified in place: pressure and velocity updated).
     * @param dt Time step size.
     * @return Number of iterations used (negative if not converged).
     */
    int solve(MACGrid& grid, double dt);

    /// @brief Get the residual after the last solve.
    double lastResidual() const { return last_residual_; }

private:
    PressureSolverConfig config_;
    double last_residual_ = 0.0;

    // Working arrays (allocated once, reused)
    Array3D<double> rhs_;       ///< Right-hand side
    Array3D<double> r_;         ///< Residual
    Array3D<double> z_;         ///< Preconditioned residual
    Array3D<double> s_;         ///< Search direction
    Array3D<double> precon_;    ///< Preconditioner (diagonal)
    Array3D<double> Ap_;        ///< Laplacian * search direction

    // Laplacian coefficients per cell (6 neighbors + diagonal)
    Array3D<double> Adiag_;     ///< Diagonal of A
    Array3D<double> Ax_;        ///< Coefficient for (i+1,j,k)
    Array3D<double> Ay_;        ///< Coefficient for (i,j+1,k)
    Array3D<double> Az_;        ///< Coefficient for (i,j,k+1)

    /// @brief Build the Laplacian matrix coefficients using ghost fluid method.
    void buildLaplacian(const MACGrid& grid, double dt);

    /// @brief Build the MIC(0) preconditioner.
    void buildPreconditioner(const MACGrid& grid);

    /// @brief Compute RHS = -(1/dt) * div(u*).
    void buildRHS(const MACGrid& grid, double dt);

    /// @brief Apply the Laplacian: result = A * x.
    void applyLaplacian(const Array3D<double>& x, Array3D<double>& result,
                        const MACGrid& grid) const;

    /// @brief Apply MIC(0) preconditioner: z = M^{-1} r.
    void applyPreconditioner(const Array3D<double>& r, Array3D<double>& z,
                             const MACGrid& grid) const;

    /// @brief Subtract pressure gradient from velocity to enforce div-free.
    void projectVelocity(MACGrid& grid, double dt);

    /// @brief Dot product over fluid cells.
    double dot(const Array3D<double>& a, const Array3D<double>& b,
               const MACGrid& grid) const;

    /// @brief Ensure working arrays match grid size.
    void ensureSize(const MACGrid& grid);
};

} // namespace sloshing
