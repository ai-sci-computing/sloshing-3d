/**
 * @file clsvof.h
 * @brief Coupled Level Set / Volume of Fluid interface tracking.
 *
 * Combines the geometric accuracy of the level set method (smooth normals,
 * easy curvature) with the volume conservation of VOF. After each advection
 * step, the VOF field corrects the level set, and the level set is
 * reinitialized to a signed distance function.
 *
 * Reference: Sussman & Puckett, JCP 162 (2000); Son & Hur, JCP 187 (2003).
 */
#pragma once

#include "sloshing/grid.h"

namespace sloshing {

/**
 * @brief Configuration for the CLSVOF coupling.
 */
/// @brief Direction of the CLSVOF volume correction.
enum class CorrectionMode {
    VOF_to_LS,   ///< Correct LS from VOF (classic CLSVOF). Conservative but diffusive.
    LS_to_VOF,   ///< Reconstruct VOF from LS. Less diffusive, LS-quality dynamics.
    None         ///< No correction. Pure independent advection.
};

struct CLSVOFConfig {
    int reinit_iterations = 1;       ///< Iterations for redistancing
    double reinit_dt_factor = 0.5;   ///< Pseudo-timestep = factor * dx
    double volume_correction = true; ///< Enable volume correction
    double correction_blend = 0.5;   ///< VOF→LS correction strength (0=none, 1=full)
    CorrectionMode correction_mode = CorrectionMode::VOF_to_LS; ///< Correction direction
};

/**
 * @brief Performs one CLSVOF coupling step.
 *
 * Call this after advecting both phi and vof. It:
 * 1. Corrects the level set using VOF volume information
 * 2. Reinitializes phi to a signed distance function
 * 3. Updates cell types
 *
 * @param grid The MAC grid (phi and vof are modified in place).
 * @param config CLSVOF parameters.
 */
void clsvofCoupling(MACGrid& grid, const CLSVOFConfig& config = {});

/**
 * @brief Reinitialize the level set to a signed distance function.
 *
 * Uses the PDE-based reinitialization method (Sussman et al. 1994):
 *   phi_t + sign(phi_0) * (|grad phi| - 1) = 0
 *
 * @param phi Level set field (modified in place).
 * @param dx,dy,dz Grid spacing.
 * @param iterations Number of pseudo-timesteps.
 * @param dt_factor Pseudo-timestep = factor * min(dx,dy,dz).
 */
void reinitializeLevelSet(Array3D<double>& phi,
                           double dx, double dy, double dz,
                           int iterations = 5, double dt_factor = 0.5);

/**
 * @brief Correct the level set using VOF volume fractions.
 *
 * For each interface cell, adjusts phi so that the volume implied by
 * the level set matches the VOF fraction. Uses the PLIC normal from
 * the level set and shifts the plane constant.
 *
 * @param grid The MAC grid (phi is modified based on vof).
 * @param blend Blending factor in [0,1] controlling correction strength.
 */
void correctLevelSetWithVOF(MACGrid& grid, double blend = 0.5);

/**
 * @brief Reconstruct VOF from the level set field.
 *
 * For each cell, computes the volume fraction implied by the current
 * phi using the level set normal and signed distance. This is the
 * reverse of correctLevelSetWithVOF: instead of degrading LS to match
 * a diffused VOF, it sharpens VOF to match the less-diffusive LS.
 *
 * @param grid The MAC grid (vof is modified based on phi).
 */
void correctVOFWithLevelSet(MACGrid& grid);

/**
 * @brief Compute total fluid volume from the VOF field.
 *
 * @param grid The MAC grid.
 * @return Total fluid volume in physical units.
 */
double computeTotalVolume(const MACGrid& grid);

/**
 * @brief Initialize level set and VOF for a flat water surface.
 *
 * Sets phi as a signed distance to the plane y = water_level,
 * and vof as the corresponding volume fractions.
 *
 * @param grid The MAC grid (phi and vof are set).
 * @param water_level Height of the water surface (meters).
 */
void initializeFlatSurface(MACGrid& grid, double water_level);

} // namespace sloshing
