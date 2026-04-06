/**
 * @file fluid_utils.h
 * @brief Utility functions for the fluid solver.
 */
#pragma once

#include <cmath>
#include <algorithm>

namespace sloshing {

/**
 * @brief Compute the ghost fluid interface fraction between two cells.
 *
 * Given level set values phi_a (fluid cell, phi < 0) and phi_b (air cell, phi > 0),
 * returns the fraction theta in (0,1] representing how far the interface is
 * from cell a toward cell b. Used for the ghost fluid pressure BC.
 *
 * @param phi_fluid Level set value in the fluid cell (should be <= 0).
 * @param phi_air Level set value in the air cell (should be > 0).
 * @return Fraction theta in (0, 1].
 */
inline double ghostFluidTheta(double phi_fluid, double phi_air) {
    // theta = |phi_fluid| / (|phi_fluid| + |phi_air|)
    double af = std::abs(phi_fluid);
    double aa = std::abs(phi_air);
    double denom = af + aa;
    if (denom < 1e-12) return 0.5;
    return std::clamp(af / denom, 0.01, 1.0);
}

/**
 * @brief Smooth Heaviside function for interface smearing.
 * @param phi Level set value.
 * @param epsilon Smearing width (typically 1.5 * dx).
 * @return Value in [0, 1]: 0 in fluid, 1 in air, smooth transition at interface.
 */
inline double smoothHeaviside(double phi, double epsilon) {
    if (phi < -epsilon) return 0.0;
    if (phi > epsilon) return 1.0;
    return 0.5 + phi / (2.0 * epsilon) + std::sin(M_PI * phi / epsilon) / (2.0 * M_PI);
}

/**
 * @brief Smooth delta function (derivative of Heaviside).
 * @param phi Level set value.
 * @param epsilon Smearing width.
 * @return Delta function approximation.
 */
inline double smoothDelta(double phi, double epsilon) {
    if (std::abs(phi) > epsilon) return 0.0;
    return 0.5 / epsilon * (1.0 + std::cos(M_PI * phi / epsilon));
}

/**
 * @brief WENO5 interpolation for level set advection.
 *
 * Fifth-order Weighted Essentially Non-Oscillatory scheme for computing
 * the spatial derivative in the level set advection equation.
 *
 * @param v0 Stencil value f_{i-2}.
 * @param v1 Stencil value f_{i-1}.
 * @param v2 Stencil value f_{i}.
 * @param v3 Stencil value f_{i+1}.
 * @param v4 Stencil value f_{i+2}.
 * @return WENO5 reconstructed value at i+1/2.
 */
inline double weno5(double v0, double v1, double v2, double v3, double v4) {
    constexpr double eps = 1e-6;

    // Three candidate stencils
    double s0 = (1.0 / 3.0) * v0 - (7.0 / 6.0) * v1 + (11.0 / 6.0) * v2;
    double s1 = -(1.0 / 6.0) * v1 + (5.0 / 6.0) * v2 + (1.0 / 3.0) * v3;
    double s2 = (1.0 / 3.0) * v2 + (5.0 / 6.0) * v3 - (1.0 / 6.0) * v4;

    // Smoothness indicators
    double b0 = (13.0 / 12.0) * (v0 - 2.0 * v1 + v2) * (v0 - 2.0 * v1 + v2)
              + 0.25 * (v0 - 4.0 * v1 + 3.0 * v2) * (v0 - 4.0 * v1 + 3.0 * v2);
    double b1 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3)
              + 0.25 * (v1 - v3) * (v1 - v3);
    double b2 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4)
              + 0.25 * (3.0 * v2 - 4.0 * v3 + v4) * (3.0 * v2 - 4.0 * v3 + v4);

    // Weights
    double a0 = 0.1 / ((eps + b0) * (eps + b0));
    double a1 = 0.6 / ((eps + b1) * (eps + b1));
    double a2 = 0.3 / ((eps + b2) * (eps + b2));
    double sum = a0 + a1 + a2;

    return (a0 * s0 + a1 * s1 + a2 * s2) / sum;
}

} // namespace sloshing
