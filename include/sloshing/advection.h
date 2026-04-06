/**
 * @file advection.h
 * @brief Advection schemes for the level set and VOF fields.
 *
 * - Level set: Semi-Lagrangian advection with WENO5 interpolation.
 * - VOF: Directional-split advection with PLIC (Piecewise Linear Interface
 *   Calculation) reconstruction using normals from the level set.
 * - Velocity: Semi-Lagrangian advection for each component.
 */
#pragma once

#include "sloshing/grid.h"

namespace sloshing {

/**
 * @brief Advects the level set field using semi-Lagrangian method.
 *
 * Traces characteristics backward in time and interpolates the level set
 * value at the departure point using trilinear interpolation.
 *
 * @param grid The MAC grid (phi is read, phi_new is written).
 * @param phi_new Output: advected level set.
 * @param dt Time step.
 */
void advectLevelSet(const MACGrid& grid, Array3D<double>& phi_new, double dt);

/**
 * @brief Advects the VOF field using directional-split PLIC.
 *
 * Uses Piecewise Linear Interface Calculation with normals derived from
 * the level set for accurate, volume-conserving advection.
 *
 * @param grid The MAC grid (vof is read, uses phi for normals).
 * @param vof_new Output: advected VOF field.
 * @param dt Time step.
 */
void advectVOF(const MACGrid& grid, Array3D<double>& vof_new, double dt);

/**
 * @brief Advects the velocity field using semi-Lagrangian method.
 *
 * Each velocity component is advected by tracing backward along the
 * full velocity field and interpolating.
 *
 * @param grid The MAC grid (velocities are updated in place).
 * @param dt Time step.
 */
void advectVelocity(MACGrid& grid, double dt);

/**
 * @brief Compute the interface normal at cell (i,j,k) from the level set.
 *
 * Uses central differences on the level set field.
 *
 * @param phi Level set field.
 * @param i,j,k Cell indices.
 * @param dx,dy,dz Grid spacing.
 * @return Normalized gradient of phi (unit normal pointing into air).
 */
glm::dvec3 computeNormal(const Array3D<double>& phi,
                          int i, int j, int k,
                          double dx, double dy, double dz);

/**
 * @brief PLIC: compute the cutting plane constant for a given normal and volume fraction.
 *
 * Given a unit normal n and a target volume fraction F in [0,1], finds the
 * constant d such that the plane n·x = d cuts the unit cube to enclose
 * exactly volume F.
 *
 * @param normal Interface normal (unit vector).
 * @param vof_fraction Target volume fraction.
 * @return Plane constant d.
 */
double plicFindPlaneConstant(const glm::dvec3& normal, double vof_fraction);

/**
 * @brief Compute the volume fraction of a unit cube below the plane n·x = d.
 *
 * @param normal Interface normal.
 * @param d Plane constant.
 * @return Volume fraction in [0, 1].
 */
double plicVolumeBelowPlane(const glm::dvec3& normal, double d);

/**
 * @brief Extrapolate fluid velocities into air cells.
 *
 * After pressure projection, air cell velocities retain unphysical values
 * (accumulated body forces). This function marks faces adjacent to at least
 * one fluid cell as "valid" and propagates those values outward layer by
 * layer so that semi-Lagrangian backtraces near the interface see a smooth
 * velocity field.
 *
 * @param grid The MAC grid (velocities modified in place).
 * @param layers Number of extrapolation layers into the air region.
 */
void extendVelocityIntoAir(MACGrid& grid, int layers = 3);

} // namespace sloshing
