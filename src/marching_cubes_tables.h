/**
 * @file marching_cubes_tables.h
 * @brief Lookup tables for the Marching Cubes algorithm.
 *
 * Edge table: maps cube configuration (8-bit) to which edges are intersected.
 * Triangle table: maps cube configuration to triangle vertex indices on edges.
 *
 * These tables are from the classic Paul Bourke / Cory Gene Bloyd implementation.
 */
#pragma once

#include <cstdint>

namespace sloshing {
namespace mc_tables {

/// @brief For each of the 256 cube configurations, which edges are intersected.
extern const uint16_t edgeTable[256];

/// @brief For each configuration, up to 5 triangles (15 edge indices, -1 terminated).
extern const int triTable[256][16];

} // namespace mc_tables
} // namespace sloshing
