/**
 * @file grid.h
 * @brief MAC (Marker-and-Cell) staggered grid for incompressible flow simulation.
 *
 * The MAC grid stores pressure and scalar fields at cell centers, and velocity
 * components at cell face centers. This eliminates checkerboard pressure
 * oscillations that plague collocated grids.
 *
 * Grid layout (2D slice for illustration):
 * @verbatim
 *   +---v---+---v---+
 *   |       |       |
 *   u   p   u   p   u
 *   |       |       |
 *   +---v---+---v---+
 *   |       |       |
 *   u   p   u   p   u
 *   |       |       |
 *   +---v---+---v---+
 * @endverbatim
 *
 * - p (pressure, level set, VOF): Ni x Nj x Nk values at cell centers
 * - u (x-velocity): (Ni+1) x Nj x Nk values at x-face centers
 * - v (y-velocity): Ni x (Nj+1) x Nk values at y-face centers
 * - w (z-velocity): Ni x Nj x (Nk+1) values at z-face centers
 */
#pragma once

#include <vector>
#include <cstddef>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <glm/glm.hpp>

namespace sloshing {

/**
 * @brief Cell material classification for the free-surface solver.
 */
enum class CellType : uint8_t {
    Fluid,  ///< Cell is fully occupied by liquid
    Air,    ///< Cell is fully occupied by gas
    Solid   ///< Cell is a wall/boundary
};

/**
 * @brief 3D array with flat storage and (i,j,k) indexing.
 * @tparam T Element type (typically double or CellType).
 */
template <typename T>
class Array3D {
public:
    Array3D() = default;

    /**
     * @brief Construct a 3D array with given dimensions.
     * @param ni Size in x-direction.
     * @param nj Size in y-direction.
     * @param nk Size in z-direction.
     * @param init_val Initial value for all elements.
     */
    Array3D(int ni, int nj, int nk, T init_val = T{})
        : ni_(ni), nj_(nj), nk_(nk), data_(static_cast<size_t>(ni) * nj * nk, init_val) {}

    /// @name Accessors
    /// @{
    T& operator()(int i, int j, int k) {
        assert(i >= 0 && i < ni_ && j >= 0 && j < nj_ && k >= 0 && k < nk_);
        return data_[index(i, j, k)];
    }

    const T& operator()(int i, int j, int k) const {
        assert(i >= 0 && i < ni_ && j >= 0 && j < nj_ && k >= 0 && k < nk_);
        return data_[index(i, j, k)];
    }
    /// @}

    /// @name Dimensions
    /// @{
    int ni() const { return ni_; }
    int nj() const { return nj_; }
    int nk() const { return nk_; }
    size_t size() const { return data_.size(); }
    /// @}

    /// @brief Fill entire array with a value.
    void fill(T val) { std::fill(data_.begin(), data_.end(), val); }

    /// @brief Direct access to underlying data.
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    /// @brief Swap contents with another array.
    void swap(Array3D& other) {
        std::swap(ni_, other.ni_);
        std::swap(nj_, other.nj_);
        std::swap(nk_, other.nk_);
        data_.swap(other.data_);
    }

private:
    size_t index(int i, int j, int k) const {
        return static_cast<size_t>(i) + static_cast<size_t>(ni_) * (j + static_cast<size_t>(nj_) * k);
    }

    int ni_ = 0, nj_ = 0, nk_ = 0;
    std::vector<T> data_;
};

/**
 * @brief MAC staggered grid holding all simulation fields.
 *
 * Manages the velocity field (u, v, w on staggered faces), pressure,
 * level set (phi), VOF fraction (vof), and cell type classification.
 */
class MACGrid {
public:
    /**
     * @brief Construct a MAC grid for a rectangular domain.
     * @param ni Number of cells in x-direction.
     * @param nj Number of cells in y-direction.
     * @param nk Number of cells in z-direction.
     * @param lx Physical length in x-direction (meters).
     * @param ly Physical length in y-direction (meters).
     * @param lz Physical length in z-direction (meters).
     */
    MACGrid(int ni, int nj, int nk, double lx, double ly, double lz);

    /// @name Grid dimensions
    /// @{
    int ni() const { return ni_; }
    int nj() const { return nj_; }
    int nk() const { return nk_; }
    double dx() const { return dx_; }
    double dy() const { return dy_; }
    double dz() const { return dz_; }
    double lx() const { return lx_; }
    double ly() const { return ly_; }
    double lz() const { return lz_; }
    /// @}

    /// @name Position queries
    /// @{

    /// @brief Cell center position.
    glm::dvec3 cellCenter(int i, int j, int k) const {
        return {(i + 0.5) * dx_, (j + 0.5) * dy_, (k + 0.5) * dz_};
    }

    /// @brief X-face center position (where u lives).
    glm::dvec3 uPos(int i, int j, int k) const {
        return {i * dx_, (j + 0.5) * dy_, (k + 0.5) * dz_};
    }

    /// @brief Y-face center position (where v lives).
    glm::dvec3 vPos(int i, int j, int k) const {
        return {(i + 0.5) * dx_, j * dy_, (k + 0.5) * dz_};
    }

    /// @brief Z-face center position (where w lives).
    glm::dvec3 wPos(int i, int j, int k) const {
        return {(i + 0.5) * dx_, (j + 0.5) * dy_, k * dz_};
    }
    /// @}

    /// @name Velocity interpolation
    /// @{

    /// @brief Trilinearly interpolate velocity at an arbitrary position.
    glm::dvec3 interpolateVelocity(const glm::dvec3& pos) const;

    /// @brief Interpolate a cell-centered scalar field at an arbitrary position.
    double interpolateScalar(const Array3D<double>& field, const glm::dvec3& pos) const;
    /// @}

    /// @name Cell classification
    /// @{

    /// @brief Classify cells as Fluid, Air, or Solid based on level set and boundary.
    void classifyCells();

    /// @brief Mark boundary cells as solid.
    void enforceBoundaryConditions();
    /// @}

    /// @name Divergence
    /// @{

    /// @brief Compute velocity divergence at cell center (i,j,k).
    double divergence(int i, int j, int k) const;

    /// @brief Compute maximum absolute divergence over all fluid cells.
    double maxDivergence() const;
    /// @}

    // --- Public fields (MAC convention) ---

    Array3D<double> u;       ///< x-velocity at x-faces: (ni+1) x nj x nk
    Array3D<double> v;       ///< y-velocity at y-faces: ni x (nj+1) x nk
    Array3D<double> w;       ///< z-velocity at z-faces: ni x nj x (nk+1)
    Array3D<double> pressure;///< Pressure at cell centers: ni x nj x nk
    Array3D<double> phi;     ///< Level set (signed distance) at cell centers
    Array3D<double> vof;     ///< Volume-of-fluid fraction at cell centers [0,1]
    Array3D<CellType> cell_type; ///< Cell classification

private:
    int ni_, nj_, nk_;
    double lx_, ly_, lz_;
    double dx_, dy_, dz_;

    /// @brief Trilinear interpolation helper for a staggered component.
    double interpolateComponent(const Array3D<double>& field,
                                const glm::dvec3& pos,
                                const glm::dvec3& offset) const;
};

} // namespace sloshing
