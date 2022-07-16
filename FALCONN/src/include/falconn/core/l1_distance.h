#ifndef __L1_DISTANCE_H__
#define __L1_DISTANCE_H__

#include <cstdint>
#include <vector>

#include <Eigen/Dense>

namespace falconn {
namespace core {

// TODO: rename to negative inner product distance?
// TODO: make a single CosineDistance class with different template
// specializations?

// Currently not support sparse
// The Dense functions assume that the data points are stored as dense
// Eigen column vectors.

template <typename CoordinateType = float>
struct L1DistanceDense {
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      VectorType;

  template <typename Derived1, typename Derived2>
  CoordinateType operator()(const Eigen::MatrixBase<Derived1>& p1,
                            const Eigen::MatrixBase<Derived2>& p2) {
    // negate the result because LSHTable assumes that smaller distances
    // are better
    return (p1 - p2).cwiseAbs().sum();
  }
};

}  // namespace core
}  // namespace falconn

#endif
