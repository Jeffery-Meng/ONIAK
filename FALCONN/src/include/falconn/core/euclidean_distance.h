#ifndef __EUCLIDEAN_DISTANCE_H__
#define __EUCLIDEAN_DISTANCE_H__

#include <cstdint>
#include <vector>
#include <iostream>

#include <Eigen/Dense>

namespace falconn {
namespace core {

// Computes *SQUARED* Euclidean distance between sparse or dense vectors.

template <typename CoordinateType = float, typename IndexType = int32_t>
struct EuclideanDistanceSparse {
  typedef std::vector<std::pair<IndexType, CoordinateType>> VectorType;

  CoordinateType operator()(const VectorType& p1, const VectorType& p2) {
    CoordinateType res = 0.0;
    IndexType ii1 = 0, ii2 = 0;
    IndexType p1size = p1.size();
    IndexType p2size = p2.size();
    CoordinateType x;

    while (ii1 < p1size || ii2 < p2size) {
      if (ii2 == p2size) {
        x = p1[ii1].second;
        res += x * x;
        ++ii1;
        continue;
      }
      if (ii1 == p1size) {
        x = p2[ii2].second;
        res += x * x;
        ++ii2;
        continue;
      }
      if (p1[ii1].first < p2[ii2].first) {
        x = p1[ii1].second;
        res += x * x;
        ++ii1;
        continue;
      }
      if (p2[ii2].first < p1[ii1].first) {
        x = p2[ii2].second;
        res += x * x;
        ++ii2;
        continue;
      }
      x = p1[ii1].second;
      x -= p2[ii2].second;
      res += x * x;
      ++ii1;
      ++ii2;
    }

    return res;
  }
};

// The Dense functions assume that the data points are stored as dense
// Eigen column vectors.

template <typename CoordinateType = float>
struct EuclideanDistanceDense {
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      VectorType;

  template <typename Derived1, typename Derived2>
  float operator()(const Eigen::MatrixBase<Derived1>& p1,
                            const Eigen::MatrixBase<Derived2>& p2) {
    
// Eigen::Matrix<float, Eigen::Dynamic, 1, Eigen::ColMajor>  p1_temp = p1.template cast<float>();
// Eigen::Matrix<float, Eigen::Dynamic, 1, Eigen::ColMajor>  p2_temp = p2.template cast<float>();
// std::cout << p1.size() << std::endl;
    return (p1 - p2).squaredNorm();
  }
};

template <typename CoordinateType = float>
struct MatrixNormDistance {
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      VectorType;
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
      MatrixType;

  template <typename Derived1, typename Derived2>
  float operator()(const Eigen::MatrixBase<Derived1>& A,
                            const Eigen::MatrixBase<Derived2>& x) {
    
// Here the roles of A and X are different!

   return eval(A, x);
  }

  template <typename Derived1, typename Derived2>
  static float eval (const Eigen::MatrixBase<Derived1>& A,
                            const Eigen::MatrixBase<Derived2>& x)   {
    
// Here the roles of A and X are different!

    auto x_head = x.head(A.cols());
    return (A*x_head).squaredNorm();
  }
};


}  // namespace core
}  // namespace falconn

#endif
