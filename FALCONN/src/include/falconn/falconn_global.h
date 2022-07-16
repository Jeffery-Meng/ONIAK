#ifndef __FALCONN_GLOBAL_H__
#define __FALCONN_GLOBAL_H__

#include <stdexcept>
#include <utility>
#include <vector>
#include <fstream>

#include <Eigen/Dense>
#include <Exception.h>
// #include "core/partition_metric.h"
// #include "core/partition.h"
// #include "core/flat_hash_table.h"
// #include "core/gaussian_hash.h"
// #include "wrapper/cpp_wrapper_impl.h"
// #include "core/multiprobe.h"
// #include "core/partitioin_hash_table.h"

#include <type_traits>

namespace falconn {
class FalconnError : public std::logic_error {
 public:
  FalconnError(const char* msg) : logic_error(msg) {}
};

constexpr int PROBES_PER_TABLE = 61;
constexpr float MARGIN = 1e-4;

using KeyType = int32_t;

namespace global {
  std::vector<std::vector<KeyType> > filtered_keys;
}

// Please change these options before compilation.


      ///
/// Common exception base class
///

///
/// General traits class for point types. Only the template specializations
/// below correspond to valid point types.
///
template <typename PointType>
struct PointTypeTraits {
  PointTypeTraits() {
    static_assert(FalseStruct<PointType>::value, "Point type not supported.");
  }

  template <typename PT>
  struct FalseStruct : std::false_type {};
};

///
/// Type for dense points / vectors. The coordinate type can be either float
/// or double (i.e., use DenseVector<float> or DenseVector<double>). In most
/// cases, float (single precision) should be sufficient.
///
template <typename CoordinateType>
using DenseVector =
    Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>;

template <typename CoordinateType>
using DenseMatrix = Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

using CoordinateType = float;
typedef DenseVector<CoordinateType> PointType;  // Type of data points
typedef DenseMatrix<CoordinateType> MatrixType;

using PointSet = std::vector<PointType>;
///
/// Traits class for accessing the corresponding scalar type.
///
template <typename CoordinateType>
struct PointTypeTraits<DenseVector<CoordinateType>> {
  typedef CoordinateType ScalarType;
};

template <typename CoordinateType>
struct PointTypeTraits<DenseMatrix<CoordinateType>> {
  typedef CoordinateType ScalarType;
};

///
/// Type for sparse points / vectors. The coordinate type can be either float
/// or double (i.e., use SparseVector<float> or SparseVector<double>). In most
/// cases, float (single precision) should be sufficient.
///
/// The elements of the vector must be sorted by the index (the first
/// component of the pair).
///
/// Optionally, you can also change the type of the coordinate indices. This
/// might be useful if you have indices that fit into an int16_t and you want
/// to save memory.
///
template <typename CoordinateType, typename IndexType = int32_t>
using SparseVector = std::vector<std::pair<IndexType, CoordinateType>>;

///
/// Traits class for accessing the corresponding scalar type.
///
template <typename CoordinateType, typename IndexT>
struct PointTypeTraits<SparseVector<CoordinateType, IndexT>> {
  typedef CoordinateType ScalarType;
  typedef IndexT IndexType;
};

///
/// Data structure for point query statistics
///
struct QueryStatistics {
  ///
  /// Average total query time
  ///
  double average_total_query_time = 0.0;
  ///
  /// Average hashing time
  ///
  double average_lsh_time = 0.0;

  double average_multiprobe_time = 0.0;
  ///
  /// Average hash table retrieval time
  ///
  double average_hash_table_time = 0.0;

  double average_sketches_time = 0.0;
  ///
  /// Average time for computing distances
  ///
  double average_distance_time = 0.0;
  ///
  /// Average number of candidates
  ///
  double average_num_candidates = 0;
  ///
  /// Average number of *unique* candidates
  ///
  double average_num_unique_candidates = 0;

  double average_num_filtered_candidates = 0;
  ///
  /// Number of queries the statistics were computed over
  ///
  int_fast64_t num_queries = 0;

  // TODO: move these to internal helper functions?
  void convert_to_totals() {
    average_total_query_time *= num_queries;
    average_lsh_time *= num_queries;
    average_hash_table_time *= num_queries;
    average_sketches_time *= num_queries;
    average_distance_time *= num_queries;
    average_num_candidates *= num_queries;
    average_num_unique_candidates *= num_queries;
    average_num_filtered_candidates *= num_queries;
    average_multiprobe_time *= num_queries;
  }

  void compute_averages() {
    if (num_queries > 0) {
      average_total_query_time /= num_queries;
      average_lsh_time /= num_queries;
      average_hash_table_time /= num_queries;
      average_sketches_time /= num_queries;
      average_distance_time /= num_queries;
      average_multiprobe_time /= num_queries;
      average_num_candidates /= num_queries;
      average_num_unique_candidates /= num_queries;
      average_num_filtered_candidates /= num_queries;
    }
  }

  void add_totals(const QueryStatistics& other) {
    average_total_query_time += other.average_total_query_time;
    average_lsh_time += other.average_lsh_time;
    average_hash_table_time += other.average_hash_table_time;
    average_sketches_time += other.average_sketches_time;
    average_distance_time += other.average_distance_time;
    average_num_candidates += other.average_num_candidates;
    average_num_unique_candidates += other.average_num_unique_candidates;
    average_num_filtered_candidates += other.average_num_filtered_candidates;
    num_queries += other.num_queries;
  }

  void reset() {
    average_total_query_time = 0.0;
    average_lsh_time = 0.0;
    average_hash_table_time = 0.0;
    average_sketches_time = 0.0;
    average_distance_time = 0.0;
    average_num_candidates = 0.0;
    average_num_unique_candidates = 0.0;
    average_num_filtered_candidates = 0.0;
    num_queries = 0;
  }
};

///
/// A struct for wrapping point data stored in a single dense data array. The
/// coordinate order is assumed to be point-by-point (row major), i.e., the
/// first dimension coordinates belong to the first point and there are
/// num_points points in total.
///
template <typename CoordinateType>
struct PlainArrayPointSet {
  const CoordinateType* data;
  int_fast32_t num_points;
  int_fast32_t dimension;
};

template <typename T>
 void ptrs_sanity_check(const std::unique_ptr<T>& p) {
    if (!p) throw FalconnError(
          strcat("Error! Pointer not correctly initialized! Type: ", typeid(T).name()));
  }

template <typename CoordinateType>
DenseMatrix<CoordinateType> read_eigen_matrix(std::ifstream& fin, int rows, int cols) {
  DenseMatrix<CoordinateType> result(rows, cols);
    for (int rr = 0; rr < rows; ++rr) {
      for (int cc = 0; cc < cols; ++cc) {
        fin >> result(rr, cc);
      }
    }
    return result;
}

template <typename CoordinateType>
DenseMatrix<CoordinateType> read_one_matrix(std::ifstream& fin, int rows, int cols) {
  static_assert(sizeof(CoordinateType) == 4);

  char buff[4];
  DenseMatrix<CoordinateType> result(rows, cols);
    for (int rr = 0; rr < rows; ++rr) {
      fin.read(buff, 4);
      for (int cc = 0; cc < cols; ++cc) {
          fin.read(reinterpret_cast<char*> (&result(rr, cc)), 4);
      }
    }
    return result;
}


// used for reading ground truth 
typedef std::vector<std::vector<int> > IntegerMatrix;
IntegerMatrix read_ground_truth(std::ifstream& fin, int K, int qn) {
  int sz;
  IntegerMatrix result;
  while (fin.read(reinterpret_cast<char*> (&sz), 4)) {
      std::vector<int> cur_gt(K);
      fin.read(reinterpret_cast<char*> (cur_gt.data()), 4 * K);
      assert(sz >= K && "Ground truth must contain at least K nearest neighbors.");
      fin.seekg((sz-K) * 4, std::ios::cur); // skip the remaining neighbors for this query
      std::sort(cur_gt.begin(), cur_gt.end());
      result.push_back(std::move(cur_gt));
  }
  assert(result.size() == qn);
  return result;
}


template <typename CoordinateType>
DenseVector<CoordinateType> read_vector(std::ifstream& fin) {
  static_assert(sizeof(CoordinateType) == 4);

  int sz;
  fin.read(reinterpret_cast<char*> (&sz), 4);
  DenseVector<CoordinateType> result(sz);
  fin.read(reinterpret_cast<char*> (&result(0)), 4 * sz);

  //TODO: test this function
 // for (int cc = 0; cc < sz; ++cc) {
//      fin.read(reinterpret_cast<char*> (&result(cc)), 4);
//  }
  
  return result;
}

class NearestNeighborQueryError : public FalconnError {
 public:
  NearestNeighborQueryError(const char* msg) : FalconnError(msg) {}
};

}  // namespace falconn

#endif
