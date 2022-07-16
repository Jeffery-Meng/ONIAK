#ifndef __PARTITION_METRIC_H__
#define __PARTITION_METRIC_H__

#include <memory>
#include <utility>
#include <vector>
#include <fstream>
#include "hash_table_helpers.h"
#include "composite_hash_table.h"
#include "Exception.h"
#include <iostream>

namespace falconn {
namespace core {

/*template <typename CoordinateType = float, typename MetricType = float>
class PartitionMetricBase {
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      VectorType;

  static MetricType eval(const VectorType& p) const = 0;
};*/

class PartitionL2Norm {
public:
  typedef float PartitionMetricType;

  template <typename CoordinateType, int Rows, int Columns>
  static float eval(const Eigen::Matrix<CoordinateType, Rows, Columns, Eigen::ColMajor>& p) {
    // TODO: return the L2 norm of a point
    return p.squaredNorm();
  }

 // template <typename DataStorage>
 /* static std::vector<float> eval_all(const DataStorage& points)
  {
    // Huayi TODO: return the L2 norms of all points in the data set.
    // Use the following iterator to pass through all points
    typename DataStorage::FullSequenceIterator iter =
          points.get_full_sequence();
    std::vector<float> L2_norm_temp;
    
    while (iter.is_valid()) {
      CoordinateType l2_now = eval(iter.get_point());
      L2_norm_temp.push_back(l2_now);
      ++iter;
      }
    return L2_norm_temp;
    }*/

  template <typename PointType>
  static std::vector<PointType> filter_data_in_range(const std::vector<PointType>& points, 
      PartitionMetricType lower, PartitionMetricType upper) {
      std::vector<PointType> filtered_points;
    
    for(auto point:points) {
      auto l2_now = eval(point);

    if (lower<l2_now && l2_now<=upper) {
        filtered_points.push_back(point);}
    }
    return filtered_points;
  }

  /*  template <typename DataStorage, typename KeyType, typename Normalizer>
  static std::vector<std::vector<KeyType> > filter_in_range(const DataStorage& points, 
                const std::vector<Normalizer>& normalizers) {
                // TODO for Huayi: write this function, which filters the KEYS of all points, 
                // whose metric values are between lower and upper
   int num_partitions = normalizers.size();
    std::vector<std::vector<KeyType> > filtered_keys(num_partitions);
    for (auto& fk : filtered_keys) {
      fk.reserve(points.size());
    }

    // std::vector<float> bin_mx_values{0};
    // for(int i=0;i<num_partitions;i++) {
    //   bin_mx_values.push_back(normalizers[pp].get_mx());
    // }

    typename DataStorage::FullSequenceIterator iter =
    points.get_full_sequence();
    
    while (iter.is_valid()) {
      
      auto l2_now = eval(iter.get_point());



      int pp = 0;
      for (; pp < num_partitions; ++pp) {  // use linear search here
    //       if(iter.get_key() < 100) 
    //   {
    //   if (pp >= filtered_keys.size()){
    //     std::cout << pp << "\t" << l2_now << "\t" << normalizers[pp].get_mx() << std::endl;
    //   }
    //  }
       // if (l2_now <= normalizers[pp].get_mx()*(1+MARGIN)) break;
        if (l2_now <= normalizers[pp].get_mx() - MARGIN) break;
      }

      NPP_ASSERT(pp < filtered_keys.size());
      normalizers[pp].normalize(iter.get_point(), l2_now);
      
      filtered_keys[pp].push_back(iter.get_key());
      //std::cout << iter.get_key() << std::endl;
      ++iter;
    }
    return filtered_keys;
  }*/


  template <typename MetricVecType>
  int compute_num_probes(const MetricVecType& p_metric, float lower, float upper, int l) const {
    // Huayi TODO: return the number of probes for this point, given the L2 distance of this point\
    // and the lower and upper bounds in the correspoding partition
    return PROBES_PER_TABLE * l;
  }
};

//add more partition metrics here


}  // namespace core
}  // namespace falconn

#endif
