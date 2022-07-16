#ifndef __CPP_WRAPPER_IMPL_H__
#define __CPP_WRAPPER_IMPL_H__

#include <atomic>
#include <random>
#include <thread>
#include <type_traits>
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <memory>
#include <cstring>
#include <Exception.h>


#include "data_storage_adapter.h"
#include "../falconn_global.h"
#include "../lsh_nn_table.h"

#include "../core/bit_packed_flat_hash_table.h"
#include "../core/composite_hash_table.h"
#include "../core/cosine_distance.h"
#include "../core/data_storage.h"
#include "../core/euclidean_distance.h"
#include "../core/flat_hash_table.h"
#include "../core/gaussian_hash.h"
#include "../core/multiprobe.h"
#include "../core/hyperplane_hash.h"
#include "../core/lsh_table.h"
#include "../core/nn_query.h"
#include "../core/polytope_hash.h"
#include "../core/probing_hash_table.h"
#include "../core/stl_hash_table.h"
#include "../core/l1_distance.h"
#include "../core/partition_metric.h"
#include "../core/partition_hash_table.h"
#include "../core/nearest_hyperplane.h"
#include "../core/math_helpers.h"
#include "../core/heap.h"


namespace falconn {
namespace wrapper {

template<typename T>
inline constexpr bool false_constexpr = false;

template <typename PointType, typename KeyType, typename DistanceType,
          typename LSHTable, typename ScalarType, typename DistanceFunction,
          typename DataStorage>
class LSHNNQueryWrapper : public LSHNearestNeighborQuery<PointType, KeyType> {
  typedef core::NearestNeighborQuery<typename LSHTable::Query, PointType,
                                     KeyType, PointType, ScalarType,
                                     DistanceFunction, DataStorage>
      NNQueryType;
 public:
  LSHNNQueryWrapper(const LSHTable& parent, int_fast64_t num_probes,
                    int_fast64_t max_num_candidates,
                    const DataStorage& data_storage)
      : num_probes_(num_probes), max_num_candidates_(max_num_candidates) {
    if (num_probes <= 0) {
      throw LSHNearestNeighborTableError(
          "Number of probes must be at least 1.");
    }
    internal_query_.reset(new typename LSHTable::Query(parent, num_probes));
    internal_nn_query_.reset(
        new NNQueryType(internal_query_.get(), data_storage));
  }

  KeyType find_nearest_neighbor(const FalconnQueryType& q) {
    return internal_nn_query_->find_nearest_neighbor(q, q, num_probes_,
                                                     max_num_candidates_);
  }

  void find_k_nearest_neighbors(const FalconnQueryType& q, int_fast64_t k,
                                std::vector<KeyType>* result) {
    internal_nn_query_->find_k_nearest_neighbors(q, q, k, num_probes_,
                                                 max_num_candidates_, result);
  }

  void find_near_neighbors(const FalconnQueryType& q, DistanceType threshold,
                           std::vector<KeyType>* result) {
    internal_nn_query_->find_near_neighbors(q, q, threshold, num_probes_,
                                            max_num_candidates_, result);
  }

  void get_candidates_with_duplicates(const FalconnQueryType& q,
                                      std::vector<KeyType>* result) {
    internal_nn_query_->get_candidates_with_duplicates(
        q, num_probes_, max_num_candidates_, result);
  }

  void get_unique_candidates(const FalconnQueryType& q, std::vector<KeyType>* result) {
    internal_nn_query_->get_unique_candidates(q, num_probes_,
                                              max_num_candidates_, result);
  }

  int_fast64_t get_num_probes() { return num_probes_; }

  void set_num_probes(int_fast64_t new_num_probes) {
    if (new_num_probes <= 0) {
      throw LSHNearestNeighborTableError(
          "Number of probes must be at least 1.");
    }
    num_probes_ = new_num_probes;
  }

  int_fast64_t get_max_num_candidates() { return max_num_candidates_; }

  void set_max_num_candidates(int_fast64_t new_max_num_candidates) {
    max_num_candidates_ = new_max_num_candidates;
  }

  void reset_query_statistics() {
    internal_nn_query_->reset_query_statistics();
  }

  QueryStatistics get_query_statistics() {
    return internal_nn_query_->get_query_statistics();
  }

  // without_threshold
  void find_k_nearest_neighbors_without_threshold(const FalconnQueryType& q, int_fast64_t k,
                        std::vector<KeyType>* result, unsigned q_cnt) override {
      find_k_nearest_neighbors( q, k, result);
  }

  //with_threshold
  void find_k_nearest_neighbors_with_threshold(const FalconnQueryType& q, int_fast64_t k,
                        std::vector<KeyType>* result, unsigned q_cnt,float t) override {
      find_k_nearest_neighbors(q, k, result);
  }

  virtual ~LSHNNQueryWrapper() {}

 protected:
  std::unique_ptr<typename LSHTable::Query> internal_query_;
  std::unique_ptr<NNQueryType> internal_nn_query_;
  int_fast64_t num_probes_;
  int_fast64_t max_num_candidates_;
};

/*template <typename PointType, typename KeyType, typename DistanceType,
          typename LSHTable, typename ScalarType, typename DistanceFunction,
          typename DataStorage>
class LSHNNQueryWrapper2 {
 public:
  LSHNNQueryWrapper2(const LSHTable& parent, int_fast64_t num_probes,
                    int_fast64_t max_num_candidates,
                    const DataStorage& data_storage)
      : num_probes_(num_probes), max_num_candidates_(max_num_candidates) {
    if (num_probes <= 0) {
      throw LSHNearestNeighborTableError(
          "Number of probes must be at least 1.");
    }
    internal_query_.reset(new typename LSHTable::Query(parent, num_probes));
    internal_nn_query_.reset(
        new FalconnNNQueryType(internal_query_.get(), data_storage));
  }


  void find_k_nearest_neighbors(const FalconnQueryType& q, int_fast64_t k,
                                std::vector<KeyType>* result, bool reset_A) {
    if (reset_A) {
      internal_nn_query_->update_A(retrieve_A(q));
    }
    
    internal_nn_query_->find_k_nearest_neighbors(retrieve_query(q), retrieve_query(q), k, num_probes_,
                                                 max_num_candidates_, result);
  }


  int_fast64_t get_num_probes() { return num_probes_; }

  void set_num_probes(int_fast64_t new_num_probes) {
    if (new_num_probes <= 0) {
      throw LSHNearestNeighborTableError(
          "Number of probes must be at least 1.");
    }
    num_probes_ = new_num_probes;
  }

  int_fast64_t get_max_num_candidates() { return max_num_candidates_; }

  void set_max_num_candidates(int_fast64_t new_max_num_candidates) {
    max_num_candidates_ = new_max_num_candidates;
  }

  void reset_query_statistics() {
    internal_nn_query_->reset_query_statistics();
  }

  QueryStatistics get_query_statistics() {
    return internal_nn_query_->get_query_statistics();
  }

 private:
  std::unique_ptr<typename LSHTable::Query> internal_query_;
  std::unique_ptr<FalconnNNQueryType> internal_nn_query_;
  int_fast64_t num_probes_;
  int_fast64_t max_num_candidates_;

  MatrixType retrieve_A(const MatrixType& A) const{
    return A.topLeftCorner(A.rows(), A.cols() - 1);
  }
  PointType retrieve_query(const MatrixType& A) const{
    return A.topRightCorner(A.rows(), 1);
  }
};*/

/*template <typename PointType, typename KeyType, typename DistanceType,
          typename LSHTable, typename ScalarType, typename DistanceFunction,
          typename DataStorage>
class LSHFilterQueryWrapper : public LSHNearestNeighborQuery<PointType, KeyType> {
  typedef core::FilteringQuery<typename LSHTable::Query, PointType,
                                     KeyType, PointType, ScalarType,
                                     DistanceFunction, DataStorage>
      NNQueryType;

 public:
  LSHFilterQueryWrapper(const LSHTable& parent, int_fast64_t num_probes,
                    int_fast64_t max_num_candidates,
                    const DataStorage& data_storage, unsigned num_filters, float recall_target)
      : num_probes_(num_probes), max_num_candidates_(max_num_candidates) {
    if (num_probes <= 0) {
      throw LSHNearestNeighborTableError(
          "Number of probes must be at least 1.");
    }
    internal_query_.reset(new typename LSHTable::Query(parent, num_probes));
    internal_nn_query_.reset(
        new NNQueryType(internal_query_.get(), data_storage,num_filters,recall_target));
  }

  KeyType find_nearest_neighbor(const FalconnQueryType& q) {
    return 0;
  }

  void find_k_nearest_neighbors(const FalconnQueryType& q, int_fast64_t k,
                                std::vector<KeyType>* result) {
     internal_nn_query_->find_k_nearest_neighbors(q, q, k, num_probes_,
                                                 max_num_candidates_, result);
  }

  void find_near_neighbors(const FalconnQueryType& q, DistanceType threshold,
                           std::vector<KeyType>* result) {
  }

  void get_candidates_with_duplicates(const FalconnQueryType& q,
                                      std::vector<KeyType>* result) {
  }

  void get_unique_candidates(const FalconnQueryType& q, std::vector<KeyType>* result) {
  }

  int_fast64_t get_num_probes() { return num_probes_; }

  void set_num_probes(int_fast64_t new_num_probes) {
    if (new_num_probes <= 0) {
      throw LSHNearestNeighborTableError(
          "Number of probes must be at least 1.");
    }
    num_probes_ = new_num_probes;
  }

  int_fast64_t get_max_num_candidates() { return max_num_candidates_; }

  void set_max_num_candidates(int_fast64_t new_max_num_candidates) {
    max_num_candidates_ = new_max_num_candidates;
  }

  void reset_query_statistics() {
    internal_nn_query_->reset_query_statistics();
  }

  // without_threshold
  void find_k_nearest_neighbors_without_threshold(const FalconnQueryType& q, int_fast64_t k,
                        std::vector<KeyType>* result, unsigned q_cnt) override {
    internal_nn_query_ -> find_k_nearest_neighbors_without_threshold(q,q,k,num_probes_,max_num_candidates_,result,q_cnt);
  }

  // with_threshold
  void find_k_nearest_neighbors_with_threshold(const FalconnQueryType& q, int_fast64_t k,
                        std::vector<KeyType>* result, unsigned q_cnt,float t) override {
      internal_nn_query_ -> find_k_nearest_neighbors_with_threshold(q,q,k,num_probes_,max_num_candidates_,result,q_cnt,t);
  }

  QueryStatistics get_query_statistics() {
    return internal_nn_query_->get_query_statistics();
  }

  virtual ~LSHFilterQueryWrapper() {}

 protected:
  std::unique_ptr<typename LSHTable::Query> internal_query_;
  std::unique_ptr<NNQueryType> internal_nn_query_;
  int_fast64_t num_probes_;
  int_fast64_t max_num_candidates_;
};*/

///  LSH Query Pool - implementation of queries 
// Multiple threads process different queries in parallel.
// DO in future: modify it to support partitions.

template <typename PointType, typename KeyType, typename DistanceType,
          typename LSHTable, typename ScalarType, typename DistanceFunction,
          typename DataStorage>
class LSHNNQueryPool : public LSHNearestNeighborQueryPool<PointType, KeyType> {
  typedef core::NearestNeighborQuery<typename LSHTable::Query, PointType,
                                     KeyType, PointType, ScalarType,
                                     DistanceFunction, DataStorage>
      NNQueryType;

 public:
  LSHNNQueryPool(const LSHTable& parent, int_fast64_t num_probes,
                 int_fast64_t max_num_candidates,
                 const DataStorage& data_storage,
                 int_fast64_t num_query_objects)
      : locks_(num_query_objects),
        num_probes_(num_probes),
        max_num_candidates_(max_num_candidates) {
    if (num_probes <= 0) {
      throw LSHNearestNeighborTableError(
          "Number of probes must be at least 1.");
    }
    if (num_query_objects <= 0) {
      throw LSHNearestNeighborTableError(
          "Number of query objects in the pool must be at least 1.");
    }
    for (int ii = 0; ii < num_query_objects; ++ii) {
      std::unique_ptr<typename LSHTable::Query> cur_query(
          new typename LSHTable::Query(parent, num_probes));
      std::unique_ptr<NNQueryType> cur_nn_query(
          new NNQueryType(cur_query.get(), data_storage));
      internal_queries_.push_back(std::move(cur_query));
      internal_nn_queries_.push_back(std::move(cur_nn_query));
      locks_[ii].clear(std::memory_order_release);
    }
  }

  KeyType find_nearest_neighbor(const PointType& q) {
    int_fast32_t query_index = get_query_index_and_lock();
    KeyType res = internal_nn_queries_[query_index]->find_nearest_neighbor(
        q, q, num_probes_, max_num_candidates_);
    unlock_query(query_index);
    return res;
  }

  void find_k_nearest_neighbors(const FalconnQueryType& q, int_fast64_t k,
                                std::vector<KeyType>* result) {
    int_fast32_t query_index = get_query_index_and_lock();
    internal_nn_queries_[query_index]->find_k_nearest_neighbors(
        q, q, k, num_probes_, max_num_candidates_, result);
    unlock_query(query_index);
  }

  void find_near_neighbors(const FalconnQueryType& q, DistanceType threshold,
                           std::vector<KeyType>* result) {
    int_fast32_t query_index = get_query_index_and_lock();
    internal_nn_queries_[query_index]->find_near_neighbors(
        q, q, threshold, num_probes_, max_num_candidates_, result);
    unlock_query(query_index);
  }

  void get_candidates_with_duplicates(const FalconnQueryType& q,
                                      std::vector<KeyType>* result) {
    int_fast32_t query_index = get_query_index_and_lock();
    internal_nn_queries_[query_index]->get_candidates_with_duplicates(
        q, num_probes_, max_num_candidates_, result);
    unlock_query(query_index);
  }

  void get_unique_candidates(const FalconnQueryType& q, std::vector<KeyType>* result) {
    int_fast32_t query_index = get_query_index_and_lock();
    internal_nn_queries_[query_index]->get_unique_candidates(
        q, num_probes_, max_num_candidates_, result);
    unlock_query(query_index);
  }

  int_fast64_t get_num_probes() { return num_probes_; }

  void set_num_probes(int_fast64_t new_num_probes) {
    if (new_num_probes <= 0) {
      throw LSHNearestNeighborTableError(
          "Number of probes must be at least 1.");
    }
    num_probes_ = new_num_probes;
  }

  int_fast64_t get_max_num_candidates() { return max_num_candidates_; }

  void set_max_num_candidates(int_fast64_t new_max_num_candidates) {
    max_num_candidates_ = new_max_num_candidates;
  }

  void reset_query_statistics() {
    for (int_fast64_t ii = 0;
         ii < static_cast<int_fast64_t>(internal_nn_queries_.size()); ++ii) {
      while (locks_[ii].test_and_set(std::memory_order_acquire))
        ;
      internal_nn_queries_[ii]->reset_query_statistics();
      locks_[ii].clear(std::memory_order_release);
    }
  }

  QueryStatistics get_query_statistics() {
    QueryStatistics res;
    for (int_fast64_t ii = 0;
         ii < static_cast<int_fast64_t>(internal_nn_queries_.size()); ++ii) {
      while (locks_[ii].test_and_set(std::memory_order_acquire))
        ;
      QueryStatistics cur_stats =
          internal_nn_queries_[ii]->get_query_statistics();
      cur_stats.convert_to_totals();
      res.add_totals(cur_stats);
      locks_[ii].clear(std::memory_order_release);
    }
    res.compute_averages();
    return res;
  }

  virtual ~LSHNNQueryPool() {}

 protected:
  int_fast32_t get_query_index_and_lock() {
    static thread_local std::minstd_rand gen((std::random_device())());
    std::uniform_int_distribution<int_fast32_t> dist(0, locks_.size() - 1);
    int_fast32_t cur_index = dist(gen);
    while (true) {
      if (!locks_[cur_index].test_and_set(std::memory_order_acquire)) {
        return cur_index;
      }
      if (cur_index == static_cast<int_fast32_t>(locks_.size()) - 1) {
        cur_index = 0;
      } else {
        cur_index += 1;
      }
    }
  }

  void unlock_query(int_fast32_t index) {
    locks_[index].clear(std::memory_order_release);
  }

  std::vector<std::unique_ptr<typename LSHTable::Query>> internal_queries_;
  std::vector<std::unique_ptr<NNQueryType>> internal_nn_queries_;
  std::vector<std::atomic_flag> locks_;
  int_fast64_t num_probes_;
  int_fast64_t max_num_candidates_;
}; 

template <typename PointType, typename KeyType, typename DistanceType,
          typename DistanceFunction, typename LSHTable, typename LSHFunction,
          typename HashTableFactory, typename CompositeHashTable,
          typename DataStorage>
class LSHNNTableWrapper : public LSHNearestNeighborTable<PointType, KeyType> {
 public:
  LSHNNTableWrapper(std::vector<std::unique_ptr<LSHFunction> > lshes,
                    std::unique_ptr<LSHTable> lsh_table,
                    std::unique_ptr<HashTableFactory> hash_table_factory,
                    std::vector<std::unique_ptr<CompositeHashTable> > composite_tables,
                    std::unique_ptr<DataStorage> data_storage)
      : lshes_(std::move(lshes)),
        lsh_table_(std::move(lsh_table)),
        hash_table_factory_(std::move(hash_table_factory)),
        composite_tables_(std::move(composite_tables)),
        data_storage_(std::move(data_storage)) {}

  void add_table() {} // deleted for now
  /*  lsh_->add_table();
    composite_hash_table_vec_->add_table();
    lsh_table_->add_table();
  }*/


// Here, num_probes is the maximum number of probes, used in the precomputation of MultiProbe
  std::unique_ptr<FalconnQueryWrapper>
  construct_query_object(int_fast64_t num_probes,
                         int_fast64_t max_num_candidates , unsigned num_filters = 0,float recall_target = 0.) 
             const override{
    assert(num_probes > 0);

    typedef typename PointTypeTraits<PointType>::ScalarType ScalarType;
   /* if constexpr(std::is_same_v<FalconnQueryWrapper, LSHNNQueryWrapper2<
                                      PointType, KeyType, DistanceType, LSHTable,
                                      ScalarType, DistanceFunction, DataStorage> >)
    {
    FalconnQueryWrapper
        nn_query(
                *lsh_table_, num_probes, max_num_candidates, *data_storage_);
    return std::move(nn_query);
    } else*/ if constexpr(std::is_same_v<FalconnQueryWrapper, LSHNNQueryWrapper<
                                      PointType, KeyType, DistanceType, LSHTable,
                                      ScalarType, DistanceFunction, DataStorage> >) {
     std::unique_ptr<FalconnQueryWrapper>  
        nn_query( new FalconnQueryWrapper(
                *lsh_table_, num_probes, max_num_candidates, *data_storage_) );
    return std::move(nn_query);                                   
    }
   else {
     static_assert(false_constexpr<PointType>, "Unsupported Query Wrapper type!");
    /*typedef typename PointTypeTraits<PointType>::ScalarType ScalarType;
    std::unique_ptr<
        LSHFilterQueryWrapper<PointType, KeyType, DistanceType, LSHTable,
                          ScalarType, DistanceFunction, DataStorage>>
        nn_query(
            new LSHFilterQueryWrapper<PointType, KeyType, DistanceType, LSHTable,
                                  ScalarType, DistanceFunction, DataStorage>(
                *lsh_table_, num_probes, max_num_candidates, *data_storage_, num_filters, recall_target));
    return std::move(nn_query);*/
  }
 }

 
  std::unique_ptr<LSHNearestNeighborQueryPool<PointType, KeyType>>
  construct_query_pool(int_fast64_t num_probes = -1,
                       int_fast64_t max_num_candidates = -1,
                       int_fast64_t num_query_objects = 0) const {
    if (num_probes <= 0) {
      num_probes = lshes_->get_l();
    }
    if (num_query_objects <= 0) {
      num_query_objects = std::max(1u, 2 * std::thread::hardware_concurrency());
    }
    typedef typename PointTypeTraits<PointType>::ScalarType ScalarType;
    std::unique_ptr<LSHNNQueryPool<PointType, KeyType, DistanceType, LSHTable,
                                   ScalarType, DistanceFunction, DataStorage>>
        nn_query_pool(
            new LSHNNQueryPool<PointType, KeyType, DistanceType, LSHTable,
                               ScalarType, DistanceFunction, DataStorage>(
                *lsh_table_, num_probes, max_num_candidates, *data_storage_,
                num_query_objects));
    return std::move(nn_query_pool);
  }

  ~LSHNNTableWrapper() {}

 protected:
  std::vector<std::unique_ptr<LSHFunction> > lshes_;
  std::unique_ptr<LSHTable> lsh_table_;
  std::unique_ptr<HashTableFactory> hash_table_factory_;
  std::vector<std::unique_ptr<CompositeHashTable> > composite_tables_;
  std::unique_ptr<DataStorage> data_storage_;
};

template <typename PointType, typename KeyType, typename PointSet>
class StaticTableFactory {
 public:
  typedef typename PointTypeTraits<PointType>::ScalarType ScalarType;

  typedef typename DataStorageAdapter<PointSet>::template DataStorage<KeyType>
      DataStorageType;

  //using CompositeTableT  = CompositeTable<HashType, KeyType, HashTable>;

  StaticTableFactory(const PointSet& points,
                     const LSHConstructionParameters& params)
      : points_(points), params_(params) {}

  std::unique_ptr<LSHNearestNeighborTable<PointType, KeyType>> setup() {
    if (params_.dimension < 1) {
      throw LSHNNTableSetupError(
          "Point dimension must be at least 1. Maybe "
          "you forgot to set the point dimension in the parameter struct?");
    }
    
    if (params_.num_setup_threads < 0) {
      throw LSHNNTableSetupError(
          "The number of setup threads cannot be "
          "negative. Maybe you forgot to set num_setup_threads in the "
          "parameter struct? A value of 0 indicates that FALCONN should use "
          "the maximum number of available hardware threads.");
    }
    

    data_storage_ = std::move(
        DataStorageAdapter<PointSet>::template construct_data_storage<KeyType>(
            points_));

    

    //ComputeNumberOfHashBits<PointType> helper;
    //num_bits_ = 22; // Current hash bits is set to 20;
    num_bits_ = params_.hash_table_width;
    n_ = data_storage_->size();

    //std::unique_ptr<LSH> lsh;
    std::unique_ptr<typename HashTable::Factory> factory(
            new typename HashTable::Factory(1 << num_bits_));
    std::vector<std::unique_ptr<CompositeTable> > partition_tables;
    std::vector<std::unique_ptr<LSH> > partition_lshes;
    std::unique_ptr<LSHTableType> lsh_table;

    assert(params_.num_partitions == params_.hash_table_params.size());

    for (const auto& par_para: params_.hash_table_params) {
         if constexpr(std::is_same_v<CompositeTable, core::StaticPartition<HashType, KeyType, 
                                      PartitionMetric::PartitionMetricType, HashTable>>)
        {
          partition_tables.emplace_back(new CompositeTable(
              par_para.l, factory.get(), par_para.partition_lower, par_para.partition_upper));                                 
        } else {
          static_assert(false_constexpr<PointType>, "Unsupported Composite table!");
        }

         if constexpr(std::is_same_v<LSH, core::GaussianHashDense<CoordinateType, HashType> >)
        {
          partition_lshes.emplace_back(std::make_unique<LSH>(params_.dimension, par_para.k, par_para.l, params_.universe,
                                          params_.seed ^ 93384688, par_para.bucket_width, 
                                          params_.bucket_id_width, params_.hash_table_width) );
        } if constexpr(std::is_same_v<LSH, core::AXequalYHash<CoordinateType, HashType> >)
        {
          //auto precomputed = LSH::read_precomputed(params_.eigen_filename, par_para.k*par_para.l,
              //                find_next_power_of_two(params_.dimension), params_.fast_rotation));
          //auto hyperplanes = read_hyperplanes();
          partition_lshes.emplace_back(std::make_unique<LSH>(params_.dimension, par_para.k, par_para.l, params_.num_rotations,
                                          params_.seed ^ 93384688, par_para.bucket_width, 
                                          params_.hash_table_width, params_.dim_Arows,
                                           params_.second_step, params_.fast_rotation) );
        } else {
          static_assert(false_constexpr<PointType>, "Unsupported LSH Function!");
        }
    }

     if constexpr(std::is_same_v<LSHTableType, core::StaticPartitionLSHTable<PointType, KeyType, LSH, HashType,
                                 CompositeTable , MultiProbe, PartitionMetric, DataStorageType> >)
    {
      auto precomputed = LSH::read_precomputed(params_.eigen_filename, params_.num_hash_funcs, params_.num_rotations,
                         params_.rotation_dim, params_.dim_Acols,
                        params_.fast_rotation);
      auto pre_hash = std::make_unique<LSH::PreHasher>(std::move(precomputed), params_.num_hash_funcs, params_.dimension,
                            params_.dim_Arows, params_.second_step, params_.fast_rotation, 
                            params_.num_rotations, params_.seed ^ 93384688);
      ptrs_sanity_check(pre_hash);
      lsh_table = std::make_unique<LSHTableType>(partition_lshes, partition_tables, *data_storage_,
                         params_.num_setup_threads, params_.load_index,
                         params_.index_path + params_.index_filename, pre_hash);
    } else {
          static_assert(false_constexpr<PointType>, "Unsupported LSH Table!");
    }

    ptrs_sanity_check(factory);
    ptrs_sanity_check(lsh_table);
    ptrs_sanity_check(data_storage_);
    assert(params_.num_partitions == partition_tables.size());
    assert(params_.num_partitions == partition_lshes.size());
    for (const auto &p : partition_tables) ptrs_sanity_check(p);
    for (const auto &p : partition_lshes) ptrs_sanity_check(p);

    table_.reset(new LSHNNTableWrapper<PointType, KeyType, ScalarType,
                                       DistanceFunc, LSHTableType,
                                       LSH, typename HashTable::Factory,
                                       CompositeTable, DataStorageType>(
        std::move(partition_lshes), std::move(lsh_table), std::move(factory),
        std::move(partition_tables), std::move(data_storage_)));  

    return std::move(table_);
  }

 private:
 


  //std::vector<MatrixT> read_hyperplanes() {
  //  if (params_.fast_rotation) {
  //    return std::vector<MatrixT>();   
      // if fast rotation is true, hyperplane is no use, just return empty vector
  //  }

    // TODO Huayi: read the Gaussain matrices corresponding to the precomputed eigenvalues
    // from a file.

  //}


  



  const PointSet& points_;
  const LSHConstructionParameters& params_;
  std::unique_ptr<DataStorageType> data_storage_;
  int_fast32_t num_bits_;
  int_fast64_t n_;
  std::unique_ptr<LSHNearestNeighborTable<PointType, KeyType>> table_ = nullptr;
};

}  // namespace wrapper
}  // namespace falconn

namespace falconn {

template <typename PointType, typename KeyType, typename PointSet>
std::unique_ptr<LSHNearestNeighborTable<PointType, KeyType>> construct_table(
    const PointSet& points, const LSHConstructionParameters& params) {
  wrapper::StaticTableFactory<PointType, KeyType, PointSet> factory(points,
                                                                    params);
  return std::move(factory.setup());
}

template <typename PointSet>
std::vector<std::vector<KeyType> > normalize_data(PointSet& points, const LSHConstructionParameters& params) {
  std::vector<typename LSH::Normalizer> normalizers;
  std::vector<std::vector<KeyType> > filtered_keys(params.num_partitions);
  normalizers.reserve(params.num_partitions);

  for (auto par: params.hash_table_params) {
    normalizers.emplace_back(par.partition_upper - MARGIN, params.dimension);
  }

  for (int ptid = 0; ptid < points.size(); ++ptid) {
    auto& pt = points[ptid];
    auto pt_metric = PartitionMetric::eval(pt);
    int pp = 0;
    for (; pp < params.num_partitions; ++pp) {  // use linear search here
      if (pt_metric <= params.hash_table_params[pp].partition_upper - MARGIN) break;
    }
#if DEBUG
    std::cout << pt_metric << "\t" << params.hash_table_params[params.num_partitions-1].partition_upper << std::endl;
#endif
    NPP_ASSERT(pp < normalizers.size() &&  "Please increase the maximum norm!");
    normalizers[pp].normalize(pt, pt_metric);
    filtered_keys[pp].push_back(ptid);
  }
  
  return filtered_keys;
}

template <typename PointSet>
CoordinateType maximum_partition_metric(const PointSet &points) {
  CoordinateType val = 0.0;
  for (const auto& pt: points) {
    auto pt_metric = PartitionMetric::eval(pt);
    if (pt_metric > val) {
      val = pt_metric;
    }
  }
  return val;
}


std::vector<std::vector<CoordinateType> > compute_ground_truth(
 bool computation_needed, std::string filename, int num_neighbors, const std::vector<PointType>& points, 
 const std::vector<FalconnQueryType>& queries)
{
  std::vector<std::vector<CoordinateType> > result(queries.size());
  std::ifstream fin; 
  std::ofstream fout;
  if (computation_needed) {
    fout.open(filename);
    fout << queries.size() << " " << num_neighbors << std::endl;
    core::SimpleHeap<float, int> heap;
    heap.resize(points.size());

    for (int qid = 0; qid < queries.size(); ++qid) {
      result[qid].resize(num_neighbors);
      heap.reset();
      // sorting by distance, so distance is the key
      
      for (const auto& pt: points) {
        float distance = DistanceFunc::eval(queries[qid], pt); 
        heap.insert_unsorted(distance, 0);
      }
      heap.heapify();

      fout << qid << "\t";
      for (int nei_id = 0; nei_id < num_neighbors; ++ nei_id) {
        float distance;
        int key;
        heap.extract_min(&distance, &key);
        result[qid][nei_id] = distance;
        fout << distance << "\t";
      }
      fout << std::endl;
    }
  } else {
    fin.open(filename);
    assert(fin);
    int num_q_file, num_neigh_file;
    fin >> num_neigh_file >> num_q_file;
    assert(num_q_file == queries.size());
    assert(num_neigh_file == num_neighbors);

    for (int qid = 0; qid < queries.size(); ++qid) {
      result[qid].resize(num_neighbors);
      for (int nei_id = 0; nei_id < num_neighbors; ++ nei_id) {
        fin >> result[qid][nei_id];
      }
    }
  }
  return result;
}


}  // namespace falconn

#endif
