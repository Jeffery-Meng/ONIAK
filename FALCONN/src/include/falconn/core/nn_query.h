#ifndef __NN_QUERY_H__
#define __NN_QUERY_H__

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <iostream>

#include "../falconn_global.h"
#include "heap.h"
#include "data_storage.h"
#include <limits>

using std::runtime_error;
using std::string;
using std::exception;


namespace falconn {
  /*
 * An auxiliary function that reads a point from a binary file that is produced
 * by a script 'prepare-dataset.sh'
 */
template <typename T>
bool read_point(FILE *file, DenseVector<T> *point) {
  int d;
  if (fread(&d, sizeof(int), 1, file) != 1) {
    return false;
  }
  T *buf = new T[d];
  if (fread(buf, sizeof(T), d, file) != (size_t)d) {
    throw runtime_error("can't read a point");
  }
  point->resize(d);
  for (int i = 0; i < d; ++i) {
    (*point)[i] = buf[i];
  }
  delete[] buf;
  return true;
}

/*
 * An auxiliary function that reads a dataset from a binary file that is
 * produced by a script 'prepare-dataset.sh'
 */
void read_dataset(string file_name, std::vector<DenseVector<float>> *dataset) {
  FILE *file = fopen(file_name.c_str(), "rb");
  if (!file) {
    throw runtime_error("can't open the file with the dataset");
  }
  DenseVector<float> p;
  dataset->clear();
  while (read_point(file, &p)) {
    dataset->push_back(p);
  }
  if (fclose(file)) {
    throw runtime_error("fclose() error");
  }
}

namespace core {



template <typename LSHTableQuery, typename LSHTablePointType,
          typename LSHTableKeyType, typename ComparisonPointType,
          typename DistanceType, typename DistanceFunction,
          typename DataStorage>
class NearestNeighborQuery {
  
 public:
 typedef DenseMatrix<CoordinateType> QueryType;
  NearestNeighborQuery(LSHTableQuery* table_query,
                       const DataStorage& data_storage)
      : table_query_(table_query), data_storage_(data_storage) {}

  LSHTableKeyType find_nearest_neighbor(const QueryType& q,
                                        const QueryType& q_comp,
                                        int_fast64_t num_probes,
                                        int_fast64_t max_num_candidates) {
    auto start_time = std::chrono::high_resolution_clock::now();

    table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                        &candidates_);
    auto distance_start_time = std::chrono::high_resolution_clock::now();

    // TODO: use nullptr for pointer types
    LSHTableKeyType best_key = -1;

    if (candidates_.size() > 0) {
      typename DataStorage::SubsequenceIterator iter =
          data_storage_.get_subsequence(candidates_);

      best_key = candidates_[0];
      DistanceType best_distance = dst_(q_comp, iter.get_point());
      ++iter;

      // printf("%d %f\n", candidates_[0], best_distance);

      while (iter.is_valid()) {
        DistanceType cur_distance = dst_(q_comp, iter.get_point());
        // printf("%d %f\n", iter.get_key(), cur_distance);
        if (cur_distance < best_distance) {
          best_distance = cur_distance;
          best_key = iter.get_key();
          // printf("  is new best\n");
        }
        ++iter;
      }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_distance =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time - distance_start_time);
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    stats_.average_distance_time += elapsed_distance.count();
    stats_.average_total_query_time += elapsed_total.count();

    return best_key;
  }

  void find_k_nearest_neighbors(const QueryType& q,
                                const QueryType& q_comp,
                                int_fast64_t k, int_fast64_t num_probes,
                                int_fast64_t max_num_candidates,
                                std::vector<LSHTableKeyType>* result) {
    if (result == nullptr) {
      throw NearestNeighborQueryError("Results vector pointer is nullptr.");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<LSHTableKeyType>& res = *result;
    res.clear();

    table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                        &candidates_);

    heap_.reset();
    heap_.resize(k);

    auto distance_start_time = std::chrono::high_resolution_clock::now();

    typename DataStorage::SubsequenceIterator iter =
        data_storage_.get_subsequence(candidates_);

    int_fast64_t initially_inserted = 0;
    for (; initially_inserted < k; ++initially_inserted) {
      if (iter.is_valid()) {
        heap_.insert_unsorted(-dst_(q_comp, iter.get_point()), iter.get_key());
        ++iter;
      } else {
        break;
      }
    }

    if (initially_inserted >= k) {
      heap_.heapify();
      while (iter.is_valid()) {
        DistanceType cur_distance = dst_(q_comp, iter.get_point());
        if (cur_distance < -heap_.min_key()) {
          heap_.replace_top(-cur_distance, iter.get_key());
        }
        ++iter;
      }
    }

    res.resize(initially_inserted);
    std::sort(heap_.get_data().begin(),
              heap_.get_data().begin() + initially_inserted);
    for (int_fast64_t ii = 0; ii < initially_inserted; ++ii) {
      res[ii] = heap_.get_data()[initially_inserted - ii - 1].data;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_distance =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time - distance_start_time);
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    stats_.average_distance_time += elapsed_distance.count();
    stats_.average_total_query_time += elapsed_total.count();
  }

  void find_near_neighbors(const QueryType& q,
                           const QueryType& q_comp,
                           DistanceType threshold, int_fast64_t num_probes,
                           int_fast64_t max_num_candidates,
                           std::vector<LSHTableKeyType>* result) {
    if (result == nullptr) {
      throw NearestNeighborQueryError("Results vector pointer is nullptr.");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<LSHTableKeyType>& res = *result;
    res.clear();

    table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                        &candidates_);
    auto distance_start_time = std::chrono::high_resolution_clock::now();

    typename DataStorage::SubsequenceIterator iter =
        data_storage_.get_subsequence(candidates_);
    while (iter.is_valid()) {
      DistanceType cur_distance = dst_(q_comp, iter.get_point());
      if (cur_distance < threshold) {
        res.push_back(iter.get_key());
      }
      ++iter;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_distance =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time - distance_start_time);
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    stats_.average_distance_time += elapsed_distance.count();
    stats_.average_total_query_time += elapsed_total.count();
  }

  void get_candidates_with_duplicates(const QueryType& q,
                                      int_fast64_t num_probes,
                                      int_fast64_t max_num_candidates,
                                      std::vector<LSHTableKeyType>* result) {
    auto start_time = std::chrono::high_resolution_clock::now();

    table_query_->get_candidates_with_duplicates(q, num_probes,
                                                 max_num_candidates, result);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    stats_.average_total_query_time += elapsed_total.count();
  }

  void get_unique_candidates(const QueryType& q,
                             int_fast64_t num_probes,
                             int_fast64_t max_num_candidates,
                             std::vector<LSHTableKeyType>* result) {
    auto start_time = std::chrono::high_resolution_clock::now();

    table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                        result);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    stats_.average_total_query_time += elapsed_total.count();
  }

  void reset_query_statistics() {
    table_query_->reset_query_statistics();
    stats_.reset();
  }

  QueryStatistics get_query_statistics() {
    QueryStatistics res = table_query_->get_query_statistics();
    res.average_total_query_time = stats_.average_total_query_time;
    res.average_distance_time = stats_.average_distance_time;

    if (res.num_queries > 0) {
      res.average_total_query_time /= res.num_queries;
      res.average_distance_time /= res.num_queries;
    }
    return res;
  }

 private:
  LSHTableQuery* table_query_;
  const DataStorage& data_storage_;
  std::vector<LSHTableKeyType> candidates_;
  DistanceFunction dst_;
  SimpleHeap<DistanceType, LSHTableKeyType> heap_;

  QueryStatistics stats_;
};

template <typename LSHTableQuery, typename LSHTablePointType,
          typename LSHTableKeyType, typename ComparisonPointType,
          typename DistanceType, typename DistanceFunction,
          typename DataStorage>
class FilteringQuery {
 public:
 typedef DenseMatrix<CoordinateType> QueryType;
  FilteringQuery(LSHTableQuery* table_query,
                       const DataStorage& data_storage, unsigned num_filters, float recall_target)
      : table_query_(table_query), data_storage_(data_storage),num_filters_(num_filters),recall_target_(recall_target) {

        // read filter data 
        std::string filename_data ("filter_data.fvecs");
        read_dataset(filename_data, &filters_data_);

        // read filter query 
        std::string filename_query ("filter_query.fvecs");
        read_dataset(filename_query ,&filters_queries_);


        // read chi-squared inverse value
        std::ifstream infile("inverse_chi_squared.txt");
        float temp1;
        int temp2;
        float temp3;
        while(infile >> temp1 >> temp2 >> temp3){
          if (temp1 == recall_target_ && temp2 == num_filters_){
              chi_squared_value_ = temp3;
            }
          } 

      }

  void find_k_nearest_neighbors(const QueryType& q,
                                const QueryType& q_comp,
                                int_fast64_t k, int_fast64_t num_probes,
                                int_fast64_t max_num_candidates,
                                std::vector<LSHTableKeyType>* result) {
    if (result == nullptr) {
      throw NearestNeighborQueryError("Results vector pointer is nullptr.");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<LSHTableKeyType>& res = *result;
    res.clear();

    table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                        &candidates_);

    heap_.reset();
    heap_.resize(k);

    auto distance_start_time = std::chrono::high_resolution_clock::now();

    typename DataStorage::SubsequenceIterator iter =
        data_storage_.get_subsequence(candidates_);

    int_fast64_t initially_inserted = 0;
    for (; initially_inserted < k; ++initially_inserted) {
      if (iter.is_valid()) {
        heap_.insert_unsorted(-dst_(q_comp, iter.get_point()), iter.get_key());
        ++iter;
      } else {
        break;
      }
    }

    if (initially_inserted >= k) {
      heap_.heapify();
      while (iter.is_valid()) {
        DistanceType cur_distance = dst_(q_comp, iter.get_point());
        if (cur_distance < -heap_.min_key()) {
          heap_.replace_top(-cur_distance, iter.get_key());
        }
        ++iter;
      }
    }

    res.resize(initially_inserted);
    std::sort(heap_.get_data().begin(),
              heap_.get_data().begin() + initially_inserted);
    for (int_fast64_t ii = 0; ii < initially_inserted; ++ii) {
      res[ii] = heap_.get_data()[initially_inserted - ii - 1].data;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_distance =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time - distance_start_time);
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    stats_.average_distance_time += elapsed_distance.count();
    stats_.average_total_query_time += elapsed_total.count();
  }

  void find_k_nearest_neighbors_without_threshold(const QueryType& q,
                                const QueryType& q_comp,
                                int_fast64_t k, int_fast64_t num_probes,
                                int_fast64_t max_num_candidates,
                                std::vector<LSHTableKeyType>* result,unsigned q_cnt) {
    if (result == nullptr) {
      throw NearestNeighborQueryError("Results vector pointer is nullptr.");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<LSHTableKeyType>& res = *result;
    res.clear();

    table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                        &candidates_);
    heap_.reset();
    heap_.resize(k);
    hash_heap_.reset();
    hash_heap_.resize(k);


    int_fast64_t initially_inserted = 0;

    // If candidates less than k, than stop and return the result
    if (candidates_.size()<=k)
    {
      typename DataStorage::SubsequenceIterator iter = data_storage_.get_subsequence(candidates_);
      for (; initially_inserted < k; ++initially_inserted) {
      if (iter.is_valid()) {
        heap_.insert_unsorted(-dst_(q_comp, iter.get_point()), iter.get_key());
        ++iter;
      } else {
        break;
      }
      }
    } 
    
    // Other cases
    else {
    // The k-th distance estimator
    // typename DataStorage::SubsequenceIterator iter =
    //     data_storage_.get_subsequence(candidates_);
    // First loop to get the estimation of kth distance
    DenseVector<float> q_filter = filters_queries_[q_cnt];
    std::vector<float> hash_distances(filters_data_.size());
    for(auto candidate:candidates_)
    {
      float cur_hash_distance = dst_(filters_data_[candidate],q_filter);
      hash_distances[candidate] = cur_hash_distance;
      if (initially_inserted <k)
      {
      hash_heap_.insert_unsorted(-cur_hash_distance, candidate);
      }
      else{
        //hash_heap_.heapify();
        if (cur_hash_distance < -hash_heap_.min_key()) {
          hash_heap_.replace_top(-cur_hash_distance,candidate);
        }
      }
      initially_inserted++;
    }


    // Get the top k canidates on hash distances
    std::vector<LSHTableKeyType> hash_candidates;
    LSHTableKeyType hash_candidate;
    float hash_distance_temp;
    for(auto i=0;i<k;i++)
    {
      hash_heap_.extract_min(&hash_distance_temp,&hash_candidate);
      hash_candidates.push_back(hash_candidate);
    }

    // The k-th distance estimator
    typename DataStorage::SubsequenceIterator iter = data_storage_.get_subsequence(hash_candidates);

    
    DistanceType estimate_distance= 0.0;
    while (iter.is_valid()) {
        DistanceType cur_distance_temp = dst_(q_comp, iter.get_point());
        if (cur_distance_temp >= estimate_distance)
        {
          estimate_distance = cur_distance_temp;
        }
        ++iter;
      } 

    // Loop through the whole dataset
    auto distance_start_time = std::chrono::high_resolution_clock::now();
    initially_inserted = 0;
    iter = data_storage_.get_subsequence(candidates_);
    int cnt_temp = 0;
    while (iter.is_valid()) {
    // The filtering step 
    if (hash_distances[iter.get_key()]<= estimate_distance*chi_squared_value_)
    // if(true)
    {
      cnt_temp++;
       DistanceType cur_distance = dst_(q_comp, iter.get_point());
       if (initially_inserted<k-1){
        heap_.insert_unsorted(-cur_distance, iter.get_key());
       } 
       else if (initially_inserted == k-1) {
          heap_.insert_unsorted(-cur_distance, iter.get_key());
          heap_.heapify();
       }
       else{
          if (cur_distance < -heap_.min_key()) {
          heap_.replace_top(-cur_distance, iter.get_key());
          heap_.heapify(); 
        }

        // update estimated distance
        if (-heap_.min_key()<estimate_distance) {
          estimate_distance = -heap_.min_key();
        }
       }
       initially_inserted++;
    } 
        ++iter;
        
    }
    initially_inserted = k;
    auto end_time1 = std::chrono::high_resolution_clock::now();
    auto elapsed_distance =
      std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time1 - distance_start_time);
    stats_.average_distance_time += elapsed_distance.count();
    }


    res.resize(initially_inserted);
    std::sort(heap_.get_data().begin(),
              heap_.get_data().begin() + initially_inserted);
    for (int_fast64_t ii = 0; ii < initially_inserted; ++ii) {
      res[ii] = heap_.get_data()[initially_inserted - ii - 1].data;

    }


    auto end_time2 = std::chrono::high_resolution_clock::now();

    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time2 -
                                                                  start_time);

    stats_.average_total_query_time += elapsed_total.count();
  }

  // Finish later
  void find_k_nearest_neighbors_with_threshold(const QueryType& q,
                                const QueryType& q_comp,
                                int_fast64_t k, int_fast64_t num_probes,
                                int_fast64_t max_num_candidates,
                                std::vector<LSHTableKeyType>* result,unsigned q_cnt, float t) {
                              }

  void reset_query_statistics() {
    table_query_->reset_query_statistics();
    stats_.reset();
  }

  QueryStatistics get_query_statistics() {
    QueryStatistics res = table_query_->get_query_statistics();
    res.average_total_query_time = stats_.average_total_query_time;
    res.average_distance_time = stats_.average_distance_time;

    if (res.num_queries > 0) {
      res.average_total_query_time /= res.num_queries;
      res.average_distance_time /= res.num_queries;
    }
    return res;
  }

 private:
  LSHTableQuery* table_query_;
  const DataStorage& data_storage_;
  std::vector<LSHTableKeyType> candidates_;
  DistanceFunction dst_;
  SimpleHeap<DistanceType, LSHTableKeyType> heap_;

  std::vector<DenseVector<float>> filters_data_; // The addtional filtering hash vectors of data
  std::vector<DenseVector<float>> filters_queries_; // The addtional filtering hash vectors of queries
  std::vector<float> hash_distances_; // A vector for all hash distances
  float chi_squared_value_; // The corresponding inverse chi squared distribution
  SimpleHeap<float, LSHTableKeyType> hash_heap_; // heap for raw hash vector
  unsigned num_filters_; // Number of filters
  float recall_target_; // target recall

  QueryStatistics stats_;
};

}  // namespace core
}  // namespace falconn

#endif
