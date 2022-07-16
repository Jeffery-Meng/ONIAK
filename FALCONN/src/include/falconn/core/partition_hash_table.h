#ifndef __PARTITION_HASH_TABLE_H__
#define __PARTITION_HASH_TABLE_H__

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <Exception.h>

#include "../falconn_global.h"
#include "data_storage.h"
#include "lsh_table.h"
#include "lsh_query_new.h"

namespace falconn {
namespace core {


template <typename PointType,  // the type of the data points to be stored
          typename KeyType,    // must be integral for a static table
          typename LSH,        // the LSH family
          typename HashType,   // type returned by a set of k LSH functions
          typename HashTable,  // the low-level hash tables
          typename MultiProbe,
          typename PartitionMetric,
          typename DataStorageType = ArrayDataStorage<PointType, KeyType>>
class StaticPartitionLSHTable {
 public:
 typedef typename PartitionMetric::PartitionMetricType PartitionMetricT;
 typedef typename LSH::QueryType QueryType;

  StaticPartitionLSHTable(const std::vector<std::unique_ptr<LSH>> & lshes, 
                  const std::vector<std::unique_ptr<HashTable>> &hash_tables, const  DataStorageType& points,
                  int_fast32_t num_setup_threads, bool load_index, std::string filename,
                  std::unique_ptr<typename LSH::PreHasher>& prehash)
      :    n_(points.size()),  points_(points), num_partitions_(lshes.size()),
          pre_hasher_(std::move(prehash)), filtered_keys_(std::move(global::filtered_keys)) {

    assert(num_partitions_ == hash_tables.size());
    assert(num_partitions_ == filtered_keys_.size());
    assert(n_ == std::accumulate(filtered_keys_.begin(), filtered_keys_.end(), 0, 
        [](int res, const std::vector<KeyType>& a){return res + a.size();}));
    lshes_.reserve(num_partitions_);
    composite_tables_.reserve(num_partitions_);
    normalizers_.reserve(num_partitions_);

    for (const auto & lsh: lshes) {
      lshes_.push_back(lsh.get());
    }
    for (const auto & ht: hash_tables) {
      composite_tables_.push_back(ht.get());
      normalizers_.emplace_back(ht->get_upper(), lshes[0]->get_dimension());
    }

    if (num_setup_threads < 0) {
      throw LSHTableError("Number of setup threads cannot be negative.");
    }
    if (num_setup_threads == 0) {
      num_setup_threads = std::max(1u, std::thread::hardware_concurrency());
    }
    int_fast32_t num_par = composite_tables_.size();
    // indexing works are divided by partition

    num_setup_threads = std::min(num_par, num_setup_threads);
    int_fast32_t num_tables_per_thread = num_par / num_setup_threads;
    int_fast32_t num_leftover_tables = num_par % num_setup_threads;

    std::vector<std::future<void>> thread_results;
    int_fast32_t next_table_range_start = 0;

    for (int_fast32_t ii = 0; ii < num_setup_threads; ++ii) {
      int_fast32_t next_table_range_end =
          next_table_range_start + num_tables_per_thread - 1;
      if (ii < num_leftover_tables) {
        next_table_range_end += 1;
      }
      
      thread_results.push_back(std::async(
          std::launch::async, &StaticPartitionLSHTable::setup_table_range, this,
          next_table_range_start, next_table_range_end, points, load_index, filename));
      next_table_range_start = next_table_range_end + 1;
    }

    for (int_fast32_t ii = 0; ii < num_setup_threads; ++ii) {
      thread_results[ii].get();
    }
  }

/*
  void add_table() {
    typename LSH::template BatchHash<DataStorageType> bh(*(this->lsh_));
    std::vector<HashType> table_hashes;
    bh.batch_hash_single_table(points_, (this->lsh_)->get_l() - 1,
                               &table_hashes);
    this->hash_table_->add_entries_for_table(table_hashes,
                                             (this->lsh_)->get_l() - 1);
  }*/



  // TODO: add query statistics back in
  class Query {
   public:
    Query(const StaticPartitionLSHTable& parent, unsigned num_probes)
        : parent_(parent), is_candidate_(parent.n_) 
        {
            for (auto lsh_ptr: parent.lshes_){
              lsh_query_.emplace_back(*lsh_ptr, num_probes);
            }
        }


    void get_candidates_with_duplicates(const QueryType& p,
                                        int_fast64_t num_probes,
                                        int_fast64_t max_num_candidates,
                                        std::vector<KeyType>* result) {}
                                        //Deleted for now
   /*   if (result == nullptr) {
        throw LSHTableError("Results vector pointer is nullptr.");
      }

      stats_.num_queries += 1;

      auto start_time = std::chrono::high_resolution_clock::now();

      lsh_query_.get_transformed_vector(p);

      auto lsh_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_lsh =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              lsh_end_time - start_time);
      stats_.average_lsh_time += elapsed_lsh.count();

      lsh_query_.get_probes_by_table(&tmp_probes_by_table_, num_probes);

      auto multiprobe_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_multiprobe =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              multiprobe_end_time - lsh_end_time);
      stats_.average_multiprobe_time += elapsed_lsh.count();

      hash_table_iterators_ =
          parent_.hash_table_->retrieve_bulk(tmp_probes_by_table_);

      int_fast64_t num_candidates = 0;
      result->clear();
      if (max_num_candidates < 0) {
        max_num_candidates = std::numeric_limits<int_fast64_t>::max();
      }
      while (num_candidates < max_num_candidates &&
             hash_table_iterators_.first != hash_table_iterators_.second) {
        num_candidates += 1;
        result->push_back(*(hash_table_iterators_.first));
        ++hash_table_iterators_.first;
      }

      auto hashing_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_hashing =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              hashing_end_time - lsh_end_time);
      stats_.average_hash_table_time += elapsed_hashing.count();

      auto sketches_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_sketches =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              sketches_end_time - hashing_end_time);
      stats_.average_sketches_time += elapsed_sketches.count();

      stats_.average_num_candidates += num_candidates;

      auto end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_total =
          std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                    start_time);
      stats_.average_total_query_time += elapsed_total.count();
    }*/

    void get_unique_candidates(const QueryType& p, int_fast64_t num_probes,
                               int_fast64_t max_num_candidates,
                               std::vector<KeyType>* result) {
      if (result == nullptr) {
        throw LSHTableError("Results vector pointer is nullptr.");
      }

      auto start_time = std::chrono::high_resolution_clock::now();
      stats_.num_queries += 1;

      DenseVector<float> balancing_metric;
      std::vector<KeyType> result_inner;
      query_counter_ += 1;

      //auto prehash_values = parent_.pre_hasher_->template pre_hash_all_query<PartitionMetric>(p, balancing_metric);

      auto lsh_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_lsh =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              lsh_end_time - start_time);
      stats_.average_lsh_time += elapsed_lsh.count();

      for (int pp = 0; pp < parent_.num_partitions_; ++pp) {
        // apply alpha and beta to the transformed query
        // Need to initalize c before transformation to use letter.
        double c;
        auto transformed_query = query_transform(p, parent_.normalizers_[pp], c);
    
        PartitionMetricT p_lower = parent_.composite_tables_[pp]->get_lower();
        PartitionMetricT p_upper = parent_.composite_tables_[pp]->get_upper();
        int l_pp = parent_.composite_tables_[pp]->get_l();
        int num_probes_p = par_metric_.compute_num_probes(balancing_metric, p_lower, p_upper, l_pp);

        auto prehash_values = parent_.pre_hasher_->template pre_hash_all_query<PartitionMetric>(transformed_query, balancing_metric);
        //auto transformed_prehash = parent_.normalizers_[pp].query_balance(prehash_values, balancing_metric); no need for this
        QueryType cI = Eigen::MatrixXf::Identity(p.cols()+1, p.cols()+1) * sqrt(c);
        auto prehash_cI = parent_.pre_hasher_->template pre_hash_all_query<PartitionMetric>(cI, balancing_metric);
        auto prehashed = prehash_values - prehash_cI;

        get_unique_candidates_internal(prehashed, pp, num_probes_p, max_num_candidates, &result_inner);
        result->insert(result->end(), result_inner.begin(), result_inner.end());

      }

      auto end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_total =
          std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                    start_time);
      stats_.average_total_query_time += elapsed_total.count();
    }

    QueryType query_transform(const QueryType& query_matrix, const typename LSH::Normalizer& normalizer, double & c_in) {
      DenseMatrix<double> transformed_query =  query_matrix.template cast<double>();
      double alpha = normalizer.get_alpha();
      transformed_query.col(transformed_query.cols()-1) /= alpha;
      auto [beta, c] = normalizer.compute_beta_c(transformed_query);
      transformed_query *= beta;
      c_in = c;
      return transformed_query.template cast<float> ();
    }

    void reset_query_statistics() { stats_.reset(); }

    QueryStatistics get_query_statistics() {
      QueryStatistics res = stats_;
      res.compute_averages();
      return res;
    }

    // TODO: add void get_candidate_sequence(const PointType& p)
    // TODO: add void get_unique_candidate_sequence(const PointType& p)

   private:
    const StaticPartitionLSHTable& parent_;
    int_fast32_t query_counter_ = 0;
    std::vector<int32_t> is_candidate_;
    std::vector<HashObjectQuery2<LSH, MultiProbe> > lsh_query_;
    std::vector<std::vector<HashType>> tmp_probes_by_table_;
    std::pair<typename HashTable::Iterator, typename HashTable::Iterator>
        hash_table_iterators_;
    PartitionMetric par_metric_;
    
    QueryStatistics stats_;


    template <typename Derived>
    void get_unique_candidates_internal(const Eigen::MatrixBase<Derived> & p, int pp, 
                                        int_fast64_t num_probes,
                                        int_fast64_t max_num_candidates,
                                        std::vector<KeyType>* result) {
      auto start_time = std::chrono::high_resolution_clock::now();

      lsh_query_[pp].get_transformed_vector(p);

      auto lsh_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_lsh =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              lsh_end_time - start_time);
      stats_.average_lsh_time += elapsed_lsh.count();

      lsh_query_[pp].get_probes_by_table(&tmp_probes_by_table_, num_probes);

      auto multiprobe_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_multiprobe =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              multiprobe_end_time - lsh_end_time);
      stats_.average_multiprobe_time += elapsed_multiprobe.count();

      //for (auto id : tmp_probes_by_table_[0]){
      //  std::cout << id << " ";
      //}
      // std::cout << std::endl;

      hash_table_iterators_ =
          parent_.composite_tables_[pp]->retrieve_bulk(tmp_probes_by_table_);
      

      int_fast64_t num_candidates = 0;
      result->clear();
      
      if (max_num_candidates < 0) {
        max_num_candidates = std::numeric_limits<int_fast64_t>::max();
      }
      while (num_candidates < max_num_candidates &&
             hash_table_iterators_.first != hash_table_iterators_.second) {
        num_candidates += 1;
        int_fast64_t cur = *(hash_table_iterators_.first);
        if (is_candidate_[cur] != query_counter_) {
          is_candidate_[cur] = query_counter_;
          result->push_back(cur);
        }

        ++hash_table_iterators_.first;
      }
      //std::cout << result->size() << std::endl;
      auto hashing_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_hashing =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              hashing_end_time - multiprobe_end_time);
      stats_.average_hash_table_time += elapsed_hashing.count();

      auto sketches_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_sketches =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              sketches_end_time - hashing_end_time);
      stats_.average_sketches_time += elapsed_sketches.count();

      stats_.average_num_candidates += num_candidates;
      stats_.average_num_unique_candidates += result->size();
    }
  };

 private:
  int_fast64_t n_;
  const int num_partitions_;
  const DataStorageType& points_;
  std::vector<LSH*> lshes_;
  std::vector<HashTable*> composite_tables_;
  std::unique_ptr<typename LSH::PreHasher> pre_hasher_;
  std::vector<typename LSH::Normalizer> normalizers_;
  std::vector< std::vector<KeyType> > filtered_keys_;
  //PartitionMetric par_met_;

  // from and to are id of hash tables
  void setup_table_range(int_fast32_t from, int_fast32_t to,
                         const DataStorageType& points, bool load_index, std::string filename) {
   
   // std::vector<std::vector<KeyType>> filtered_keys;
    std::ofstream fout;
   
    if (!load_index) {
     // filtered_keys =  PartitionMetric::template 
      //filter_in_range<DataStorageType, KeyType, typename LSH::Normalizer>(points, normalizers_);
      fout.open(filename);
    }

    std::cout << "Number of Points in each partition: " << std::endl;
    for (int pp = 0; pp < normalizers_.size(); ++pp) {
      std::cout << "Upper Limit: " << normalizers_[pp].get_mx() << "\t#" << filtered_keys_[pp].size() << std::endl;
    }


    for (int_fast32_t pp = from; pp <= to; ++pp) { // pp is the id of the partition
    // WARNING: currently parallel loading or saving is NOT supported!
    // In the future, different threads need to access different files.
      if (load_index){
        std::ifstream fin(filename);
        NPP_ENFORCE(fin);
        for (int_fast32_t ii = 0; ii < this->composite_tables_[pp]->get_l(); ++ii) {
          this->composite_tables_[pp]->add_entries_from_stream(fin, ii);
        }
      } else {
        PartitionMetricT p_lower = this->composite_tables_[pp]->get_lower();
        PartitionMetricT p_upper = this->composite_tables_[pp]->get_upper();

        
        
        //std::vector<KeyType> filtered_keys = filter_in_range(point_par_metrics, p_lower, p_upper);
        std::vector<HashType> table_hashes;
        
        typename LSH::template BatchHash<DataStorageType> bh(*(this->lshes_[pp]), *pre_hasher_);
        for (int_fast32_t ii = 0; ii < this->composite_tables_[pp]->get_l(); ++ii) {
          
          bh.batch_hash_single_table(points, ii, filtered_keys_[pp], &table_hashes);
        // TODO: May not need to compute for all hash values in the future
        // bh.batch_hash_single_table(points, ii, filtered_keys, &table_hashes);
          
          this->composite_tables_[pp]->add_entries_in_keys(table_hashes, filtered_keys_[pp], ii);
          this->composite_tables_[pp]->dump_table_to_stream(fout, ii);
        }
      }
    }
  }

};

}  // namespace core
}  // namespace falconn

#endif
