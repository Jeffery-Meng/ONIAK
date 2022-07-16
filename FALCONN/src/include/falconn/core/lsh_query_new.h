#ifndef __LSH_QUERY_NEW_H__
#define __LSH_QUERY_NEW_H__

#include "../falconn_global.h"
#include "lsh_function_helpers.h"
#include <fstream>

#include <chrono>

namespace falconn {
namespace core {


// Helper class that contains the actual per-query state of an LSH function
// object (the transformed input point, the temporary datat of the
// transformation, and the multiprobe object).
// The helper class also has functions for retrieving the probing sequence,
// either in a "lazy" probe-by-probe way or with a "batch" method for a fixed
// number of probes.
template <typename HashFunction, class MultiProbe>
class HashObjectQuery2 {
 private:
  typedef typename HashFunction::TransformedVectorType TransformedVectorType;
  typedef typename HashFunction::HashTransformation HashTransformation;

 public:
  typedef typename HashFunction::HashType HashType;
  typedef typename HashFunction::VectorType VectorType;
  typedef typename HashFunction::QueryType QueryType;
  typedef typename HashFunction::DataType DataType;

  class ProbingSequenceIterator
      : public std::iterator<std::forward_iterator_tag,
                             std::pair<HashType, int_fast32_t> > {
   public:
    ProbingSequenceIterator(HashObjectQuery2* parent = nullptr)
        : parent_(parent) {
      if (parent_ != nullptr) {
        if (!parent_->multiprobe_.get_next_probe(&cur_val_.first,
                                                 &cur_val_.second)) {
          parent_ = nullptr;
        }
      }
    }

    // TODO: should also check cur_val for general use?
    bool operator==(const ProbingSequenceIterator& rhs) const {
      return parent_ == rhs.parent_;
    }

    // TODO: should also check cur_val for general use?
    bool operator!=(const ProbingSequenceIterator& rhs) const {
      return parent_ != rhs.parent_;
    }

    typename std::iterator<std::forward_iterator_tag,
                           std::pair<HashType, int_fast32_t>>::reference
    operator*() const {
      return cur_val_;
    }

    typename std::iterator<std::forward_iterator_tag,
                           std::pair<HashType, int_fast32_t>>::pointer
    operator->() {
      return &cur_val_;
    }

    ProbingSequenceIterator& operator++() {
      if (!parent_->multiprobe_.get_next_probe(&cur_val_.first,
                                               &cur_val_.second)) {
        parent_ = nullptr;
      }
      return *this;
    }

   private:
    HashObjectQuery2* parent_;
    std::pair<HashType, int_fast32_t> cur_val_;
  };

  HashObjectQuery2(const HashFunction& parent, unsigned num_probes)
      : parent_(parent), multiprobe_(parent, num_probes), hash_transformation_(parent) {
    parent_.reserve_transformed_vector_memory(&transformed_vector_);
  }

  template <typename Derived>
  std::pair<ProbingSequenceIterator, ProbingSequenceIterator>
  get_probing_sequence(const Eigen::MatrixBase<Derived> & point) {
    hash_transformation_.apply(point, &transformed_vector_);
    multiprobe_.setup_probing(transformed_vector_, -1);
    return std::make_pair(ProbingSequenceIterator(this),
                          ProbingSequenceIterator(nullptr));
  }

  template <typename Derived>
  void get_transformed_vector(const Eigen::MatrixBase<Derived>& point){
    hash_transformation_.apply(point, &transformed_vector_);
  }

  //template <typename Derived>
  //void get_pre_transformation(const Eigen::MatrixBase<Derived>& point){
  //  hash_transformation_.apply_pre(point, &transformed_vector_);
  //}

  void get_probes_by_table(std::vector<std::vector<HashType>>* probes,
                           int_fast64_t num_probes) {
    if (num_probes < parent_.get_l()) {
      throw LSHFunctionError(
          "Number of probes must be at least "
          "the number of tables.");
    }

   // static std::ofstream fout("multiprobe_bucket.txt");

    if (static_cast<int_fast64_t>(probes->size()) != parent_.get_l()) {
      probes->resize(parent_.get_l());
    }
    for (size_t ii = 0; ii < probes->size(); ++ii) {
      (*probes)[ii].clear();
    }

    multiprobe_.setup_probing(transformed_vector_, num_probes);

    int_fast32_t cur_table;
    HashType cur_probe;


    for (int_fast64_t ii = 0; ii < num_probes; ++ii) {
      if (!multiprobe_.get_next_probe(&cur_probe, &cur_table)) {
        continue;
      }
      // printf("%u %d\n", cur_probe, cur_table);
      
     // if (ii < 10){
     //   fout << cur_probe << "\t";
      //}
      (*probes)[cur_table].push_back(cur_probe);
    }
 // fout << std::endl;
  }

 private:
  const HashFunction& parent_;
  MultiProbe multiprobe_;
  HashTransformation hash_transformation_;
  TransformedVectorType transformed_vector_;

};

}  // namespace core
}  // namespace falconn

#endif
