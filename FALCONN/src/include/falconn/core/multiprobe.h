#ifndef __MULTIPROBE_H__
#define __MULTIPROBE_H__

#include <vector>
#include "Eigen/Dense"
#include "wyhash32.h"
#include <fstream>

#include "heap.h"
#include "gaussian_hash.h"
#include "lsh_function_helpers.h"
// Set number of hash bits to 20
//const int hash_bits_sep = 21;
template<class C, class T>
auto contains(const C v, const T& x)
-> decltype(end(v), true)
{
    return end(v) != std::find(begin(v), end(v), x);
}

// template <typename CoordinateType>
// inline int_fast32_t hash_round_sep(CoordinateType coor, int_fast32_t levels){
//   int_fast32_t half_bucket_num = 1 << (levels-1);
//   int_fast32_t bucket_id = static_cast<int_fast32_t>(std::floor(coor)) + half_bucket_num;
//   if (bucket_id < 0){
//     bucket_id = 0;
//   } else if (bucket_id >= 2 * half_bucket_num){
//     bucket_id = 2 * half_bucket_num - 1;
//   }
//   return bucket_id;
// }


namespace falconn {
namespace core {

template <typename HashFunction>
class MultiProbeBase{
public:
    typedef typename HashFunction::CoordinateType CoordinateType;
     typedef typename HashFunction::HashType HashType;
     typedef  Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      TransformedVectorType;
    virtual void setup_probing(const TransformedVectorType& hash_vector,
                       int_fast64_t num_probes) = 0;
    virtual bool get_next_probe(HashType* cur_probe, int_fast32_t* cur_table) = 0;

};

/*template <typename HashFunction>
class CustomizedMultiProbe : public MultiProbeBase<HashFunction>{
   public:
     typedef typename HashFunction::CoordinateType CoordinateType;
     typedef typename HashFunction::HashType HashType;
     typedef typename HashFunction::HashTransformation HashTran;
     typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      TransformedVectorType;

    CustomizedMultiProbe(const HashFunction& parent)
        : hash_tran_(parent),
          k_(parent.k_),
          l_(parent.l_),
          num_probes_(0),
          cur_probe_counter_(0),
          sorted_hyperplane_indices_(parent.l_),
          bucket_id_width_(parent.bucket_id_width_),
          bucket_num_(1 << bucket_id_width_),
          hash_mask_(bucket_num_ - 1),
          hash_seed_(parent.seed_hash2_),
          main_table_probe_(parent.l_) {
      for (int_fast32_t ii = 0; ii < l_; ++ii) {
        sorted_hyperplane_indices_[ii].resize(2 * k_);
        // 0 to k-1 means rounding up, and k to 2k-1 means rounding down
        for (int_fast32_t jj = 0; jj < 2 * k_; ++jj) {
          // indices from 0 to k, not sorted by delta now
          sorted_hyperplane_indices_[ii][jj] = jj;
        }
      }

      distance_vector_.resize(2 * l_ * k_);
    }

//  set up the heaps for hash_vector (personalized probing sequence for hash_vector)
    void setup_probing(const TransformedVectorType& hash_vector,
                       int_fast64_t num_probes) override {
      hash_vector_ = &hash_vector;
      num_probes_ = num_probes;
      cur_probe_counter_ = -1;

      hash_tran_.round(hash_vector, main_table_probe_);

// indices for main table (non-multi-probe) buckets
        // main table probes are enough
      if (num_probes_ >= 0 && num_probes_ <= l_) {
        return;
      }
// get distance vector
      for (int_fast32_t ii = 0; ii < l_; ++ii){
        for (int_fast32_t jj = 0; jj < 2 * k_; ++jj) {
          distance_vector_(2 * ii * k_ + jj) = 
              distance_to_boundary(hash_vector(ii * k_ + jj % k_), jj >= k_);
        }
      }

//  sort the dimensions
// Get the sorted hash perturbation vector

      for (int_fast32_t ii = 0; ii < l_; ++ii) {
        GaussianComparator comp(distance_vector_, 2 * ii * k_); // sort within the ii-th table
        std::sort(sorted_hyperplane_indices_[ii].begin(),
                  sorted_hyperplane_indices_[ii].end(), comp);
      }

      if (num_probes_ >= 0) {
        heap_.resize(2 * num_probes_);
      }
      heap_.reset();
      for (int_fast32_t ii = 0; ii < l_; ++ii) {
        // insert the best perturbation vector (stored in hash_mask) of every hash table
        // score is squared distance (same to Gaussian multiprobe)
        int_fast32_t cur = 0;
        int_fast32_t best_index = next_valid_perturbation(ii, cur);
        CoordinateType score = distance_vector_[2 * ii * k_ + best_index];
        score = score * score;
        HashType wipe_mask = ~(hash_mask_ << bucket_id_width_ * (best_index % k_));
        HashType pert_mask = retrieve_hash_value(ii, best_index) + (best_index<k_? 1: -1);
        pert_mask <<= bucket_id_width_ * (best_index % k_);
        heap_.insert_unsorted(score, ProbeCandidate(ii, wipe_mask, pert_mask, cur));
      }
      heap_.heapify();
    }

  // return probe (the bucket of this probe) and table (by pointer)
    bool get_next_probe(HashType* cur_probe, int_fast32_t* cur_table) override {
      cur_probe_counter_ += 1;

      if (num_probes_ >= 0 && cur_probe_counter_ >= num_probes_) {
        // printf("out of probes\n");en
        return false;
      }

      if (cur_probe_counter_ < l_) {
        // printf("initial probes %lld\n", cur_probe_counter_);
        *cur_probe = main_table_probe_[cur_probe_counter_];
        *cur_table = cur_probe_counter_;
        return true;
      }

      if (heap_.empty()) {
        return false;
      }

      CoordinateType cur_score;
      ProbeCandidate cur_candidate;
      heap_.extract_min(&cur_score, &cur_candidate);
      *cur_table = cur_candidate.table_;
      int_fast32_t cur_index =
          sorted_hyperplane_indices_[*cur_table][cur_candidate.last_index_];
      *cur_probe = (main_table_probe_[*cur_table] & cur_candidate.wipe_mask_) | cur_candidate.pert_mask_;
      int_fast32_t cur_position = cur_candidate.last_index_ + 1;

      int_fast32_t next_index = next_valid_perturbation(*cur_table, cur_position);
      HashType new_pert = retrieve_hash_value(*cur_table, next_index) + (next_index<k_? 1: -1);
// maintain the heap structure
      if (next_index >= 0) {  // invalid if next_index < 0
        // shift: swapping out the last flipped index
        HashType next_wipe = cur_candidate.wipe_mask_, next_pert = cur_candidate.pert_mask_;
        CoordinateType next_score =
            cur_score - (distance_vector_[*cur_table * k_ * 2 + cur_index] *
                         distance_vector_[*cur_table * k_ * 2 + cur_index]);
        next_score += (distance_vector_[*cur_table * k_ * 2+ next_index] *
                       distance_vector_[*cur_table * k_ * 2+ next_index]);
        swap_out(next_wipe, next_pert, next_index);
        if (swap_in(next_wipe, next_pert, next_index, new_pert)){
          heap_.insert(next_score, ProbeCandidate(*cur_table, next_wipe, next_pert,
                                                next_index));
        }

        // expand: adding a new flipped index
        next_wipe = cur_candidate.wipe_mask_;
        next_pert = cur_candidate.pert_mask_;
        next_score =
            cur_score +(distance_vector_[*cur_table * k_ * 2 + next_index] *
                       distance_vector_[*cur_table * k_ * 2 + next_index]);
        if (swap_in(next_wipe, next_pert, next_index, new_pert)){
          heap_.insert(next_score, ProbeCandidate(*cur_table, next_wipe, next_pert,
                                                next_index));
        }
      }

      return true;
    }

       // last_index is used for generating the next perturbation vector
   // it is the largest (latest) dimension of the perturbation vector
   // wipe_mask is 1->unchanged 0-> changed, used to wipe out 
   // pert_mask is 0->unchanged [new hash bucket]-> changed
    class ProbeCandidate {
     public:
      ProbeCandidate(int_fast32_t table = 0, HashType wipe_mask = 0, HashType pert_mask = 0,
                     int_fast32_t last_index = 0)
          : table_(table), wipe_mask_(wipe_mask), pert_mask_(pert_mask), last_index_(last_index) {}

      int_fast32_t table_;
      HashType pert_mask_, wipe_mask_;
      int_fast32_t last_index_;
    };

    // checks against invalid perturbations: moving left of 0 or moving right of bucket_num
    // returns next valid perturbation after cur
    // invalid candidates are skipped
    int_fast32_t next_valid_perturbation(
        int_fast32_t l, int_fast32_t& cur){
      while(cur < 2 * k_){
        int_fast32_t candidate = sorted_hyperplane_indices_[l][cur];
        HashType hash_value_k = retrieve_hash_value(l, candidate);
        if ((candidate < k_ && hash_value_k < bucket_num_ -1) || (candidate >= k_ && hash_value_k > 0))
          return candidate;

        ++cur;
      }
      return -1; // not valid candidates, all posssible dimensions have been explored
    }
//  sort the indices in increasing order of absolute hash value
// The first (hash) dimension is the one closest to the boundary
    class GaussianComparator {
     public:
     // values is a set of distances
      GaussianComparator(const TransformedVectorType& values,
                           int_fast32_t offset)
          : values_(values), offset_(offset){}

      bool operator()(int_fast32_t ii, int_fast32_t jj) const {
        return values_[offset_ + ii] < values_[offset_ + jj];
      }

     private:
      const TransformedVectorType& values_;
      int_fast32_t offset_;
    };

    static CoordinateType distance_to_boundary(CoordinateType p, bool up_or_down){
      if (up_or_down){
        return std::ceil(p) - p;
      } else {
        return p - std::floor(p);
      }
    }

  // retrieves the hash value of the l-th table, k-th function from main_table_probe
    HashType retrieve_hash_value(int_fast32_t l, int_fast32_t k){
      k %= k_;
      HashType hash_value_k = main_table_probe_[l] & (hash_mask_ << k * bucket_id_width_);
      return hash_value_k >> k * bucket_id_width_;
    }

    void swap_out(HashType& wipe_mask, HashType& pert_mask, int_fast32_t k){
      k %= k_;
      wipe_mask |= hash_mask_ << k * bucket_id_width_; // reset wipe_mask bits to 1
      pert_mask &= ~wipe_mask;   // reset pert_mask bits to 0
    }

    // checks conflicting perturbation (+1 and -1 on the same dimension)
    bool swap_in(HashType& wipe_mask, HashType& pert_mask, int_fast32_t k, HashType new_pert){
      k %= k_;
      if ((~wipe_mask & hash_mask_ << k * bucket_id_width_) != 0) return false; // confliction
      wipe_mask &= ~(hash_mask_ << k * bucket_id_width_); // reset wipe_mask bits to 0
      pert_mask |= new_pert << k * bucket_id_width_;   // switch new perturbation value in
      return true;
    }

    int_fast32_t k_;
    int_fast32_t l_;
    int_fast64_t num_probes_;
    int_fast64_t cur_probe_counter_;
    uint_fast32_t hash_seed_;
    int_fast32_t bucket_id_width_, bucket_num_;
    // a mask used to retrieve the hash value of a certain hash function
    const HashType hash_mask_;
    // [table][index] sorted by increasing order to boundary
    std::vector<std::vector<int_fast32_t>> sorted_hyperplane_indices_;
    // l center buckets that are probed without multiprobing
    std::vector<HashType> main_table_probe_;
    SimpleHeap<CoordinateType, ProbeCandidate> heap_;
    // h(data), center of probing
    const TransformedVectorType* hash_vector_;
    HashTran hash_tran_;

    TransformedVectorType distance_vector_;
};*/


// Precomputed Sequence
template <typename HashFunction>
class PreComputedMultiProbe : public MultiProbeBase<HashFunction>{
   public:
     typedef typename HashFunction::CoordinateType CoordinateType;
     typedef typename HashFunction::HashType HashType;
     typedef typename HashFunction::HashTransformation HashTran;
     typedef typename HashFunction::QueryType QueryType;
     typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      TransformedVectorType;

    PreComputedMultiProbe(const HashFunction& parent, unsigned num_probes)
        : hash_tran_(parent),
          k_(parent.get_k()),
          l_(parent.get_l()),
          num_probes_(num_probes),
          cur_probe_counter_(0),
          sorted_hyperplane_indices_(parent.get_l()),
          //bucket_id_width_(parent.bucket_id_width_),
          bucket_width_(parent.get_bucket_width()),
          //bucket_num_(1 << bucket_id_width_),
          hash_seed_(parent.get_seed()),
          //hash_mask_(bucket_num_ - 1),
          hash_width_(parent.get_hash_width()),
          main_table_probe_(parent.get_l()) {
      for (int_fast32_t ii = 0; ii < l_; ++ii) {
        sorted_hyperplane_indices_[ii].resize(2 * k_);
        // 0 to k-1 means rounding up, and k to 2k-1 means rounding down
        for (int_fast32_t jj = 0; jj < 2 * k_; ++jj) {
          // indices from 0 to k, not sorted by delta now
          sorted_hyperplane_indices_[ii][jj] = jj;
        }

      }

    hash_vector_.resize(l_ * k_);

    //std::cout << "multiprobe \t" << hash_seed_ << std::endl;

    SimpleHeap<double,std::vector<int_fast32_t>> heap_temp;
      // Generate the precomputed perturbation vector
      // each table number of probes,set to 100 now
      int num_probes_each_table = num_probes / l_ - 1;
      // insert the best perturbation vector (stored in hash_mask) of every hash table
      // score is the precomputed value
      std::vector<int_fast32_t> temp_pert;
      temp_pert.push_back(1);
      double cur_score = 1*(1+1)*1.0/((4*(k_+1)*(k_+2))*1.0*(bucket_width_*bucket_width_));
      double temp_score;
      heap_temp.reset();
      heap_temp.insert_unsorted(cur_score,temp_pert);
      heap_temp.heapify();

      for(int_fast32_t i =0; i< num_probes_each_table;i++){
        int label = 0;
        while(label != 1) {
           heap_temp.extract_min(&cur_score, &temp_pert);
           // Shift on the vector
           std::vector<int_fast32_t> temp_shift_pert;
           temp_score = 0;
           for(int j=0;j<temp_pert.size();j++) {
             temp_shift_pert.push_back((j<temp_pert.size()-1)? temp_pert[j] : temp_pert[j]+1);
             temp_score += (temp_shift_pert.back()<=k_)? 1.0*temp_shift_pert.back()*(temp_shift_pert.back()+1)/(4*(k_+1)*(k_+2)*1.0)*1.0*(bucket_width_*bucket_width_): 
              1.0*(bucket_width_*bucket_width_)*(1-(2*k_+1-temp_shift_pert.back())*1.0/(k_+1)+(2*k_+1-temp_shift_pert.back())*(2*k_+2-temp_shift_pert.back())*1.0/(4*(k_+1)*(k_+2)));
           }
          heap_temp.insert_unsorted(temp_score,temp_shift_pert);
          heap_temp.heapify();

          // Expand the vector
          std::vector<int_fast32_t> temp_expand_pert;
          temp_score = 0;
          for(int j=0;j<=temp_pert.size();j++) {
              // <= sign is justifiable here, access is checked by the following tertiary expression
             temp_expand_pert.push_back((j<temp_pert.size())? temp_pert[j] : temp_pert[j-1]+1);
             temp_score += (temp_expand_pert.back()<=k_)? 1.0*temp_expand_pert.back()*(temp_expand_pert.back()+1)/(4*(k_+1)*(k_+2)*1.0)*1.0*(bucket_width_*bucket_width_): 
              1.0*(bucket_width_*bucket_width_)*(1-(2*k_+1-temp_expand_pert.back())*1.0/(k_+1)+(2*k_+1-temp_expand_pert.back())*(2*k_+2-temp_expand_pert.back())*1.0/(4*(k_+1)*(k_+2)));
           }
          heap_temp.insert_unsorted(temp_score,temp_expand_pert);
          heap_temp.heapify();
          // Verify the vector
          int temp_label = 1;
          for(int i=0;i<temp_pert.size();i++) {
              if (contains(temp_pert,temp_pert[i]) && contains(temp_pert,2*k_+1-temp_pert[i]))
              {
               temp_label = 0;
              }
            }
        if (temp_pert.back() > 2*k_) temp_label =0;
        if (temp_label == 1) break;
        }
        probes_vecs_.push_back(temp_pert);
      }

      distance_vector_.resize(2 * l_ * k_);
    }

//  set up the heaps for hash_vector (personalized probing sequence for hash_vector)
    void setup_probing(const TransformedVectorType& hash_vector,
                       int_fast64_t num_probes) override {
      // hash_vector_ is the hash value of each hash functions in each hash table
      //static std::ofstream fout("probe_hash.txt");
      for (int_fast32_t ii = 0; ii < l_; ++ii){
        for (int_fast32_t jj = 0; jj < k_; ++jj) {
          hash_vector_[ii * k_ + jj] = 
            static_cast<int_fast32_t>(std::floor(hash_vector(ii * k_ + jj)));
            //fout << hash_vector_[ii * k_ + jj] << "\t";
        }
      }

      //fout << std::endl;
      assert(num_probes > 0 && num_probes <= l_*probes_vecs_.size() + l_);

      num_probes_ = num_probes;
      cur_probe_counter_ = -1;

      hash_tran_.round(hash_vector, main_table_probe_);

// indices for main table (non-multi-probe) buckets
        // main table probes are enough
      if (num_probes_ >= 0 && num_probes_ <= l_) {
        return;
      }
// get distance vector
      for (int_fast32_t ii = 0; ii < l_; ++ii){
        for (int_fast32_t jj = 0; jj < 2 * k_; ++jj) {
          distance_vector_(2 * ii * k_ + jj) = 
              distance_to_boundary(hash_vector(ii * k_ + jj % k_), jj < k_);
        }
      }

//  sort the dimensions
      for (int_fast32_t ii = 0; ii < l_; ++ii) {
        GaussianComparator comp(distance_vector_, 2 * ii * k_); // sort within the ii-th table
        std::sort(sorted_hyperplane_indices_[ii].begin(),
                  sorted_hyperplane_indices_[ii].end(), comp);
      }

    }

  // return probe (the bucket of this probe) and table (by pointer)
    bool get_next_probe(HashType* cur_probe, int_fast32_t* cur_table) override {
      cur_probe_counter_ += 1;
      if (num_probes_ >= 0 && cur_probe_counter_ >= num_probes_) {
        // printf("out of probes\n");
        return false;
      }

      if (cur_probe_counter_ < l_) {
        // printf("initial probes %lld\n", cur_probe_counter_);
        *cur_probe = main_table_probe_[cur_probe_counter_];
        *cur_table = cur_probe_counter_;
        return true;
      }

      // In case empty
      if (probes_vecs_.empty()) {
        return false;
      }

      *cur_table  = cur_probe_counter_ % l_; // Current table position
      int cur_probes_num; 
      cur_probes_num = int(cur_probe_counter_ *1.0 / l_) - 1; // Current number of probe in each table
      *cur_probe = main_table_probe_[*cur_table] ;
      // The temp_hash_vector is the hash vector corresponding to the current table
      std::vector<int_fast32_t>::const_iterator first = hash_vector_.begin() + (*cur_table) * k_ ;
      std::vector<int_fast32_t>::const_iterator last = hash_vector_.begin() + (*cur_table + 1) * k_ ;
      std::vector<int> temp_hash_vector(first,last);

      // Now we generate the corresponding hash value of current perturbation vector
      for (int i=0;i<probes_vecs_[cur_probes_num].size();i++) {
        int pert_pos = probes_vecs_[cur_probes_num][i] - 1; //The sorted index to change
        int_fast32_t real_pos = sorted_hyperplane_indices_[*cur_table][pert_pos]; // The real position index to change
        // Current hash value in this hash function
        int_fast32_t cur_hash_value = hash_vector_[*cur_table * k_ + real_pos % k_];
        // Check overflow
        // if((cur_hash_value == 0) && (real_pos >= k_)) {return false;}
        // if((cur_hash_value == bucket_num_-1) && (real_pos < k_)) {return false;}
        int_fast32_t new_pert = cur_hash_value + (real_pos<k_? 1: -1); // The new perturbation
        // Change the corresponding hash vector
        temp_hash_vector[real_pos % k_] = new_pert;
        // // Generate the new pert_mask and change the corresponding positions
        // int_fast32_t k = real_pos % k_; 
        // HashType wipe_mask,pert_mask;
        // wipe_mask = ~(hash_mask_ << (k_-k-1) * bucket_id_width_); // reset wipe_mask corresponding bits to 0
        // pert_mask = new_pert << (k_-k-1) * bucket_id_width_;   // switch new perturbation value in
        // *cur_probe =  (*cur_probe & wipe_mask) | pert_mask;
      }

      // calculated the new hash value
      //for (auto bucket: temp_hash_vector){
       // std::cout << bucket << " ";
      //}
      
      *cur_probe = wyhash32(temp_hash_vector.data(), this->k_ * sizeof(int), hash_seed_);
      //std::cout << sizeof(int_fast32_t) << std::endl;
      *cur_probe &= (1<<hash_width_)-1;


//       CoordinateType cur_score;
//       ProbeCandidate cur_candidate;
//       heap_.extract_min(&cur_score, &cur_candidate);
//       *cur_table = cur_candidate.table_;
//       int_fast32_t cur_index =
//           sorted_hyperplane_indices_[*cur_table][cur_candidate.last_index_];
//       *cur_probe = (main_table_probe_[*cur_table] & cur_candidate.wipe_mask_) | cur_candidate.pert_mask_;
//       int_fast32_t cur_position = cur_candidate.last_index_ + 1;

//       int_fast32_t next_index = next_valid_perturbation(*cur_table, cur_position);
//       HashType new_pert = retrieve_hash_value(*cur_table, next_index) + (next_index<k_? 1: -1);
// // maintain the heap structure
//       if (next_index >= 0) {  // invalid if next_index < 0
//         // shift: swapping out the last flipped index
//         HashType next_wipe = cur_candidate.wipe_mask_, next_pert = cur_candidate.pert_mask_;
//         CoordinateType next_score =
//             cur_score - (distance_vector_[*cur_table * k_ * 2 + cur_index] *
//                          distance_vector_[*cur_table * k_ * 2 + cur_index]);
//         next_score += (distance_vector_[*cur_table * k_ * 2+ next_index] *
//                        distance_vector_[*cur_table * k_ * 2+ next_index]);
//         swap_out(next_wipe, next_pert, next_index);
//         if (swap_in(next_wipe, next_pert, next_index, new_pert)){
//           heap_.insert(next_score, ProbeCandidate(*cur_table, next_wipe, next_pert,
//                                                 next_index));
//         }

//         // expand: adding a new flipped index
//         next_wipe = cur_candidate.wipe_mask_;
//         next_pert = cur_candidate.pert_mask_;
//         next_score =
//             cur_score +(distance_vector_[*cur_table * k_ * 2 + next_index] *
//                        distance_vector_[*cur_table * k_ * 2 + next_index]);
//         if (swap_in(next_wipe, next_pert, next_index, new_pert)){
//           heap_.insert(next_score, ProbeCandidate(*cur_table, next_wipe, next_pert,
//                                                 next_index));
//         }
//       }

      return true;
    }

       // last_index is used for generating the next perturbation vector
   // it is the largest (latest) dimension of the perturbation vector
   // wipe_mask is 1->unchanged 0-> changed, used to wipe out 
   // pert_mask is 0->unchanged [new hash bucket]-> changed
    class ProbeCandidate {
     public:
      ProbeCandidate(int_fast32_t table = 0, HashType wipe_mask = 0, HashType pert_mask = 0,
                     int_fast32_t last_index = 0)
          : table_(table), wipe_mask_(wipe_mask), pert_mask_(pert_mask), last_index_(last_index) {}

      int_fast32_t table_;
      HashType pert_mask_, wipe_mask_;
      int_fast32_t last_index_;
    };
    // checks against invalid perturbations: moving left of 0 or moving right of bucket_num
    // returns next valid perturbation after cur
    // invalid candidates are skipped
    /*int_fast32_t next_valid_perturbation(
        int_fast32_t l, int_fast32_t& cur){
      while(cur < 2 * k_){
        int_fast32_t candidate = sorted_hyperplane_indices_[l][cur];
        HashType hash_value_k = retrieve_hash_value(l, candidate);
        if ((candidate < k_ && hash_value_k < bucket_num_ -1) || (candidate >= k_ && hash_value_k > 0))
          return candidate;

        ++cur;
      }
      return -1; // not valid candidates, all posssible dimensions have been explored
    }*/
//  sort the indices in increasing order of absolute hash value
// The first (hash) dimension is the one closest to the boundary
    class GaussianComparator {
     public:
     // values is a set of distances
      GaussianComparator(const TransformedVectorType& values,
                           int_fast32_t offset)
          : values_(values), offset_(offset){}

      bool operator()(int_fast32_t ii, int_fast32_t jj) const {
        return values_[offset_ + ii] < values_[offset_ + jj];
      }

     private:
      const TransformedVectorType& values_;
      int_fast32_t offset_;
    };

    static CoordinateType distance_to_boundary(CoordinateType p, bool up_or_down){
      if (up_or_down){
        return std::ceil(p) - p;
      } else {
        return p - std::floor(p);
      }
    }

  //retrieves the hash value of the l-th table, k-th function from main_table_probe
  /*   HashType retrieve_hash_value(int_fast32_t l, int_fast32_t k){
      k %= k_;
      HashType hash_value_k = main_table_probe_[l] & (hash_mask_ << k * bucket_id_width_);
      return hash_value_k >> k * bucket_id_width_;
    }

    void swap_out(HashType& wipe_mask, HashType& pert_mask, int_fast32_t k){
      k %= k_;
      wipe_mask |= hash_mask_ << k * bucket_id_width_; // reset wipe_mask bits to 1
      pert_mask &= ~wipe_mask;   // reset pert_mask bits to 0
    }

    // checks conflicting perturbation (+1 and -1 on the same dimension)
    bool swap_in(HashType& wipe_mask, HashType& pert_mask, int_fast32_t k, HashType new_pert){
      k %= k_;
      if ((~wipe_mask & hash_mask_ << k * bucket_id_width_) != 0) return false; // confliction
      wipe_mask &= ~(hash_mask_ << k * bucket_id_width_); // reset wipe_mask bits to 0
      pert_mask |= new_pert << k * bucket_id_width_;   // switch new perturbation value in
      return true;
    }*/

    int_fast32_t k_;
    int_fast32_t l_;
    int_fast64_t num_probes_;
    int_fast64_t cur_probe_counter_;
    float bucket_width_;
    uint_fast32_t hash_seed_;
    //int_fast32_t bucket_id_width_, bucket_num_;
    // a mask used to retrieve the hash value of a certain hash function
    //const HashType hash_mask_;
    // [table][index] sorted by increasing order to boundary
    std::vector<std::vector<int_fast32_t>> sorted_hyperplane_indices_;
    // l center buckets that are probed without multiprobing
    std::vector<HashType> main_table_probe_;
    // The pre computed probes_vecs_;
    std::vector<std::vector<int_fast32_t>> probes_vecs_;
    SimpleHeap<CoordinateType, ProbeCandidate> heap_;
    // h(data), center of probing
    std::vector<int_fast32_t> hash_vector_;
    HashTran hash_tran_;
    int_fast32_t hash_width_;

    TransformedVectorType distance_vector_;
};
}}

#endif