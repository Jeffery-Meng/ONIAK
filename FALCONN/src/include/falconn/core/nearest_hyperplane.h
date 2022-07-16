#ifndef __NEAREST_HYPERPLANE_H__
#define __NEAREST_HYPERPLANE_H__

#include "gaussian_hash.h"
#include "polytope_hash.h"
#include "math_helpers.h"
#include <cmath>
#include <exception>
#include <utility>

namespace falconn{
  namespace core {

template <typename CoordinateT = float, typename HashT = uint32_t>
class AXequalYHash
{
 public:
  static constexpr int ADDITIONAL_DIMENSIONS = 2;
  typedef CoordinateT CoordinateType;
  typedef HashT HashType;
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor> VectorType;
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      DerivedVectorT;
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      TransformedVectorType;
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::ColMajor>
          MatrixType;

  typedef MatrixType QueryType;
  typedef DerivedVectorT DataType;
  typedef std::vector<DerivedVectorT> VectorofVectorType;

  const MatrixType& get_hyperplanes() const { return hyperplanes_; }

  const DerivedVectorT& get_translation() const {return translation_;}

  AXequalYHash(int dim,  int_fast32_t k, int_fast32_t l, int_fast32_t num_rotations,
                      uint_fast64_t seed, float w, int_fast32_t hash_width, 
                       int_fast32_t dim_Arows, bool second_step,
                bool fast_rotation)
      : 
      dim_(dim), k_(k), l_(l), bucket_width_(w), rotation_dim_(find_next_power_of_two(dim)),
       fht_helper_(rotation_dim_), num_rotations_(num_rotations), fast_rotation_(fast_rotation),
       hash_width_(hash_width), sqrt_rotation_dim_(1/sqrt(rotation_dim_)),
       
      seed_(seed), gen_(seed), second_step_(second_step), dim_Arows_(dim_Arows){

        // IMPORTANT: even if fast_rotation (at query time) is false, we still use it in indexing
 

      wy_seed_ = static_cast<uint32_t> (gen_());

      std::uniform_real_distribution<CoordinateType> uniform_dist(0.0, this->bucket_width_);
      int jj = 0;
      translation_.resize(this->k_ * this->l_);
      for (int ll = 0; ll < this->l_; ++ll){
        for (int kk = 0; kk < this->k_; ++kk) {
          translation_(jj) = uniform_dist(gen_);
          ++jj;
        }
      }
    }    /*eigenvalues_(std::move(precomputed.eigenvalues)),
       random_signs_(std::move(precomputed.signs)),
       hyperplanes_(std::move(precomputed.gaussians)),
    std::normal_distribution<CoordinateType> gauss(0.0, 1.0); 
    
    std::uniform_int_distribution<int> zero_one(0, 1);

// dim: data dimension
    //if (!fast_rotation_) hyperplanes_.resize(this->k_ * this->l_, this->dim_);
    
    random_signs_.resize(k_ * l_ * num_rotations_);
    if (second_step_) gaussian_proj_.resize(k_ * l_, dim_Arows_);

// generate and fix RVs 
  int jj = 0, rs_idx = 0;
  
    
      for (int rr = 0; rr < num_rotations_; ++rr) {
        random_signs_[rs_idx].resize(rotation_dim_);
        for (int ii = 0; ii < rotation_dim_; ++ii) {
          random_signs_[rs_idx](ii) = zero_one(gen_)? sqrt_rotation_dim_: -sqrt_rotation_dim_;
        }
        ++rs_idx;
      }

      if (second_step_) {
        for (int ii = 0; ii < dim_Arows_; ++ii) {
          gaussian_proj_(jj, ii) = gauss(gen_);
        }
      }

      ++jj;
    }
  }
  */


  
 


  // TODO: specialize template for faster batch hyperplane setup (if the batch
  // vector type is also an Eigen matrix, we can just do a single matrix-matrix
  // multiplication.)
  
  
// (Ax+b)/w
  template <typename Derived>
  void raw_hash_all_query(const Eigen::MatrixBase<Derived>& pre_hash,
                                        DerivedVectorT* res) const {

    // 1. compute A^T * A (one step) or U * A (two steps)
    // 2. Frobenius inner product with a Gaussian random matrix
    std::ofstream f_result("query_hash value.txt",std::ios_base::app);

    DerivedVectorT& result = *res;
  /*  result.setZero(k_ * l_);

    if (fast_rotation_) {
      if (second_step_) {
        MatrixType mat_u = gaussian_proj_ * query_A; // (k*l) x dim matrix of U
        int jj = 0;
        for (int ll = 0; ll < l_; ++ll){
          for (int kk = 0; kk < k_; ++kk) {
            auto rotated_vec = rotate_one_vector(mat_u.row(jj), kk, ll); 
            result(jj) = rotated_vec.cwiseProduct(rotated_vec).cwiseProduct(eigenvalues_[jj]).sum();
            ++jj;
          }
        }
      } else { // one step scheme
        int jj = 0;
        for (int ll = 0; ll < l_; ++ll){
          for (int kk = 0; kk < k_; ++kk) {
            //res(jj) = 0;
            for (int aa = 0; aa < dim_Arows_; ++aa) {
              auto rotated_vec = rotate_one_vector(query_A.row(aa), kk, ll); 
              result(jj) += rotated_vec.cwiseProduct(rotated_vec).cwiseProduct(eigenvalues_[jj]).sum();
            }
            ++jj;
          }
        }
      }
    } else { // direct product. WARNING: this may be VERY slow.
      if (second_step_) {
        MatrixType mat_u = gaussian_proj_ * query_A;
        int jj = 0;
        for (int ll = 0; ll < l_; ++ll){
          for (int kk = 0; kk < k_; ++kk) {
            auto vec_u = mat_u.row(jj);
            auto mat_utu = vec_u.transpose() * vec_u;
            result(jj) = mat_utu.cwiseProduct(hyperplanes_[jj]).sum();
            ++jj;
          }
        }
      } else {  // one step
          auto mat_ata = query_A.transpose() * query_A;
          int jj = 0;
          for (int ll = 0; ll < l_; ++ll){
            for (int kk = 0; kk < k_; ++kk) {
              result(jj) = mat_ata.cwiseProduct(hyperplanes_[jj]).sum();
              ++jj;
            }
          }
      }
      
    }*/
    // std::cout << pre_hash[0] << " " << pre_hash[1] << std::endl;
    // f_result << pre_hash << '\n';
    result = (-pre_hash + translation_) / this->bucket_width_;
  }

  void raw_hash_single_index( int_fast32_t l,     DerivedVectorT* res) const {
    // TODO: check whether middleRows is as fast as building a memory mapconst DerivedVectorT& vec_x,
    // manually.
    DerivedVectorT& result = *res;
    /*result.setZero(k_);

    // fast rotation is always enabled for indexing
    // the indexing part is the same for one step or two steps

    int jj = l * k_;
    for (int kk = 0; kk < k_; ++kk) {
      auto rotated_vec = rotate_one_vector(vec_x, kk, l); 
      result(kk) = rotated_vec.cwiseProduct(rotated_vec).cwiseProduct(eigenvalues_[jj]).sum();
      ++jj;
    }*/
    
    result = (result + translation_.middleRows(l*k_, k_)) / this->bucket_width_;
  }

  void sketch_vector(DerivedVectorT&& vec_x,
                                          int_fast32_t l,
                                          DerivedVectorT* res) const {
    // TODO: used for debugging
    DerivedVectorT& result = *res;
    result.setZero(k_);
    float nor = vec_x.norm();
    

    // fast rotation is always enabled for indexing
    // the indexing part is the same for one step or two steps

    int jj = l * k_;
    for (int kk = 0; kk < k_; ++kk) {
      auto rotated_vec = rotate_one_vector_f(vec_x, kk, l); 
      float nor_ro = rotated_vec.norm();
      result(kk) = (rotated_vec.cwiseProduct(rotated_vec).cwiseProduct(eigenvalues_[jj])).sum();
      ++jj;
    }
    
    //result = (result + translation_.middleRows(l*k_, k_)) / this->bucket_width_;
  }

  void hash(const MatrixType& query_A, std::vector<HashType>* result,
            DerivedVectorT* tmp_hash_vector = nullptr) const {
    bool allocated = false;
    if (tmp_hash_vector == nullptr) {
      allocated = true;
      tmp_hash_vector = new DerivedVectorT(this->k_ * this->l_);
    }

    raw_hash_all_query(query_A, tmp_hash_vector);

    std::vector<HashType>& res = *result;
    std::vector<int> res_rounded(this->k_, 0);
    if (res.size() != static_cast<size_t>(this->l_)) {
      res.resize(this->l_);
    }
    for (int_fast32_t ii = 0; ii < this->l_; ++ii) {
      for (int_fast32_t jj = 0; jj < this->k_; ++jj) {
        res_rounded[jj] = static_cast<int_fast32_t>(std::floor((*tmp_hash_vector)[ii * this->k_ + jj]));
      }
      res[ii] = wyhash32(res_rounded.data(), this->k_ * sizeof(int), wy_seed_);
      res[ii] &= (1<<this->hash_width_)-1;
    }

    if (allocated) {
      delete tmp_hash_vector;
    }
  }


  HashType compute_hash_single_table(const DerivedVectorT& v,uint_fast32_t seed) const {
    //static std::ofstream fout("index_round.txt");
    HashType res = 0;
    std::vector<int> res_rounded(v.size(), 0);
    for (int_fast32_t jj = 0; jj < v.size(); ++jj) {
      res_rounded[jj] =  static_cast<int_fast32_t>(std::floor(v[jj]));
      //fout << res_rounded[jj] << "\t";
    }
    //fout << std::endl;
    res = wyhash32(res_rounded.data(), v.size() * sizeof(int), seed);
    res &= (1<<hash_width_)-1;
    return res;
  }

  void hash_to_bucket(const DerivedVectorT& hash_vec, std::vector<HashType>& res) const{
    //static std::ofstream fout("main_bucket.txt");
    if (res.size() != static_cast<size_t>(this->l_)) {
      res.resize(this->l_);
    }
    std::vector<int> res_rounded(this->k_, 0);
    for (int_fast32_t ii = 0; ii < this->l_; ++ii) {
      for (int_fast32_t jj = 0; jj < this->k_; ++jj) {
        res_rounded[jj] =  static_cast<int_fast32_t>(std::floor(hash_vec[ii * this->k_ + jj]));
      }
      res[ii] = wyhash32(res_rounded.data(), this->k_ * sizeof(int), wy_seed_);
      res[ii] &= (1<<this->hash_width_) - 1;
      //fout << res[ii] << "\t";
    }
    //fout << std::endl;
  }

  int get_k() const {
    return k_;
  }
  int get_l() const {
    return l_;
  }

  CoordinateType get_bucket_width() const {
    return bucket_width_;
  }

  uint_fast32_t get_seed() const {
    return wy_seed_;
  }

  int_fast32_t get_hash_width() const {
    return hash_width_;
  }

  int_fast32_t get_dimension() const {
    return dim_;
  }

  int_fast32_t get_rotation_dim() const {
    return rotation_dim_;
  }


void reserve_transformed_vector_memory(TransformedVectorType* tv) const {
    tv->resize(k_ * l_);
  } 

  struct AXequalYPrecomputed {
    std::vector<DerivedVectorT> eigenvalues, random_signs;
    std::vector<MatrixType> gaussians;
  };

  static  AXequalYPrecomputed read_precomputed(std::string path, int num_hash_funcs, int num_rotations,
                int rotation_dim, int dimension, bool fast_rotation=true) {
    // TODO Huayi: read precomputed eigenvalues from a file

    AXequalYPrecomputed precomputed;
    // std::cout  << "Current path: "  << path.c_str() << std::endl;
    std::ifstream fin(path+"eigenvalues.txt");
    NPP_ENFORCE(fin);
    
   // int num_hash_funcs = params_.hash_table_params[0].k * params_.hash_table_params[0].l;
   // int rotation_dim = core::find_next_power_of_two(params_.dimension);
   // std::vector<ColumnVecT> result(num_hash_funcs);
    precomputed.eigenvalues.resize(num_hash_funcs);

    for (int hh = 0; hh < num_hash_funcs; ++hh) {
      precomputed.eigenvalues[hh].resize(rotation_dim);
      for (int dd = 0; dd < rotation_dim; ++dd) {
        fin >> precomputed.eigenvalues[hh](dd);
      }
    }
    fin.close();

    fin.open(path+"signs.txt");
    bool health = bool(fin);
    precomputed.random_signs.resize(num_hash_funcs * num_rotations);

    for (int hh = 0; hh < num_hash_funcs * num_rotations; ++hh) {
      precomputed.random_signs[hh].resize(rotation_dim);
      for (int dd = 0; dd < rotation_dim; ++dd) {
        fin >> precomputed.random_signs[hh](dd);
      }
    }
    fin.close();

    if (!fast_rotation) {
      fin.open(path+"gaussians.txt");
      NPP_ENFORCE(fin);
      precomputed.gaussians.resize(num_hash_funcs);

      for (int hh = 0; hh < num_hash_funcs; ++hh) {
        // each gaussian random matrix is dimension x dimension
        // because this part (among the rotation_dim x rotation_dim matrix)
        // is used for cwise
        precomputed.gaussians[hh].resize(dimension, dimension);
        for (int dd = 0; dd < rotation_dim; ++dd) {
          for (int cc = 0; cc < rotation_dim; ++cc) {
            read_in_range(fin, cc, dd, dimension, precomputed.gaussians[hh]);
          } 
        }
      }
    }

    return precomputed;
  }

  class PreHasher {
    std::vector<MatrixType> hyperplanes_;
    MatrixType gaussian_proj_;
    bool second_step_, fast_rotation_;
    std::vector<DerivedVectorT> random_signs_, eigenvalues_;
    int num_hash_funcs_, rotation_dim_, dim_Arows_, num_rotations_, dim_;
    cp_hash_helpers::FHTHelper<CoordinateType> fht_helper_;
    std::mt19937 gen_;

    public: 
    PreHasher(AXequalYPrecomputed&& precomputed, int num_hash_funcs,  int dim, int dim_Arows,
              bool second_step, bool fast_rotation,  int num_rotations,
              uint32_t seed):
      eigenvalues_(precomputed.eigenvalues),
       random_signs_(precomputed.random_signs), dim_Arows_(dim_Arows), dim_(dim),
       hyperplanes_(precomputed.gaussians), rotation_dim_(find_next_power_of_two(dim)),
      second_step_(second_step), fast_rotation_(fast_rotation), num_hash_funcs_(num_hash_funcs),
      fht_helper_(rotation_dim_), gen_(seed), num_rotations_(num_rotations)
      {
       assert(eigenvalues_.size()== num_hash_funcs_);
        assert(eigenvalues_[0].size() == rotation_dim_);
        assert(random_signs_.size()== num_hash_funcs_*num_rotations);
        assert(random_signs_[0].size() == rotation_dim_);
        if (!fast_rotation_) {
          assert(hyperplanes_.size()== num_hash_funcs_);
          assert(hyperplanes_[0].rows() == dim-1);
          assert(hyperplanes_[0].cols() == dim-1);
        }

        std::normal_distribution<CoordinateType> gauss(0.0, 1.0); 
        if (second_step_) {
          gaussian_proj_.resize(num_hash_funcs_, dim_Arows_);
        
          for (int hh = 0; hh < num_hash_funcs_; ++hh) {
            for (int ii = 0; ii < dim_Arows_; ++ii) {
              gaussian_proj_(hh, ii) = gauss(gen_);
            }
          }
          
        }
      }


      template <typename PartitionMetric>
    DerivedVectorT pre_hash_all_query(const MatrixType& query_A, 
          DerivedVectorT& balancing_metric) const {

      // 1. compute A^T * A (one step) or U * A (two steps)
      // 2. Frobenius inner product with a Gaussian random matrix

      DerivedVectorT result;
      result.setZero(num_hash_funcs_);
     // std::ofstream fout("matrix_new.txt",std::ios_base::app);
     // fout << query_A << '\n';

      // Note: if second step is on, then fast rotation is used
      // 

      if (second_step_) {
        MatrixType mat_u = gaussian_proj_ * query_A; // (k*l) x dim matrix of U
        balancing_metric = mat_u.rowwise().squaredNorm();
        // the squared L2 norm of each row

        for (int jj = 0; jj < num_hash_funcs_; ++jj){
          auto rotated_vec = rotate_one_vector(mat_u.row(jj), jj); 
          result(jj) = rotated_vec.cwiseProduct(rotated_vec).cwiseProduct(eigenvalues_[jj]).sum();
        }
      } else { // one step scheme
        if (fast_rotation_) {
          for (int jj = 0; jj < num_hash_funcs_; ++jj){
            for (int aa = 0; aa < query_A.rows(); ++aa) {
              auto rotated_vec = rotate_one_vector(query_A.row(aa), jj); 
              // Todo: Might need to check
              result(jj) += rotated_vec.cwiseProduct(rotated_vec).cwiseProduct(eigenvalues_[jj]).sum();
            }
          }
        } else {
          MatrixType mat_ata = query_A.transpose() * query_A;
          //balancing_metric = DerivedVectorT::Constant(num_hash_funcs_, mat_ata.squaredNorm());

          for (int jj = 0; jj < num_hash_funcs_; ++jj) {
              result(jj) = mat_ata.cwiseProduct(hyperplanes_[jj]).sum();
          }
        }
        
      }

      return result;
    }

    void pre_hash_single_index(const DerivedVectorT& vec_x,
                                          int_fast32_t l, int_fast32_t k_num,
                                          DerivedVectorT* res) const {
    // TODO: check whether middleRows is as fast as building a memory map
    // l : the id of hash table
    // manually.
    DerivedVectorT& result = *res;
    result.setZero(k_num);

    // fast rotation is always enabled for indexing
    // the indexing part is the same for one step or two steps

    int jj = l * k_num;
    for (int kk = 0; kk < k_num; ++kk) {
      auto rotated_vec = rotate_one_vector(vec_x, jj); 
      result(kk) = rotated_vec.cwiseProduct(rotated_vec).cwiseProduct(eigenvalues_[jj]).sum();
      ++jj;
    }
  }


    private:



    DerivedVectorT rotate_one_vector(const DerivedVectorT& vec, int_fast32_t hf_idx) const {
      DerivedVectorT result = embed_dims(vec);

      int pattern = hf_idx * num_rotations_;
      for (int_fast32_t rot = 0; rot < num_rotations_; ++rot) {
            result = result.cwiseProduct(random_signs_[pattern]);
            ++pattern;
            fht_helper_.apply(result.data());
            //result = result / sqrt_rotation_dim_;
      }
      return result;
    }

    DerivedVectorT embed_dims(const DerivedVectorT& vec) const {
      DerivedVectorT result(rotation_dim_);
      result.head(vec.size()) = vec;
      result.tail(rotation_dim_ - dim_).setZero();
      return result;
    }


  };

  class Normalizer {
    const double mx_squared_; // Maximum L2 norm 
    const int dimension_;

    public: 
    typedef CoordinateType CoordiT;
    Normalizer(CoordinateType mx, int dim) : mx_squared_(mx), dimension_(dim) {}

    void normalize(DerivedVectorT& point, CoordinateType l2_now) const {
      point(dimension_-1) = sqrt(mx_squared_ - l2_now);
      float alpha = get_alpha();
      point  = point / (float) alpha;
      point(dimension_ - 2) = -1.f;
    }

    double get_alpha() const {
      return sqrt(mx_squared_) * pow(dimension_, -1./6.);
    }

    CoordinateType get_mx() const {return mx_squared_;}

   /*template <typename Derived>
    DerivedVectorT query_balance(const Eigen::MatrixBase<Derived> & query, const DerivedVectorT& balancing_metrics) const {
      // : balance the query and the normalized data.
      // CAUTION: balancing metrics are not meaningful if fast rotation is true.

      DerivedVectorT coef = (balancing_metrics / (mx_squared_*mx_squared_)).cwiseSqrt();
      return query.cwiseQuotient(coef);
      // assume there is no division by zero
    }*/



    template <typename Derived>
    std::pair<double, double> compute_beta_c(const Eigen::MatrixBase<Derived> & kernel) const {
      double norm_kernel = kernel.squaredNorm();
      double norm_kernelTkernel = (kernel.transpose() * kernel).squaredNorm();
      double temp = (dimension_ * norm_kernelTkernel - norm_kernel * norm_kernel) / (dimension_ -1);
      double universe = sqrt(1 + mx_squared_ / get_alpha() / get_alpha());
      double beta = pow(temp, 0.25) / universe;
      double c = (norm_kernel + beta * beta * universe * universe) / dimension_;
      beta = 1 / beta;
      c *= beta * beta;
      return std::make_pair(beta, c);
    }


  };

  class HashTransformation {
   public:
    HashTransformation(const AXequalYHash& parent) : parent_(parent) {}

    // apply y=(Ax+b)/w
    // This is hash transformation without quantization

    // make use of Eigen's lazy evaluation
    template <typename Derived>
    void apply(const Eigen::MatrixBase<Derived>& A, DerivedVectorT* result) const {
      parent_.raw_hash_all_query(A, result);
    }

   // template <typename Derived>
   // void apply_pre(const Eigen::DenseBase<Derived>& A, DerivedVectorT* result) const {
   //   parent_.pre_hash_all_query(A, result);
    //}

    void round(const DerivedVectorT& hash_vec, std::vector<HashType>& res){
      parent_.hash_to_bucket(hash_vec, res);
    }

   private:
   // reference to the hash transformation
    const AXequalYHash& parent_;
  };


template <typename BatchVectorType>
  class BatchHash {
   public:
    BatchHash( AXequalYHash& parent, const PreHasher& pre_h)
        : parent_(parent), tmp_vector_(parent.get_k()), pre_hash_(pre_h){}

    // hash a set of points, using the hash functions of the l-th table
    void batch_hash_single_table(const BatchVectorType& points, int_fast32_t l,
                                 std::vector<HashType>* res) {

      int_fast64_t nn = points.size();
      if (static_cast<int_fast64_t>(res->size()) != nn) {
        res->resize(nn);
      }
      typename BatchVectorType::FullSequenceIterator iter =
          points.get_full_sequence();

    #ifdef DEBUG
    std::cout << parent_.seed_hash2_ << std::endl;
    #endif 
      
      for (int_fast64_t ii = 0; ii < nn; ++ii) {
        pre_hash_.pre_hash_single_index(iter.get_point(), l, parent_.k_,
                                                   &tmp_vector_);
        parent_.raw_hash_single_index(l,  &tmp_vector_);

        (*res)[ii] = parent_.compute_hash_single_table(tmp_vector_,parent_.wy_seed_);
        
      //  if (ii < 200) {
      //     fout << "index:" << ii << "id"<< tmp_vector_(0) << "\t";
      //   }
        
        ++iter;
      }
      //fout << std::endl;
    }

    template <typename KeyType>
    void batch_hash_single_table(const BatchVectorType& points, int_fast32_t l, const std::vector<KeyType>& keys,
                                 std::vector<HashType>* res) {
    // std::string filename = "index_bucket-" + std::to_string(l) + ".txt";
    // std::ofstream fout(filename);
    //  std::ofstream fout1("read_point.txt");
      
    int_fast64_t nn = keys.size();
    // std::cout << keys.size() << std::endl;
      if (static_cast<int_fast64_t>(res->size()) != nn) {
        res->resize(nn);
      }
      auto iter =  points.get_subsequence(keys);
      
      for (int_fast64_t ii = 0; ii < nn; ++ii) {
        pre_hash_.pre_hash_single_index(iter.get_point(), l, parent_.k_,
                                                   &tmp_vector_);
        parent_.raw_hash_single_index(l,  &tmp_vector_);

        (*res)[ii] = parent_.compute_hash_single_table(tmp_vector_, parent_.wy_seed_);
        
       //if (ii < 10000) {
          //fout<< "index:" << ii << " Bucket id: "<< (*res)[ii]  << "\n";
        // }
        
        ++iter;
      }
      //fout << std::endl;
   }

   private:
    AXequalYHash& parent_;
    DerivedVectorT tmp_vector_;
    const PreHasher& pre_hash_;
  };


private:

  static bool read_in_range(std::ifstream& fin, int cc, int dd, int dimension, MatrixType& matrix) {
    bool condi = (cc < dimension)  && (dd < dimension);
    CoordinateType dummy;
    CoordinateType& sink = condi? matrix(dd, cc): dummy;
    fin >> sink;
    return condi;
  }
/* void compute_rotated_vectors(
      const VectorT& v, TransformedVectorType* result) const {
    int_fast32_t pattern = 0;
    for (int_fast32_t ii = 0; ii < l_; ++ii) {
      for (int_fast32_t jj = 0; jj < k_; ++jj) {
        DerivedVectorT& cur_vec = (*result)[ii * k_ + jj];
        static_cast<const Derived*>(this)->embed(v, ii, jj, &cur_vec);

        for (int_fast32_t rot = 0; rot < num_rotations_; ++rot) {
          cur_vec = cur_vec.cwiseProduct(random_signs_[pattern]);
          ++pattern;
          fht->apply(cur_vec.data());
        }
      }
    }
  }

  DerivedVectorT rotate_one_vector_f(const DerivedVectorT& vec, int_fast32_t k, int_fast32_t l) const {
    DerivedVectorT result = vec; //= embed_dims(vec);

    int pattern = (l * k_ + k) * num_rotations_;
    for (int_fast32_t rot = 0; rot < num_rotations_; ++rot) {
          result = result.cwiseProduct(random_signs_[pattern]);
          ++pattern;
          fht_helper_.apply(result.data());
          //result = result / sqrt_rotation_dim_;
    }
    return result;
  }*/

  



private:



  // Gaussian Matrix A
  std::vector<MatrixType> hyperplanes_;
  // Gaussian projections u (if two-step is used)
  MatrixType gaussian_proj_;

  bool second_step_, fast_rotation_; // whether two-step or one-step
      // uniform vector b 
  DerivedVectorT translation_;
  // random signs used in random rotations and the eigenvalues of A
  // They together define the Gaussian matrix A
  std::vector<DerivedVectorT> random_signs_, eigenvalues_;
  std::mt19937_64 gen_;  
  uint_fast64_t seed_;

  uint32_t wy_seed_;

  int dim_, num_rotations_, rotation_dim_, dim_Arows_;
  // number of hash functions per table
  int_fast32_t k_, hash_width_;
  // number of hash tables
  int_fast32_t l_;
  // denumerator w
  float bucket_width_;
  // log2 of number of buckets on each dimension / number of bits of each bucket id for each hash function
  cp_hash_helpers::FHTHelper<CoordinateType> fht_helper_;
  float sqrt_rotation_dim_;
};

}
}
#endif