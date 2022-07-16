
template <typename PointType>
struct PointTypeTraitsInternal {};

// TODO: get rid of these type trait classes once CosineDistance and the LSH
// classes are specialized on PointType (if we want to specialize on point
// type).
template <typename CoordinateType>
class PointTypeTraitsInternal<DenseVector<CoordinateType>> {
 public:
  typedef core::CosineDistanceDense<CoordinateType> CosineDistance;
  typedef core::EuclideanDistanceDense<CoordinateType> EuclideanDistance;
  typedef core::L1DistanceDense<CoordinateType> L1Distance;
  template <typename HashType>
  using HPHash = core::HyperplaneHashDense<CoordinateType, HashType>;
  template <typename HashType>
  using CPHash = core::CrossPolytopeHashDense<CoordinateType, HashType>;
  typedef CoordinateType CoorT;

  template <typename HashType>
  static std::unique_ptr<CPHash<HashType>> construct_cp_hash(
      const LSHConstructionParameters& params) {
    std::unique_ptr<CPHash<HashType>> res(new CPHash<HashType>(
        params.dimension, params.k, params.l, params.num_rotations,
        params.last_cp_dimension, params.seed ^ 93384688));
    return std::move(res);
  }

  /*template <typename HashType>
  static std::unique_ptr<GaHash<HashType>> construct_ga_hash(
      const LSHConstructionParameters& params) {
    std::unique_ptr<GaHash<HashType>> res(new GaHash<HashType>(
        params.dimension, params.k, params.l, params.seed ^ 93384688, 
        params.bucket_width, params.bucket_id_width));
    return std::move(res);
  }*/
};

template <typename CoordinateType, typename IndexType>
class PointTypeTraitsInternal<SparseVector<CoordinateType, IndexType>> {
 public:
  typedef core::CosineDistanceSparse<CoordinateType, IndexType> CosineDistance;
  typedef core::EuclideanDistanceSparse<CoordinateType, IndexType>
      EuclideanDistance;
  template <typename HashType>
  using HPHash =
      core::HyperplaneHashSparse<CoordinateType, HashType, IndexType>;
  template <typename HashType>
  using CPHash =
      core::CrossPolytopeHashSparse<CoordinateType, HashType, IndexType>;
  typedef CoordinateType CoorT;

  template <typename HashType>
  static std::unique_ptr<CPHash<HashType>> construct_cp_hash(
      const LSHConstructionParameters& params) {
    std::unique_ptr<CPHash<HashType>> res(new CPHash<HashType>(
        params.dimension, params.k, params.l, params.num_rotations,
        params.feature_hashing_dimension, params.last_cp_dimension,
        params.seed ^ 93384688));
    return std::move(res);
  }
};


template <typename PointType>
struct ComputeNumberOfHashFunctions {
  static void compute(int_fast32_t, LSHConstructionParameters*) {
    static_assert(FalseStruct<PointType>::value, "Point type not supported.");
  }
  template <typename T>
  struct FalseStruct : std::false_type {};
};

template <typename CoordinateType>
struct ComputeNumberOfHashFunctions<DenseVector<CoordinateType>> {
  static void compute(int_fast32_t number_of_hash_bits,
                      LSHConstructionParameters* params) {
    if (params->lsh_family == LSHFamily::Hyperplane) {
      params->k = number_of_hash_bits;
    } else if (params->lsh_family == LSHFamily::CrossPolytope) {
      if (params->dimension <= 0) {
        throw LSHNNTableSetupError(
            "Vector dimension must be set to determine "
            "the number of dense cross polytope hash functions.");
      }
      int_fast32_t rotation_dim =
          core::find_next_power_of_two(params->dimension);
      core::cp_hash_helpers::compute_k_parameters_for_bits(
          rotation_dim, number_of_hash_bits, &(params->k),
          &(params->last_cp_dimension));
    } else if (params->lsh_family == LSHFamily::Gaussian) {
        params->k = number_of_hash_bits / params->bucket_id_width;
      } else {
      throw LSHNNTableSetupError(
          "Cannot set paramters for unknown hash "
          "family.");
    }
  }
};

template <typename CoordinateType, typename IndexType>
struct ComputeNumberOfHashFunctions<SparseVector<CoordinateType, IndexType>> {
  static void compute(int_fast32_t number_of_hash_bits,
                      LSHConstructionParameters* params) {
    if (params->lsh_family == LSHFamily::Hyperplane) {
      params->k = number_of_hash_bits;
    } else if (params->lsh_family == LSHFamily::CrossPolytope) {
      if (params->feature_hashing_dimension <= 0) {
        throw LSHNNTableSetupError(
            "Feature hashing dimension must be set to "
            "determine  the number of sparse cross polytope hash functions.");
      }
      // TODO: add check here for power-of-two feature hashing dimension
      // (or allow non-power-of-two feature hashing dimension in the CP hash)
      int_fast32_t rotation_dim =
          core::find_next_power_of_two(params->feature_hashing_dimension);
      core::cp_hash_helpers::compute_k_parameters_for_bits(
          rotation_dim, number_of_hash_bits, &(params->k),
          &(params->last_cp_dimension));
    } else {
      throw LSHNNTableSetupError(
          "Cannot set paramters for unknown hash "
          "family.");
    }
  }
};

// this is default template
// if this template is instantiated -> error : not supported type
template <typename PointType>
struct ComputeNumberOfHashBits {
  static int_fast32_t compute(const LSHConstructionParameters&) {
    static_assert(FalseStruct<PointType>::value, "Point type not supported.");
    return 0;
  }
  template <typename T>
  struct FalseStruct : std::false_type {};
};

template <typename CoordinateType>
struct ComputeNumberOfHashBits<DenseVector<CoordinateType>> {
  static int_fast32_t compute(const LSHConstructionParameters& params) {
    if (params.k <= 0) {
      throw LSHNNTableSetupError(
          "Number of hash functions k must be at least "
          "1 to determine the number of hash bits.");
    }
    if (params.lsh_family == LSHFamily::Hyperplane) {
      return params.k;
    } else if (params.lsh_family == LSHFamily::CrossPolytope) {
      if (params.dimension <= 0) {
        throw LSHNNTableSetupError(
            "Vector dimension must be set to determine "
            "the number of dense cross polytope hash bits.");
      }
      if (params.last_cp_dimension <= 0) {
        throw LSHNNTableSetupError(
            "Last cross-polytope dimension must be set "
            "to determine the number of dense cross polytope hash bits.");
      }
      return core::cp_hash_helpers::compute_number_of_hash_bits(
          params.dimension, params.last_cp_dimension, params.k);
    } else 
    if (params.lsh_family == LSHFamily::Gaussian){
      return params.k * params.bucket_id_width;
    } else {
      throw LSHNNTableSetupError(
          "Cannot compute number of hash bits for "
          "unknown hash family.");
    }
  }
};

template <typename CoordinateType, typename IndexType>
struct ComputeNumberOfHashBits<SparseVector<CoordinateType, IndexType>> {
  static int_fast32_t compute(const LSHConstructionParameters& params) {
    if (params.k <= 0) {
      throw LSHNNTableSetupError(
          "Number of hash functions k must be at least "
          "1 to determine the number of hash bits.");
    }
    if (params.lsh_family == LSHFamily::Hyperplane) {
      return params.k;
    } else if (params.lsh_family == LSHFamily::CrossPolytope) {
      if (params.feature_hashing_dimension <= 0) {
        throw LSHNNTableSetupError(
            "Feature hashing dimension must be set to "
            "determine the number of dense cross polytope hash bits.");
      }
      if (params.last_cp_dimension <= 0) {
        throw LSHNNTableSetupError(
            "Last cross-polytope dimension must be set "
            "to determine the number of dense cross polytope hash bits.");
      }
      return core::cp_hash_helpers::compute_number_of_hash_bits(
          params.feature_hashing_dimension, params.last_cp_dimension, params.k);
    } else {
      throw LSHNNTableSetupError(
          "Cannot compute number of hash bits for "
          "unknown hash family.");
    }
  }
};

template <typename PointType>
struct GetDefaultParameters {
  static LSHConstructionParameters get(int_fast64_t, int_fast32_t,
                                       DistanceFunction, bool) {
    static_assert(FalseStruct<PointType>::value, "Point type not supported.");
    LSHConstructionParameters tmp;
    return tmp;
  }
  template <typename T>
  struct FalseStruct : std::false_type {};
};

template <typename CoordinateType>
struct GetDefaultParameters<DenseVector<CoordinateType>> {
  static LSHConstructionParameters get(int_fast64_t dataset_size,
                                       int_fast32_t dimension,
                                       DistanceFunction distance_function,
                                       bool is_sufficiently_dense) {
    LSHConstructionParameters result;
    result.dimension = dimension;
    result.distance_function = distance_function;
    result.lsh_family = LSHFamily::CrossPolytope;

    result.num_rotations = 2;
    if (is_sufficiently_dense) {
      result.num_rotations = 1;
    }

    result.l = 10;
    result.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
    result.num_setup_threads = 0;

    int_fast32_t number_of_hash_bits = 1;
    while ((1 << (number_of_hash_bits + 2)) <= dataset_size) {
      ++number_of_hash_bits;
    }
    compute_number_of_hash_functions<DenseVector<CoordinateType>>(
        number_of_hash_bits, &result);

    return result;
  }
};

template <typename CoordinateType>
struct GetDefaultParameters<SparseVector<CoordinateType>> {
  static LSHConstructionParameters get(int_fast64_t dataset_size,
                                       int_fast32_t dimension,
                                       DistanceFunction distance_function,
                                       bool) {
    LSHConstructionParameters result;
    result.dimension = dimension;
    result.distance_function = distance_function;
    result.lsh_family = LSHFamily::CrossPolytope;
    result.feature_hashing_dimension = 1024;
    result.num_rotations = 2;

    result.l = 10;
    result.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
    result.num_setup_threads = 0;

    int_fast32_t number_of_hash_bits = 1;
    while ((1 << (number_of_hash_bits + 2)) <= dataset_size) {
      ++number_of_hash_bits;
    }
    compute_number_of_hash_functions<SparseVector<CoordinateType>>(
        number_of_hash_bits, &result);

    return result;
  }
};

// decide on HashType 32 or 64 bit
  void setup0() {
    if (num_bits_ <= 32) {
      typedef uint32_t HashType;
      HashType tmp;
      setup1(std::make_tuple(tmp));
    } /*else if (num_bits_ <= 64) {
      typedef uint64_t HashType;
      HashType tmp;
      setup1(std::make_tuple(tmp));
    } */else {
      throw LSHNNTableSetupError(
          "More than 64 hash bits are currently not "
          "supported.");
    }
  }



// decide on lsh family
  template <typename V>
  void setup1(V vals) {
    typedef typename std::tuple_element<kHashTypeIndex, V>::type HashType;
/*
    if (params_.lsh_family == LSHFamily::Hyperplane) {
      typedef typename wrapper::PointTypeTraitsInternal<
          PointType>::template HPHash<HashType>
          LSH;
      std::unique_ptr<LSH> lsh(new LSH(params_.dimension, params_.k, params_.l,
                                       params_.seed ^ 93384688));
      setup2(std::tuple_cat(vals, std::make_tuple(std::move(lsh))));
    } else if (params_.lsh_family == LSHFamily::CrossPolytope) {
      if (params_.num_rotations < 0) {
        throw LSHNNTableSetupError(
            "The number of pseudo-random rotations for "
            "the cross polytope hash must be non-negative. Maybe you forgot to "
            "set num_rotations in the parameter struct?");
      }
      if (params_.last_cp_dimension <= 0) {
        throw LSHNNTableSetupError(
            "The last cross polytope dimension for "
            "the cross polytope hash must be at least 1. Maybe you forgot to "
            "set last_cp_dimension in the parameter struct?");
      }

      // TODO: for sparse vectors, also check feature_hashing_dimension here (it
      // is checked in the CP hash class, but the error message is less
      // verbose).

      typedef typename wrapper::PointTypeTraitsInternal<
          PointType>::template CPHash<HashType>
          LSH;
      std::unique_ptr<LSH> lsh(
          std::move(wrapper::PointTypeTraitsInternal<
                    PointType>::template construct_cp_hash<HashType>(params_)));
      setup2(std::tuple_cat(vals, std::make_tuple(std::move(lsh))));
    }else*/
      if (params_.lsh_family == LSHFamily::Gaussian) {
      // switch for different type of Gaussian
      if (params_.gauss_type == GaussianFunctionType::Cauchy) {
      typedef typename wrapper::PointTypeTraitsInternal<PointType>::CoorT CoordinateType;
      typedef typename core::GaussianHashDense<CoordinateType, HashType> LSH;
      std::unique_ptr<LSH> lsh(new LSH(params_.dimension, params_.k, params_.l,params_.universe,
                                       params_.seed ^ 93384688, params_.bucket_width, 
                                       params_.bucket_id_width, params_.hash_table_width));
      setup2(std::tuple_cat(vals, std::make_tuple(std::move(lsh))));
    }
    else if (params_.gauss_type == GaussianFunctionType::L1Precompute) {
      typedef typename wrapper::PointTypeTraitsInternal<PointType>::CoorT CoordinateType;
      typedef typename core::ToWHashDense<CoordinateType, HashType> LSH;
      std::unique_ptr<LSH> lsh(new LSH(params_.dimension, params_.k, params_.l,params_.universe,params_.step,
                                       params_.seed ^ 93384688, params_.bucket_width, 
                                       params_.bucket_id_width, params_.hash_table_width));
      setup2(std::tuple_cat(vals, std::make_tuple(std::move(lsh))));
    }
    else if (params_.gauss_type == GaussianFunctionType::L1DyadicSim) {
      typedef typename wrapper::PointTypeTraitsInternal<PointType>::CoorT CoordinateType;
      typedef typename core::RangeSummableGaussian<CoordinateType, HashType> LSH;
      std::unique_ptr<LSH> lsh(new LSH(params_.dimension, params_.k, params_.l,params_.universe,
                                       params_.seed ^ 93384688, params_.bucket_width, 
                                       params_.bucket_id_width, params_.hash_table_width));
      setup2(std::tuple_cat(vals, std::make_tuple(std::move(lsh))));
    }
    // else {
    //         throw LSHNNTableSetupError(
    //       "Unknown hash function type. Maybe you forgot to set "
    //       "the hash function type in the parameter struct?");
    // }
      //::template GaHash<HashType>
       ////   LSH;  hardcode as dense vector
    }
      else {
      throw LSHNNTableSetupError(
          "Unknown hash family. Maybe you forgot to set "
          "the hash family in the parameter struct?");
    }
  
  }

  template <typename V>
  void setup2(V vals) {
    /*if (params_.distance_function == DistanceFunction::NegativeInnerProduct) {
      typedef
          typename wrapper::PointTypeTraitsInternal<PointType>::CosineDistance
              DistanceFunc;
      DistanceFunc tmp;
      setup3(std::tuple_cat(std::move(vals), std::make_tuple(tmp)));
    } else*/ if (params_.distance_function ==
               DistanceFunction::EuclideanSquared) {
      typedef typename wrapper::PointTypeTraitsInternal<
          PointType>::EuclideanDistance DistanceFunc;
      DistanceFunc tmp;
      setup3(std::tuple_cat(std::move(vals), std::make_tuple(tmp)));
    } /*else if (params_.distance_function ==
               DistanceFunction::L1Norm) {
      typedef typename wrapper::PointTypeTraitsInternal<
          PointType>::L1Distance DistanceFunc;
      DistanceFunc tmp;
      setup3(std::tuple_cat(std::move(vals), std::make_tuple(tmp)));
    } */
    else {
      throw LSHNNTableSetupError(
          "Unknown distance function. Maybe you forgot "
          "to set the hash family in the parameter struct?");
    }
  }

  template <typename V>
  void setup3(V vals) {
    typedef typename std::tuple_element<kHashTypeIndex, V>::type HashType;

    if (params_.storage_hash_table == StorageHashTable::FlatHashTable) {
      typedef core::FlatHashTable<HashType> HashTable;
      std::unique_ptr<typename HashTable::Factory> factory(
          new typename HashTable::Factory(1 << num_bits_));

      std::vector<std::unique_ptr<CompositeTableT> > composite_table_vec;

      for (const auto& par_para: params_.hash_table_params){
        constexpr if (std::is_same_v<CompositeTableT, StaticPartition<HashType, KeyType, 
                                      PartitionMetric::PartitionMetricType, HashTable>>)
        {
          composite_table_vec.emplace_back(new CompositeTableT(
              par_para.l, factory.get(), par_para.partition_lower, par_para.partition_upper));                                 
        }
      }
      setup4(std::tuple_cat(std::move(vals),
                            std::make_tuple(std::move(factory)),
                            std::make_tuple(std::move(composite_table_vec))));
    }/* else if (params_.storage_hash_table ==
               StorageHashTable::BitPackedFlatHashTable) {
      typedef core::BitPackedFlatHashTable<HashType> HashTable;
      std::unique_ptr<typename HashTable::Factory> factory(
          new typename HashTable::Factory(1 << num_bits_, n_));

      typedef core::StaticCompositeHashTable<HashType, KeyType, HashTable>
          CompositeTable;
      std::unique_ptr<CompositeTable> composite_table(
          new CompositeTable(params_.l, factory.get()));
      setup4(std::tuple_cat(std::move(vals),
                            std::make_tuple(std::move(factory)),
                            std::make_tuple(std::move(composite_table))));
    } else if (params_.storage_hash_table == StorageHashTable::STLHashTable) {
      typedef core::STLHashTable<HashType> HashTable;
      std::unique_ptr<typename HashTable::Factory> factory(
          new typename HashTable::Factory());

      typedef core::StaticCompositeHashTable<HashType, KeyType, HashTable>
          CompositeTable;
      std::unique_ptr<CompositeTable> composite_table(
          new CompositeTable(params_.l, factory.get()));
      setup4(std::tuple_cat(std::move(vals),
                            std::make_tuple(std::move(factory)),
                            std::make_tuple(std::move(composite_table))));
    } else if (params_.storage_hash_table ==
               StorageHashTable::LinearProbingHashTable) {
      typedef core::StaticLinearProbingHashTable<HashType, KeyType> HashTable;
      std::unique_ptr<typename HashTable::Factory> factory(
          new typename HashTable::Factory(2 * n_));

      typedef core::StaticCompositeHashTable<HashType, KeyType, HashTable>
          CompositeTable;
      std::unique_ptr<CompositeTable> composite_table(
          new CompositeTable(params_.l, factory.get()));
      setup4(std::tuple_cat(std::move(vals),
                            std::make_tuple(std::move(factory)),
                            std::make_tuple(std::move(composite_table))));
    }*/ else {
      throw LSHNNTableSetupError(
          "Unknown storage hash table type. Maybe you "
          "forgot to set the hash table type in the parameter struct?");
    }
  }

  template <typename V>
  void setup4(V vals) {
    /*if (params_.multi_probe == MultiProbeType::Legacy){
      setup_final(std::move(vals));
    } else */
    /*if (params_.multi_probe == MultiProbeType::Customized) {
      setup_final2<core::CustomizedMultiProbe>(std::move(vals));
    } else*/ if (params_.multi_probe == MultiProbeType::Precomputed) {
      setup_final2<core::PreComputedMultiProbe>(std::move(vals));
    } 
    else {
      throw LSHNNTableSetupError(
          "Unknown multiprobe type. Maybe you "
          "forgot to set the hash table type in the parameter struct?");
    }
    
  }

  /*template <typename V>
  void setup_final(V vals) {
    typedef typename std::tuple_element<kHashTypeIndex, V>::type HashType;

    typedef
        typename std::tuple_element<kLSHFamilyIndex, V>::type LSHPointerType;
    typedef typename LSHPointerType::element_type LSHType;

    typedef typename std::tuple_element<kDistanceFunctionIndex, V>::type
        DistanceFunctionType;

    typedef typename std::tuple_element<kHashTableFactoryIndex, V>::type
        HashTableFactoryPointerType;
    typedef
        typename HashTableFactoryPointerType::element_type HashTableFactoryType;

    //typedef typename std::tuple_element<kCompositeHashTableIndex, V>::type
    //    CompositeHashTablePointerType;
    //typedef typename CompositeHashTablePointerType::element_type
    //    CompositeHashTableType;

    std::unique_ptr<LSHType>& lsh = std::get<kLSHFamilyIndex>(vals);
    std::unique_ptr<HashTableFactoryType>& factory =
        std::get<kHashTableFactoryIndex>(vals);
    //std::unique_ptr<CompositeHashTableType>& 
    std::vector<std::unique_ptr<CompositeTableT> > composite_table_vec =
        std::get<kCompositeHashTableIndex>(vals);

    typedef core::StaticLSHTable<PointType, KeyType, LSHType, HashType,
                                 CompositeHashTableType, DataStorageType>
        LSHTableType;
    std::unique_ptr<LSHTableType> lsh_table(
        new LSHTableType(lsh.get(), composite_table.get(), *data_storage_,
                         params_.num_setup_threads));

    table_.reset(new LSHNNTableWrapper<PointType, KeyType, ScalarType,
                                       DistanceFunctionType, LSHTableType,
                                       LSHType, HashTableFactoryType,
                                       CompositeHashTableType, DataStorageType>(
        std::move(lsh), std::move(lsh_table), std::move(factory),
        std::move(composite_table), std::move(data_storage_)));
  }*/

// Here the V
  template <template<typename> typename MultiProbe, typename V>
  void setup_final2(V vals) {
    typedef typename std::tuple_element<kHashTypeIndex, V>::type HashType;

    typedef
        typename std::tuple_element<kLSHFamilyIndex, V>::type LSHPointerType;
    typedef typename LSHPointerType::element_type LSHType;

    typedef typename std::tuple_element<kDistanceFunctionIndex, V>::type
        DistanceFunctionType;

    typedef typename std::tuple_element<kHashTableFactoryIndex, V>::type
        HashTableFactoryPointerType;
    typedef
        typename HashTableFactoryPointerType::element_type HashTableFactoryType;

    //typedef typename std::tuple_element<kCompositeHashTableIndex, V>::type
    //    CompositeHashTablePointerType;
    //typedef typename CompositeHashTablePointerType::element_type
     //   CompositeHashTableType;

    std::unique_ptr<LSHType>& lsh = std::get<kLSHFamilyIndex>(vals);
    std::unique_ptr<HashTableFactoryType>& factory =
        std::get<kHashTableFactoryIndex>(vals);
    //std::unique_ptr<CompositeHashTableType>& composite_table =
    std::vector<std::unique_ptr<CompositeTableT> > composite_table_vec =
        std::get<kCompositeHashTableIndex>(vals);

    typedef core::StaticLSHTable2<PointType, KeyType, LSHType, HashType,
                                 CompositeHashTableType , MultiProbe, DataStorageType>
        LSHTableType;
    std::unique_ptr<LSHTableType> lsh_table(
        new LSHTableType(lsh.get(), composite_table.get(), *data_storage_,
                         params_.num_setup_threads, params_.load_index, params_.index_filename));

    table_.reset(new LSHNNTableWrapper<PointType, KeyType, ScalarType,
                                       DistanceFunctionType, LSHTableType,
                                       LSHType, HashTableFactoryType,
                                       CompositeHashTableType, DataStorageType>(
        std::move(lsh), std::move(lsh_table), std::move(factory),
        std::move(composite_table_vec), std::move(data_storage_)));
  }


template <typename PointType>
void compute_number_of_hash_functions(int_fast32_t number_of_hash_bits,
                                      LSHConstructionParameters* params) {
  wrapper::ComputeNumberOfHashFunctions<PointType>::compute(number_of_hash_bits,
                                                            params);
}

template <typename PointType>
LSHConstructionParameters get_default_parameters(
    int_fast64_t dataset_size, int_fast32_t dimension,
    DistanceFunction distance_function, bool is_sufficiently_dense) {
  return wrapper::GetDefaultParameters<PointType>::get(
      dataset_size, dimension, distance_function, is_sufficiently_dense);
}


  const static int_fast32_t kHashTypeIndex = 0;
  const static int_fast32_t kLSHFamilyIndex = 1;
  const static int_fast32_t kDistanceFunctionIndex = 2;
  const static int_fast32_t kHashTableFactoryIndex = 3;
  const static int_fast32_t kCompositeHashTableIndex = 4;

  if (params_.k < 1) {
      throw LSHNNTableSetupError(
          "The number of hash functions k must be at "
          "least 1. Maybe you forgot to set k in the parameter struct?");
    }
    if (params_.l < 1) {
      throw LSHNNTableSetupError(
          "The number of hash tables l must be at "
          "least 1. Maybe you forgot to set l in the parameter struct?");
    }

    if (params_.lsh_family == LSHFamily::Unknown) {
      throw LSHNNTableSetupError("The hash family is not specified.");
    }
    if (params_.distance_function == DistanceFunction::Unknown) {
      throw LSHNNTableSetupError("The distance function is not specified.");
    }
    if (params_.storage_hash_table == StorageHashTable::Unknown) {
      throw LSHNNTableSetupError("The storage type is not specified.");
    }
    if (params_.lsh_family == LSHFamily::CrossPolytope) {
      if (params_.last_cp_dimension < 1) {
        throw LSHNNTableSetupError(
            "Forgot to set last_cp_dimension in the parameter struct.");
      }
      if (params_.num_rotations < 1) {
        throw LSHNNTableSetupError(
            "Forgot to set num_rotations in the parameter struct.");
      }
      if (params_.feature_hashing_dimension < -1) {
        throw LSHNNTableSetupError(
            "Invalid value for the feature hashing dimension.");
      }
    } else if (params_.lsh_family == LSHFamily::Gaussian){
      if (params_.bucket_id_width < 0){
        throw LSHNNTableSetupError(
            "Invalid value for the bucket id width.");
      }
      if (params_.bucket_width < 0){
        throw LSHNNTableSetupError(
            "Invalid value for the bucket width.");
      }
    }

    if (params_.multi_probe == MultiProbeType::Unknown) {
      throw LSHNNTableSetupError("The multiprobe type is not specified.");
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
    