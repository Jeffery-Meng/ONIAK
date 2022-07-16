#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <fstream>
#include <algorithm>
#include "nlohmann/json.hpp"
#include "lsh_nn_table.h"
#include "core/math_helpers.h"

namespace falconn{

using json = nlohmann::json;

LSHConstructionParameters read_config(const char * filename) {
    std::ifstream json_f(filename);
    json config;
    json_f >> config;

    LSHConstructionParameters para;

    para.dimension = config.at("dimension");
    if (config.contains("seed")) {
        para.seed = config["seed"];
    }
    if (config.contains("number of rotations")) {
        para.num_rotations = config.at("number of rotations");
    }
    if (config.contains("hash table width")) {
        para.hash_table_width = config.at("hash table width");
    }
    para.load_index = config.at("load index");

    if (config.contains("index filename")) {
        para.index_filename = config["index filename"];
    }
    if (config.contains("index path")) {
        para.index_path = config["index path"];
    }
    if (config.contains("eigenvalue filepath")) {
        para.eigen_filename = config["eigenvalue filepath"];
    }

    if (config.contains("number of partitions")) {
        para.num_partitions = config["number of partitions"];
    }
    
    auto hash_params = config.at("hash table parameters");
    for (const auto& par_param : hash_params) {
        CompositeHashTableParameters ch_para;
        ch_para.k = par_param.at("k");
        ch_para.l = par_param.at("l");
        ch_para.bucket_width = par_param.at("bucket width");
        ch_para.partition_lower = par_param.at("lower");
        ch_para.partition_upper = par_param.at("upper");
        para.hash_table_params.push_back(ch_para);
        
        int kbyl = ch_para.k * ch_para.l;
        if (kbyl > para.num_hash_funcs) {
            para.num_hash_funcs = kbyl;
        }
    }

    std::sort(para.hash_table_params.begin(), para.hash_table_params.end(), 
            [](const CompositeHashTableParameters& a, const CompositeHashTableParameters &b) {
                return a.partition_upper < b.partition_upper;
            });

    //if (config.contains("second step")) {
    //    para.second_step = config["second step"];
    //}
    para.second_step = false;  // Enforce single step ONIAK
    
    if (config.contains("fast rotation")) {
        para.fast_rotation = config["fast rotation"];
    }
    if (config.contains("number of query rows")) {
        para.dim_Arows = config["number of query rows"];
    }

    para.rotation_dim = core::find_next_power_of_two(para.dimension);
    para.dim_Acols = para.dimension - 1;
    para.num_setup_threads = 1;

    if (config.contains("allow overwrite")) {
        para.allow_overwrite = config["allow overwrite"];
    }
    if (config.contains("ground truth file")) {
        para.gnd_filename = config["ground truth file"];
    }
    if (config.contains("compute ground truth")) {
        para.compute_gound_truth = config["compute ground truth"];
    }
    if (config.contains("number of neighbors")) {
        para.num_neighbors = config["number of neighbors"];
    } 
    if (config.contains("training size")) {
        para.num_points = config["training size"];
    } 
    if (config.contains("testing size")) {
        para.num_queries = config["testing size"];
    } 
    if (config.contains("data filename")) {
        para.data_filename = config["data filename"];
    } 
    if (config.contains("query filename")) {
        para.query_filename = config["query filename"];
    }
    if (config.contains("kernel filename")) {
        para.kernel_filename = config["kernel filename"];
    } 

    para.raw_dimension = para.dimension - ADDITIONAL_DIMENSIONS;

    if (config.contains("result filename")) {
        para.result_filename = config["result filename"];
    } 
    if (config.contains("summary path")) {   // print out summary 
                                            // recall + candidate #, used for parameter tuning
        para.summary_path = config["summary path"];
    } 
    if (config.contains("use single kernel")) {                                        
        para.single_kernel = config["use single kernel"];
    } 
    if (config.contains("input transformed queries")) {                                  
        para.transformed_queries = config["input transformed queries"];
    } 
    if (config.contains("candidate filename")) {                                  
        para.candidate_filename = config["candidate filename"];
    } 
    if (config.contains("number of prefilter")) {                                  
        para.num_prehash_filters = config["number of prefilter"];
    } 
    if (config.contains("ratio of prefilter")) {                                  
        para.prefilter_ratio = config["ratio of prefilter"];
    } 

    return para;

}

}

#endif