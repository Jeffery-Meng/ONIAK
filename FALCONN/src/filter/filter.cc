#include "nlohmann/json.hpp"
#include <string>
#include <fstream>
#include "fvecs.hpp"
#include "cpu_timer.hpp"
#include <iostream>
#include <random>
#include <utility>
#include <numeric>

// Implementation of JLT-based filter.

using json = nlohmann::json;

///////////////////////////////////////////////////////////
// Global variables 

const unsigned seed = 0x312f373d;
std::mt19937 rng;
int n, qn, dim, dim_Arows, dim_Frows, K;
std::string data_filename, query_filename, kernel_filename, gt_filename, candi_filename;
std::ifstream data_in, query_in, kernel_in, gt_in, candi_in;

FloatMatrix data_, queries_, kernel_, filter_; 
IntegerMatrix gt_;

CPUTimer timer_;

double finalist_ratio, recall_sum;
int_fast64_t total_num_candi, total_num_final;
bool linear_scan;

/////////////////////////////////////////////////////////////////////////////

float distance_func(const FloatMatrix & transformed_kernel, const FloatVector& data, const FloatVector& query) {
    auto temp = transformed_kernel * (data - query);
    return temp.squaredNorm();
}

FloatMatrix random_normal_matrix(std::mt19937& rng, int rows, int cols) {
    FloatMatrix result(rows, cols);
    std::normal_distribution<float> gauss;

    for (int rr = 0; rr < rows; ++rr) {
        for (int cc = 0; cc < cols; ++cc) {
            result(rr, cc) = gauss(rng);
        }
    }
    return result;
}

void run_query(int qid) {
    using PairT = std::pair<int, float>;
    kernel_ = read_one_matrix(kernel_in, dim_Arows, dim);
    IntegerVector candidate;
    if (linear_scan) {
        candidate.resize(n);
        std::iota(candidate.begin(), candidate.end(), 0);
    } else {
        candidate = read_vector(candi_in);
    }

    std::vector<PairT> filtered_candi;
    std::vector<int>& ground_truth = gt_[qid];
    std::vector<int> finalist, v_intersection;

    total_num_candi += candidate.size();
    int_fast64_t num_finalist = candidate.size() * finalist_ratio;
    total_num_final += num_finalist;

    timer_.start();
    // Step 1: compute multiplied kernel
    FloatMatrix transformed_kernel = filter_ * kernel_;
    // Step 2: compute the filtered distance for each transformed kernel
    std::transform(candidate.begin(), candidate.end(), std::back_inserter(filtered_candi), 
        [&](int candi) {return std::make_pair(candi, distance_func(transformed_kernel, data_.row(candi), 
                queries_.row(qid)));});

    // Step 3: sort by filtered distance
    std::sort(filtered_candi.begin(), filtered_candi.end(), [](const PairT& a, const PairT& b) {
            return a.second < b.second;
    });

    std::transform(filtered_candi.begin(), filtered_candi.begin() + num_finalist, std::back_inserter(finalist), 
            [](const PairT& a) {return a.first;});
    timer_.stop();  

    // check ground truth - This part is not included in  running time

    std::sort(finalist.begin(), finalist.end());
    std::set_intersection(ground_truth.begin(),
       ground_truth.end(), finalist.begin(), finalist.end(), std::back_inserter(v_intersection));
    double recall = (float) v_intersection.size() / (float) K;
    recall_sum += recall;
}


int main (int argc, char* argv[]) {
    std::string filename = argv[1];
    std::string outputname = argv[2];
    std::ifstream json_f(filename);
    json config;
    json_f >> config;

    try {
        n = config["training size"];
        qn = config["testing size"];
        dim = config["dimension"];
        dim -= 2; // the dimension number in config is dimension + 2
        dim_Arows = config["number of query rows"];
        dim_Frows = config["number of filters"];
        K = config["number of neighbors"];
        finalist_ratio = config["ratio of finalists"];

        data_filename = config["data filename"];
        query_filename = config["query filename"];
        kernel_filename = config["kernel filename"];
        gt_filename = config["ground truth file"];
        candi_filename = config["candidate filename"];
        linear_scan = (candi_filename == "linear scan");

    } 
        catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        std::cerr << "Some required fields are missing in the config file! Aborted" << std::endl;
        exit(1);
    }
    catch (...) {
        std::cerr << "Some required fields are missing in the config file! Aborted" << std::endl;
        exit(1);
    }

    std::cout << "Reading data..." << std::endl;
    data_in = open_binary(data_filename);
    data_ = read_one_matrix(data_in, n, dim);
    query_in = open_binary(query_filename);
    queries_ = read_one_matrix(query_in, qn, dim);
    gt_in = open_binary(gt_filename);
    gt_ = read_ground_truth(gt_in, K, qn);
    candi_in = open_binary(candi_filename);
    kernel_in = open_binary(kernel_filename);
 
    filter_ = random_normal_matrix(rng, dim_Frows, dim_Arows);

    total_num_candi = 0;
    total_num_final = 0;
    recall_sum = 0.;

    for (int qid = 0; qid < qn; ++qid) {
        std::cout << "query id: " << qid << std::endl;
        run_query(qid);
    }
    
    // print results.
    // n, dim , qn, K, finalist_ratio, dim_Frows
    // total_num_candi, total_num_final, recall, running time
    std::ofstream fout(outputname);

    fout << "n\tdimension\tqn\tK\tfinalist_ratio\tnumber of filters" << std::endl;
    print_one_line(fout, n, dim, qn, K, finalist_ratio, dim_Frows);
    fout << "ratio fo candidates\tratio of finalists\trecall\tcpu time" << std::endl;
    print_one_line(fout, (double) total_num_candi/ ((double) n * qn) , 
        (double) total_num_final/ ((double) n * qn), recall_sum / qn, timer_.watch());
    return 0;
}