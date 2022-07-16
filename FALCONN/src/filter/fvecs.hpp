#ifndef __FVECS_HPP__
#define __FVECS_HPP__
#include <Eigen/Dense>
#include <fstream>
#include <cassert>
#include <vector>
#include <string>

template<typename T>
using EigenMatrix =  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using FloatMatrix =  EigenMatrix<float>;
using FloatVector =  Eigen::VectorXf;
using IntegerVector = std::vector<int>;
using IntegerMatrix = std::vector<std::vector<int> >;

template<typename T>
void print_one_line(std::ostream& os, T value) {
    os << value << std::endl;
}

template<typename T, typename... Targs>
void print_one_line(std::ostream& os, T value, Targs... Fargs) {
    os << value << "\t";
    print_one_line(os, Fargs...);
}

std::ifstream open_binary(std::string filename) {
    return std::ifstream(filename, std::ios::binary);
}

// read operations from binary fvecs files
// used for reading data, query, kernel, 
FloatMatrix read_one_matrix(std::ifstream& fin, int rows, int cols) {
  char buff[4];
  FloatMatrix result(rows, cols);
    for (int rr = 0; rr < rows; ++rr) {
      fin.read(buff, 4);
      fin.read(reinterpret_cast<char*> (&result(rr, 0)), 4 * cols);
    }
    return result;
}

// used for reading candidates
IntegerVector read_vector(std::ifstream& fin) {
  int sz;
  fin.read(reinterpret_cast<char*> (&sz), 4);
  IntegerVector result(sz);
  fin.read(reinterpret_cast<char*> (result.data()), 4 * sz);
  return result;
}

// used for reading ground truth 
IntegerMatrix read_ground_truth(std::ifstream& fin, int K, int qn) {
  int sz;
  IntegerMatrix result;
  while (fin.read(reinterpret_cast<char*> (&sz), 4)) {
      IntegerVector cur_gt(K);
      fin.read(reinterpret_cast<char*> (cur_gt.data()), 4 * K);
      assert(sz >= K && "Ground truth must contain at least K nearest neighbors.");
      fin.seekg((sz-K) * 4, std::ios::cur); // skip the remaining neighbors for this query
      std::sort(cur_gt.begin(), cur_gt.end());
      result.push_back(std::move(cur_gt));
  }
  assert(result.size() == qn);
  return result;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& matrix){
    for(const auto& row : matrix){
         os << row << "\n";
    }
    return os;
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const std::pair<T1, T2>& pr){
    os << "(" << pr.first << ",\t" << pr.second << ")";
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec){
    for(const T& elem : vec){
         os << elem << "\t";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const Eigen::VectorXf & container){
    for(const auto& elem : container){
         os << elem << "\t";
    }
    return os;
}


#endif