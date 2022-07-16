#ifndef _VECTOR_UTILS_HPP_
#define _VECTOR_UTILS_HPP_

#include <vector>
#include "Exception.h"
#include <sstream>
#include <utility>
#include <numeric>
#include <algorithm>


namespace VectorUtils{

template <typename T>
std::vector<std::vector<T> > to_matrix(const std::vector<T>& vector, int rows, int cols) {
    NPP_ASSERT(vector.size() == rows * cols);
    std::vector<std::vector<T> > result(rows);

    for (int i = 0; i < rows; ++i) {
        result[i].assign(vector.begin() + i*cols, vector.begin() + (i+1)*cols);
    }
    return result;
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

template <typename T>
void pack(std::string &line, std::vector<T>& vec) {
    std::istringstream line_stream(line);

    vec.clear();
    while (true) {
        T val;
        line_stream >> val;
        if (line_stream) {
            vec.push_back(val);
        } else {
            break;
        }
    }
}

template <typename T>
std::istream& operator>>(std::istream& is, std::vector<T>& vec){
    // use endline as separator
    std::string line;
    std::getline(is, line);
    pack(line, vec);
    while (vec.size() == 0){
        std::getline(is, line);
        pack(line, vec);
    }
    return is;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& matrix){
    for(const auto& row : matrix){
         os << row << "\n";
    }
    return os;
}

// read to vector: at most n elements
template <typename T>
int read_vector(std::istream& is, std::vector<T>& vec, int n){
    for(int i = 0; i < n; ++i){
        T val;
        is >> val;
        if (is){
            vec.push_back(val);
        } else {
            break;
        }
    }
    return vec.size();
}
   
// read n by n matrix
template <typename T>
int read_matrix(std::istream& is, std::vector<std::vector<T>>& matrix, int n){
    int result = 0;
    matrix.resize(n);
    for(int i = 0; i < n; ++i){
        result += read_vector(is, matrix[i], n);
    }
    return result;
}

template <typename T>
int vector_sum(const std::vector<T>& vec){
    return std::accumulate(vec.begin(), vec.end(), 0);
}

template <typename T>
int matrix_sum(const std::vector<std::vector<T>>& mat){
    return std::accumulate(mat.begin(), mat.end(), 0, 
        [](const int tmp, const std::vector<T>& vec){
            return tmp + vector_sum(vec);
    });
}

template <typename T>
int max_row_sum(const std::vector<std::vector<T>>& mat){
    std::vector<int> row_sums(mat.size(), 0);
    for (int row = 0; row < mat.size(); ++row){
        row_sums[row] = vector_sum(mat[row]);
    }
    return *std::max_element(row_sums.begin(), row_sums.end());
}

template <typename T>
int max_column_sum(const std::vector<std::vector<T>>& mat){
    if (mat.size() == 0) return -1;
    std::vector<int> col_sums(mat[0].size(), 0);
    for (int col = 0; col < mat[0].size(); ++col){
        int sum = 0;
        for (int row = 0; row < mat.size(); ++row) 
            sum += mat[row][col];
        col_sums[col] = sum;
    }
    return *std::max_element(col_sums.begin(), col_sums.end());
}

} // end namespace VectorUtils

#endif