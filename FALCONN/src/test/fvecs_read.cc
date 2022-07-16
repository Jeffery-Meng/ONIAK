#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <VectorUtils.hpp>

using namespace VectorUtils;
using namespace std;
template <typename CoordinateType>
using DenseMatrix = Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;


template <typename CoordinateType>
DenseMatrix<CoordinateType> read_one_matrix(std::ifstream& fin, int rows, int cols) {
  static_assert(sizeof(CoordinateType) == 4);

  char buff[4];
  DenseMatrix<CoordinateType> result(rows, cols);
    for (int rr = 0; rr < rows; ++rr) {
      fin.read(buff, 4);
      for (int cc = 0; cc < cols; ++cc) {
          fin.read(reinterpret_cast<char*> (&result(rr, cc)), 4);
      }
    }
    return result;
}

int main () {
    std::ifstream fin("/home/gtnetuser/alt/half_kernels/audio.fvecs", ios::binary);
    auto matrix = read_one_matrix<float>(fin, 192, 192);

    //cout << matrix.row(145);

    matrix = read_one_matrix<float>(fin, 192, 192);
    cout << matrix.row(23);

    return 0;
}