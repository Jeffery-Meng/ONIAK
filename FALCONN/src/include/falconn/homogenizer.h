#ifndef __HOMOGENIZER_H__
#define __HOMOGENIZER_H__

#include <cassert>
#include "lsh_nn_table.h"

/* Convert Heterogeneous searching |Ax-y|_2 to Homogeneous Searching |Ax|_2 */

namespace falconn {

    template <typename PointSet>
    void homogenize_data(PointSet& points, int dimension) {
        // dimension is the number of dimensions in the homogenized data
        // for example, if the original dataset has dimension 100
        // here the dimension is 101
        assert(dimension > 1 && dimension < points[0].size());
        for (auto& pt : points) {
            pt(dimension-1) = -1.0;
        }
    }

    std::vector<MatrixType> queries_from_single_A(const DenseMatrix<CoordinateType>& mat_A, 
        const PointSet& points) {
            std::vector<MatrixType> result;
            result.reserve(points.size());

            for (const auto& pt: points) {
                DenseMatrix<CoordinateType> que(mat_A.rows(), mat_A.cols()+1);
                que.topLeftCorner(mat_A.rows(), mat_A.cols()) = mat_A;
                que.topRightCorner(mat_A.rows(), 1) = mat_A * pt;
                result.push_back(std::move(que));
            }

            return result;
        }
    
    DenseMatrix<CoordinateType> combine_one(const DenseMatrix<CoordinateType>& mat_A, const PointType& y)
    {   
        // Just combine, do not transform
        DenseMatrix<CoordinateType> que(mat_A.rows(), mat_A.cols()+1);
                que.topLeftCorner(mat_A.rows(), mat_A.cols()) = mat_A;
                que.topRightCorner(mat_A.rows(), 1) = y;
        return que;
    }

    DenseMatrix<CoordinateType> combine_one_transform(const DenseMatrix<CoordinateType>& mat_A, const PointType& y)
    {   
        // Just combine, do not transform
        DenseMatrix<CoordinateType> que(mat_A.rows(), mat_A.cols()+1);
                que.topLeftCorner(mat_A.rows(), mat_A.cols()) = mat_A;
                que.topRightCorner(mat_A.rows(), 1) = mat_A * y;
        return que;
    }

}

#endif