
#ifndef C_LUBRICATION_EIGEN_TYPEDEFS_H
#define C_LUBRICATION_EIGEN_TYPEDEFS_H

#ifndef DOUBLE_PRECISION
#define SINGLE_PRECISION
#endif

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::MatrixXi IMatrix;

#ifdef SINGLE_PRECISION
using real = float;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::Vector3f Vector3;
typedef Eigen::Matrix3f Matrix3;
#else
using real = double;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::Vector3d Vector3;
typedef Eigen::Matrix3d Matrix3;
#endif

typedef Eigen::SparseMatrix<real> SpMatrix;

#endif // C_LUBRICATION_EIGEN_TYPEDEFS_H
