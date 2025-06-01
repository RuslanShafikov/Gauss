
#ifndef GAUSS_H
#define GAUSS_H
#include <Eigen/Dense>
#include <string>

bool matrixReader(const std::string& filename,
                 Eigen::MatrixXd& A,
                 Eigen::VectorXd& b);

bool findIndependentRows(const Eigen::MatrixXd& A,
                         const Eigen::VectorXd& b,
                         Eigen::MatrixXd& reducedA,
                         Eigen::VectorXd& reducedB);

bool gaussSolver(const Eigen::MatrixXd& A,
                 const Eigen::VectorXd& b,
                 Eigen::VectorXd& x_particular,
                 Eigen::MatrixXd& null_space_basis);

bool matrixWriter(const std::string& filename,
                 const Eigen::VectorXd& x_particular,
                 const Eigen::MatrixXd& null_space_basis);


#endif //GAUSS_H
