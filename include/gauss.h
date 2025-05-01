
#ifndef GAUSS_H
#define GAUSS_H
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
using namespace Eigen;

struct Solution {
    VectorXd particular;
    MatrixXd kernel;
    bool isInconsistent = false;
};

Solution solveSystem(const MatrixXd& A, const VectorXd& b);
MatrixXd readCSV(const std::string& filename, int& rows, int& cols);
void writeCSV(const std::string& filename, const Solution& solution);
#endif //GAUSS_H
