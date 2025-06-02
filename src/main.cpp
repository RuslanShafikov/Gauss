#include "../include/gauss.h"
#include <iostream>
#include <filesystem>

int main() {
    Eigen::MatrixXd A;
    Eigen::VectorXd b, x_particular;
    Eigen::MatrixXd null_space_basis;

    if (!matrixReader("../data/input.csv", A, b)) {
        std::cerr << "Failed to read matrix.\n";
        return 1;
    }

    if (!gaussSolver(A, b, x_particular, null_space_basis)) {
        std::cerr << "System is inconsistent or has no solution.\n";
        return 1;
    }

    std::string output_path = "../data/output.csv";

    if (!matrixWriter(output_path, x_particular, null_space_basis)) {
        std::cerr << "Failed to write solution.\n";
        return 1;
    }

    return 0;
}