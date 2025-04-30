#include "../include/gauss.h"
#include <iostream>


int main() {
    try {
        int rows, cols;
        MatrixXd fullMatrix = readCSV("input.csv", rows, cols);

        if (cols < 2) {
            std::cerr << "Need at least 2 columns" << std::endl;
            return 1;
        }

        MatrixXd A = fullMatrix.leftCols(cols - 1);
        VectorXd b = fullMatrix.rightCols(1);

        Solution solution = solveSystem(A, b);
        writeCSV("output.csv", solution);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}