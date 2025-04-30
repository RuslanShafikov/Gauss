#include "../include/gauss.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <string>


MatrixXd readCSV(const std::string &filename, int &rows, int &cols) {
    std::ifstream file(filename);
    std::vector<double> matrixEntries;
    rows = 0;
    cols = 0;

    std::string line;
    while (getline(file, line)) {
        line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());
        std::stringstream ss(line);
        std::string cell;
        int currentCols = 0;
        while (getline(ss, cell, ',')) {
            matrixEntries.push_back(stod(cell));
            currentCols++;
        }
        if (rows == 0) cols = currentCols;
        else if (currentCols != cols) throw std::runtime_error("Column mismatch");
        rows++;
    }

    return Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(matrixEntries.data(), rows, cols);
}

void writeCSV(const std::string &filename, const Solution& solution) {
    std::ofstream file(filename);
    if (solution.isInconsistent) {
        file << "System is inconsistent";
        return;
    }

    file << "Particular solution:\n";
    for (int i = 0; i < solution.particular.size(); ++i) {
        file << solution.particular(i);
        if (i < solution.particular.size() - 1) file << ",";
    }

    if (solution.kernel.size() > 0) {
        file << "\n\nKernel basis:\n";
        for (int i = 0; i < solution.kernel.rows(); ++i) {
            for (int j = 0; j < solution.kernel.cols(); ++j) {
                file << solution.kernel(i, j);
                if (j < solution.kernel.cols() - 1) file << ",";
            }
            file << "\n";
        }
    }
}

Solution solveSystem(const MatrixXd& A, const VectorXd& b) {
    Solution solution;
    MatrixXd augmented(A.rows(), A.cols() + 1);
    augmented << A, b;

    FullPivLU<MatrixXd> luA(A);
    FullPivLU<MatrixXd> luAug(augmented);

    if (luA.rank() != luAug.rank()) {
        solution.isInconsistent = true;
        return solution;
    }

    if (luA.rank() == A.cols()) {
        solution.particular = A.colPivHouseholderQr().solve(b);
    } else {
        solution.particular = luA.solve(b);
        solution.kernel = luA.kernel();
    }

    return solution;
}