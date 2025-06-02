#include "../include/gauss.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

bool matrixReader(const std::string& filename, Eigen::MatrixXd& A, Eigen::VectorXd& b) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    std::vector<std::vector<double>> values;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(ss, cell, ',')) {
            std::stringstream cleaner(cell);
            double value;
            cleaner >> value;
            row.push_back(value);
        }

        if (!row.empty()) {
            values.push_back(row);
        }
    }

    if (values.empty()) {
        return false;
    }

    int rows = values.size();
    int cols = values[0].size();

    if (cols < 2) {
        return false;
    }

    A.resize(rows, cols - 1);
    b.resize(rows);

    for (int i = 0; i < rows; ++i) {
        if ((int)values[i].size() != cols) {
            return false;
        }

        for (int j = 0; j < cols - 1; ++j) {
            A(i, j) = values[i][j];
        }
        b(i) = values[i][cols - 1];
    }

    return true;
}

bool findIndependentRows(const Eigen::MatrixXd& A,
                         const Eigen::VectorXd& b,
                         Eigen::MatrixXd& reducedA,
                         Eigen::VectorXd& reducedB) {
    int n = A.rows();
    int m = A.cols();

    Eigen::MatrixXd Aug(n, m+1);
    Aug.leftCols(m) = A;
    Aug.col(m) = b;

    std::vector<int> independent_rows;
    const double eps = 1e-12;

    int r = 0;
    for (int j = 0; j < m; j++) {
        if (r >= n) {
            break;
        }
        int pivot_row = -1;
        double max_val = eps;
        for (int i = r; i < n; i++) {
            if (std::abs(Aug(i, j)) > max_val) {
                max_val = std::abs(Aug(i, j));
                pivot_row = i;
            }
        }

        if (pivot_row == -1) {
            continue;
        }

        if (pivot_row != r) {
            Aug.row(r).swap(Aug.row(pivot_row));
        }

        independent_rows.push_back(r);
        double pivot_val = Aug(r, j);

        for (int i = r+1; i < n; i++) {
            double factor = Aug(i, j) / pivot_val;
            for (int k = j; k < m+1; k++) {
                Aug(i, k) -= factor * Aug(r, k);
            }
        }

        r++;
    }

    for (int i = r; i < n; i++) {
        if (std::abs(Aug(i, m)) > eps) {
            bool all_zeros = true;
            for (int j = 0; j < m; j++) {
                if (std::abs(Aug(i, j)) > eps) {
                    all_zeros = false;
                    break;
                }
            }
            if (all_zeros) {
                return false;
            }
        }
    }

    int num_independent = independent_rows.size();
    reducedA.resize(num_independent, m);
    reducedB.resize(num_independent);

    for (int i = 0; i < num_independent; i++) {
        int row_idx = independent_rows[i];
        reducedA.row(i) = Aug.block(row_idx, 0, 1, m);
        reducedB(i) = Aug(row_idx, m);
    }

    return true;
}

bool gaussSolver(const Eigen::MatrixXd& A,
                 const Eigen::VectorXd& b,
                 Eigen::VectorXd& x_particular,
                 Eigen::MatrixXd& null_space_basis) {
    Eigen::MatrixXd reducedA;
    Eigen::VectorXd reducedB;

    if (!findIndependentRows(A, b, reducedA, reducedB)) {
        return false;
    }

    int n = reducedA.rows();
    int m = reducedA.cols();

    Eigen::MatrixXd Aug(n, m+1);
    Aug.leftCols(m) = reducedA;
    Aug.col(m) = reducedB;

    std::vector<int> pivot_cols;
    std::vector<bool> is_pivot(m, false);
    const double eps = 1e-12;

    int r = 0;
    for (int j = 0; j < m; j++) {
        if (r >= n) {
            break;
        }

        int max_row = r;
        for (int i = r+1; i < n; i++) {
            if (std::abs(Aug(i, j)) > std::abs(Aug(max_row, j))) {
                max_row = i;
            }
        }

        if (std::abs(Aug(max_row, j)) < eps) {
            continue;
        }

        Aug.row(r).swap(Aug.row(max_row));
        pivot_cols.push_back(j);
        is_pivot[j] = true;

        double pivot_val = Aug(r, j);
        for (int k = j; k < m+1; k++) {
            Aug(r, k) /= pivot_val;
        }

        for (int i = 0; i < n; i++) {
            if (i == r) {
                continue;
            }
            double factor = Aug(i, j);
            for (int k = j; k < m+1; k++) {
                Aug(i, k) -= factor * Aug(r, k);
            }
        }
        r++;
    }

    x_particular = Eigen::VectorXd::Zero(m);
    for (size_t i = 0; i < pivot_cols.size(); i++) {
        int col_idx = pivot_cols[i];
        x_particular(col_idx) = Aug(i, m);
    }

    std::vector<int> free_cols;
    for (int j = 0; j < m; j++) {
        if (!is_pivot[j]) {
            free_cols.push_back(j);
        }
    }

    null_space_basis = Eigen::MatrixXd(m, free_cols.size());
    for (int k = 0; k < free_cols.size(); k++) {
        int free_col = free_cols[k];
        Eigen::VectorXd basis_vec = Eigen::VectorXd::Zero(m);
        basis_vec(free_col) = 1;

        for (size_t i = 0; i < pivot_cols.size(); i++) {
            int pivot_col = pivot_cols[i];
            basis_vec(pivot_col) = -Aug(i, free_col);
        }

        null_space_basis.col(k) = basis_vec;
    }

    return true;
}

bool matrixWriter(const std::string& filename,
                 const Eigen::VectorXd& x_particular,
                 const Eigen::MatrixXd& null_space_basis) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    for (int i = 0; i < x_particular.size(); i++) {
        file << x_particular(i);
        if (i < x_particular.size() - 1) {
            file << ",";
        }
    }
    file << "\n";

    for (int col = 0; col < null_space_basis.cols(); col++) {
        for (int row = 0; row < null_space_basis.rows(); row++) {
            file << null_space_basis(row, col);
            if (row < null_space_basis.rows() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    return true;
}