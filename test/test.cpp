#include "../include/gauss.h"
#include <iostream>
#include <gtest/gtest.h>
#include <fstream>
#include <random>
using namespace Eigen;

TEST(ReadCSVTest, Test1) {
    std::string filename = "test_readcsv_valid.csv";
    std::ofstream file(filename);
    file << "1,2,3\n4,5,6\n7,8,9";
    file.close();
    int rows, cols;
    MatrixXd result = readCSV(filename, rows, cols);

    MatrixXd expected(3, 3);
    expected << 1, 2, 3,
               4, 5, 6,
               7, 8, 9;

    ASSERT_EQ(rows, 3);
    ASSERT_EQ(cols, 3);
    ASSERT_TRUE(result.isApprox(expected, 1e-6));
    remove(filename.c_str());

}

TEST(ReadCSVTest, Test2) {
    std::string filename = "test_readcsv_invalid.csv";
    std::ofstream file(filename);
    file << "1,2,3\n4,5\n7,8,9";
    file.close();
    int rows, cols;
    EXPECT_THROW({
        readCSV(filename, rows, cols);
    }, std::runtime_error);
    remove(filename.c_str());

}

TEST(WriteCSVTest, Test3) {
    Solution solution;
    solution.isInconsistent = true;
    std::string filename = "test_write_inconsistent.csv";
    writeCSV(filename, solution);

    std::ifstream file(filename);
    std::string content((std::istreambuf_iterator<char>(file)), {});
    ASSERT_NE(content.find("System is inconsistent"), std::string::npos);
    remove(filename.c_str());

}

TEST(SolveSystemTest, Test5) {
    MatrixXd A = MatrixXd::Zero(2, 2);
    VectorXd b(2);
    b << 1, 1;
    Solution solution = solveSystem(A, b);
    ASSERT_TRUE(solution.isInconsistent);
}

TEST(SolveSystemTest, Test6) {
    MatrixXd A(1, 3);
    A << 1, 1, 1;
    VectorXd b(1);
    b << 3;
    Solution solution = solveSystem(A, b);
    ASSERT_FALSE(solution.isInconsistent);
    ASSERT_EQ(solution.kernel.cols(), 2);
    for(int i = 0; i < solution.kernel.cols(); ++i) {
        double dot = A.row(0).dot(solution.kernel.col(i));
        ASSERT_NEAR(dot, 0.0, 1e-6);
    }

}

TEST(ReadCSVTest, EmptyFile) {
    std::string filename = "test_empty.csv";
    std::ofstream(filename).close();
    int rows, cols;
    MatrixXd result = readCSV(filename, rows, cols);
    ASSERT_EQ(rows, 0);
    ASSERT_EQ(cols, 0);
    remove(filename.c_str());
}

TEST(ReadCSVTest, Test7) {
    std::string filename = "test_invalid.csv";
    std::ofstream file(filename);
    file << "a,b,c\n1,2.5,three";
    file.close();
    int rows, cols;
    EXPECT_THROW({
        readCSV(filename, rows, cols);
    }, std::invalid_argument);
    remove(filename.c_str());

}

TEST(ReadCSVTest, WithSpacesAndTabs) {
    std::string filename = "test_whitespace.csv";
    std::ofstream file(filename);
    file << " 1 , 2 \t, 3.5\n\t4.2,5, 6.9";
    file.close();
    int rows, cols;
    MatrixXd result = readCSV(filename, rows, cols);

    MatrixXd expected(2, 3);
    expected << 1, 2, 3.5,
               4.2, 5, 6.9;

    ASSERT_EQ(rows, 2);
    ASSERT_EQ(cols, 3);
    ASSERT_TRUE(result.isApprox(expected, 1e-6));
    remove(filename.c_str());

}

TEST(SolveSystemTest, OneMoreTest) {
    MatrixXd A(1, 2);
    A << 2, 3;
    VectorXd b(1);
    b << 8;
    Solution solution = solveSystem(A, b);

    ASSERT_FALSE(solution.isInconsistent);
    ASSERT_EQ(solution.kernel.cols(), 1);
    ASSERT_NEAR((A * solution.particular)(0), 8.0, 1e-6);

}

TEST(SolveSystemTest, Test9) {
    MatrixXd A(2, 2);
    A << 1, 0.5,
    0.5, 1.0/3.0;
    VectorXd b(2);
    b << 1.5, 0.833333;
    Solution solution = solveSystem(A, b);

    VectorXd expected(2);
    expected << 1, 1;
    ASSERT_TRUE(solution.particular.isApprox(expected, 1e-3));

}

TEST(SolveSystemTest, Test10) {
    MatrixXd A(2, 1);
    A << 1, 2;
    VectorXd b(2);
    b << 1, 2;
    Solution solution = solveSystem(A, b);

    ASSERT_FALSE(solution.isInconsistent);
    ASSERT_NEAR(solution.particular[0], 1.0, 1e-6);

}


TEST(ReadCSVTest, Test11) {
    std::string filename = "test_scientific.csv";
    std::ofstream file(filename);
    file << "1e3,-2.5e-3,3.14\n-4.2E+1,5,6.9E2";
    file.close();
    int rows, cols;
    MatrixXd result = readCSV(filename, rows, cols);

    MatrixXd expected(2, 3);
    expected << 1000, -0.0025, 3.14,
               -42, 5, 690;

    ASSERT_TRUE(result.isApprox(expected, 1e-6));
    remove(filename.c_str());

}

TEST(ReadCSVTest, Test12) {
    std::string filename = "test_malformed.csv";
    std::ofstream file(filename);
    file << "1,,3\n4,5.2.1,6";
    file.close();
    int rows, cols;
    EXPECT_THROW({
        readCSV(filename, rows, cols);
    }, std::invalid_argument);
    remove(filename.c_str());

}

TEST(SolveSystemTest, Test13) {
    MatrixXd A(2, 2);
    A << 1, 1, 2, 2;
    VectorXd b(2);
    b << 1, 2;
    Solution solution = solveSystem(A, b);

    ASSERT_FALSE(solution.isInconsistent);
    ASSERT_NEAR(solution.particular[0] + solution.particular[1], 1.0, 1e-6);
    ASSERT_EQ(solution.kernel.cols(), 1);

}

TEST(WriteCSVTest, Test14) {
    Solution solution;
    solution.isInconsistent = false;
    solution.particular = Vector3d(1, 2, 3);
    solution.kernel.resize(3, 0); // Empty kernel
    std::string filename = "test_nokernel.csv";
    writeCSV(filename, solution);

    std::ifstream file(filename);
    std::string content((std::istreambuf_iterator<char>(file)), {});

    ASSERT_NE(content.find("Particular solution:\n1,2,3"), std::string::npos);
    ASSERT_EQ(content.find("Kernel basis:"), std::string::npos);
    remove(filename.c_str());

}

TEST(ReadCSVTest, Test15) {
    std::string filename = "test_large.csv";
    std::ofstream file(filename);
    file << "123456789,987654321\n-123456789.987654321,3.141592653589793";
    file.close();
    int rows, cols;
    MatrixXd result = readCSV(filename, rows, cols);

    MatrixXd expected(2, 2);
    expected << 123456789, 987654321,
               -123456789.987654321, 3.141592653589793;

    ASSERT_TRUE(result.isApprox(expected, 1e-9));
    remove(filename.c_str());

}

//@note very big random numbers
TEST(SolveSystemTest, LargeRandomSystems) {
    const int SEED = 42;
    const int MAX_SIZE = 100;
    std::mt19937 gen(SEED);
    std::uniform_int_distribution<> size_dist(10, MAX_SIZE);
    std::uniform_real_distribution<> value_dist(-10.0, 10.0);

    for(int test_num = 0; test_num < 5; ++test_num) {
        int rows = size_dist(gen);
        int cols = size_dist(gen);
        MatrixXd A = MatrixXd::Zero(rows, cols);
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                A(i, j) = value_dist(gen);
            }
        }

        VectorXd b = VectorXd::Zero(rows);
        for(int i = 0; i < rows; ++i) {
            b(i) = value_dist(gen);
        }

        Solution solution = solveSystem(A, b);

        if(!solution.isInconsistent) {
            VectorXd residual = A * solution.particular - b;
            ASSERT_NEAR(residual.norm(), 0.0, 1e-5)
                << "Residual norm: " << residual.norm()
                << "\nMatrix size: " << rows << "x" << cols;

            if(solution.kernel.cols() > 0) {
                for(int i = 0; i < solution.kernel.cols(); ++i) {
                    VectorXd kernel_vec = solution.kernel.col(i);
                    VectorXd product = A * kernel_vec;
                    ASSERT_NEAR(product.norm(), 0.0, 1e-5)
                        << "Kernel vector " << i
                        << " failed, norm: " << product.norm();
                }
            }
        }
    }
}

TEST(SolveSystemTest, LargeRankDeficientSystem) {
    const int SEED = 123;
    const int SIZE = 50;
    const int RANK = 30;

    std::mt19937 gen(SEED);
    std::uniform_real_distribution<> value_dist(-1.0, 1.0);

    MatrixXd U = MatrixXd::Random(SIZE, RANK).householderQr().householderQ();
    MatrixXd V = MatrixXd::Random(RANK, SIZE).householderQr().householderQ();
    MatrixXd A = U * V.transpose();

    VectorXd x = VectorXd::Random(SIZE);
    VectorXd b = A * x;

    Solution solution = solveSystem(A, b);

    ASSERT_FALSE(solution.isInconsistent);
    VectorXd residual = A * solution.particular - b;
    ASSERT_NEAR(residual.norm(), 0.0, 1e-8);
    int expected_nullity = SIZE - RANK;
    ASSERT_EQ(solution.kernel.cols(), expected_nullity);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}