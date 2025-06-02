#include "../include/gauss.h"
#include <gtest/gtest.h>
#include <fstream>
#include <random>
#include <filesystem>

using namespace Eigen;
namespace fs = std::filesystem;

TEST(GaussTest, ReadValidCSV) {
    std::string filename = "test_readcsv_valid.csv";
    std::ofstream file(filename);
    file << "1,2,3\n4,5,6\n7,8,9";
    file.close();

    MatrixXd A;
    VectorXd b;
    bool success = matrixReader(filename, A, b);

    MatrixXd expectedA(3, 2);
    expectedA << 1,2,4,5,7,8;
    VectorXd expectedB(3);
    expectedB << 3,6,9;

    ASSERT_TRUE(success);
    ASSERT_EQ(A.rows(), 3);
    ASSERT_EQ(A.cols(), 2);
    ASSERT_EQ(b.size(), 3);
    ASSERT_TRUE(A.isApprox(expectedA, 1e-6));
    ASSERT_TRUE(b.isApprox(expectedB, 1e-6));

    fs::remove(filename);
}

TEST(GaussTest, ReadInvalidCSV) {
    std::string filename = "test_readcsv_invalid.csv";
    std::ofstream file(filename);
    file << "1,2,3\n4,5\n7,8,9";
    file.close();

    MatrixXd A;
    VectorXd b;
    bool success = matrixReader(filename, A, b);

    ASSERT_FALSE(success);
    fs::remove(filename);
}

TEST(GaussTest, WriteSolution) {
    VectorXd x_particular(3);
    x_particular << 1, 2, 3;
    MatrixXd null_space(3, 2);
    null_space << 4,5,6,7,8,9;

    std::string filename = "test_write.csv";
    bool success = matrixWriter(filename, x_particular, null_space);
    ASSERT_TRUE(success);

    // Verify file content
    std::ifstream file(filename);
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    ASSERT_EQ(lines.size(), 3);
    ASSERT_EQ(lines[0], "1,2,3");
    ASSERT_EQ(lines[1], "4,6,8");
    ASSERT_EQ(lines[2], "5,7,9");

    fs::remove(filename);
}

TEST(GaussTest, InconsistentSystem) {
    MatrixXd A(2, 2);
    A << 1,1,1,1;
    VectorXd b(2);
    b << 1,2;

    VectorXd x_particular;
    MatrixXd null_space;
    bool success = gaussSolver(A, b, x_particular, null_space);

    ASSERT_FALSE(success);
}

TEST(GaussTest, UnderdeterminedSystem) {
    MatrixXd A(1, 3);
    A << 1,1,1;
    VectorXd b(1);
    b << 3;

    VectorXd x_particular;
    MatrixXd null_space;
    bool success = gaussSolver(A, b, x_particular, null_space);

    ASSERT_TRUE(success);
    ASSERT_NEAR(A.row(0).dot(x_particular), 3.0, 1e-6);
    ASSERT_EQ(null_space.cols(), 2);

    // Verify null space vectors
    for(int i = 0; i < null_space.cols(); ++i) {
        VectorXd basis = null_space.col(i);
        ASSERT_NEAR(A.row(0).dot(basis), 0.0, 1e-6);
    }
}

TEST(GaussTest, ReadCSVWithWhitespace) {
    std::string filename = "test_whitespace.csv";
    std::ofstream file(filename);
    file << " 1 , 2 \t, 3.5\n\t4.2,5, 6.9";
    file.close();

    MatrixXd A;
    VectorXd b;
    bool success = matrixReader(filename, A, b);

    MatrixXd expectedA(2, 2);
    expectedA << 1,2,4.2,5;
    VectorXd expectedB(2);
    expectedB << 3.5,6.9;

    ASSERT_TRUE(success);
    ASSERT_TRUE(A.isApprox(expectedA, 1e-6));
    ASSERT_TRUE(b.isApprox(expectedB, 1e-6));
    fs::remove(filename);
}

TEST(GaussTest, SimpleSystemSolution) {
    MatrixXd A(1, 2);
    A << 2,3;
    VectorXd b(1);
    b << 8;

    VectorXd x_particular;
    MatrixXd null_space;
    bool success = gaussSolver(A, b, x_particular, null_space);

    ASSERT_TRUE(success);
    ASSERT_NEAR(A.row(0).dot(x_particular), 8.0, 1e-6);
    ASSERT_EQ(null_space.cols(), 1);
}

TEST(GaussTest, FloatingPointPrecision) {
    MatrixXd A(2, 2);
    A << 1, 0.5,
         0.5, 1.0/3.0;
    VectorXd b(2);
    b << 1.5, 0.833333;

    VectorXd x_particular;
    MatrixXd null_space;
    bool success = gaussSolver(A, b, x_particular, null_space);

    ASSERT_TRUE(success);
    VectorXd expected(2);
    expected << 1,1;
    ASSERT_TRUE(x_particular.isApprox(expected, 1e-3));
}

TEST(GaussTest, ScientificNotation) {
    std::string filename = "test_scientific.csv";
    std::ofstream file(filename);
    file << "1e3,-2.5e-3,3.14\n-4.2E+1,5,6.9E2";
    file.close();

    MatrixXd A;
    VectorXd b;
    bool success = matrixReader(filename, A, b);

    MatrixXd expectedA(2, 2);
    expectedA << 1000, -0.0025, -42, 5;
    VectorXd expectedB(2);
    expectedB << 3.14, 690;

    ASSERT_TRUE(success);
    ASSERT_TRUE(A.isApprox(expectedA, 1e-6));
    ASSERT_TRUE(b.isApprox(expectedB, 1e-6));
    fs::remove(filename);
}

TEST(GaussTest, RankDeficientSystem) {
    MatrixXd A(2, 2);
    A << 1,1,2,2;
    VectorXd b(2);
    b << 1,2;

    VectorXd x_particular;
    MatrixXd null_space;
    bool success = gaussSolver(A, b, x_particular, null_space);

    ASSERT_TRUE(success);
    ASSERT_NEAR(x_particular[0] + x_particular[1], 1.0, 1e-6);
    ASSERT_EQ(null_space.cols(), 1);
}

TEST(GaussTest, LargeRandomSystems) {
    const int SEED = 42;
    const int MAX_SIZE = 100;
    std::mt19937 gen(SEED);
    std::uniform_int_distribution<> size_dist(10, MAX_SIZE);
    std::uniform_real_distribution<> value_dist(-10.0, 10.0);

    for(int test_num = 0; test_num < 5; ++test_num) {
        int rows = size_dist(gen);
        int cols = size_dist(gen);
        MatrixXd A = MatrixXd::Random(rows, cols);
        VectorXd b = VectorXd::Random(rows);

        VectorXd x_particular;
        MatrixXd null_space;
        bool success = gaussSolver(A, b, x_particular, null_space);

        if(success) {
            VectorXd residual = A * x_particular - b;
            ASSERT_NEAR(residual.norm(), 0.0, 1e-5)
                << "Residual norm: " << residual.norm()
                << "\nMatrix size: " << rows << "x" << cols;

            for(int i = 0; i < null_space.cols(); ++i) {
                VectorXd kernel_vec = null_space.col(i);
                VectorXd product = A * kernel_vec;
                ASSERT_NEAR(product.norm(), 0.0, 1e-5)
                    << "Kernel vector " << i
                    << " failed, norm: " << product.norm();
            }
        }
    }
}

TEST(GaussTest, LargeRankDeficientSystem) {
    const int SEED = 123;
    const int SIZE = 50;
    const int RANK = 30;

    std::mt19937 gen(SEED);
    MatrixXd U = MatrixXd::Random(SIZE, RANK);
    MatrixXd V = MatrixXd::Random(RANK, SIZE);
    MatrixXd A = U * V;
    VectorXd x = VectorXd::Random(SIZE);
    VectorXd b = A * x;

    VectorXd x_particular;
    MatrixXd null_space;
    bool success = gaussSolver(A, b, x_particular, null_space);

    ASSERT_TRUE(success);
    VectorXd residual = A * x_particular - b;
    ASSERT_NEAR(residual.norm(), 0.0, 1e-8);
    int expected_nullity = SIZE - RANK;
    ASSERT_EQ(null_space.cols(), expected_nullity);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}