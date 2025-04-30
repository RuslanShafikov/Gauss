cmake_minimum_required(VERSION 3.20)
project(GaussSolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen
find_package(Eigen3 REQUIRED)

# Google Test
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# Main executable
add_executable(gauss_solver
        src/main.cpp
        src/gauss.cpp
)
target_include_directories(gauss_solver PRIVATE include)
target_link_libraries(gauss_solver Eigen3::Eigen)

# Test executable
add_executable(gauss_test
        test/test.cpp
        src/gauss.cpp
)
target_include_directories(gauss_test PRIVATE include)
target_link_libraries(gauss_test
        GTest::gtest_main
        Eigen3::Eigen
)