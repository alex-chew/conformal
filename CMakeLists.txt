cmake_minimum_required(VERSION 2.8)
project(Conformal)

set(CMAKE_CXX_FLAGS "-Wall -std=c++14")
add_executable(conformal conformal.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(conformal ${OpenCV_LIBS})

find_package(Armadillo REQUIRED)
target_link_libraries(conformal lapack openblas)
