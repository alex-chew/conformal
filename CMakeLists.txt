cmake_minimum_required(VERSION 2.8)
project(Conformal)

set(CMAKE_CXX_FLAGS "-Wall -std=c++14")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(hello hello.cpp)
target_link_libraries(hello ${OpenCV_LIBS})
