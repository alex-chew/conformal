cmake_minimum_required(VERSION 2.8)
project(Conformal)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic")
add_executable(conformal conformal.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(conformal ${OpenCV_LIBS})

find_package(LAPACK REQUIRED)
target_link_libraries(conformal ${LAPACK_LIBRARIES})

find_package(BLAS REQUIRED)
target_link_libraries(conformal ${BLAS_LIBRARIES})

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(conformal ${ARMADILLO_LIBRARIES})
add_definitions(-DARMA_DONT_USE_WRAPPER)
