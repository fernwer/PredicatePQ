cmake_minimum_required(VERSION 3.14)
project(PredicatePQ CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -fopenmp")

find_package(faiss REQUIRED)
find_package(OpenMP REQUIRED)

set(BLA_VENDOR Intel10_64lp)
find_package(BLAS)
find_package(LAPACK)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_executable(PredicatePQ_Demo main.cpp ${SOURCES})

target_link_libraries(PredicatePQ_Demo PRIVATE faiss OpenMP::OpenMP_CXX)

if(BLAS_FOUND AND LAPACK_FOUND)
    message(STATUS "Optimized BLAS/LAPACK (e.g., Intel MKL) found and enabled.")
    target_link_libraries(PredicatePQ_Demo PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
else()
    message(WARNING "Intel MKL not found. Performance may be degraded. Falling back to default backend.")
endif()