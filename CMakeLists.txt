cmake_minimum_required(VERSION 3.14)
project(PredicatePQ CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")

find_package(faiss REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_executable(PredicatePQ main.cpp ${SOURCES})
target_link_libraries(PredicatePQ PRIVATE faiss OpenMP::OpenMP_CXX)