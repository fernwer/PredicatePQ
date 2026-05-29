cmake_minimum_required(VERSION 3.20)
project(predicatepq LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

option(PREDICATEPQ_ENABLE_IO_URING "Enable io_uring refiner" ON)
option(PREDICATEPQ_ENABLE_TESTS "Build tests" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(FAISS REQUIRED)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(yaml-cpp REQUIRED)

if(PREDICATEPQ_ENABLE_IO_URING)
  find_library(URING_LIB uring REQUIRED)
endif()

add_library(predicatepq
  src/predicate/predicate.cpp
  src/scalar/arrow_scalar_store.cpp
  src/planner/planner.cpp
  src/core/cluster_reduce.cpp
  src/core/engine.cpp
  src/pq/pq_scanner.cpp
  src/pq/simd_transpose.cpp
  src/io/refiner_mmap.cpp
  src/index/build_layout.cpp
  src/update/update_manager.cpp
)

if(PREDICATEPQ_ENABLE_IO_URING)
  target_sources(predicatepq PRIVATE src/io/refiner_io_uring.cpp)
  target_compile_definitions(predicatepq PUBLIC PREDICATEPQ_USE_IO_URING=1)
  target_link_libraries(predicatepq PUBLIC ${URING_LIB})
endif()

target_include_directories(predicatepq
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(predicatepq
  PUBLIC
    Threads::Threads
    OpenMP::OpenMP_CXX
    FAISS::faiss
    Arrow::arrow_shared
    Parquet::parquet_shared
    yaml-cpp
)

# 可执行程序
add_executable(build_index apps/build_index.cpp)
target_link_libraries(build_index PRIVATE predicatepq)

add_executable(query_cli apps/query_cli.cpp)
target_link_libraries(query_cli PRIVATE predicatepq)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/apps/benchmark.cpp")
  add_executable(benchmark apps/benchmark.cpp)
  target_link_libraries(benchmark PRIVATE predicatepq)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/apps/microbench_transpose.cpp")
  add_executable(microbench_transpose apps/microbench_transpose.cpp)
  target_link_libraries(microbench_transpose PRIVATE predicatepq)
endif()

# 编译优化
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(predicatepq PRIVATE -O3 -march=native -fopenmp)
endif()

# 安装（可选）
install(TARGETS predicatepq build_index query_cli
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)