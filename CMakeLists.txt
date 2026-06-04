cmake_minimum_required(VERSION 3.20)
project(predicatepq LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

option(PREDICATEPQ_ENABLE_IO_URING "Enable io_uring refiner" ON)

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

add_library(predicatepq_core
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
  target_sources(predicatepq_core PRIVATE src/io/refiner_io_uring.cpp)
  target_compile_definitions(predicatepq_core PUBLIC PREDICATEPQ_USE_IO_URING=1)
  target_link_libraries(predicatepq_core PUBLIC ${URING_LIB})
endif()

target_include_directories(predicatepq_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(predicatepq_core
  PUBLIC
    Threads::Threads
    OpenMP::OpenMP_CXX
    FAISS::faiss
    Arrow::arrow_shared
    Parquet::parquet_shared
    yaml-cpp
)

add_executable(predicatepq src/main.cpp)
target_link_libraries(predicatepq PRIVATE predicatepq_core)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(predicatepq_core PRIVATE -O3 -march=native -fopenmp)
endif()

install(TARGETS predicatepq predicatepq_core
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)