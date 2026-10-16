cmake_minimum_required(VERSION 3.20)
project(parallel_lj CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

enable_testing()

add_executable(test_lj tests/test_lj.cpp)
target_include_directories(test_lj PRIVATE src)
add_test(NAME test_lj COMMAND test_lj)

add_executable(test_md tests/test_md.cpp)
target_include_directories(test_md PRIVATE src)
add_test(NAME test_md COMMAND test_md)

add_executable(test_neighbor tests/test_neighbor.cpp)
target_include_directories(test_neighbor PRIVATE src)
add_test(NAME test_neighbor COMMAND test_neighbor)
