cmake_minimum_required(VERSION 3.16)
project(tensor_autodiff_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(tinyad src/tensor.cpp)
target_include_directories(tinyad PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(tinyad PRIVATE -Wall -Wextra -Wpedantic)

add_executable(linear_adam examples/linear_adam.cpp)
target_link_libraries(linear_adam PRIVATE tinyad)

include(CTest)
if(BUILD_TESTING)
    add_executable(tensor_tests tests/test_tensor.cpp)
    target_link_libraries(tensor_tests PRIVATE tinyad)
    target_compile_options(tensor_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME tensor_tests COMMAND tensor_tests)
    add_test(NAME linear_adam_example COMMAND linear_adam)
endif()
