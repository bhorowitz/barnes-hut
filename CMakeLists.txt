cmake_minimum_required(VERSION 3.21)
project(diff-barnes-hut CXX)

set(CMAKE_BUILD_TYPE Release)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70) # matches compute capability 7.0 on this machine

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA OPTIONAL)
endif()

include(eigen)
include(pybind11)
find_package(Python REQUIRED COMPONENTS Interpreter)

pybind11_add_module(sbh src/sbh.cpp)
target_link_libraries(sbh PUBLIC Eigen3::Eigen)
target_include_directories(sbh PUBLIC include)

if(CMAKE_CUDA_COMPILER)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import jax.ffi; print(jax.ffi.include_dir())"
    OUTPUT_VARIABLE JAXLIB_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  pybind11_add_module(sbh_gpu src/sbh_gpu.cu)
  target_link_libraries(sbh_gpu PUBLIC Eigen3::Eigen stdc++)
  target_include_directories(sbh_gpu PUBLIC include)
  target_compile_options(sbh_gpu PUBLIC --extended-lambda -lineinfo --use_fast_math)

  add_library(sbh_ffi SHARED src/sbh_ffi.cu)
  target_include_directories(sbh_ffi PRIVATE include ${JAXLIB_INCLUDE_DIR})
  target_link_libraries(sbh_ffi PUBLIC Eigen3::Eigen stdc++)
  target_compile_options(sbh_ffi PUBLIC --extended-lambda -lineinfo --use_fast_math)
endif()
