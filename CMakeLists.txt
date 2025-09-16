cmake_minimum_required(VERSION 3.20)
project(lubrication)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/" "$ENV{CONDA_PREFIX}/share/cmake/Modules")
set(CMAKE_BUILD_TYPE Release)
list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")


enable_language(CXX)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)

set(SOURCES src/c_lubrication.cpp)

set(BLA_VENDOR Generic)
# find_package(BLAS REQUIRED)
# find_package(LAPACK REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package (Eigen3 REQUIRED NO_MODULE)
include_directories(${Python_INCLUDE_DIRS})
# include_directories($ENV{CONDA_PREFIX}/include)

# if(APPLE) # for pybind11 on mac
  # set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS
      # "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -undefined dynamic_lookup")
# endif()

if(DOUBLEPRECISION)
  add_compile_definitions(PUBLIC DOUBLE_PRECISION)
endif()

if(NOT Python_SOABI)
  message(FATAL_ERROR "Python_SOABI is empty. Set it to the output of 'python-config --extension-suffix'")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR)
message(STATUS "python executable: ${Python_EXECUTABLE}")
message(STATUS "nanobind_DIR: ${nanobind_DIR}")
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(c_lubrication.${Python_SOABI} ${SOURCES})

set_target_properties(c_lubrication.${Python_SOABI} PROPERTIES SUFFIX ".so") # force .so suffix for mac
set_target_properties(c_lubrication.${Python_SOABI} PROPERTIES PREFIX "") # remove the lib prefix

target_compile_options(c_lubrication.${Python_SOABI} PRIVATE -fPIC -O3) # -Wall

target_link_libraries(c_lubrication.${Python_SOABI} PRIVATE Eigen3::Eigen)

install(TARGETS c_lubrication.${Python_SOABI} DESTINATION ${Python_SITEARCH}/Lubrication)
# file(GLOB PYTHON_SOURCES src/*.py)
install(FILES ${PYTHON_SOURCES} DESTINATION ${Python_SITEARCH}/Lubrication)