cmake_minimum_required(VERSION 3.18)
project(RamulatorAtomicCandidate LANGUAGES CXX)

# This recipe is trusted. Only the named model files are supplied by the agent;
# it cannot supply CMake code, compiler flags or additional include directories.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT MODEL_SOURCES)
  message(FATAL_ERROR "MODEL_SOURCES must name the submitted model files")
endif()
add_library(candidate MODULE ${MODEL_SOURCES})
target_include_directories(candidate PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/api")
target_compile_options(candidate PRIVATE -O3)
target_link_options(candidate PRIVATE -Wl,-z,defs)
set_target_properties(candidate PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/output"
)
