cmake_minimum_required(VERSION 3.1)

set(CMAKE_BUILD_TYPE Release)
set(LIB_ONLY ON)
set(WITH_PYTHON_BINDINGS ON CACHE BOOL "enable Python bindings")

add_subdirectory(third_party/OpenNMTTokenizer)
