cmake_minimum_required(VERSION 3.5)

set( CMAKE_CXX_FLAGS  ${CMAKE_CXX_FLAGS}  "-g -std=c++11" )
file(GLOB SRC_FILES *.cpp )
add_library(host ${SRC_FILES})
target_include_directories (host PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


