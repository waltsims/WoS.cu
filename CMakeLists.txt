
cmake_minimum_required(VERSION 2.8)

project(WoS.cu)

option(DEBUG "compiling with debuging output" OFF)
option(OUT "compiling for export/output" OFF)

if(DEBUG)
	message( "Debugging flag being set")
	add_definitions(-DDEBUG)
endif()

if(OUT)
	message( "OUT flag being set")
	add_definitions(-DOUT)
endif()

# Specify the binary name and the source files to build
include_directories(inc)
add_subdirectory(src)
add_executable(wos src)
target_link_libraries(wos src)


