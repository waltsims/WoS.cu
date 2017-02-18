# CMake configuration for the ShapeRegistration project
cmake_minimum_required(VERSION 2.8)

project(WoS.cu)

option(DEBUG "compiling with debuging output" OFF)
option(OUT "compiling for export/output" OFF)

set(STANDARD_FLAGS ${STANDARD_FLAGS}  --compiler-options -Wall -std=c++11 -lineinfo)

# Check if CUDA is available
	message( "creating Makefile for lightning speed GPU implementation")
	find_package(CUDA REQUIRED)
	if(CUDA_FOUND)

		# Set the compilation flags for CUDA
		set ( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${STANDARD_FLAGS}  --ptxas-options=-v,-maxrregcount=50 --use_fast_math
		-Wno-deprecated-gpu-targets)

		if(DEBUG)
			message( "Debugging flag being set")
			add_definitions(-DDEBUG)
		endif()
		if(OUT)
			message( "OUT flag being set")
			add_definitions(-DOUT)
		endif()
		# Specify the binary name and the source files to build
		add_subdirectory(src/host)
		file(GLOB SRC src/wos_thrust.cuh src/helper.cpp  main.cu src/export.cpp  src/params.cpp src/parse.cpp )
		file(GLOB LIB src/wos_thrust.h src/helper.hpp src/clock.h src/export.h	src/params.h src/parse.h src/reduce_kernel.cuh src/wos_wrapper.cuh src/wos_native.cuh)
		file(GLOB INC inc/helper_string.h inc/helper_cuda.h)
		cuda_add_executable(wos ${SRC} ${LIB} ${INC})
		target_link_libraries(wos ${CUDA_LIBARIES} host)

	else()
		# CUDA not found
		message(CRITICAL_ERROR "CUDA cannot be found, please ensure CUDA dependencies are installed")

	endif()
