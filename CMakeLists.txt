# CMake configuration for the ShapeRegistration project
cmake_minimum_required(VERSION 2.8)

project(WoS.cu)

option(GPU "compile for lightning GPU speeds" ON)
option(DEBUG "compiling with debuging output" ON)
option(PLOT "compiling for plot output" ON)

set(STANDARD_FLAGS ${STANDARD_FLAGS} --std=c++11 --compiler-options -Wall)


# Check if CUDA is available
	message( "creating Makefile for lightning speed GPU implementation")
	find_package(CUDA REQUIRED)
	if(CUDA_FOUND)

		# Set the compilation flags for CUDA
		set ( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${STANDARD_FLAGS} --ptxas-options=-v --use_fast_math
			-Wno-deprecated-gpu-targets )

		if(DEBUG)
			message( "Debugging flag being set")
			add_definitions(-DDEBUG)
		endif()
		# Specify the binary name and the source files to build
		cuda_add_executable(wos src/clock.h src/wos.cu src/wos_kernel.cuh src/reduce_kernel.cuh)
		target_link_libraries(wos ${CUDA_LIBARIES} )

	else()
		# CUDA not found
		message(CRITICAL_ERROR "CUDA cannot be found, please ensure CUDA dependencies are installed")

	endif()
