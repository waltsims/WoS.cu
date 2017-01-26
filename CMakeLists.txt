# CMake configuration for the ShapeRegistration project
cmake_minimum_required(VERSION 2.8)

project(WoS.cu)

option(DEBUG "compiling with debuging output" OFF)
option(PLOT "compiling for plot output" OFF)
option(CPU_REDUCE "complete reduction on CPU" OFF)

set(STANDARD_FLAGS ${STANDARD_FLAGS} --std=c++11 --compiler-options -Wall -lineinfo)


# Check if CUDA is available
	message( "creating Makefile for lightning speed GPU implementation")
	find_package(CUDA REQUIRED)
	if(CUDA_FOUND)

		# Set the compilation flags for CUDA
		set ( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${STANDARD_FLAGS} --ptxas-options=-v,-maxrregcount=50 --use_fast_math
		-Wno-deprecated-gpu-targets)

		if(DEBUG)
			message( "Debugging flag being set")
			add_definitions(-DDEBUG)
		endif()
		if(PLOT)
			message( "Plot flag being set")
			add_definitions(-DPLOT)
		endif()
		if(CPU_REDUCE)
			message( "Plot flag being set")
			add_definitions(-DCPU_REDUCE)
		endif()
		# Specify the binary name and the source files to build
		file(GLOB SRC src/helper.cpp src/main.cu src/wos_kernel.cuh src/params.cpp src/reduce_kernel.cuh src/parse.cpp )
		file(GLOB LIB src/helper.hpp src/clock.h src/params.h src/parse.h)
		cuda_add_executable(wos ${SRC} ${LIB}  )
		target_link_libraries(wos ${CUDA_LIBARIES} )

	else()
		# CUDA not found
		message(CRITICAL_ERROR "CUDA cannot be found, please ensure CUDA dependencies are installed")

	endif()
