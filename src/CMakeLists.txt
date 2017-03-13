
cmake_minimum_required(VERSION 2.8)

set(STANDARD_FLAGS ${STANDARD_FLAGS}  --compiler-options -Wall -lineinfo -D_MWAITXINTRIN_H_INCLUDED)

# Check if CUDA is available
	message( "creating Makefile for lightning speed GPU implementation")
	find_package(CUDA REQUIRED)
	if(CUDA_FOUND)

		# Set the compilation flags for CUDA
		set ( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${STANDARD_FLAGS}  --ptxas-options=-v,-maxrregcount=50 --use_fast_math
		-Wno-deprecated-gpu-targets)

		# Specify the binary name and the source files to build
		add_subdirectory(host)

		file(GLOB SRC wos_thrust.cuh cpuReduce.cpp helper.cpp  main.cu export.cpp  params.cpp)
		file(GLOB LIB wos_thrust.h helper.h cpuReduce.h clock.h export.h params.h wos_wrapper.cuh wos_native.cuh)
		cuda_add_library(src ${SRC} ${LIB} ${INC})
		target_link_libraries(src ${CUDA_LIBARIES} host)
		target_include_directories (src PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
		set_property(TARGET src PROPERTY CXX_STANDARD 14)

	else()
		# CUDA not found
		message(CRITICAL_ERROR "CUDA cannot be found, please ensure CUDA dependencies are installed")

	endif()
