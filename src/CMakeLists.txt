
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

		file(GLOB SRC gpu_config.cu wos_thrust.cuh cpu_reduce.cpp helper.cpp  main.cu data_logger.cpp  parameters.cpp)
		file(GLOB LIB gpu_config.h wos_thrust.h helper.h cpu_reduce.h clock.h data_logger.h parameters.h wos_wrapper.cuh wos_native.cuh)
		cuda_add_library(src ${SRC} ${LIB} ${INC})
		target_link_libraries(src ${CUDA_LIBARIES} host)
		target_include_directories (src PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
		set_property(TARGET src PROPERTY CXX_STANDARD 14)

	else()
		# CUDA not found
		message(CRITICAL_ERROR "CUDA cannot be found, please ensure CUDA dependencies are installed")

	endif()
