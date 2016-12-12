# WoS.cu
a GPU based implementation of the Walking on Spheres Algorithm for a masters thesis at the Technical University of Munich

## Author
- Walter Simson 

## Requirements

The following libraries are required

*CUDA compiler NVCC

## Setup

We use `cmake` to generate a user friendly Makefile and to support various operating system setups.

To create a Makefile with CMake run the folling in the base project directory:
```sh
    cmake . 
```
Please mind the period at the end. This process needs to be repeated only if the CMakeList.txt has been modified.

Once the Makefile is generated, you can build the project:
```sh
    make
```

## Usage:

*coming soon*

## Style Guide

For development you should install `clang`. We use `clang`'s `clang-format` to keep our coding style consistent.

We also use [google's C++ style guide](http://google.github.io/styleguide/cppguide.html). It
is a reasonable guideline, used by a large organization to great success and at the same time it is an interesting read and can maybe even teach you something about C++.

You can use the script scripts/format to automatically reformat all the source files. Alternatively, try one of the integrations of clang-format into various editors/IDEs. 
