#include <cstddef>
#include <iostream>
#include <stdio.h>

// Beautify outpt
#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_YELLOW "\x1b[33m"
#define ANSI_BLUE "\x1b[34m"
#define ANSI_MAGENTA "\x1b[35m"
#define ANSI_CYAN "\x1b[36m"
#define ANSI_RESET "\x1b[0m"

extern "C" bool isPow2(unsigned int x);

// Finds the next largest power 2 number
extern "C" size_t nextPow2(size_t x);

void printInfo(const char *info);

void printError(const char *info);

void printTitle();
