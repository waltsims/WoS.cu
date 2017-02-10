#include "helper.hpp"

#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

extern "C" bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

// Finds the next largest power 2 number
extern "C" size_t nextPow2(size_t x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

void printInfo(const char *info) {

  printf(ANSI_YELLOW "INFO:  " ANSI_RESET);
  printf(info, "\n");
  std::cout << std::endl;
}

void printError(const char *info) {

  printf(ANSI_RED "ERROR:  " ANSI_RESET);
  printf(info, "\n");
  std::cout << std::endl;
}

void printTitle() {
  // More Beautification
  // clang-format off
std::cout << ANSI_MAGENTA "     __      __      "                   ANSI_MAGENTA "_________                   "                                                          ANSI_RESET
          << std::endl;
std::cout << ANSI_MAGENTA "    /  \\    /  \\" ANSI_CYAN "____"     ANSI_MAGENTA "/   _____/  "                     ANSI_CYAN  "    ____  "   ANSI_MAGENTA "__ __   "    ANSI_RESET
          << std::endl;
std::cout << ANSI_MAGENTA "    \\   \\/\\/   " ANSI_CYAN "/  _ \\"  ANSI_MAGENTA "_____   \\  "                     ANSI_CYAN "  _/ ___\\"    ANSI_MAGENTA "|  |  \\   " ANSI_RESET
          << std::endl;
std::cout << ANSI_MAGENTA "     \\        "    ANSI_CYAN "( 〈_〉)" ANSI_MAGENTA "        \\  "                     ANSI_CYAN   " \\  \\___"  ANSI_MAGENTA "|  |  /   "  ANSI_RESET
          << std::endl;
std::cout << ANSI_MAGENTA "      \\__/\\  / "  ANSI_CYAN "\\____/"  ANSI_MAGENTA "_______  / " ANSI_BLUE  "/\\ "    ANSI_CYAN    "\\___  〉"   ANSI_MAGENTA  "___/   "   ANSI_RESET
          << std::endl;
std::cout << ANSI_MAGENTA  "           \\/               \\/ "                                 ANSI_BLUE " \\/   "  ANSI_CYAN       "  \\/         "                     ANSI_RESET
          << std::endl;
  // clang-format on
}

void printTiming(float prep, float computation, float total, float finish) {
  // Time output
  printf("TIMING: \n");

  printf(" ----------------------------------------------------------------"
         "----------------------------------------\n");
  printf("|%-25s|%-25s|%-25s|%-25s|\n", "memory init time[sec]",
         "GPU computation time[sec]", "total exicution time[sec]",
         "memory finish time[sec]");
  printf(" ----------------------------------------------------------------"
         "----------------------------------------\n");
  printf("|%-25f|%-25f|%-25f|%-25f|\n ", prep, computation, total, finish);
  printf(" ----------------------------------------------------------------"
         "---------------------------------------\n");
}
void testResults(float gpu_result, Parameters &p) {
  // TODO: dynamic table output function
  // this is a hot mess

  // Basic testing

  printf("\nSIMULATION SUMMARY: \n\n");
  printf("VALUES: \n");

  printf(" ----------------------------------------------------------------"
         "------------------------------\n");
  printf("|%-15s|%-15s|%-15s|%-15s|%-15s|%-15s|\n", "totalPaths",
         "desired value", "resulting value", "epsilon", "delta", "status");
  printf(" ----------------------------------------------------------------"
         "------------------------------\n");

  float EPS = 0.00001;
  float x0 = (float)p.wos.x0.value;

  if (fabs(x0 - 0.0) < EPS) {
    float desired = /*0.042535;*/ 0.041561;
    // float desired = 0.0;
    // desired = (float)(d_eps != 0.01) * 0.039760;
    // desired = (float)(d_eps == 0.01) * 0.042535;
    // Julia value [0.0415682]
    if (fabs(gpu_result - desired) < EPS) {
      printf("|%-15d|%-15f|%-15f|%-15f|%-15f|", p.wos.totalPaths, desired,
             gpu_result, EPS, fabs(gpu_result - desired));
      printf(ANSI_GREEN "%-14s" ANSI_RESET, "TEST PASSED!");
      printf("|\n");
    } else {
      printf("|%-15d|%-15f|%-15f|%-15f|%-15f|", p.wos.totalPaths, desired,
             gpu_result, EPS, fabs(gpu_result - desired));
      printf(ANSI_RED "%-14s" ANSI_RESET, "TEST FAILED!");
      printf("|\n");
    }
  } else if (fabs(x0 - 1.0) < EPS) {
    float desired = 0.5;
    if (fabs(gpu_result - desired) < EPS) {
      printf("|%-15d|%-15f|%-15f|%-15f|%-15f|", p.wos.totalPaths, desired,
             gpu_result, EPS, fabs(gpu_result - desired));
      printf(ANSI_GREEN "%-14s" ANSI_RESET, "TEST PASSED!");
      printf("|\n");
    } else {
      printf("|%-15d|%-15f|%-15f|%-15f|%-15f|", p.wos.totalPaths, desired,
             gpu_result, EPS, fabs(gpu_result - desired));
      printf(ANSI_RED "%-14s" ANSI_RESET, "TEST FAILED!");
      printf("|\n");
    }
  }
  printf(" ----------------------------------------------------------------"
         "------------------------------\n");
}
