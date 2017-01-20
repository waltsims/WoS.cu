#include "helper.hpp"

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
