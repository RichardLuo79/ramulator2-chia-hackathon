// Trusted input-format probe: no model code, trace data, or simulation.
#include <cstddef>
#include <cstdio>
#include "trace_instruction.h"

int main() {
  std::printf("%zu\n", sizeof(input_instr));
}
