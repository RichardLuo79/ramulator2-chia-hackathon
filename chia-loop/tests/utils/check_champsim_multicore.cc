// Offline identity regression for the finite-job ChampSim trace-reader patch.
// Compile against that source tree's inc/ directory; no trace payload required.
#include "tracereader.h"
#include <cassert>
#include <cstddef>
#include <cstdint>

extern const std::size_t NUM_CPUS = 4;

struct Reader {
  ooo_model_instr operator()() { return ooo_model_instr{0, input_instr{}}; }
};

int main()
{
  champsim::tracereader a{Reader{}, 0}, b{Reader{}, 1};
  assert(a().instr_id == 0);
  assert(a().instr_id == 4);
  assert(b().instr_id == 1);
  assert(a().instr_id == 8);
  assert(b().instr_id == 5);
  champsim::tracereader reordered_a{Reader{}, 0}, reordered_b{Reader{}, 1};
  assert(reordered_b().instr_id == 1);
  assert(reordered_b().instr_id == 5);
  assert(reordered_a().instr_id == 0);
  assert(reordered_a().instr_id == 4);
  assert(reordered_a().instr_id == 8);
}
