// Compile against a patched ChampSim include tree. No LLM or simulator run.
#include "measurement_protocol.h"
#include "repeatable.h"
#include "tracereader.h"
#include <cassert>

extern const std::size_t NUM_CPUS = 4;
struct ShortReader {
  unsigned count=0;
  explicit ShortReader(unsigned) {}
  bool eof() const { return count==3; }
  ooo_model_instr operator()() { ++count; return ooo_model_instr{0, input_instr{}}; }
};
int main()
{
  champsim::retirement_watchdog w;
  assert(!w.sample(0));
  assert(!w.sample(1)); // arbitrarily low but positive IPC is progress
  assert(!w.sample(1));
  assert(w.sample(1));
  assert(!w.sample(2));
  champsim::tracereader r{champsim::repeatable<ShortReader,unsigned>{0}, 2};
  for (unsigned i=0;i<100;++i)
    assert(r().instr_id==4*i+2);
  assert(r.repeats()==33);
}
