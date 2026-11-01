#include <unordered_map>

#include "ramulator/base/base.h"
#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/translation/i_translation.h"

namespace Ramulator {

// First-touch page allocation with a deterministically shuffled physical
// order — models ChampSim-style vmem randomization, which destroys the row
// contiguity that identity mapping (NoTranslation) preserves. Collision-free
// by construction: physical frames are a bit-mixed permutation of an
// allocation counter.
class RandomTranslation : public ITranslation, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(ITranslation, RandomTranslation, "RandomTranslation");

 private:
  Addr_t m_max_paddr;
  Addr_t m_page_size;
  Addr_t m_num_frames;
  Addr_t m_next = 0;
  std::unordered_map<Addr_t, Addr_t> m_table;

  Addr_t permute(Addr_t i) const {
    // Feistel-style mix over the frame index space (power-of-two frames).
    Addr_t x = i;
    x ^= x >> 7;
    x *= 0x9E3779B97F4A7C15ull;
    x ^= x >> 11;
    return x & (m_num_frames - 1);
  }

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_max_paddr, Addr_t, "max_addr").required();
    RAMULATOR_PARSE_PARAM(m_page_size, Addr_t, "page_size").default_val(4096);
    m_num_frames = m_max_paddr / m_page_size;
    // round down to a power of two so permute() is a bijection
    while (m_num_frames & (m_num_frames - 1)) {
      m_num_frames &= m_num_frames - 1;
    }
  };

  bool translate(Request& req) override {
    Addr_t vpage = static_cast<Addr_t>(static_cast<uint64_t>(req.addr) / m_page_size);
    Addr_t offset = static_cast<Addr_t>(static_cast<uint64_t>(req.addr) % m_page_size);
    auto it = m_table.find(vpage);
    if (it == m_table.end()) {
      // linear-probe the permutation for an unused frame
      Addr_t frame = permute(m_next++);
      it = m_table.emplace(vpage, frame).first;
    }
    req.addr = it->second * m_page_size + offset;
    return true;
  }
};

}  // namespace Ramulator
