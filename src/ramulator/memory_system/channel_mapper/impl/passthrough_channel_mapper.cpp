#include "ramulator/base/base.h"
#include "ramulator/memory_system/channel_mapper/i_channel_mapper.h"

namespace Ramulator {

// Pass-through channel mapper for addr_vec-native frontends.
// Assumes addr_vec[0] is already set by the frontend (or defaults to 0).
// Copies req.addr to req.intra_channel_addr unchanged.
class PassThroughChannelMapper final : public IChannelMapper, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IChannelMapper, PassThroughChannelMapper, "PassThroughChannelMapper");

 public:
  void init() override {}

  void setup(int num_channels, int tx_offset) override {}

  void apply(Request& req) const override {
    if (req.addr_vec.empty()) {
      req.addr_vec.resize(1, 0);
    } else if (req.addr_vec[0] < 0) {
      req.addr_vec[0] = 0;
    }
    req.intra_channel_addr = req.addr;
  }
};

}  // namespace Ramulator
