from m5.SimObject import *
from m5.params import *
from m5.objects.AbstractMemory import *

class Ramulator2(AbstractMemory):
  type = "Ramulator2"
  cxx_class = "gem5::memory::Ramulator2"
  cxx_header = "mem/ramulator2/ramulator2.hh"

  port = ResponsePort("The port for receiving memory requests and sending responses")
  ramulator_config = Param.String("Ramulator 2 configuration (set automatically by ramulator.gem5.Memory)")
  event_driven = Param.Bool(False, "Skip controller cycles certified idle by idle_ticks()")
  clock_edge_first = Param.Bool(False, "Process the current DRAM edge before admission; implied by event_driven")
