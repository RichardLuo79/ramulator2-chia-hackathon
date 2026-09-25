# Site-specific values for esp_xcelium.yaml. Copy to esp_site_env.sh
# (gitignored; esp_cluster.sh auto-sources it) and fill in for your site.
# Keep real hostnames and install paths out of version control.
export CHIA_ESP_SIM_HOST=simhost.example.edu     # machine with the CAD tools
export CHIA_ESP_HEAD_IP=10.0.0.1                 # head machine's IP, routable from the sim host
# -v flags for the CAD tool trees. Bind concrete subtrees, not an autofs
# root (that yields empty stubs in the container).
export CHIA_ESP_CAD_MOUNT_OPTS="-v /path/to/cadence:/path/to/cadence:ro -v /path/to/xilinx:/path/to/xilinx:ro"
# PATH + license setup. Also export XILINX_VIVADO=<vivado install root> here:
# the ESP docker image's esp_env.sh stubs it to "/", which mangles the
# $(XILINX_VIVADO)/data/... paths ESP's sim flow compiles (e.g. glbl.v).
export CHIA_ESP_CAD_ENV="source /path/to/cad/env.bashrc"
export CHIA_ESP_VIVADO_BIN=/path/to/Vivado/<ver>/bin      # for the Xilinx simlib compile
# Bin dir providing vivado_hls (last shipped with Vivado 2020.1), for the
# ESP Vivado HLS accelerator flow. May be an older install than the above.
export CHIA_ESP_VIVADO_HLS_BIN=/path/to/Vivado/2020.1/bin
# Only for examples/esp_accel_loop: .claude config dir (with credentials)
# mounted into the LLM worker container.
export CHIA_ESP_CLAUDE_CONFIG=/path/to/.claude

# FPGA flow (read by the fpga e2e driver, passed to the workspace members).
# The endpoints are as seen FROM the worker: ssh -L tunnels to the machine
# holding the JTAG/UART/board-Ethernet make them all look like localhost,
# needing no holes in that machine's firewall.
export CHIA_ESP_FPGA_HOST=127.0.0.1               # hw_server host
export CHIA_ESP_HW_SERVER_PORT=13121              # hw_server port
export CHIA_ESP_UART_HOST=127.0.0.1               # TCP-exposed UART console
export CHIA_ESP_UART_PORT=14001
export CHIA_ESP_ESPLINK_IP=127.0.0.1              # SoC EDCL IP (or a relay to it)
# Vivado bin dir for the PROGRAMMING step only: hw_server accepts clients
# at or below its own version, which may rule out the synthesis Vivado.
export CHIA_ESP_VIVADO_PROG_BIN=/path/to/Vivado/<ver>/bin
# Optional Vivado bin dir for SYNTHESIS: the board scripts pin exact IP
# versions, so an era-matched install may be required (see the note in
# fpga_e2e_test.py). Unset to use the worker's default vivado.
export CHIA_ESP_VIVADO_SYN_BIN=/path/to/Vivado/<ver>/bin
# Shared ESP workspace: a host directory holding the ESP checkout, bind-
# mounted into the container at this SAME path and visible natively to the
# head's bare Vivado worker. Populate once from the image's ESP tree.
# Drivers pass it as esp_root so container (configure/build) and host
# (synth) share one tree. Vivado bakes absolute paths, so path must match.
export CHIA_ESP_SHARED_WORKSPACE=/path/to/esp_shared
# Env for the head's Vivado-synthesis worker: synth Vivado on PATH, its
# matching XILINX_VIVADO root, and the Xilinx license server. Single-quoted
# so $PATH expands on the head, not at source time.
export CHIA_ESP_SYN_ENV='export PATH=/path/to/Vivado/<ver>/bin:$PATH && export XILINX_VIVADO=/path/to/Vivado/<ver> && export XILINXD_LICENSE_FILE=<port>@<license-server>'
