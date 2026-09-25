"""Local unit tests for the ESP nodes — no cluster, no ESP image.

A ``@ChiaFunction`` called directly (not via ``chia_remote``) runs in the
caller's process, and ``require_colocated=False`` skips placement-group
reservation, so everything here runs under plain pytest against a stub
``socs/<board>/Makefile`` and accgen script that mimic ESP's contracts
(esp-config consumes ``.esp_config``, soft/linux populate
``soft-build/<cpu>/``, accgen and the per-accelerator/sim targets leave
inspectable outputs).

Run:
    pytest chia/esp/test/test_esp_local.py
"""

import os

import pytest

from chia.esp.esp_workspace import EspWorkspaceNode, board_dir, with_acc_tile
from chia.esp.state_def import EspAccelSpec

BOARD = "testboard"

# Stub Makefile mimicking the ESP targets Phase 1 drives. Recipe lines are
# tab-indented (make requires it).
STUB_MAKEFILE = "\n".join([
    "soft:",
    "\tmkdir -p soft-build/ariane",
    "\tprintf 'PROM' > soft-build/ariane/prom.bin",
    "\tprintf 'SYSTEST' > soft-build/ariane/systest.bin",
    "\techo 'SMP=$(SMP)' > smp.txt",
    "",
    "linux:",
    "\tmkdir -p soft-build/ariane",
    "\thead -c 2048 /dev/zero > soft-build/ariane/linux.bin",
    "",
    "esp-config:",
    "\ttest -f socgen/esp/.esp_config",
    "\tcp socgen/esp/.esp_config socgen.marker",
    "",
    "xmsim:",
    "\ttest -f xcelium/xmsim.in",
    "\tcp xcelium/xmsim.in xmsim_ran.txt",
    "\techo 'TEST_PROGRAM=$(TEST_PROGRAM)' >> xmsim_ran.txt",
    "\t@echo '*** SIM PASSED ***'",
    "",
    "xmsim-distclean:",
    "\trm -rf xcelium",
    "\ttouch distcleaned.txt",
    "",
    "chiatest_rtl-hls:",
    "\ttouch hls_ran.txt",
    "",
    "chiatest_vivado-hls:",
    "\ttouch vivado_hls_ran.txt",
    "",
    "chiatest_stratus-hls:",
    "\ttouch stratus_hls_ran.txt",
    "",
    "chiatest_sysc_catapult-hls:",
    "\ttouch catapult_hls_ran.txt",
    "",
    "chiatest_rtl-baremetal:",
    "\tmkdir -p soft-build/ariane/baremetal",
    "\tprintf 'EXE' > soft-build/ariane/baremetal/chiatest.exe",
    "",
    "fail:",
    "\texit 1",
    "",
    "sleepy:",
    "\tsleep 30",
    "",
    "vivado-syn:",
    "\t@read ans && echo $$ans > overwrite_answer.txt || true",
    "\techo \"$$PATH\" > syn_path.txt",
    "\tmkdir -p vivado/esp-testboard.runs/impl_1",
    "\tprintf 'BIT' > vivado/esp-testboard.runs/impl_1/top.bit",
    "\tprintf 'TIMING OK' > vivado/esp-testboard.runs/impl_1/top_timing_summary_routed.rpt",
    "\tln -sf vivado/esp-testboard.runs/impl_1/top.bit top.bit",
    "",
    "fpga-program:",
    "\techo 'HOST=$(FPGA_HOST) PORT=$(XIL_HW_SERVER_PORT)' > programmed.txt",
    "\techo \"$$PATH\" > prog_path.txt",
    "",
    "fpga-run:",
    "\techo 'ESPLINK_IP=$(ESPLINK_IP)' > fpga_ran.txt",
    "",
    "fpga-run-linux:",
    "\ttouch fpga_ran_linux.txt",
    "",
    # Stub esplink build: drops a fake esplink that logs its args (into the
    # board dir, esplink's cwd), so the direct dram_image load path is
    # testable.
    "esplink:",
    "\tmkdir -p socgen/esp soft-build/ariane",
    "\tprintf 'PROM' > soft-build/ariane/prom.bin",
    "\tprintf '#!/bin/sh\\necho \"$$*\" >> esplink_calls.txt\\n' > socgen/esp/esplink",
    "\tchmod +x socgen/esp/esplink",
    "",
])

# A second board whose `soft` forgets systest.bin and whose `vivado-syn`
# forgets the bitstream (missing-output paths).
STUB_MAKEFILE_INCOMPLETE = "\n".join([
    "soft:",
    "\tmkdir -p soft-build/ariane",
    "\tprintf 'PROM' > soft-build/ariane/prom.bin",
    "",
    "vivado-syn:",
    "\ttrue",
    "",
])


@pytest.fixture
def esp_root(tmp_path):
    """A fake ESP checkout with two stub boards."""
    for board, makefile in ((BOARD, STUB_MAKEFILE),
                            ("incompleteboard", STUB_MAKEFILE_INCOMPLETE)):
        bdir = tmp_path / "socs" / board
        bdir.mkdir(parents=True)
        (bdir / "Makefile").write_text(makefile)
    return str(tmp_path)


@pytest.fixture
def ws():
    """An unpinned workspace node: no ray, members run in-process."""
    return EspWorkspaceNode(require_colocated=False)


# ---------------------------------------------------------------------------
# EspWorkspaceNode
# ---------------------------------------------------------------------------

def test_make_success_and_listing(esp_root, ws):
    bdir = board_dir(esp_root, BOARD)
    res = ws.make(bdir, "soft", list_dir=os.path.join(bdir, "soft-build"))
    assert res.success and res.returncode == 0
    assert res.target == "soft" and res.work_dir == bdir
    assert res.listing == {"ariane/prom.bin": 4, "ariane/systest.bin": 7}


def test_make_no_listing_by_default(esp_root, ws):
    res = ws.make(board_dir(esp_root, BOARD), "soft")
    assert res.success and res.listing == {}


def test_make_vars_reach_the_makefile(esp_root, ws):
    bdir = board_dir(esp_root, BOARD)
    assert ws.make(bdir, "soft", make_vars={"SMP": "1"}).success
    assert open(os.path.join(bdir, "smp.txt")).read().strip() == "SMP=1"


def test_make_failure(esp_root, ws):
    res = ws.make(board_dir(esp_root, BOARD), "fail")
    assert not res.success and res.returncode != 0


def test_make_timeout(esp_root, ws):
    res = ws.make(board_dir(esp_root, BOARD), "sleepy", timeout_seconds=1)
    assert not res.success and res.returncode == -1
    assert "timed out" in res.stderr


def test_put_file_bytes_str_and_parents(esp_root, ws):
    bdir = board_dir(esp_root, BOARD)
    p1 = ws.put_file(bdir, ".esp_config", "CPU ariane\n")
    assert open(p1).read() == "CPU ariane\n"
    p2 = ws.put_file(bdir, "hw/src/kernel.cpp", b"// bytes")
    assert p2 == os.path.join(bdir, "hw/src/kernel.cpp")
    assert open(p2, "rb").read() == b"// bytes"


def test_put_file_refuses_escape(esp_root, ws):
    with pytest.raises(ValueError, match="escapes"):
        ws.put_file(board_dir(esp_root, BOARD), "../../evil.txt", "x")


def test_remove_file_and_tree(esp_root, ws):
    bdir = board_dir(esp_root, BOARD)
    ws.put_file(bdir, "hls-work/proj/out.v", "module m; endmodule")
    assert ws.remove(bdir, "hls-work") is True
    assert not os.path.exists(os.path.join(bdir, "hls-work"))
    assert ws.remove(bdir, "hls-work") is False       # already gone
    ws.put_file(bdir, "one.txt", "x")
    assert ws.remove(bdir, "one.txt") is True


def test_remove_refuses_escape(esp_root, ws):
    with pytest.raises(ValueError, match="escapes"):
        ws.remove(board_dir(esp_root, BOARD), "../../..")


def test_collect_globs_dedup_and_cap(esp_root, ws):
    bdir = board_dir(esp_root, BOARD)
    assert ws.make(bdir, "soft").success
    # Overlapping patterns dedup; the 1-byte... use 5-byte cap: prom.bin (4)
    # ships, systest.bin (7) is size-capped into `skipped`.
    col = ws.collect(bdir, ["soft-build/**/*.bin", "soft-build/ariane/prom.bin"],
                     max_bytes_per_file=5)
    assert col.files == {"soft-build/ariane/prom.bin": "PROM"}
    assert col.skipped == {"soft-build/ariane/systest.bin": 7}
    assert "soft-build/ariane/systest.bin" in col.listing


# ---------------------------------------------------------------------------
# configure
# ---------------------------------------------------------------------------

def test_configure_from_text(esp_root, ws):
    res = ws.configure(esp_root, BOARD, esp_config="CPU ariane\nNOC 2x2\n")
    assert res.success
    assert res.esp_config == "CPU ariane\nNOC 2x2\n"
    marker = os.path.join(res.board_dir, "socgen.marker")
    assert open(marker).read() == res.esp_config


def test_configure_from_path(esp_root, ws, tmp_path):
    saved = tmp_path / "saved_esp_config"
    saved.write_text("CPU leon3\n")
    res = ws.configure(esp_root, BOARD, esp_config_path=str(saved))
    assert res.success and res.esp_config == "CPU leon3\n"


def test_configure_requires_exactly_one_source(esp_root, ws):
    with pytest.raises(ValueError, match="exactly one"):
        ws.configure(esp_root, BOARD)
    with pytest.raises(ValueError, match="exactly one"):
        ws.configure(esp_root, BOARD, esp_config="x", esp_config_path="/y")


def test_configure_unknown_board(esp_root, ws):
    with pytest.raises(FileNotFoundError, match="board dir"):
        ws.configure(esp_root, "no-such-board", esp_config="x")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def test_build_soft_inlines_binaries(esp_root, ws):
    art = ws.build(esp_root, BOARD, cpu="ariane")
    assert art.success and art.target == "soft" and art.missing == []
    assert art.binaries == {"prom.bin": b"PROM", "systest.bin": b"SYSTEST"}
    assert art.kept == {}
    assert art.soft_build_dir.endswith("soft-build/ariane")


def test_build_linux_keeps_big_outputs(esp_root, ws):
    art = ws.build(esp_root, BOARD, cpu="ariane", target="linux",
                   inline_max_bytes=100)
    assert art.success
    assert art.binaries == {} and art.kept == {"linux.bin": 2048}
    assert os.path.getsize(os.path.join(art.soft_build_dir, "linux.bin")) == 2048


def test_build_smp_passes_make_var(esp_root, ws):
    art = ws.build(esp_root, BOARD, cpu="ariane", smp=True)
    assert art.success
    smp_txt = os.path.join(board_dir(esp_root, BOARD), "smp.txt")
    assert open(smp_txt).read().strip() == "SMP=1"


def test_build_missing_output_fails(esp_root, ws):
    art = ws.build(esp_root, "incompleteboard", cpu="ariane")
    assert not art.success and art.returncode == 0
    assert art.missing == ["systest.bin"]
    assert "missing" in art.stderr


def test_build_wrong_cpu_falls_back_to_only_subdir(esp_root, ws):
    art = ws.build(esp_root, BOARD, cpu="leon3")
    assert art.success and art.cpu == "ariane"
    assert art.binaries["prom.bin"] == b"PROM"


def test_build_rejects_unknown_target(esp_root, ws):
    with pytest.raises(ValueError, match="target"):
        ws.build(esp_root, BOARD, target="bitstream")


# ---------------------------------------------------------------------------
# sim
# ---------------------------------------------------------------------------

def test_sim_writes_batch_input_and_runs(esp_root, ws):
    res = ws.sim(esp_root, BOARD)
    assert res.success and res.pass_matched is None
    ran = open(os.path.join(board_dir(esp_root, BOARD), "xmsim_ran.txt")).read()
    assert "run" in ran and "exit" in ran          # batch script reached the sim
    assert "TEST_PROGRAM=\n" in ran                # no TEST_PROGRAM by default


def test_sim_custom_input_and_test_program(esp_root, ws):
    res = ws.sim(esp_root, BOARD, sim_input="run 10 ms\nexit\n",
                 test_program="./soft-build/ariane/baremetal/fft.exe")
    assert res.success and res.test_program.endswith("fft.exe")
    ran = open(os.path.join(board_dir(esp_root, BOARD), "xmsim_ran.txt")).read()
    assert ran.startswith("run 10 ms\nexit\n")
    assert "TEST_PROGRAM=./soft-build/ariane/baremetal/fft.exe" in ran


def test_sim_pass_pattern(esp_root, ws):
    assert ws.sim(esp_root, BOARD, pass_pattern=r"\*\*\* SIM PASSED").pass_matched
    res = ws.sim(esp_root, BOARD, pass_pattern="TEST FAILED MARKER")
    assert not res.success and res.pass_matched is False and res.returncode == 0


def test_sim_clean_runs_distclean_before_writing_input(esp_root, ws):
    res = ws.sim(esp_root, BOARD, clean=True)
    assert res.success
    bdir = board_dir(esp_root, BOARD)
    assert os.path.exists(os.path.join(bdir, "distcleaned.txt"))
    # The input script was written after the clean, so the sim saw it.
    assert os.path.exists(os.path.join(bdir, "xmsim_ran.txt"))


def test_sim_unknown_board(esp_root, ws):
    with pytest.raises(FileNotFoundError, match="board dir"):
        ws.sim(esp_root, "no-such-board")


def test_sim_seeds_xcelium_cds_defaults(esp_root, ws, tmp_path):
    # Fake Xcelium install: <root>/tools/bin/64bit/xmvhdl + default cds.lib.
    xc = tmp_path / "xc_install"
    (xc / "tools/bin/64bit").mkdir(parents=True)
    xmvhdl = xc / "tools/bin/64bit/xmvhdl"
    xmvhdl.write_text("#!/bin/sh\n")
    xmvhdl.chmod(0o755)
    default_cds = xc / "tools/xcelium/files/cds.lib"
    default_cds.parent.mkdir(parents=True)
    default_cds.write_text("DEFINE std ...\n")

    env = {"PATH": f"{xc}/tools/bin/64bit:/usr/bin:/bin"}
    assert ws.sim(esp_root, BOARD, env=env).success
    cache_cds = os.path.join(esp_root, ".cache/xcelium/cds.lib")
    line = f"softinclude {default_cds}"
    assert open(cache_cds).read().startswith(line)

    # Idempotent, and preserves content compile_simlib appends later.
    with open(cache_cds, "a") as f:
        f.write("DEFINE unisim /x/unisim\n")
    assert ws.sim(esp_root, BOARD, env=env).success
    content = open(cache_cds).read()
    assert content.count(line) == 1 and "DEFINE unisim" in content


def test_sim_skips_seeding_without_xmvhdl(esp_root, ws):
    assert ws.sim(esp_root, BOARD, env={"PATH": "/usr/bin:/bin"}).success
    assert not os.path.exists(os.path.join(esp_root, ".cache/xcelium/cds.lib"))


# ---------------------------------------------------------------------------
# accgen / accel / with_acc_tile
# ---------------------------------------------------------------------------

# Mimics accgen.sh's contract: consumes prompt answers from stdin, creates
# the skeleton dir, and echoes what it read (so tests can assert the piping).
STUB_ACCGEN = "\n".join([
    "#!/bin/sh",
    "read name; read flow; read path; read id",
    'case "$flow" in',
    '    V) dir="accelerators/vivado_hls/${name}_vivado";;',
    '    S) dir="accelerators/stratus_hls/${name}_stratus";;',
    '    C) dir="accelerators/catapult_hls/${name}_sysc_catapult";;',
    '    *) dir="accelerators/rtl/${name}_rtl";;',
    "esac",
    '[ -d "$dir" ] && echo "exists" && exit 2',
    'mkdir -p "$dir/hw"',
    'echo "name=$name flow=$flow id=$id" > "$dir/hw/args.txt"',
    'echo "generated $name"',
])


@pytest.fixture
def accgen_root(esp_root):
    script = os.path.join(esp_root, "tools/accgen/accgen.sh")
    os.makedirs(os.path.dirname(script))
    with open(script, "w") as f:
        f.write(STUB_ACCGEN)
    os.chmod(script, 0o755)
    return esp_root


def test_spec_to_answers_order():
    spec = EspAccelSpec(name="chiatest", device_id="04A")
    answers = spec.to_answers().split("\n")
    assert answers[:4] == ["chiatest", "R", "", "04A"]
    assert answers[4:20] == [""] * 16


# (flow, accgen answer letter, skeleton dir, stub hls marker file). The
# skeleton basename doubles as make_name; catapult's carries accgen's
# hardcoded SystemC language tag.
FLOW_CASES = [
    ("vivado", "V", "accelerators/vivado_hls/chiatest_vivado",
     "vivado_hls_ran.txt"),
    ("stratus", "S", "accelerators/stratus_hls/chiatest_stratus",
     "stratus_hls_ran.txt"),
    ("catapult", "C", "accelerators/catapult_hls/chiatest_sysc_catapult",
     "catapult_hls_ran.txt"),
]


@pytest.mark.parametrize("flow,letter,acc_dir,_marker", FLOW_CASES)
def test_spec_flow_derivations(flow, letter, acc_dir, _marker):
    spec = EspAccelSpec(name="chiatest", flow=flow)
    assert spec.acc_dir == acc_dir
    assert spec.make_name == acc_dir.rsplit("/", 1)[1]
    assert spec.to_answers().split("\n")[1] == letter


def test_spec_rejects_unknown_flow():
    with pytest.raises(ValueError, match="flow"):
        EspAccelSpec(name="x", flow="hls4ml").to_answers()


def test_accgen_runs_and_lists(accgen_root, ws):
    res = ws.accgen(accgen_root, EspAccelSpec(name="chiatest", device_id="04A"))
    assert res.success and res.acc_dir.endswith("accelerators/rtl/chiatest_rtl")
    assert res.listing == {"hw/args.txt": len("name=chiatest flow=R id=04A\n")}
    args = open(os.path.join(res.acc_dir, "hw/args.txt")).read()
    assert args == "name=chiatest flow=R id=04A\n"


@pytest.mark.parametrize("flow,letter,acc_dir,_marker", FLOW_CASES)
def test_accgen_flow_dirs(accgen_root, ws, flow, letter, acc_dir, _marker):
    res = ws.accgen(accgen_root, EspAccelSpec(name="chiatest", flow=flow))
    assert res.success and res.acc_dir.endswith(acc_dir)
    args = open(os.path.join(res.acc_dir, "hw/args.txt")).read()
    assert args == f"name=chiatest flow={letter} id=\n"


def test_accgen_overwrite(accgen_root, ws):
    spec = EspAccelSpec(name="chiatest")
    assert ws.accgen(accgen_root, spec).success
    assert not ws.accgen(accgen_root, spec).success          # exists -> refuses
    assert ws.accgen(accgen_root, spec, overwrite=True).success


def test_accgen_failure(accgen_root, ws):
    os.chmod(os.path.join(accgen_root, "tools/accgen/accgen.sh"), 0o644)
    res = ws.accgen(accgen_root, EspAccelSpec(name="nope"))
    assert not res.success and res.listing == {}


def test_accel_targets(esp_root, ws):
    make_name = EspAccelSpec(name="chiatest").make_name
    assert make_name == "chiatest_rtl"
    hls = ws.accel(esp_root, BOARD, make_name, "hls")
    assert hls.success and hls.name == make_name and hls.action == "hls"
    assert os.path.exists(os.path.join(board_dir(esp_root, BOARD), "hls_ran.txt"))
    bm = ws.accel(esp_root, BOARD, make_name, "baremetal")
    assert bm.success
    exe = os.path.join(board_dir(esp_root, BOARD),
                       "soft-build/ariane/baremetal/chiatest.exe")
    assert open(exe).read() == "EXE"
    assert not ws.accel(esp_root, BOARD, make_name, "nosuch").success


@pytest.mark.parametrize("flow,_letter,_acc_dir,marker", FLOW_CASES)
def test_accel_flow_targets(esp_root, ws, flow, _letter, _acc_dir, marker):
    make_name = EspAccelSpec(name="chiatest", flow=flow).make_name
    assert ws.accel(esp_root, BOARD, make_name, "hls").success
    assert os.path.exists(os.path.join(board_dir(esp_root, BOARD), marker))


SAMPLE_CONFIG = "\n".join([
    "CPU_ARCH = ariane",
    "TILE_0_0 = 0 mem mem 0 0 0",
    "TILE_1_0 = 2 empty empty 0 0 0",
    "POWER_0_0 = mem 0 0 0 0 0 0 0 0 0 0 0 0 ",
    "POWER_1_0 = empty 0 0 0 0 0 0 0 0 0 0 0 0 ",
])


def test_with_acc_tile_rewrites_tile_and_power():
    out = with_acc_tile(SAMPLE_CONFIG, "chiatest_rtl", 1, 0)
    assert "TILE_1_0 = 2 acc CHIATEST_RTL 0 0 0 basic_dma64 0 sld\n" in out + "\n"
    assert "POWER_1_0 = CHIATEST_RTL 0 0 0 0 0 0 0 0 0 0 0 0 " in out
    # Untouched lines survive verbatim.
    assert "TILE_0_0 = 0 mem mem 0 0 0" in out
    assert "POWER_0_0 = mem 0" in out


def test_with_acc_tile_hls_impl_point():
    out = with_acc_tile(SAMPLE_CONFIG, "chiatest_vivado", 1, 0, impl="dma64_w32")
    assert "TILE_1_0 = 2 acc CHIATEST_VIVADO 0 0 0 dma64_w32 0 sld\n" in out + "\n"


def test_with_acc_tile_missing_tile_raises():
    with pytest.raises(ValueError, match="TILE_3_3"):
        with_acc_tile(SAMPLE_CONFIG, "x", 3, 3)


# ---------------------------------------------------------------------------
# synth / fpga_program / fpga_run
# ---------------------------------------------------------------------------

def test_synth_bitstream_and_reports(esp_root, ws):
    res = ws.synth(esp_root, BOARD)
    assert res.success and res.bitstream.endswith("top.bit")
    assert os.path.exists(res.bitstream)
    assert res.reports == {
        "vivado/esp-testboard.runs/impl_1/top_timing_summary_routed.rpt":
            "TIMING OK"}
    # ESP's setup recipe prompts to overwrite an existing project; the
    # member answers headlessly: reuse by default, "y" on request.
    answer = os.path.join(board_dir(esp_root, BOARD), "overwrite_answer.txt")
    assert open(answer).read().strip() == "n"
    assert ws.synth(esp_root, BOARD, overwrite_project=True,
                    vivado_bin="/fake/syn/bin").success
    assert open(answer).read().strip() == "y"
    syn_path = os.path.join(board_dir(esp_root, BOARD), "syn_path.txt")
    assert open(syn_path).read().startswith("/fake/syn/bin")


def test_synth_missing_bitstream_fails(esp_root, ws):
    res = ws.synth(esp_root, "incompleteboard")
    assert not res.success and res.returncode == 0 and res.bitstream is None
    assert "not produced" in res.stderr


def test_fpga_program_vars_and_vivado_bin(esp_root, ws):
    bdir = board_dir(esp_root, BOARD)
    res = ws.fpga_program(esp_root, BOARD, fpga_host="127.0.0.1",
                          hw_server_port=13121, vivado_bin="/fake/vivado/bin")
    assert res.success and res.target == "fpga-program"
    assert open(os.path.join(bdir, "programmed.txt")).read().strip() == \
        "HOST=127.0.0.1 PORT=13121"
    assert open(os.path.join(bdir, "prog_path.txt")).read().startswith(
        "/fake/vivado/bin")


@pytest.fixture
def fake_uart():
    """A TCP server that plays a boot transcript to whoever connects."""
    import socket as _socket
    import threading as _threading
    srv = _socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def _serve():
        conn, _ = srv.accept()
        conn.sendall(b"ESP-Ariane boot loader\nHello from ESP!\n")
        # Hold the socket open: a real console does not EOF after printing.
        _threading.Event().wait(10)
        conn.close()

    t = _threading.Thread(target=_serve, daemon=True)
    t.start()
    yield port
    srv.close()


def test_fpga_run_pass_pattern(esp_root, ws, fake_uart):
    res = ws.fpga_run(esp_root, BOARD, uart_host="127.0.0.1",
                      uart_port=fake_uart, esplink_ip="192.168.1.99",
                      pass_pattern="Hello from ESP")
    assert res.success and res.pass_matched
    assert "boot loader" in res.uart
    ran = open(os.path.join(board_dir(esp_root, BOARD), "fpga_ran.txt")).read()
    assert ran.strip() == "ESPLINK_IP=192.168.1.99"


def test_fpga_run_uart_timeout(esp_root, ws, fake_uart):
    res = ws.fpga_run(esp_root, BOARD, uart_host="127.0.0.1",
                      uart_port=fake_uart, esplink_ip="192.168.1.99",
                      pass_pattern="NEVER PRINTED", uart_timeout_seconds=1)
    assert not res.success and res.pass_matched is False
    assert res.returncode == 0                     # the load itself worked
    assert "Hello from ESP!" in res.uart           # transcript still captured


def test_fpga_run_unreachable_uart(esp_root, ws):
    res = ws.fpga_run(esp_root, BOARD, uart_host="127.0.0.1", uart_port=1,
                      esplink_ip="192.168.1.99", pass_pattern="x")
    assert not res.success and "could not connect to UART" in res.stderr


def test_fpga_run_linux_target(esp_root, ws, fake_uart):
    res = ws.fpga_run(esp_root, BOARD, uart_host="127.0.0.1",
                      uart_port=fake_uart, esplink_ip="192.168.1.99",
                      linux=True)
    assert res.success and res.pass_matched is None
    assert os.path.exists(os.path.join(board_dir(esp_root, BOARD),
                                       "fpga_ran_linux.txt"))


def test_fpga_run_custom_dram_image(esp_root, ws, fake_uart):
    # The direct path builds esplink and drives reset/PROM/DRAM/reset with a
    # caller-supplied image (e.g. an accelerator self-test), not systest.
    bdir = board_dir(esp_root, BOARD)
    res = ws.fpga_run(esp_root, BOARD, uart_host="127.0.0.1",
                      uart_port=fake_uart, esplink_ip="192.168.1.99",
                      dram_image="/w/soft-build/ariane/baremetal/chiatest.bin",
                      pass_pattern="Hello from ESP")
    assert res.success and res.pass_matched
    calls = open(os.path.join(bdir, "esplink_calls.txt")).read()
    assert "--reset" in calls
    assert "--brom -i" in calls and "prom.bin" in calls
    assert "--dram -i /w/soft-build/ariane/baremetal/chiatest.bin" in calls
    # systest's make target was NOT used.
    assert not os.path.exists(os.path.join(bdir, "fpga_ran.txt"))
