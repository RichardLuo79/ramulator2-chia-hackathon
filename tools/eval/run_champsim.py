"""ChampSim runner: builds a clean library, generates reference configs, and
invokes the hash-recorded bridge runner (resources/champsim_bridge/cs_gentest.py).

The bridge runner refuses dirty source trees, so the library is built in a
throwaway clean worktree at the current HEAD. The oracle and candidate use
equal 64/64 buffer capacities; oracle-only scheduling settings remain on the
oracle. --record-requests threads per-request trace recording
through cs_gentest (per-run derived configs); match the resulting
<trace>_{oracle,eval}_req.csv.ch0 pairs with tools/eval/match_champsim.py.

Usage: python tools/eval/run_champsim.py [--workers N] [--record-requests]
"""
import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from eval import config as C

BUILD_WT = pathlib.Path("/tmp/r2-eval-build")


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def ensure_clean_build():
    head = C.git_rev()
    if BUILD_WT.exists():
        at = subprocess.run(["git", "-C", str(BUILD_WT), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
        if at != head:
            sh(["git", "-C", str(C.REPO), "worktree", "remove", "--force", str(BUILD_WT)])
    if not BUILD_WT.exists():
        sh(["git", "-C", str(C.REPO), "worktree", "add", "--detach", str(BUILD_WT), head])
    sh(["cmake", "-B", str(BUILD_WT / "build"), "-S", str(BUILD_WT),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG"], stdout=subprocess.DEVNULL)
    sh(["cmake", "--build", str(BUILD_WT / "build"), "-j", "12"],
       stdout=subprocess.DEVNULL)
    # the build's codegen step reformats tracked generated sources; revert
    # the churn so the provenance guard sees a clean tree
    sh(["git", "-C", str(BUILD_WT), "checkout", "--", "."])
    lib = BUILD_WT / "libramulator.so"
    assert lib.exists(), lib
    return BUILD_WT


def gen_configs(cfgdir):
    cfgdir.mkdir(parents=True, exist_ok=True)
    sh([sys.executable, str(C.REPO / "resources/champsim_bridge/gen_configs.py"),
        str(cfgdir), "--standards", "DDR5", "--force"])
    for name in ("oracle_DDR5.json", "candidate_DDR5.json"):
        p = cfgdir / name
        d = json.load(open(p))
        config = C.REFERENCE if name.startswith("oracle_") else C.CANDIDATE_RESOURCES
        d["memory_system"]["controllers"][0].update(config)
        json.dump(d, open(p, "w"), indent=1)
        print(f"patched {name} with reference config")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--candidate-kw", default="",
                    help="JSON overrides patched into the candidate config")
    ap.add_argument(
        "--outdir-suffix", default="",
        help="suffix for the output dir on override runs")
    ap.add_argument("--record-requests", action="store_true")
    a = ap.parse_args()
    if not 1 <= a.workers <= 12:
        ap.error("--workers must be in [1, 12]")
    wt = ensure_clean_build()
    cfgdir = C.OUT / "champsim" / ("configs" + a.outdir_suffix)
    gen_configs(cfgdir)
    if a.candidate_kw:
        p = cfgdir / "candidate_DDR5.json"
        d = json.load(open(p))
        d["memory_system"]["controllers"][0].update(json.loads(a.candidate_kw))
        json.dump(d, open(p, "w"), indent=1)
        print("patched candidate config with overrides:", a.candidate_kw)
    outdir = C.OUT / "champsim" / ("DDR5" + a.outdir_suffix)
    sh([sys.executable, str(C.REPO / "resources/champsim_bridge/cs_gentest.py"),
        "--champsim", str(C.CHAMPSIM_BIN),
        "--trace-dir", str(C.CHAMPSIM_TRACES),
        "--oracle-config", str(cfgdir / "oracle_DDR5.json"),
        "--candidate-config", str(cfgdir / "candidate_DDR5.json"),
        "--oracle-library-dir", str(wt),
        "--candidate", f"eval={wt}",
        "--output-dir", str(outdir),
        "--ticks-per-8", "12",
        "--workers", str(a.workers)]
       + (["--record-requests"] if a.record_requests else []))
    print("CHAMPSIM_EVAL_DONE")


if __name__ == "__main__":
    main()
