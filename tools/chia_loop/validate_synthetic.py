"""No-LLM native synthetic adapter validation; fixtures are not evolved designs.

Only generic generator axes are used. Never imports another run's candidates,
scenario files or findings. The test DSO changes a constant solely to detect
accidental use of the checkout's controller. This command cannot launch an LLM.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

from tools.chia_loop import gemini_loop as G, real_core as P, real_eval as E, recovery as R, loop_config as L
from tools.chia_loop.core import atomic_write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=pathlib.Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    with R.cpu_lease(E.REPO / "eval_out/chia/.cpu_leases", 3):
        root.mkdir(parents=True, exist_ok=False)
        L.install(root)
        status = {"record_type": "synthetic_integration_fixture", "status": "running",
            "paid_generation_calls": 0, "evolved_design": False, "started_at": time.time()}
        atomic_write_json(root / "validation.json", status)
        try:
            source = (E.REPO / P.MUTABLE).read_text()
            _, regions = P.regions(source)
            if (regions["INCLUDES"].strip() or re.sub(r"\s+", "", regions["CODE"]) !=
                    "voidinit_model(){}Clk_tpredict_departure(constRequest&req){returnm_clk+m_latency;}"):
                raise RuntimeError("validation requires the unchanged CHIA skeleton")
            E.command(["cmake", "--build", E.REPO / "build-bench", "--target", "ramulator", "--parallel", "3"],
                      root / "logs/optimized_build.log", timeout=600)
            E.prepare_runtime(root)
            plugin = E.compile_candidate(root, source, root / "seed")
            parent = {"source_path": str(root / "seed/atomic_controller.cpp"), "plugin": plugin,
                      "sha256": P.sha(source), "label": "seed"}
            # These exercise generic axes, not imported model-specific diagnoses.
            cases = [{"mlp": 1}, {"row_run": 1, "bank_spread": 8},
                {"wb_mode": 1, "wfrac_pct": 50}, {"wb_mode": 2, "wfrac_pct": 50, "wb_batch": 4},
                {"wb_mode": 3, "wfrac_pct": 50, "bank_spread": 4},
                {"wb_mode": 4, "wfrac_pct": 25, "pp_lead": 20},
                {"streams": 2, "share_region": True, "bank_spread": 8, "bank_random": True, "row_random": True,
                 "stream_params": [{"dep_frac": .5, "think_time": 3}, {"jitter": 10}]},
                {"mode_switch": 5000, "mode_b": {"mlp": 4, "row_run": 4}, "phase_on": 1000,
                 "phase_off": 100, "phase_off_random": 100}]
            outputs = []
            for index in range(0, len(cases), 4):
                outputs += G.inspect_tool(root, parent, {"tool": "synthetic_diagnostics", "cases": cases[index:index+4], "limit": 2})["results"]
                print(json.dumps({"completed_cases": len(outputs)}), flush=True)
            cached = G.inspect_tool(root, parent, {"tool": "synthetic_diagnostics", "cases": [cases[0]], "limit": 1})["results"][0]
            assert cached["cached"] and cached["case_id"] == outputs[0]["case_id"]
            fixture = source.replace("return m_clk + m_latency;", "return m_clk + m_latency + 17;")
            P.validate_source(source, fixture)
            directory = root / "candidates/plugin_identity_fixture"
            directory.mkdir(parents=True)
            (directory / "atomic_controller.cpp").write_text(fixture)
            fixture_plugin = E.compile_candidate(root, fixture, directory / "build")
            alternate = {"source_path": str(directory / "atomic_controller.cpp"), "plugin": fixture_plugin,
                         "sha256": P.sha(fixture), "label": "plugin_identity_fixture"}
            changed = G.inspect_tool(root, alternate, {"tool": "synthetic_diagnostics", "cases": [cases[0]], "limit": 0})["results"][0]
            assert not changed["cached"] and changed["case_id"] != cached["case_id"]
            assert changed["candidate"]["avg_read_latency"] == cached["candidate"]["avg_read_latency"] + 17
            archive_count = E.verify_run_archives(root)
            atomic_write_json(root / "selection_frozen.json", {"fixture_only": True})
            try:
                G.inspect_tool(root, parent, {"tool": "synthetic_diagnostics", "cases": [{}]})
            except RuntimeError:
                pass
            else:
                raise AssertionError("diagnostics incorrectly allowed after freeze")
            status.update(status="completed", generic_cases=len(outputs), plugin_identity_fixture=True,
                cached_case_verified=True, post_freeze_access_denied=True, trace_archives_verified=archive_count,
                simulation_cases=2 * (len(outputs) + 1), reads_per_stream=50000)
        except BaseException as exc:
            status.update(status="failed", error=str(exc)[-4000:])
            raise
        finally:
            status["finished_at"] = time.time()
            atomic_write_json(root / "validation.json", status)
            print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
