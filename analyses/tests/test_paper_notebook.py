"""Paper-only tests; no inference, simulator execution, or external evidence."""
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import shutil
import statistics
import xml.etree.ElementTree as ET

import nbformat
from PIL import Image, ImageChops
import pytest

from import_paper_snapshot import active_tex, caption_markdown
import paper_notebook
import paper_plots
from paper_evidence import load_bundle, validate, verify_exports

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "results/paper/paper-v1"


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(ROOT)


def test_verbatim_captions_and_order(bundle):
    manifest, _ = bundle
    nb = nbformat.read(ROOT / "analyses/chia_campaign_results.ipynb", as_version=4)
    assert len(nb.cells) == 17
    assert nb.cells[0].cell_type == "code"
    assert nb.cells[0].execution_count == 1
    for item, title, code in zip(manifest["figures"], nb.cells[1::2], nb.cells[2::2]):
        assert title.cell_type == "markdown" and code.cell_type == "code"
        assert title.source == f'## Fig. {item["number"]}. {caption_markdown(item["caption_latex"])}'
        assert code.source == paper_notebook.figure_source(item)
        assert code.execution_count == item["number"] + 1
        svg = [o for o in code.outputs if "image/svg+xml" in o.get("data", {})]
        assert len(svg) == 1
        assert not any(o.output_type == "error" for o in code.outputs)
    assert "organizaiton" in nb.cells[13].source  # Paper wording is not silently edited.
    assert active_tex("% unused\\caption{omit}\nkept \\% text") == "\nkept \\% text"


def test_no_hidden_payload_or_external_paths(bundle):
    nb = nbformat.read(ROOT / "analyses/chia_campaign_results.ipynb", as_version=4)
    for cell in nb.cells:
        assert not cell.metadata.get("jupyter", {}).get("source_hidden", False)
        assert "base64.b64decode" not in cell.source
        assert not re.search(r"/(?:home|Users)/|g[s]://|https?://", cell.source)
        assert "subprocess" not in cell.source
    for path in BUNDLE.rglob("*"):
        if path.is_file():
            assert not re.search(r"/(?:home|Users)/|a[3]-[A-Za-z0-9_-]+", path.read_text())


def test_all_case_populations_and_decisions(bundle):
    _, d = bundle
    assert len(d["evidence"]["single_core_test"]) == 8 * 24
    assert len(d["convergence"]["rows"]) == 4 * 2 * (11 + 10)
    assert len(d["convergence"]["decisions"]) == 40
    assert len(d["extensions"]["gem5"]) == 7 * 15
    assert len(d["extensions"]["hardware"]) == 2 * 9 * 24
    assert len(d["extensions"]["multicore"]) == 10 * 2 * 8
    assert len(d["additions"]["speed"]) == 300  # Two different same-host oracles.
    assert all(len(r["repetitions"]) == 5 for r in d["additions"]["speed"])
    assert set(r["mode"] for r in paper_plots.gem5_mode_means(d["extensions"])) == {"SE", "FS"}


@pytest.mark.parametrize("change", ["missing_accuracy", "duplicate_accuracy", "nan", "source", "selection",
                                   "oracle", "paired_population", "speed_oracle", "speed_repeat", "speed_source",
                                   "usage", "usage_scope", "gem5", "hardware", "multicore"])
def test_reject_invalid_evidence(bundle, change):
    _, original = bundle
    d = copy.deepcopy(original)
    if change == "missing_accuracy": d["evidence"]["single_core_test"].pop()
    elif change == "duplicate_accuracy": d["evidence"]["single_core_test"].append(d["evidence"]["single_core_test"][0])
    elif change == "nan": d["evidence"]["single_core_test"][0]["mae_L"] = float("nan")
    elif change == "source": d["evidence"]["sources"]["opus_single_core"]["source"] += "changed"
    elif change == "selection": d["convergence"]["decisions"][-1]["selected"] = "0" * 64
    elif change == "oracle": d["evidence"]["oracle_equivalence"][0]["equal"] = False
    elif change == "paired_population": d["evidence"]["single_core_test"][0]["pairs"] += 1
    elif change == "speed_oracle":
        next(r for r in d["additions"]["speed"] if r["arm"] == "oracle")["median"] *= 1.1
    elif change == "speed_repeat": d["additions"]["speed"][0]["repetitions"].pop()
    elif change == "speed_source":
        d["additions"]["speed_cohorts"][1]["models"]["opus_single_core"]["candidate"]["candidate_id"] = "0" * 64
    elif change == "usage": d["additions"]["usage"]["tables"]["headline"][0]["input_tokens"] += 10
    elif change == "usage_scope": d["additions"]["usage"]["rounds"] = [1, 15]
    else: d["extensions"][change].pop()
    with pytest.raises(ValueError): validate(d)


@pytest.mark.parametrize("failure", ["missing", "corrupt", "linked", "renderer"])
def test_missing_corrupt_or_linked_inputs_fail(tmp_path, failure):
    copy_root = tmp_path / "relocated"
    shutil.copytree(BUNDLE, copy_root / "results/paper/paper-v1")
    (copy_root / "analyses").mkdir()
    for name in ("paper_checks", "paper_plots", "paper_workflow"):
        shutil.copy2(ROOT / "analyses" / (name + ".py"), copy_root / "analyses")
    for name in ("requirements.txt", "requirements.lock.txt"):
        shutil.copy2(ROOT / "analyses" / name, copy_root / "analyses")
    target = copy_root / "results/paper/paper-v1/paper-additions.json"
    if failure == "missing": target.unlink()
    elif failure == "corrupt": target.write_bytes(target.read_bytes() + b" ")
    elif failure == "linked":
        target.unlink(); target.symlink_to(BUNDLE / target.name)
    else: (copy_root / "analyses/paper_plots.py").write_text("# changed")
    with pytest.raises(ValueError): load_bundle(copy_root)


def test_usage_gaps_and_same_host_speed_preserved(bundle):
    _, d = bundle
    headline = {r["campaign"]: r for r in d["additions"]["usage"]["tables"]["headline"]}
    assert all(headline[c]["unknown_usage_records"] > 0 for c in ("astra", "deepseek"))
    assert headline["gemini"]["observed_cost_lower_usd"] < headline["gemini"]["observed_cost_upper_usd"]
    assert "subscription" in d["additions"]["usage"]["campaigns"]["opus"]["billing"]
    means = {arm: statistics.geometric_mean(r["oracle_relative_speedup"]
             for r in d["additions"]["speed"] if r["arm"] == arm)
             for arm in ("astra_single_core", "deepseek_single_core", "gemini_single_core", "opus_single_core")}
    assert means == pytest.approx(dict(astra_single_core=31.63405021686087,
        deepseek_single_core=21.73151628520131, gemini_single_core=23.601955584112037,
        opus_single_core=6.437460163859206), rel=1e-12)


def test_command_always_builds_all_eight_figures(monkeypatch):
    spec = importlib.util.spec_from_file_location("paper_commands", ROOT / "scripts/artifact.py")
    commands = importlib.util.module_from_spec(spec); spec.loader.exec_module(commands)
    calls = []
    monkeypatch.setattr(paper_notebook, "build", lambda *a, **k: calls.append((a, k)))
    commands.reproduce_figures([])
    assert len(calls) == 1
    assert "supplement" not in str(calls)
    assert "dram-speed-addition" not in inspect.getsource(commands.reproduce_figures)


def test_exported_values_pixels_and_layout(bundle):
    manifest, _ = bundle
    output = ROOT / "analyses/figures"
    report = json.loads((output / "manifest.json").read_bytes())
    assert len(report["figures"]) == 8
    for f, r in zip(manifest["figures"], report["figures"]):
        name = f["name"]
        assert r["png_matches_paper"]
        assert hashlib.sha256((output / (name + ".png")).read_bytes()).hexdigest() == f["paper_exports"]["png"]
        tree = ET.parse(output / (name + ".svg"))
        assert not tree.findall(".//{http://www.w3.org/2000/svg}image")
        image = Image.open(output / (name + ".png")).convert("RGB")
        box = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
        assert box is not None
        if f["number"] > 1:
            assert max(box[1], image.height - box[3]) * 72 / 200 <= 5.5
        for table in f["reference_tables"]:
            assert (output / table).read_bytes() == (BUNDLE / "reference-tables" / table).read_bytes()
