#!/usr/bin/env python3
"""Generate the Astra single-core feature-removal variants from the frozen model.

Each variant is the frozen model.cpp with exact, asserted string replacements (each old string must
occur exactly once) plus a one-line header comment naming the removed mechanism. The control is a
byte-for-byte copy with no header, so its candidate identity equals the published endpoint.
parameters.json is copied byte for byte ("{}") for every variant.

  make_variants.py --model DIR --out DIR      (writes DIR/<NN-name>/{model.cpp,parameters.json,VARIANT.json})
"""
import argparse, difflib, hashlib, json, shutil
from pathlib import Path

# (id, name, description, header rationale, [(old, new), ...]); line numbers refer to the frozen file.
VARIANTS = [
    ("00", "control", "unmodified frozen model (byte-identical copy)", None, []),
    ("01", "no-row-state",
     "open-row proxy removed: hit=false, cold=true for every access, so row_cost = tRCD always (L72)",
     "open-row proxy removed (hit=false, cold=true): every access pays row_cost = tRCD; all hit-gated rules see a miss",
     [("bool cold=b.row<0, hit=b.row==a.levels[row_level];",
       "bool cold=true, hit=false;  // ABLATION 01")]),
    ("02", "no-batch-ring",
     "8-entry recent-row ring / batch-hit rule disabled: batch_hit never set (L88); ring bookkeeping left inert",
     "8-entry recent-row ring batch-hit rule disabled (batch_hit always false)",
     [("batch_hit=backlog>ccd && now-b.recent[i].seen < std::max(rc,backlog);",
       "batch_hit=false;  // ABLATION 02")]),
    ("03", "no-class-rows",
     "separate read-row/write-row classes disabled: class_hit = hit (L82)",
     "separate read/write row classes disabled (class_hit = hit only)",
     [("bool class_hit=hit || b.class_row[kind]==a.levels[row_level];",
       "bool class_hit=hit;  // ABLATION 03")]),
    ("04", "no-write-recovery",
     "write-recovery floor (nCWL+nBL+nWR+nRP+nRCD = 180 after a write before a conflict) disabled (L108)",
     "write-recovery floor (nCWL+nBL+nWR+nRP+nRCD after a write, before a conflict) disabled",
     [("if(b.last_kind==1 && !hit) {",
       "if(false && b.last_kind==1 && !hit) {  // ABLATION 04")]),
    ("05", "no-wr-rd-turnaround",
     "same-bank write-to-read floor (nCWL+nBL+nWTRL = 64) disabled (L124)",
     "same-bank same-row write-to-read floor (nCWL+nBL+nWTRL) disabled",
     [("if(kind==0 && b.last_kind==1 && hit && now>b.last_arrival) {",
       "if(false && kind==0 && b.last_kind==1 && hit && now>b.last_arrival) {  // ABLATION 05")]),
    ("06", "no-act-throttle",
     "rank (max(nRRDS,nFAW/4)) and bank-group (nRRDL) activation clocks disabled: A = 0, clocks never advance (L140)",
     "rank and bank-group activation clocks disabled (activation_wait A = 0)",
     [("if(!class_hit && !batch_hit) {",
       "if(false && !class_hit && !batch_hit) {  // ABLATION 06")]),
    ("07", "no-bankgroup-clock",
     "per-bank-group column clock disabled: group_wait = 0 (L162); its update stays inert",
     "per-bank-group column clock disabled (group_wait = 0)",
     [("double group_wait=std::max(0.0,groups[gi]-now);",
       "double group_wait=0;  // ABLATION 07")]),
    ("08", "no-bus",
     "channel data-bus clocks disabled: channel_wait = 0, i.e. C = t (L161); bank-only model, write tax becomes inert",
     "channel data-bus clocks disabled (channel_wait = 0, C = t): bank-only model",
     [("double channel_wait=std::max(0.0,shared-now);",
       "double channel_wait=0;  // ABLATION 08")]),
    ("09", "no-write-tax",
     "amortized write turnaround tax disabled: write_tax = 0 (L61)",
     "amortized write turnaround tax disabled (write_tax = 0)",
     [("write_tax=(timing(\"nRTW\",base)+timing(\"nWTRL\",timing(\"nWTR_L\",ccd)))/batch;",
       "write_tax=0*batch;  // ABLATION 09")]),
    ("10", "bank-less",
     "all per-bank state disabled: hit=false/cold=true (row_cost = tRCD), class_hit=false, backlog=0 (no batch hits), "
     "bank readiness = t+tRCD+A with no b.ready+service term; write-recovery/WR-RD/read-run rules become inert; "
     "channel, bank-group column and activation clocks kept (L72, L82, L83, L150-151)",
     "all per-bank state disabled (closed-bank cost, no bank backlog); channel, bank-group and activation clocks kept",
     [("bool cold=b.row<0, hit=b.row==a.levels[row_level];",
       "bool cold=true, hit=false;  // ABLATION 10"),
      ("bool class_hit=hit || b.class_row[kind]==a.levels[row_level];",
       "bool class_hit=false;  // ABLATION 10"),
      ("double backlog=std::max(0.0,b.ready-now);",
       "double backlog=0;  // ABLATION 10"),
      ("double ready=std::max(double(now)+row_cost+activation_wait,\n"
       "                          b.ready+service*bank_scale);",
       "double ready=double(now)+row_cost+activation_wait;  // ABLATION 10")]),
    ("11", "fix-channel-startup",
     "NOT a removal: L169 channel work clock no longer carries the requester's row startup; "
     "c.work = max(t, c.work) + transfer_service (nBL, plus write_tax for writes) instead of max(t+row_cost, c.work) + "
     "transfer_service; c.transfer unchanged (the two channel clocks now evolve identically)",
     "FIX (not a removal): channel work clock advances from max(t, c.work)+transfer_service, not max(t+row_cost, c.work)",
     [("c.work=std::max(double(now)+row_cost,c.work)+transfer_service;",
       "c.work=std::max(double(now),c.work)+transfer_service;  // ABLATION 11 (fix)")]),
]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    src = (a.model / "model.cpp").read_text()
    params = (a.model / "parameters.json").read_bytes()
    if sha(src.encode()) != "9f6256bf90af6756eb6238b08d0288723c9ed040e3c87033d40ca6d33696cdcc":
        raise SystemExit("frozen model.cpp hash differs from the published endpoint")
    index = []
    for vid, name, desc, header, edits in VARIANTS:
        text = src
        for old, new in edits:
            if text.count(old) != 1:
                raise SystemExit(f"{vid}: replacement target must occur exactly once: {old!r}")
            text = text.replace(old, new)
        if header:
            text = f"// ABLATION {vid} {name}: {header}.\n" + text
        d = a.out / f"{vid}-{name}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.cpp").write_text(text)
        (d / "parameters.json").write_bytes(params)
        body = text.splitlines()[1:] if header else text.splitlines()
        diff = [l for l in difflib.unified_diff(src.splitlines(), body, lineterm="", n=0)
                if l[:1] in "+-" and not l.startswith(("+++", "---"))]
        removed = sum(l.startswith("-") for l in diff)
        added = sum(l.startswith("+") for l in diff)
        meta = dict(id=vid, name=name, description=desc, header=header, edits=[dict(old=o, new=n) for o, n in edits],
                    lines_changed=removed, lines_removed=removed, lines_added=added,
                    model_cpp_sha256=sha(text.encode()), parameters_sha256=sha(params),
                    frozen_model_cpp_sha256=sha(src.encode()))
        (d / "VARIANT.json").write_text(json.dumps(meta, indent=1) + "\n")
        (d / "model.diff").write_text("\n".join(difflib.unified_diff(
            src.splitlines(), text.splitlines(), "frozen/model.cpp", f"{vid}-{name}/model.cpp", lineterm="")) + "\n")
        index.append(meta)
        print(f"{vid} {name:22s} lines_changed={removed} sha={meta['model_cpp_sha256'][:12]}")
    (a.out / "variants.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
