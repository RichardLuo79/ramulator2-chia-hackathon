"""Prepare the approved frozen-model study; execution uses the existing worker.

Only four mix slots change. Selection uses membership, never measured errors.
The old study remains an immutable source of inputs, maps and model snapshots.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from ramulator_chia.framework.archive import describe_payload
from ramulator_chia.framework.identity import file_sha256
from ramulator_chia.framework.snapshots import snapshot
from ramulator_chia.framework.dram import MODEL_FILES

SEED = "dpc4-continuous-contention-20260915-v1"
EXCLUDED = "sierra_a_4_0006"
REPLACEMENTS = {
    "test-c4_google_03": (0, "sierra_a_4_0013"),
    "test-c8-04": (0, "sierra_a_3_0001"),
    "test-c8-05": (3, "sierra_a_4_0005"),
    "test-c8-07": (3, "sierra_a_6_0001"),
}
PILOTS = ["validation-c1-gap_pr", "test-c4_google_03", "test-c8-07"]


def order(name):
    return hashlib.sha256((SEED + "\0" + name).encode()).hexdigest()


def select_cases(old):
    mixes = []
    for row in old["mixes"]:
        item = {k: v for k, v in row.items() if k != "placement"}
        item["programs"] = list(row["programs"])
        if row["name"] in REPLACEMENTS:
            slot, replacement = REPLACEMENTS[row["name"]]
            if item["programs"][slot] != EXCLUDED:
                raise ValueError("source mix differs from the reviewed membership")
            item["programs"][slot] = replacement
        mixes.append(item)
    for stage, expected in (("validation", 14), ("test", 24)):
        pool = {n for m in mixes if m["stage"] == stage for n in m["programs"]}
        if len(pool) != expected or EXCLUDED in pool:
            raise ValueError("unexpected workload pool")
        for cores in (4, 8):
            group = [m for m in mixes if (m["stage"], m["cores"]) == (stage, cores)]
            if (len(group) != 8 or len({tuple(sorted(m["programs"])) for m in group}) != 8 or
                    {n for m in group for n in m["programs"]} != pool or
                    any(len(set(m["programs"])) != cores for m in group)):
                raise ValueError("mixes lost coverage, uniqueness or distinct programs")
            local = [m["name"] for m in group if m["name"] in PILOTS]
            local += [m["name"] for m in sorted(group, key=lambda m: order(m["name"]))
                      if m["name"] not in local][:2-len(local)]
            for m in group:
                m["site"] = "local" if m["name"] in local else "cloud"
        singles = [dict(name=f"{stage}-c1-{n}", stage=stage, cores=1, programs=[n]) for n in sorted(pool)]
        count = 4 if stage == "validation" else 6
        local = [m["name"] for m in singles if m["name"] in PILOTS]
        local += [m["name"] for m in sorted(singles, key=lambda m: order(m["name"]))
                  if m["name"] not in local][:count-len(local)]
        mixes.extend(dict(m, site="local" if m["name"] in local else "cloud") for m in singles)
    if Counter(m["site"] for m in mixes) != dict(local=18, cloud=52):
        raise ValueError("host assignment differs from the approved 144/416 runs")
    return mixes


def prepare(args):
    from ramulator_chia.eval.champsim_placement import read, save, CAPACITY, ARMS
    root, prior = args.output, args.prior
    old = read(prior / "protocol.json")
    mixes = select_cases(old)
    root.mkdir(parents=True, exist_ok=True)
    save(root / "mixes.json", mixes)
    names = sorted({n for m in mixes for n in m["programs"]})

    def reuse(name):
        row = old["inputs"][name]
        src = prior / row["path"]
        if file_sha256(src) != row["inventory"]["stored_sha256"]:
            raise ValueError("input archive changed: " + name)
        digest, size = hashlib.sha256(), 0
        with gzip.open(src, "rb") as stream:
            while block := stream.read(4 << 20):
                digest.update(block)
                size += len(block)
        if (digest.hexdigest() != row["inventory"]["decoded_sha256"] or
                size != 23_000_000 * 64):
            raise ValueError("decoded input changed: " + name)
        for path in (row["path"], "pages/" + name + ".txt.gz"):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(prior / path, target)
        if file_sha256(root / "pages" / (name + ".txt.gz")) != row["pages_sha256"]:
            raise ValueError("page inventory changed: " + name)
        save(root / "inputs" / (name + ".json"), row)
        print("VERIFIED", name, flush=True)
        return name, row

    with ThreadPoolExecutor(max_workers=min(4, args.cpus)) as pool:
        inputs = dict(pool.map(reuse, names))
    old_maps = {m["name"]: m for m in old["mixes"]}
    for mix in mixes:
        name = mix["name"]
        destination = root / "maps" / (name + ".txt.gz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name in old_maps and name not in REPLACEMENTS:
            shutil.copyfile(prior / "maps" / destination.name, destination)
            if file_sha256(destination) != old_maps[name]["placement"]["stored_sha256"]:
                raise ValueError("existing placement changed")
        elif not destination.exists():
            page_file, map_file = destination.with_suffix('.pages'), destination.with_suffix('.txt')
            with page_file.open('wb') as out:
                out.write(f"CHAMPSIM_PAGES_V1 {mix['cores']} 4096 {CAPACITY}\n".encode())
                for core, n in enumerate(mix['programs']):
                    with gzip.open(root / 'pages' / (n+'.txt.gz'), 'rb') as pages:
                        for line in pages:
                            out.write(str(core).encode()+b' '+line)
            binary = getattr(args, f"source{mix['cores']}") / f"bin/champsim-{mix['cores']}core"
            env = {k:v for k,v in os.environ.items() if k not in ('RAMULATOR_CONFIG', 'CHAMPSIM_PLACEMENT_FILE')}
            with destination.with_suffix('.log').open('w') as log:
                subprocess.run([str(binary), '--prepare-placement', str(page_file), str(map_file)],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            raw = map_file.read_bytes()
            destination.write_bytes(gzip.compress(raw, compresslevel=3, mtime=0))
            if gzip.decompress(destination.read_bytes()) != raw:
                raise ValueError('placement compression failed')
            page_file.unlink()
            map_file.unlink()
        mix['placement'] = asdict(describe_payload(destination, 'placement.txt', codec='gzip').member)
    for selection in old['selections'].values():
        candidate = selection['candidate']
        if snapshot(prior/'candidates'/candidate['candidate_id'], root/'candidates', MODEL_FILES,
                    maximum_bytes=1<<20) != candidate:
            raise ValueError('frozen model source changed')
    shutil.copyfile(prior/'mess.txt', root/'mess.txt')
    if file_sha256(root/'mess.txt') != old['mess_sha256']:
        raise ValueError('MESS curve changed')
    protocol = dict(old, schema_version=2, seed=SEED, mixes=mixes, inputs=inputs, pilots=PILOTS,
        completion_policy='background-replay', request_window='per-core-admission-ordinal-v1',
        source_protocol_sha256=file_sha256(prior/'protocol.json'), expected_runs=560,
        exclusion=dict(workload=EXCLUDED, reason='operator-approved protocol change after slow execution; not corrupt input'),
        replacements=REPLACEMENTS, arms=ARMS)
    save(root/'protocol.json', protocol)
    print('PREPARED', len(mixes), 'cases / 560 runs', flush=True)


def verify_remote_inputs(root, prior, runtime):
    """Reuse retained compressed traces, after checking both representations."""
    protocol = json.loads((root/'protocol.json').read_text())
    manifest = json.loads((runtime/'runtime_manifest.json').read_text())
    if manifest['source_inventory'] != protocol['runtime_source_inventory']:
        raise ValueError('retained Ramulator runtime has different source inputs')

    def verify(item):
        name, row = item
        source, destination = prior/row['path'], root/row['path']
        if file_sha256(source) != row['inventory']['stored_sha256']:
            raise ValueError('retained compressed trace changed: '+name)
        digest, size = hashlib.sha256(), 0
        with gzip.open(source, 'rb') as stream:
            while block := stream.read(4 << 20):
                digest.update(block)
                size += len(block)
        if size != 23_000_000*64 or digest.hexdigest() != row['inventory']['decoded_sha256']:
            raise ValueError('retained decoded trace changed: '+name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if file_sha256(destination) != row['inventory']['stored_sha256']:
            raise ValueError('copied trace changed: '+name)
        print('VERIFIED retained input', name, flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, protocol['inputs'].items()))
    for mix in protocol['mixes']:
        actual = describe_payload(root/'maps'/(mix['name']+'.txt.gz'), 'placement.txt', codec='gzip')
        if asdict(actual.member) != mix['placement']:
            raise ValueError('copied placement differs: '+mix['name'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('verify-inputs',))
    for name in ('output', 'prior', 'runtime'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    verify_remote_inputs(args.output, args.prior, args.runtime)
