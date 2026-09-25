"""Acquire the published instruction prefixes and rebuild their placement maps.

Only complete, hash-checked files are published. Download fragments stay in
``.downloads`` until a later invocation completes them. No simulator is run.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import lzma
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zlib
from urllib.parse import quote

from .layout import ROOT

BLOCK = 1024 * 1024
RANGE_BYTES = 8 * BLOCK


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def safe_path(root, relative):
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError(
            "input path must be relative and remain inside the data directory"
        )
    path = Path(root) / name
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("input paths must not contain symlinks")
    return path


def space_check(directory, required, reserve):
    if reserve < 0 or shutil.disk_usage(directory).free < reserve + required:
        raise RuntimeError(
            "insufficient disk space; free space or select another data directory"
        )


@contextmanager
def preparation_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another input preparation owns this data directory"
            ) from exc
        yield


def decoded_identity(path, expected_bytes=None):
    digest = hashlib.sha256()
    size = 0
    with gzip.open(path, "rb") as stream:
        while block := stream.read(BLOCK):
            size += len(block)
            digest.update(block)
    if expected_bytes is not None and size != expected_bytes:
        raise ValueError("decoded input length differs from the frozen record count")
    return digest.hexdigest(), size


def response_headers(path):
    """Read the final HTTP response, after proxy CONNECT and redirects."""
    blocks = re.split(r"\r?\n\r?\n", path.read_text())
    headers = [b for b in blocks if b.startswith("HTTP/")]
    if not headers:
        raise ValueError("download has no HTTP response headers")
    lines = headers[-1].splitlines()
    return int(lines[0].split()[1]), {
        k.lower(): v.strip()
        for k, v in (line.split(":", 1) for line in lines[1:] if ":" in line)
    }


def download_range(url, source, prefix, *, reserve):
    """Append verified range responses; retain an interrupted response's bytes."""
    stop = int(source["upstream_range_bytes"])
    offset = prefix.stat().st_size if prefix.exists() else 0
    if offset > stop:
        raise ValueError("download fragment is longer than the declared source range")
    while offset < stop:
        end = min(stop, offset + RANGE_BYTES) - 1
        space_check(prefix.parent, 2 * (end - offset + 1), reserve)
        chunk = prefix.with_suffix(".chunk")
        headers = prefix.with_suffix(".headers")
        command = [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--retry",
            "2",
            "--retry-max-time",
            "120",
            "--connect-timeout",
            "30",
            "--max-time",
            "90",
            "--speed-limit",
            "1024",
            "--speed-time",
            "20",
            "--range",
            f"{offset}-{end}",
            "--max-filesize",
            str(end - offset + 1),
            "--header",
            'If-Match: "' + source["etag"].strip('"') + '"',
            "--dump-header",
            str(headers),
            "--output",
            str(chunk),
            url,
        ]
        if url.startswith("https://"):
            command[1:1] = ["--proto", "=https", "--proto-redir", "=https"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=240)
        status, received = response_headers(headers)
        expected_range = f"bytes {offset}-{end}/{source['size_bytes']}"
        if (
            status != 206
            or received.get("content-range") != expected_range
            or received.get("etag", "").strip('"') != source["etag"].strip('"')
        ):
            raise ValueError(
                "upstream object changed or server ignored the requested range"
            )
        size = chunk.stat().st_size if chunk.exists() else 0
        if not 0 < size <= end - offset + 1:
            raise ValueError("invalid downloaded range length")
        # Even if curl timed out, authenticated response headers and the final
        # decoded hash make these bytes safe to retain for a resumed download.
        with prefix.open("ab") as out, chunk.open("rb") as src:
            shutil.copyfileobj(src, out, BLOCK)
            out.flush()
            os.fsync(out.fileno())
        chunk.unlink()
        offset += size
        if result.returncode or offset != end + 1:
            raise RuntimeError("download interrupted; rerun the same command to resume")


def prepare_trace(root, relative, source, base_url, *, reserve):
    target = safe_path(root, relative)
    expected_bytes = source["prefix_instructions"] * 64
    scratch = safe_path(root, ".downloads/" + Path(relative).stem)
    scratch.mkdir(parents=True, exist_ok=True)
    identity = {"url": base_url + quote(source["path"], safe="/"), **source}
    receipt = scratch / "source.json"
    if receipt.exists() and json.loads(receipt.read_text()) != identity:
        raise ValueError("download metadata changed; use a new data directory")
    if not receipt.exists():
        write_json(receipt, identity)
    if target.exists():
        logical, size = decoded_identity(target, expected_bytes)
        if logical != source["decoded_sha256"]:
            raise ValueError("existing trace has the wrong decoded SHA256")
        return {"sha256": sha(target), "decoded_sha256": logical, "decoded_bytes": size}
    space_check(root, expected_bytes + source["upstream_range_bytes"], reserve)
    prefix = scratch / "upstream.partial"
    download_range(identity["url"], source, prefix, reserve=reserve)
    opener = lzma.open if source["path"].endswith(".xz") else gzip.open
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    digest = hashlib.sha256()
    written = 0
    try:
        with opener(prefix, "rb") as src, partial.open("wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0
            ) as out:
                while written < expected_bytes:
                    block = src.read(min(BLOCK, expected_bytes - written))
                    if not block:
                        raise ValueError(
                            "upstream range contains too few instruction records"
                        )
                    out.write(block)
                    digest.update(block)
                    written += len(block)
    except (EOFError, lzma.LZMAError, zlib.error) as exc:
        raise ValueError("upstream prefix is truncated or corrupt") from exc
    if digest.hexdigest() != source["decoded_sha256"]:
        raise ValueError("downloaded instruction prefix has the wrong SHA256")
    logical, size = decoded_identity(partial, expected_bytes)
    if logical != source["decoded_sha256"]:
        raise ValueError("compressed prefix verification failed")
    partial.replace(target)
    return {"sha256": sha(target), "decoded_sha256": logical, "decoded_bytes": size}


def prepare_pages(root, relative, scanner):
    """Stream a verified prefix through the existing C++ instruction scanner."""
    trace = safe_path(root, relative)
    target = safe_path(root, ".pages/" + Path(relative).stem + ".txt")
    receipt = target.with_suffix(".json")
    identity = {"trace_sha256": sha(trace), "scanner_sha256": sha(scanner)}
    if target.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved == {**identity, "pages_sha256": sha(target)}:
            return target
        raise ValueError("cached page inventory changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial")
    with temporary.open("wb") as out:
        process = subprocess.Popen([str(scanner)], stdin=subprocess.PIPE, stdout=out)
        try:
            with gzip.open(trace, "rb") as src:
                shutil.copyfileobj(src, process.stdin, BLOCK)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("instruction page scanner failed")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    with temporary.open("rb") as stream:
        if stream.readline() != b"23000000\n":
            raise ValueError("page scanner received a different instruction population")
    temporary.replace(target)
    write_json(receipt, {**identity, "pages_sha256": sha(target)})
    return target


def prepare_map(root, relative, case, pages, binary, expected):
    target = safe_path(root, relative)
    if target.exists():
        logical, size = decoded_identity(target)
        if logical != expected:
            raise ValueError("existing placement map differs from the frozen identity")
        return {"sha256": sha(target), "decoded_sha256": logical, "decoded_bytes": size}
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="placement-", dir=root / ".pages"))
    inventory, mapping = scratch / "pages.txt", scratch / "map.txt"
    with inventory.open("wb") as out:
        out.write(f"CHAMPSIM_PAGES_V1 {len(case.programs)} 4096 {8 << 30}\n".encode())
        for core, program in enumerate(case.programs):
            with pages[program].open("rb") as stream:
                next(stream)
                for line in stream:
                    out.write(f"{core} ".encode() + line)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"CHAMPSIM_PLACEMENT_FILE", "RAMULATOR_CONFIG"}
    }
    with (scratch / "prepare.log").open("wb") as log:
        subprocess.run(
            [str(binary), "--prepare-placement", str(inventory), str(mapping)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if sha(mapping) != expected:
        raise ValueError("generated placement differs from the frozen identity")
    partial = target.with_suffix(target.suffix + ".partial")
    with mapping.open("rb") as src, partial.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=3, mtime=0
        ) as out:
            shutil.copyfileobj(src, out, BLOCK)
    logical, size = decoded_identity(partial)
    if logical != expected:
        raise ValueError("compressed placement verification failed")
    partial.replace(target)
    # Remove only these generated intermediates, after verified publication.
    inventory.unlink()
    mapping.unlink()
    return {"sha256": sha(target), "decoded_sha256": logical, "decoded_bytes": size}


def prepare(configuration, data_root, work_root, *, workers=2, reserve_gib=32):
    if not 1 <= workers <= 4 or reserve_gib < 0:
        raise ValueError("use 1–4 preparation workers and a nonnegative disk reserve")
    root = Path(data_root).absolute()
    safe_path(root, "prepared-inputs.json")
    cohort = configuration.experiment.evaluation.champsim
    if cohort is None or not cohort.cases:
        raise ValueError("input preparation requires the published ChampSim cohort")
    sources_path = ROOT / "results/manifests/input-sources.json"
    sources = json.loads(sources_path.read_text())
    if sources["base_url"] != "https://traces.rbera.com/dpc4/":
        raise ValueError("unexpected trace download source")
    expected = {
        r["path"]: r
        for r in json.loads((ROOT / "results/manifests/inputs.json").read_text())[
            "files"
        ]
    }
    # Validate the complete request before downloading anything.
    relative = {}
    for program, trace in cohort.traces.items():
        name = Path(trace.path).relative_to(root).as_posix()
        if sources["traces"][name]["decoded_sha256"] != trace.decoded_sha256:
            raise ValueError("configuration selects an unrecognized trace identity")
        relative[program] = name
    for case in cohort.cases.values():
        name = Path(case.placement.path).relative_to(root).as_posix()
        if expected[name]["decoded_sha256"] != case.placement.decoded_sha256:
            raise ValueError("configuration selects an unrecognized placement identity")
        binary = Path(cohort.builds[str(len(case.programs))].binary)
        if not binary.is_file():
            raise FileNotFoundError(
                "build ChampSim before preparing inputs: " + str(binary)
            )
    rows = {}
    with preparation_lock(root):
        receipt_path = root / "prepared-inputs.json"
        if receipt_path.exists():
            rows = json.loads(receipt_path.read_text())["files"]
        scanner = Path(work_root).absolute() / "input-tools/trace-pages"
        scanner.parent.mkdir(parents=True, exist_ok=True)
        scanner_source = ROOT / "ramulator/resources/champsim_bridge/trace_pages.cc"
        scanner_header = (
            ROOT / "ramulator/integrations/champsim/source/inc/trace_instruction.h"
        )
        scanner_receipt = scanner.with_suffix(".json")
        if not scanner.exists():
            subprocess.run(
                [
                    "g++",
                    "-O3",
                    "-DNDEBUG",
                    "-std=c++20",
                    "-I" + str(scanner_header.parent),
                    str(scanner_source),
                    "-o",
                    str(scanner),
                ],
                check=True,
            )
            write_json(
                scanner_receipt,
                {
                    "source_sha256": sha(scanner_source),
                    "header_sha256": sha(scanner_header),
                    "sha256": sha(scanner),
                },
            )
        if json.loads(scanner_receipt.read_text()) != {
            "source_sha256": sha(scanner_source),
            "header_sha256": sha(scanner_header),
            "sha256": sha(scanner),
        }:
            raise ValueError("page scanner changed; select a new work directory")
        reserve = int(reserve_gib * 1024**3)
        space_check(root, workers * (23_000_000 * 64 + 256 * BLOCK), reserve)

        def one(item):
            program, name = item
            row = prepare_trace(
                root,
                name,
                sources["traces"][name],
                sources["base_url"],
                reserve=reserve,
            )
            page = prepare_pages(root, name, scanner)
            return program, name, row, page

        pages = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(one, item) for item in relative.items()]
            try:
                for future in as_completed(pending):
                    program, name, row, page = future.result()
                    rows[name] = {
                        **row,
                        "reference_sha256": expected[name]["sha256"],
                        "kind": "trace",
                    }
                    pages[program] = page
                    print("Verified trace:", program, flush=True)
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
        for name, case in cohort.cases.items():
            relative_map = Path(case.placement.path).relative_to(root).as_posix()
            space_check(root, 256 * BLOCK, reserve)
            row = prepare_map(
                root,
                relative_map,
                case,
                pages,
                Path(cohort.builds[str(len(case.programs))].binary),
                case.placement.decoded_sha256,
            )
            rows[relative_map] = {
                **row,
                "reference_sha256": expected[relative_map]["sha256"],
                "kind": "placement",
            }
            print("Verified placement:", name, flush=True)
        result = {
            "schema_version": 1,
            "source_manifest_sha256": sha(sources_path),
            "integrity": sources["integrity"],
            "files": rows,
        }
        write_json(receipt_path, result)
        return result
