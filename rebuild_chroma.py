from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from indexer import KnowledgeIndex, SUPPORTED_EXTENSIONS


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Vertiv"
CHROMA = ROOT / "data" / "chroma"
PYTHON = Path(sys.executable)
TIMEOUT_SECONDS = 120

KNOWN_SKIPS = {
    "Power Infrastructure/UPS_Systems/UPS_Systems/HiPulse/500kVA_UM.pdf",
    "Power Infrastructure/UPS_Systems/UPS_Systems/Miscellaneous Technical Manuscripts/vertiv_critical_power_consultant_kit_2019.pdf",
    "Thermal_Management/Thermal_Management/Precision_Cooling/Chillers/AFC/Liebert-AFC PD-EN-EMEA-10023174MAN_ENG.pdf",
    "Thermal_Management/Thermal_Management/Precision_Cooling/Chillers/AFC/Liebert-AFC-UM-EN-EMEA-10014089MAN_ENG 8.pdf",
}


CHILD_CODE = r"""
import sys
from pathlib import Path
from indexer import KnowledgeIndex

relative = sys.argv[1]
root = Path('Vertiv')
ix = KnowledgeIndex(root, Path('data/chroma'))
path = root / relative
try:
    ix._index_file(path, relative, force=False)
except Exception as exc:
    ix._record_error(path, relative, exc)
    print(f'ERROR {relative}: {exc}', flush=True)
"""


def manifest_entry(relative: str) -> dict | None:
    manifest = CHROMA.parent / "chroma_manifest.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data.get("documents", {}).get(relative)


def is_current(path: Path, relative: str) -> bool:
    entry = manifest_entry(relative)
    if not entry:
        return False
    stat = path.stat()
    return (
        entry.get("size") == stat.st_size
        and entry.get("modified_ns") == stat.st_mtime_ns
        and entry.get("status") in {"ready", "error"}
    )


def main() -> None:
    ix = KnowledgeIndex(DATASET, CHROMA)
    files = sorted(
        (p for p in DATASET.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )
    seen = {p.relative_to(DATASET).as_posix() for p in files}

    for number, path in enumerate(files, 1):
        relative = path.relative_to(DATASET).as_posix()
        if relative in KNOWN_SKIPS:
            print(f"[{number}/{len(files)}] SKIP {relative}", flush=True)
            ix._record_error(path, relative, RuntimeError("Skipped: PDF parser hangs on this file"))
            continue
        if is_current(path, relative):
            continue
        print(f"[{number}/{len(files)}] {relative}", flush=True)
        try:
            subprocess.run(
                [str(PYTHON), "-X", "utf8", "-c", CHILD_CODE, relative],
                cwd=ROOT,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            ix._record_error(path, relative, RuntimeError(f"Skipped: parser timeout after {TIMEOUT_SECONDS}s"))
            print(f"TIMEOUT {relative}", flush=True)

    # Child processes update the manifest while the parent is still running.
    # Reload before pruning so the parent does not overwrite successful child
    # entries with its older in-memory manifest.
    ix._manifest = ix._load_manifest()
    ix._remove_missing(seen)
    print("FINAL", ix.stats(), flush=True)


if __name__ == "__main__":
    main()
