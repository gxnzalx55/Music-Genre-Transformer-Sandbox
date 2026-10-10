"""Download and extract FMA dataset archives from the official mirror."""
from __future__ import annotations

import hashlib
import logging
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

FMA_BASE_URL = "https://os.unil.cloud.switch.ch/fma"

ARCHIVES: dict[str, str] = {
    "small":    "fma_small.zip",
    "medium":   "fma_medium.zip",
    "metadata": "fma_metadata.zip",
}
CHECKSUMS_FILE = "checksums.txt"


def download_fma(output_dir: str | Path, subset: str = "small") -> None:
    """Download fma_{subset}.zip and fma_metadata.zip to output_dir.

    Supports resume via HTTP Range requests (partial downloads saved as .part).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in [ARCHIVES[subset], ARCHIVES["metadata"], CHECKSUMS_FILE]:
        dest = output_dir / filename
        if dest.exists():
            log.info("Already downloaded: %s", filename)
            continue
        _download_file(f"{FMA_BASE_URL}/{filename}", dest)


def _download_file(url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    resume_from = part.stat().st_size if part.exists() else 0

    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    with requests.get(url, stream=True, headers=headers, timeout=30) as r:
        if r.status_code == 416:
            part.rename(dest)
            return
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + resume_from
        mode = "ab" if resume_from else "wb"
        with (
            open(part, mode) as f,
            tqdm(
                total=total,
                initial=resume_from,
                unit="B",
                unit_scale=True,
                desc=dest.name,
                miniters=1,
            ) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=65_536):
                f.write(chunk)
                pbar.update(len(chunk))
    part.rename(dest)


def verify_checksums(data_dir: Path, subset: str = "small") -> bool:
    """Verify MD5 checksums against checksums.txt.

    Returns True if all files pass; logs mismatches and returns False otherwise.
    """
    checksums_path = data_dir / CHECKSUMS_FILE
    if not checksums_path.exists():
        log.warning("checksums.txt not found — skipping verification")
        return True

    expected: dict[str, str] = {}
    with open(checksums_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 2:
                md5, name = parts
                expected[name] = md5

    all_ok = True
    for filename in [ARCHIVES[subset], ARCHIVES["metadata"]]:
        path = data_dir / filename
        if not path.exists():
            log.error("Missing: %s", filename)
            all_ok = False
            continue
        digest = _md5(path)
        if expected.get(filename) and digest != expected[filename]:
            log.error("Checksum mismatch: %s (expected %s, got %s)",
                      filename, expected[filename], digest)
            all_ok = False
        else:
            log.info("OK: %s", filename)
    return all_ok


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_archives(data_dir: Path) -> None:
    """Extract all .zip files in data_dir in-place."""
    for zpath in sorted(data_dir.glob("*.zip")):
        dest = data_dir / zpath.stem
        if dest.exists():
            log.info("Already extracted: %s", zpath.name)
            continue
        log.info("Extracting %s ...", zpath.name)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(data_dir)
        log.info("Done: %s", zpath.name)
