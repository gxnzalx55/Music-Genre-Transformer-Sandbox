"""Spotify playlists via spotDL: track list (metadata) and full-track download."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


def is_spotify_url(url: str) -> bool:
    return "open.spotify.com" in url or url.startswith("spotify:")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    exe = Path(cmd[0])
    if exe.exists():                     # relative paths with "/" fail in CreateProcess
        cmd = [str(exe.resolve())] + cmd[1:]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def list_playlist(url: str, spotdl_exe: str, ffmpeg: str) -> list[dict]:
    """Track metadata of a playlist/album/track URL: id, title, channel, duration (s)."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "list.spotdl"
        proc = _run([spotdl_exe, "save", url, "--save-file", str(out), "--ffmpeg", ffmpeg])
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"spotdl save failed: {(proc.stderr or proc.stdout)[-500:]}")
        songs = json.loads(out.read_text(encoding="utf-8"))
    return [{
        "id": s["song_id"],
        "title": s["name"],
        "channel": ", ".join(s["artists"]),
        "duration": float(s["duration"]),
        "url": s["url"],
    } for s in songs]


def download_full(entry: dict, tmp_dir: Path, spotdl_exe: str, ffmpeg: str,
                  fmt: str = "m4a") -> Path:
    """Download one full track with spotDL (it matches it on YouTube Music/YouTube)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    template = str(tmp_dir / "{track-id}.{output-ext}")
    proc = _run([spotdl_exe, "download", entry["url"], "--ffmpeg", ffmpeg,
                 "--format", fmt, "--output", template])
    matches = [p for p in tmp_dir.glob(f"{entry['id']}.*") if p.suffix != ".part"]
    if not matches:
        raise RuntimeError(f"spotdl download failed: {(proc.stderr or proc.stdout)[-300:]}")
    return matches[0]
