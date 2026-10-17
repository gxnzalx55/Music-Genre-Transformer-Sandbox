"""Playlist listing and partial-audio download with yt-dlp."""
from __future__ import annotations

from pathlib import Path

PLACEHOLDER_TITLES = {"[Private video]", "[Deleted video]"}


def list_playlist(url: str) -> list[dict]:
    """Flat listing: one dict per entry with id, title, channel, duration."""
    import yt_dlp

    opts = {"extract_flat": "in_playlist", "quiet": True, "no_warnings": True,
            "ignoreerrors": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = []
    for e in (info or {}).get("entries") or []:
        if not e or not e.get("id"):
            continue
        entries.append({
            "id": e["id"],
            "title": e.get("title") or "",
            "channel": e.get("channel") or e.get("uploader") or "",
            "duration": e.get("duration"),
        })
    return entries


def is_unavailable(entry: dict) -> bool:
    return entry["title"] in PLACEHOLDER_TITLES


def fetch_metadata(video_id: str) -> dict:
    """Full metadata of a single video (used when the flat listing lacks duration)."""
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}",
                                download=False)
    return {
        "id": video_id,
        "title": info.get("title") or "",
        "channel": info.get("channel") or info.get("uploader") or "",
        "duration": info.get("duration"),
    }


def download_segment(
    video_id: str,
    start: float,
    end: float,
    tmp_dir: Path,
    ffmpeg: str,
    retries: int = 3,
) -> Path:
    """Download only [start, end] of the best audio stream; keep original codec."""
    import yt_dlp
    from yt_dlp.utils import download_range_func

    tmp_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio",
        "outtmpl": str(tmp_dir / f"{video_id}.%(ext)s"),
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
        "ffmpeg_location": ffmpeg,
        "retries": retries,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}",
                                download=True)
    downloads = info.get("requested_downloads") or []
    if downloads and Path(downloads[0]["filepath"]).exists():
        return Path(downloads[0]["filepath"])
    matches = [p for p in tmp_dir.glob(f"{video_id}.*") if p.suffix != ".part"]
    if not matches:
        raise RuntimeError("download produced no file")
    return matches[0]

