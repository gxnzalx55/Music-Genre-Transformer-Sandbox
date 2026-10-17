"""CLI: build a custom dataset of 30 s log-mel segments from a YouTube or Spotify playlist."""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MANIFEST_FIELDS = ["track_id", "source", "title", "channel", "duration", "start", "end", "path", "genre", "playlist"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download a 30 s segment per song of a YouTube or Spotify playlist "
                    "and store it as a log-mel .npy.")
    p.add_argument("url", nargs="?", default=None, help="YouTube or Spotify playlist URL")
    p.add_argument("--genre", default=None,
                   help="Genre label for every song of URL (one playlist = one class)")
    p.add_argument("--playlist", action="append", default=[], metavar="GENRE=URL",
                   help="Extra playlist with its genre; repeatable, e.g. "
                        "--playlist Reggaeton=https://open.spotify.com/playlist/...")
    p.add_argument("--out", default="data/custom")
    p.add_argument("--config", default="configs/ingest/default.yaml")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N new songs (for testing)")
    p.add_argument("--dry-run", action="store_true",
                   help="List the playlist and planned segments, download nothing")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel downloads (default 1, to be gentle with YouTube)")
    return p.parse_args()


class Store:
    """On-disk state: mels/, manifest.csv, archive.txt, failed.log (thread-safe)."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.mels = out / "mels"
        self.tmp = out / "tmp"
        self.manifest = out / "manifest.csv"
        self.archive = out / "archive.txt"
        self.failed = out / "failed.log"
        self.lock = threading.Lock()
        self.mels.mkdir(parents=True, exist_ok=True)
        self._migrate_manifest()
        self.done = self._load_done()
        self.genres = self._load_genres()

    def _migrate_manifest(self) -> None:
        """Manifests written before labelling lack genre/playlist: add them, empty."""
        if not self.manifest.exists():
            return
        with open(self.manifest, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            header = reader.fieldnames
        if header == MANIFEST_FIELDS:
            return
        tmp = self.manifest.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, restval="")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, self.manifest)

    def _load_genres(self) -> dict[str, str]:
        genres: dict[str, str] = {}
        if self.manifest.exists():
            with open(self.manifest, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    genres[row["track_id"]] = row["genre"]
        return genres

    def _load_done(self) -> set[str]:
        done: set[str] = set()
        if self.archive.exists():
            for line in self.archive.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(line.split()[-1])
        # A crash between the manifest row and the archive line must not duplicate rows.
        if self.manifest.exists():
            with open(self.manifest, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if (self.out / row["path"]).exists():
                        done.add(row["track_id"])
        return done

    def save(self, row: dict, mel) -> None:
        import numpy as np

        final = self.mels / f"{row['track_id']}.npy"
        tmp = self.mels / f"{row['track_id']}.npy.tmp"
        with open(tmp, "wb") as fh:          # file handle: np.save won't append ".npy"
            np.save(fh, mel)
        os.replace(tmp, final)
        with self.lock:
            new = not self.manifest.exists()
            with open(self.manifest, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            with open(self.archive, "a", encoding="utf-8") as fh:
                fh.write(f"{row['source']} {row['track_id']}\n")
            self.done.add(row["track_id"])
            self.genres[row["track_id"]] = row["genre"]

    def fail(self, track_id: str, reason) -> None:
        with self.lock, open(self.failed, "a", encoding="utf-8") as fh:
            fh.write(f"{track_id}\t{' '.join(str(reason).split())}\n")


def process_one(entry: dict, cfg, dataset_cfg, store: Store, ffmpeg: str,
                source: str = "youtube", genre: str = "", playlist: str = "") -> str:
    """Handle one song. Returns 'ok' or 'failed'; never raises."""
    from data.ingest.segment import audio_length, audio_to_mel, compute_segment
    from data.ingest.youtube import download_segment, fetch_metadata, is_unavailable

    tid = entry["id"]
    audio = None
    try:
        if is_unavailable(entry):
            raise RuntimeError(f"unavailable ({entry['title']})")
        if not entry.get("duration"):
            entry = {**entry, **fetch_metadata(tid)}
        seg = compute_segment(entry["duration"], cfg.offset_ratio,
                              cfg.segment_seconds, cfg.min_duration)
        if seg is None:
            raise RuntimeError(f"too short ({entry['duration']}s)")
        start, end = seg
        if source == "spotify":
            from data.ingest.spotify import download_full
            audio = download_full(entry, store.tmp, cfg.spotdl_exe, ffmpeg, cfg.spotdl_format)
            real = audio_length(audio, ffmpeg)
            if abs(real - entry["duration"]) > cfg.duration_tolerance:
                raise RuntimeError(f"wrong match: downloaded {real:.0f}s vs "
                                   f"Spotify {entry['duration']:.0f}s")
            mel = audio_to_mel(audio, dataset_cfg, cfg.segment_seconds,
                               ffmpeg=ffmpeg, start=start)
        else:
            audio = download_segment(tid, start, end, store.tmp, ffmpeg, cfg.retries)
            mel = audio_to_mel(audio, dataset_cfg, cfg.segment_seconds, ffmpeg=ffmpeg)
        store.save({
            "track_id": tid, "source": source, "title": entry["title"],
            "channel": entry["channel"], "duration": entry["duration"],
            "start": round(start, 2), "end": round(end, 2), "path": f"mels/{tid}.npy",
            "genre": genre, "playlist": playlist,
        }, mel)
        return "ok"
    except Exception as exc:
        store.fail(tid, exc)
        return "failed"
    finally:
        if audio is not None:
            audio.unlink(missing_ok=True)
        if store.tmp.exists():
            for leftover in store.tmp.glob(f"{tid}.*"):
                leftover.unlink(missing_ok=True)
        time.sleep(random.uniform(cfg.sleep_min, cfg.sleep_max))


def parse_jobs(args: argparse.Namespace) -> list[tuple[str, str]]:
    """(genre, url) pairs from the positional URL + --genre and every --playlist GENRE=URL."""
    jobs: list[tuple[str, str]] = []
    if args.url:
        jobs.append(((args.genre or "").strip(), args.url))
    elif args.genre:
        sys.exit("--genre needs a playlist URL")
    for item in args.playlist:
        genre, sep, url = item.partition("=")
        if not sep or not genre.strip() or not url.strip():
            sys.exit(f"--playlist expects GENRE=URL, got {item!r}")
        jobs.append((genre.strip(), url.strip()))
    if not jobs:
        sys.exit("Give a playlist URL (with --genre) or one or more --playlist GENRE=URL")
    if not args.dry_run:
        for genre, url in jobs:
            if not genre:
                sys.exit(f"--genre is required (one playlist = one class label): {url}")
    return jobs


def ingest_playlist(url: str, genre: str, args, cfg, dataset_cfg, store: Store,
                    ffmpeg: str) -> tuple[int, int, int]:
    """Ingest one playlist. Returns (ok, skipped, failed)."""
    from tqdm import tqdm

    from data.ingest.segment import compute_segment
    from data.ingest.spotify import is_spotify_url
    from data.ingest.spotify import list_playlist as list_spotify
    from data.ingest.youtube import list_playlist

    source = "spotify" if is_spotify_url(url) else "youtube"
    if source == "spotify":
        entries = list_spotify(url, cfg.spotdl_exe, ffmpeg)
    else:
        entries = list_playlist(url)
    print(f"\n[{genre or '-'}] {url}\nPlaylist: {len(entries)} entries")

    if args.dry_run:
        for e in entries:
            seg = (compute_segment(e["duration"], cfg.offset_ratio,
                                   cfg.segment_seconds, cfg.min_duration)
                   if e["duration"] else None)
            plan = f"{seg[0]:.1f}-{seg[1]:.1f}s" if seg else "(sin duracion / no apta)"
            print(f"{e['id']}  {e['duration']}  {e['title'][:60]}  {plan}")
        return 0, 0, 0

    skipped = sum(1 for e in entries if e["id"] in store.done)
    todo = [e for e in entries if e["id"] not in store.done]
    clash = [e for e in entries if e["id"] in store.done
             and store.genres.get(e["id"]) not in ("", genre)]
    if clash:
        print(f"Aviso: {len(clash)} pistas ya estaban ingeridas con otro genero y conservan "
              f"su etiqueta original (p. ej. {clash[0]['title']!r}: "
              f"{store.genres[clash[0]['id']]!r}).")
    if args.limit is not None:
        todo = todo[: args.limit]

    ok = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool,             tqdm(total=len(todo), unit="song") as pbar:
        futures = [pool.submit(process_one, e, cfg, dataset_cfg, store, ffmpeg, source,
                               genre, url) for e in todo]
        try:
            for fut in as_completed(futures):
                if fut.result() == "ok":
                    ok += 1
                else:
                    failed += 1
                pbar.update(1)
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            raise
    return ok, skipped, failed


def main() -> None:
    from omegaconf import OmegaConf

    from data.ingest.segment import get_ffmpeg

    args = parse_args()
    jobs = parse_jobs(args)
    cfg = OmegaConf.load(args.config)
    dataset_cfg = OmegaConf.load(cfg.mel_dataset_cfg)
    t0 = time.time()
    ffmpeg = get_ffmpeg()
    store = Store(Path(args.out))

    ok = skipped = failed = 0
    try:
        for genre, url in jobs:
            o, s_, f_ = ingest_playlist(url, genre, args, cfg, dataset_cfg, store, ffmpeg)
            ok, skipped, failed = ok + o, skipped + s_, failed + f_
    except KeyboardInterrupt:
        print("\nInterrumpido; relanza el mismo comando para continuar.")
    finally:
        if store.tmp.exists():
            for f in store.tmp.glob("*"):
                f.unlink(missing_ok=True)
            try:
                store.tmp.rmdir()
            except OSError:
                pass

    if args.dry_run:
        return
    print(f"\nProcesados: {ok} | ya existentes: {skipped} | fallidos: {failed} | "
          f"tiempo: {time.time() - t0:.0f}s")
    if failed:
        print(f"Detalle de fallos en {store.failed}")


if __name__ == "__main__":
    main()
