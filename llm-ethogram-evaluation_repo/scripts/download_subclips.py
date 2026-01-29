#!/usr/bin/env python3
import re, time, random, subprocess
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

OUT_DIR = "clips"
FORMAT = 'bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]'
MERGE_FMT = "mp4"

@dataclass(frozen=True)
class ClipSpec:
    species: str
    url: str
    ranges: List[Tuple[str,str]]  # each is (start,end) duration exactly 120s

def parse_time_to_seconds(t: str) -> int:
    parts = t.strip().split(":")
    if len(parts)==2:
        mm, ss = parts
        return int(mm)*60 + int(ss)
    if len(parts)==3:
        hh, mm, ss = parts
        return int(hh)*3600 + int(mm)*60 + int(ss)
    raise ValueError(f"Bad time format: {t}")

def seconds_to_hhmmss(sec: int) -> str:
    hh = sec // 3600
    sec %= 3600
    mm = sec // 60
    ss = sec % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def hhmmss_compact(sec: int) -> str:
    hh = sec // 3600
    sec %= 3600
    mm = sec // 60
    ss = sec % 60
    return f"{hh:02d}{mm:02d}{ss:02d}"

def extract_youtube_id(url: str) -> str:
    m = re.search(r"[?&]v=([^&]+)", url)
    if not m:
        raise ValueError(f"Could not extract video id from URL: {url}")
    return m.group(1)

def run_cmd(cmd: List[str], max_retries: int = 5) -> None:
    base_delay = 2.0
    jitter = 1.5
    backoff = 2.0
    attempt = 0
    while True:
        try:
            time.sleep(base_delay + random.uniform(0, jitter))
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return
        except subprocess.CalledProcessError:
            attempt += 1
            if attempt >= max_retries:
                raise
            wait = (base_delay * (backoff ** attempt)) + random.uniform(0, jitter)
            time.sleep(wait)

def download_10s_subclips(spec: ClipSpec) -> None:
    vid = extract_youtube_id(spec.url)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    for (r_start, r_end) in spec.ranges:
        start_sec = parse_time_to_seconds(r_start)
        end_sec = parse_time_to_seconds(r_end)
        if end_sec - start_sec != 120:
            raise ValueError(f"Range must be 120s: {r_start}-{r_end}")
        for i in range(12):
            s0 = start_sec + i*10
            s1 = s0 + 10
            ts0 = seconds_to_hhmmss(s0)
            ts1 = seconds_to_hhmmss(s1)
            out_tmpl = str(Path(OUT_DIR) / f"{spec.species}_yt-{vid}_t{hhmmss_compact(s0)}-t{hhmmss_compact(s1)}.%(ext)s")
            cmd = [
                "yt-dlp", "--no-playlist",
                "--download-sections", f"*{ts0}-{ts1}",
                "-f", FORMAT,
                "--merge-output-format", MERGE_FMT,
                "-o", out_tmpl,
                spec.url,
            ]
            run_cmd(cmd)

if __name__ == "__main__":
    print("Edit this script or import it and call download_10s_subclips() with ClipSpec entries.")
