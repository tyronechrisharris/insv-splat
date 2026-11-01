"""GPU-accelerated motion-gated & sharpness-ranked keyframe selection for .insv videos."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class KeyframeCandidate:
    """Scored candidate timestamp in an .insv stream."""

    timestamp_s: float
    sharpness: float
    motion_from_prev: float


def _decode_thumbnail(
    insv_path: Path,
    ts: float,
    thumb_size: int = 256,
    use_cuda: bool = True,
) -> np.ndarray | None:
    """Fast I-frame seek decode of Stream 0 center region to grayscale numpy array."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if use_cuda:
        cmd += ["-hwaccel", "cuda"]
    cmd += [
        "-ss",
        f"{ts:.3f}",
        "-i",
        str(insv_path),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-vf",
        f"scale={thumb_size}:{thumb_size}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    res = subprocess.run(cmd, capture_output=True, check=False)
    expected = thumb_size * thumb_size
    if res.returncode != 0 or len(res.stdout) < expected:
        if use_cuda:
            return _decode_thumbnail(insv_path, ts, thumb_size=thumb_size, use_cuda=False)
        return None
    return np.frombuffer(res.stdout[:expected], dtype=np.uint8).reshape((thumb_size, thumb_size))


def select_keyframes(
    insv_path: Path,
    duration_s: float,
    max_keyframes: int = 110,
    probe_interval_s: float = 2.0,
    min_motion_score: float = 4.5,
    min_time_gap_s: float = 0.75,
    local_sharpness_ratio: float = 0.68,
    workers: int = 8,
    use_cuda: bool = True,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> tuple[list[KeyframeCandidate], dict]:
    """Scan video at coarse/fine intervals, skip stationary/hovering & blurry frames, and pick top keyframes.

    Returns:
        (selected_keyframes, stats_dict)
    """
    t_end = min(duration_s - 0.5, end_s) if end_s is not None else max(0.5, duration_s - 0.5)
    t_start = max(0.0, start_s)
    probe_times = np.arange(t_start, t_end, probe_interval_s, dtype=np.float64).tolist()
    if not probe_times:
        probe_times = [t_start]

    thumb_size = 256
    # Circular mask inside fisheye active area (radius <= 0.82) so black corners don't skew metrics
    yy, xx = np.ogrid[:thumb_size, :thumb_size]
    c = (thumb_size - 1) / 2.0
    valid_mask = ((xx - c) ** 2 + (yy - c) ** 2) <= (0.82 * c) ** 2

    decoded: dict[float, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_decode_thumbnail, insv_path, ts, thumb_size, use_cuda): ts
            for ts in probe_times
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Probing INSV motion/sharpness"):
            ts = futs[fut]
            arr = fut.result()
            if arr is not None:
                decoded[ts] = arr

    ordered_ts = sorted(decoded.keys())
    if not ordered_ts:
        raise RuntimeError(f"Failed to decode any thumbnails from {insv_path}")

    # Compute Laplacian sharpness and cumulative visual motion along timeline
    all_candidates: list[KeyframeCandidate] = []
    active_pool: list[KeyframeCandidate] = []
    skipped_stationary = 0

    anchor_img = cv2.GaussianBlur(decoded[ordered_ts[0]], (5, 5), 0).astype(np.float32)
    anchor_ts = ordered_ts[0]
    lap0 = cv2.Laplacian(decoded[ordered_ts[0]], cv2.CV_32F)
    sharp0 = float(np.var(lap0[valid_mask]))
    first_cand = KeyframeCandidate(timestamp_s=ordered_ts[0], sharpness=sharp0, motion_from_prev=999.0)
    all_candidates.append(first_cand)
    active_pool.append(first_cand)

    for ts in ordered_ts[1:]:
        gray = decoded[ts]
        blur = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        motion = float(np.mean(np.abs(blur[valid_mask] - anchor_img[valid_mask])))
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        sharp = float(np.var(lap[valid_mask]))
        cand = KeyframeCandidate(timestamp_s=ts, sharpness=sharp, motion_from_prev=motion)
        all_candidates.append(cand)

        if motion >= min_motion_score and (ts - anchor_ts) >= min_time_gap_s:
            active_pool.append(cand)
            anchor_img = blur
            anchor_ts = ts
        else:
            skipped_stationary += 1

    # Reject instantaneous yaw/shutter motion-blurred frames using a sliding local median window
    # so smooth low-altitude concrete/stucco scenes are never unfairly penalized vs forest scenes.
    dropped_blur = 0
    if len(active_pool) > max_keyframes and local_sharpness_ratio > 0.0:
        sharp_arr = np.array([c.sharpness for c in active_pool], dtype=np.float64)
        win_rad = 6
        locally_clean: list[KeyframeCandidate] = []
        for idx, cand in enumerate(active_pool):
            lo = max(0, idx - win_rad)
            hi = min(len(active_pool), idx + win_rad + 1)
            local_med = float(np.median(sharp_arr[lo:hi]))
            if cand.sharpness >= local_sharpness_ratio * local_med:
                locally_clean.append(cand)
            else:
                dropped_blur += 1
        if len(locally_clean) >= max_keyframes:
            active_pool = locally_clean
        else:
            dropped_blur = 0

    # Partition active_pool into `max_keyframes` uniform motion-arc buckets and pick sharpest in each
    if len(active_pool) <= max_keyframes:
        selected = active_pool
    else:
        selected = []
        edges = np.linspace(0, len(active_pool), max_keyframes + 1, dtype=int)
        for b in range(max_keyframes):
            chunk = active_pool[edges[b] : edges[b + 1]]
            if chunk:
                best = max(chunk, key=lambda c: c.sharpness)
                selected.append(best)

    selected.sort(key=lambda c: c.timestamp_s)
    stats = {
        "probed_frames": len(ordered_ts),
        "active_motion_frames": len(active_pool),
        "skipped_stationary_frames": skipped_stationary,
        "dropped_local_blur_frames": dropped_blur,
        "selected_keyframes": len(selected),
        "probe_interval_s": probe_interval_s,
        "min_motion_score": min_motion_score,
        "time_span_s": [round(selected[0].timestamp_s, 2), round(selected[-1].timestamp_s, 2)],
    }
    return selected, stats
