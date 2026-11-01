"""Server-side eGPU 3D Gaussian Splatting Web Viewer (FastAPI + WebSockets + CUDA/Triton).

Renders 3.4M+ Gaussians directly on the host discrete eGPU and streams progressive
low-latency interactive frames + high-resolution still refinements to any browser on the LAN.
"""

from __future__ import annotations

import asyncio
import json
import math
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from .renderer import GaussianSceneGPU


def resolve_splat_and_sparse(target: Path) -> tuple[Path, Path | None]:
    """Resolve a .ply file and optional COLMAP sparse reconstruction directory from a path."""
    target = target.expanduser().resolve()
    if target.is_file() and target.suffix.lower() == ".ply":
        # Check sibling/parent run layout: <run>/exports/*.ply -> <run>/colmap_dataset/sparse/0
        sparse_cand = target.parent.parent / "colmap_dataset" / "sparse" / "0"
        return target, (sparse_cand if sparse_cand.is_dir() else None)

    if target.is_dir():
        final_ply = target / "exports" / "splat_final.ply"
        if not final_ply.exists():
            plys = sorted(target.rglob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not plys:
                raise FileNotFoundError(f"No .ply splat files found inside {target}")
            final_ply = plys[0]
        sparse_cand = target / "colmap_dataset" / "sparse" / "0"
        if not sparse_cand.is_dir():
            alt_sparse = target / "sparse" / "0"
            sparse_cand = alt_sparse if alt_sparse.is_dir() else None
        return final_ply, sparse_cand

    raise FileNotFoundError(f"Path not found: {target}")


def detect_lan_ips() -> list[str]:
    """Return non-loopback IPv4 addresses of this machine for LAN/Tailscale browser access."""
    ips: list[str] = []
    try:
        out = subprocess.check_output(["hostname", "-I"], text=True, timeout=2.0).strip()
        for tok in out.split():
            if (
                "." in tok
                and not tok.startswith("127.")
                and not tok.startswith("172.17.")
                and not tok.startswith("192.168.122.")
                and not tok.startswith("192.168.125.")
            ):
                ips.append(tok)
    except Exception:
        pass
    if not ips:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ips.append(s.getsockname()[0])
        except Exception:
            ips.append("127.0.0.1")
    return ips


class GPUTelemetryCache:
    """Lightweight cached GPU telemetry (polled at most every 2s to avoid per-frame overhead)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
        self._last_poll = 0.0
        self._cached: dict[str, Any] = {}

    def get(self) -> dict[str, Any]:
        now = time.monotonic()
        if now - self._last_poll < 2.0 and self._cached:
            return self._cached
        self._last_poll = now
        vram_alloc_mb = round(torch.cuda.memory_allocated(self.device) / (1024**2), 1)
        vram_res_mb = round(torch.cuda.memory_reserved(self.device) / (1024**2), 1)
        info: dict[str, Any] = {
            "gpu_name": self.gpu_name,
            "vram_allocated_mb": vram_alloc_mb,
            "vram_reserved_mb": vram_res_mb,
        }
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=1.0,
            ).strip()
            first = out.splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 5:
                info.update(
                    {
                        "gpu_util_pct": float(parts[0]),
                        "gpu_mem_used_mb": float(parts[1]),
                        "gpu_mem_total_mb": float(parts[2]),
                        "gpu_temp_c": float(parts[3]),
                        "gpu_power_w": float(parts[4]),
                    }
                )
        except Exception:
            pass
        self._cached = info
        return info


def build_scene_navigation(
    scene: GaussianSceneGPU, sparse_dir: Path | None
) -> dict[str, Any]:
    """Compute world horizon basis (H1, H2, up_world), curated presets, and flight cameras."""
    center = np.asarray(scene.bounds_median, dtype=np.float64)
    q10 = np.asarray(scene.bounds_q10, dtype=np.float64)
    q90 = np.asarray(scene.bounds_q90, dtype=np.float64)
    scene_span = float(max(np.linalg.norm(q90 - q10), 1.0))

    cameras_out: list[dict[str, Any]] = []
    presets: list[dict[str, Any]] = []

    if sparse_dir is not None and sparse_dir.is_dir():
        try:
            rec = pycolmap.Reconstruction(str(sparse_dir))
            pts = (
                np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64)
                if len(rec.points3D) > 0
                else np.empty((0, 3), dtype=np.float64)
            )
            if len(pts) > 50:
                center = np.median(pts, axis=0)

            raw_cams = []
            nadir_fwds = []
            obl_downs = []
            for im in sorted(rec.images.values(), key=lambda x: x.name):
                T_cw = im.cam_from_world()
                R_cw = T_cw.rotation.matrix()
                pos = -R_cw.T @ T_cw.translation
                cam = rec.cameras[im.camera_id]
                fov_y = math.degrees(2.0 * math.atan((cam.height / 2.0) / cam.params[0]))
                np3d_attr = getattr(im, "num_points3D", 0)
                n_pts = int(np3d_attr() if callable(np3d_attr) else np3d_attr)
                raw_cams.append((im.name, pos, R_cw, fov_y, n_pts))
                if "nadir" in im.name.lower():
                    nadir_fwds.append(R_cw[2])
                else:
                    obl_downs.append(R_cw[1])

            if raw_cams:
                dists = np.array([np.linalg.norm(c[1] - center) for c in raw_cams])
                med_dist = float(max(np.median(dists), 1.0))
                inlier_cams = [c for c, d in zip(raw_cams, dists) if d <= max(med_dist * 5.0, 20.0)]
                if not inlier_cams:
                    inlier_cams = raw_cams

                nadir_in = [c[2][2] for c in inlier_cams if "nadir" in c[0].lower()]
                down_vec = (
                    np.mean(nadir_in, axis=0)
                    if nadir_in
                    else np.mean([c[2][1] for c in inlier_cams], axis=0)
                )
                up_world = -down_vec / max(np.linalg.norm(down_vec), 1e-8)

                ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                if abs(float(np.dot(ref, up_world))) > 0.9:
                    ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                H1 = ref - float(np.dot(ref, up_world)) * up_world
                H1 /= np.linalg.norm(H1)
                H2 = np.cross(H1, up_world)
                H2 /= np.linalg.norm(H2)

                for idx, (name, pos, R_cw, fov_y, n_pts) in enumerate(inlier_cams):
                    f = R_cw[2]
                    sin_p = float(np.clip(np.dot(f, up_world), -0.9998, 0.9998))
                    pitch = math.asin(sin_p)
                    pitch = max(math.radians(-89.2), min(math.radians(89.2), pitch))
                    if abs(sin_p) > 0.96:
                        up_scr = -R_cw[1]
                        yaw = math.atan2(float(np.dot(up_scr, H2)), float(np.dot(up_scr, H1)))
                    else:
                        yaw = math.atan2(float(np.dot(f, H2)), float(np.dot(f, H1)))

                    orbit_dist = 2.5
                    if len(pts) > 20:
                        pc = (pts - pos) @ R_cw.T
                        vis = (
                            (pc[:, 2] > 0.15)
                            & (np.abs(pc[:, 0]) < pc[:, 2] * 0.95)
                            & (np.abs(pc[:, 1]) < pc[:, 2] * 0.95)
                        )
                        if int(vis.sum()) >= 8:
                            orbit_dist = float(np.clip(np.percentile(pc[vis, 2], 45), 0.35, 18.0))

                    f_lvl = (
                        math.cos(pitch) * (math.cos(yaw) * H1 + math.sin(yaw) * H2)
                        + math.sin(pitch) * up_world
                    )
                    target_pt = pos + f_lvl * orbit_dist
                    alt = float(np.dot(pos - center, up_world))
                    cameras_out.append(
                        {
                            "index": idx,
                            "name": Path(name).with_suffix("").as_posix().replace("/", "_"),
                            "pos": [round(float(x), 5) for x in pos],
                            "target": [round(float(x), 5) for x in target_pt],
                            "yaw": round(yaw, 5),
                            "pitch": round(pitch, 5),
                            "dist": round(orbit_dist, 4),
                            "fov": round(min(max(fov_y, 55.0), 86.0), 2),
                            "alt": round(alt, 3),
                            "points": int(n_pts),
                            "is_nadir": "nadir" in name.lower(),
                        }
                    )

                # Build curated altitude-stratified presets (exclude skyward-tilted cameras)
                obliques = [
                    c for c in cameras_out if not c["is_nadir"] and c["pitch"] < -0.15
                ]
                nadirs = [c for c in cameras_out if c["is_nadir"]]
                pool_candidates = obliques if obliques else cameras_out
                alts_arr = np.array([c["alt"] for c in pool_candidates], dtype=np.float64)
                a_lo, a_mid1, a_mid2, a_hi = np.percentile(alts_arr, [15, 40, 70, 95])

                low_cands = [c for c in pool_candidates if c["alt"] <= a_mid1]
                core_center = (
                    np.mean([c["pos"] for c in low_cands], axis=0)
                    if low_cands
                    else center
                )

                def _pick_best(cands: list[dict], avoid_pos: list[np.ndarray] | None = None) -> dict | None:
                    if not cands:
                        return None
                    ranked = sorted(cands, key=lambda c: c["points"], reverse=True)
                    if avoid_pos:
                        for c in ranked:
                            p = np.asarray(c["pos"])
                            if all(np.linalg.norm(p - ap) > 1.2 for ap in avoid_pos):
                                return c
                    return ranked[0]

                chosen_positions: list[np.ndarray] = []

                high_obl = _pick_best([c for c in pool_candidates if a_mid2 <= c["alt"] <= a_hi + 0.5])
                if high_obl:
                    presets.append({**high_obl, "label": "Aerial Overview"})
                    chosen_positions.append(np.asarray(high_obl["pos"]))

                low_obl1 = _pick_best(low_cands, chosen_positions)
                if low_obl1:
                    presets.append({**low_obl1, "label": "Porch & Detail"})
                    chosen_positions.append(np.asarray(low_obl1["pos"]))

                low_obl2 = _pick_best(low_cands, chosen_positions)
                if low_obl2:
                    presets.append({**low_obl2, "label": "Patio & Pool"})
                    chosen_positions.append(np.asarray(low_obl2["pos"]))

                max_horiz = max(scene_span * 0.25, 3.5)

                def _near_core(cands: list[dict]) -> list[dict]:
                    near = [
                        c
                        for c in cands
                        if float(
                            np.linalg.norm(
                                (np.asarray(c["pos"]) - core_center)
                                - float(np.dot(np.asarray(c["pos"]) - core_center, up_world)) * up_world
                            )
                        )
                        <= max_horiz
                    ]
                    return near if near else cands

                mid_cands = [c for c in pool_candidates if a_mid1 < c["alt"] < a_mid2]
                mid_obl = _pick_best(_near_core(mid_cands), chosen_positions)
                if mid_obl:
                    presets.append({**mid_obl, "label": "Mid-Flight"})
                    chosen_positions.append(np.asarray(mid_obl["pos"]))

                if nadirs:
                    n_alts = np.array([c["alt"] for c in nadirs])
                    n_cut = float(np.percentile(n_alts, 55))
                    high_nads = [c for c in nadirs if c["alt"] >= n_cut]
                    high_nad = _pick_best(_near_core(high_nads))
                    if high_nad:
                        presets.append({**high_nad, "label": "Roof Nadir"})

                return {
                    "up_world": [float(x) for x in up_world],
                    "H1": [float(x) for x in H1],
                    "H2": [float(x) for x in H2],
                    "scene_center": [float(x) for x in center],
                    "scene_span": round(scene_span, 3),
                    "presets": presets,
                    "cameras": cameras_out,
                }
        except Exception as exc:
            print(f"[warn] Could not parse sparse cameras from {sparse_dir}: {exc}", flush=True)

    # Fallback when viewing a standalone .ply without COLMAP sparse/0
    up_world = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    H1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    H2 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dist = max(scene_span * 0.65, 2.0)
    yaw, pitch = 0.0, math.radians(-28.0)
    f_lvl = math.cos(pitch) * H1 + math.sin(pitch) * up_world
    pos = center - f_lvl * dist
    default_preset = {
        "index": 0,
        "label": "Scene Overview",
        "name": "overview",
        "pos": [float(x) for x in pos],
        "target": [float(x) for x in center],
        "yaw": yaw,
        "pitch": pitch,
        "dist": round(dist, 4),
        "fov": 72.0,
        "alt": round(dist * (-math.sin(pitch)), 3),
        "points": 0,
        "is_nadir": False,
    }
    return {
        "up_world": [float(x) for x in up_world],
        "H1": [float(x) for x in H1],
        "H2": [float(x) for x in H2],
        "scene_center": [float(x) for x in center],
        "scene_span": round(scene_span, 3),
        "presets": [default_preset],
        "cameras": [default_preset],
    }


VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
<title>insv-splat · Server eGPU 3DGS Viewer</title>
<style>
  :root {
    --brand-light: #005bb5;
    --brand-dark: #4cc2ff;
    --surface-light: #f4f6f9;
    --surface-dark: #0f131a;
    --panel-light: rgba(248, 250, 252, 0.92);
    --panel-dark: rgba(17, 22, 31, 0.88);
    --border-light: #64748b;
    --border-dark: #5a6b85;
    --text-light: #0f172a;
    --text-dark: #f1f5f9;
    --muted-light: #334155;
    --muted-dark: #cbd5e1;

    --accent-color: var(--brand-light);
    --surface-color: var(--surface-light);
    --panel-bg: var(--panel-light);
    --border-color: var(--border-light);
    --text-color: var(--text-light);
    --muted-color: var(--muted-light);

    @media (prefers-color-scheme: dark) {
      --accent-color: var(--brand-dark);
      --surface-color: var(--surface-dark);
      --panel-bg: var(--panel-dark);
      --border-color: var(--border-dark);
      --text-color: var(--text-dark);
      --muted-color: var(--muted-dark);
    }

    @supports (color: light-dark(white, black)) {
      --accent-color: light-dark(var(--brand-light), var(--brand-dark));
      --surface-color: light-dark(var(--surface-light), var(--surface-dark));
      --panel-bg: light-dark(var(--panel-light), var(--panel-dark));
      --border-color: light-dark(var(--border-light), var(--border-dark));
      --text-color: light-dark(var(--text-light), var(--text-dark));
      --muted-color: light-dark(var(--muted-light), var(--muted-dark));
    }

    color-scheme: light dark;
    accent-color: var(--accent-color);
    color: var(--text-color);
  }

  @media (forced-colors: active) {
    :root {
      --surface-color: Canvas;
      --panel-bg: Canvas;
      --border-color: CanvasText;
      --text-color: CanvasText;
      --muted-color: CanvasText;
      --accent-color: Highlight;
    }
  }

  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0a0d12;
  }

  /* Complex 3D media viewport forced into dark scheme with explicit inherited color tokens */
  .viewer-shell {
    color-scheme: only dark;
    background: var(--surface-color);
    color: var(--text-color);
    accent-color: var(--accent-color);
    position: relative;
    width: 100vw;
    height: 100vh;
    user-select: none;
    touch-action: none;
  }

  #viewport {
    display: block;
    width: 100%;
    height: 100%;
    cursor: grab;
    background: #0b0e14;
  }
  #viewport:active { cursor: grabbing; }

  .hud-top {
    position: absolute;
    top: 12px;
    left: 12px;
    right: 12px;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: flex-start;
    gap: 10px;
    pointer-events: none;
    z-index: 10;
  }

  .hud-badge-bar {
    pointer-events: auto;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    background: var(--panel-bg);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 7px 12px;
    backdrop-filter: blur(10px);
    font-size: 13px;
    line-height: 1.3;
  }

  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid var(--border-color);
    background: rgba(255, 255, 255, 0.06);
  }

  .metric {
    color: var(--muted-color);
    font-variant-numeric: tabular-nums;
  }
  .metric strong {
    color: var(--text-color);
    font-weight: 600;
  }

  .qual-tag {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 2px 7px;
    border-radius: 6px;
    border: 1px solid var(--accent-color);
    color: var(--text-color);
    background: rgba(76, 194, 255, 0.16);
  }

  .top-actions {
    pointer-events: auto;
    display: flex;
    gap: 8px;
  }

  button, select {
    font: inherit;
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-color);
    background: var(--panel-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 6px 11px;
    cursor: pointer;
    transition: background 120ms ease, border-color 120ms ease;
  }
  button:hover, select:hover {
    border-color: var(--accent-color);
    background: rgba(76, 194, 255, 0.14);
  }
  button:focus-visible, select:focus-visible, input:focus-visible {
    outline: 2px solid var(--accent-color);
    outline-offset: 2px;
  }
  button.active {
    background: rgba(76, 194, 255, 0.24);
    border-color: var(--accent-color);
  }

  .control-dock {
    position: absolute;
    bottom: 14px;
    left: 14px;
    width: min(370px, calc(100vw - 28px));
    max-height: calc(100vh - 90px);
    overflow-y: auto;
    background: var(--panel-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 12px 14px;
    backdrop-filter: blur(12px);
    z-index: 10;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .control-dock.collapsed {
    display: none;
  }

  .section-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted-color);
    margin: 0 0 5px 0;
  }

  .preset-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .preset-grid button {
    padding: 5px 9px;
    font-size: 12px;
  }

  .row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    font-size: 12.5px;
  }
  .row label {
    color: var(--muted-color);
    min-width: 88px;
  }
  .row input[type="range"] {
    flex: 1;
    cursor: pointer;
  }
  .row .val {
    min-width: 44px;
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
  }

  .stepper-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .stepper-row input[type="range"] { flex: 1; }

  .hint-bar {
    position: absolute;
    bottom: 14px;
    right: 14px;
    background: var(--panel-bg);
    border: 1px solid var(--border-color);
    border-radius: 9px;
    padding: 6px 11px;
    font-size: 12px;
    color: var(--muted-color);
    pointer-events: none;
    z-index: 9;
  }
  @media (max-width: 760px) {
    .hint-bar { display: none; }
  }

  dialog#shortcuts-dialog {
    color-scheme: only dark;
    background: #131924;
    color: var(--text-color);
    accent-color: var(--accent-color);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 18px 20px;
    max-width: 440px;
    width: calc(100vw - 32px);
  }
  dialog#shortcuts-dialog::backdrop {
    background: rgba(0, 0, 0, 0.62);
  }
  .shortcut-list {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 7px 14px;
    font-size: 13px;
    margin: 12px 0 16px 0;
  }
  kbd {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11.5px;
    padding: 2px 6px;
    border-radius: 5px;
    border: 1px solid var(--border-color);
    background: rgba(255, 255, 255, 0.08);
  }
</style>
</head>
<body>
<div class="viewer-shell" id="app">
  <canvas id="viewport"></canvas>

  <div class="hud-top">
    <div class="hud-badge-bar" role="status" aria-live="polite">
      <span class="status-pill" id="conn-pill">
        <span aria-hidden="true" id="conn-icon">◌</span>
        <span id="conn-text">Connecting…</span>
      </span>
      <span class="qual-tag" id="qual-badge">INIT</span>
      <span class="metric">Render <strong id="stat-ms">--</strong> ms</span>
      <span class="metric"><strong id="stat-fps">--</strong> FPS</span>
      <span class="metric">Splats <strong id="stat-splats">--</strong></span>
      <span class="metric" id="stat-gpu">eGPU</span>
    </div>

    <div class="top-actions">
      <button type="button" id="btn-toggle-dock" aria-expanded="true" aria-controls="control-dock">Controls</button>
      <button type="button" id="btn-snapshot" title="Save current rendered frame">Snapshot</button>
      <button type="button" id="btn-fullscreen" title="Toggle Fullscreen (F)">Fullscreen</button>
      <button type="button" commandfor="shortcuts-dialog" command="show-modal">Help</button>
    </div>
  </div>

  <aside class="control-dock" id="control-dock" aria-label="3DGS Camera and Render Controls">
    <div>
      <div class="section-title">Camera Bookmarks</div>
      <div class="preset-grid" id="preset-grid"></div>
    </div>

    <div id="flight-section">
      <div class="section-title">Flight Path (<span id="cam-label">1 / 1</span>)</div>
      <div class="stepper-row">
        <button type="button" id="btn-prev-cam" aria-label="Previous reconstructed camera">◀</button>
        <input type="range" id="cam-slider" min="0" max="0" value="0" aria-label="Flight camera timeline" />
        <button type="button" id="btn-next-cam" aria-label="Next reconstructed camera">▶</button>
      </div>
    </div>

    <div>
      <div class="section-title">Navigation &amp; Quality</div>
      <div class="preset-grid" style="margin-bottom:8px;">
        <button type="button" id="mode-orbit" class="active">Orbit Mode</button>
        <button type="button" id="mode-fly">Free Fly</button>
        <button type="button" id="btn-level">Level Horizon</button>
      </div>

      <div class="row">
        <label for="sel-quality">Stream Tier</label>
        <select id="sel-quality" style="flex:1;">
          <option value="balanced" selected>Balanced (720p → 1080p HD)</option>
          <option value="ultra">Ultra (900p → 1440p QHD)</option>
          <option value="fast">Fast Wi-Fi (540p → 1080p HD)</option>
        </select>
      </div>
    </div>

    <div class="row">
      <label for="rng-fov">Field of View</label>
      <input type="range" id="rng-fov" min="40" max="105" step="1" value="75" />
      <span class="val" id="val-fov">75°</span>
    </div>

    <div class="row">
      <label for="rng-scale">Splat Scale</label>
      <input type="range" id="rng-scale" min="0.30" max="1.40" step="0.02" value="1.00" />
      <span class="val" id="val-scale">1.00×</span>
    </div>

    <div class="row">
      <label for="rng-speed">Fly Speed</label>
      <input type="range" id="rng-speed" min="0.2" max="8.0" step="0.1" value="2.2" />
      <span class="val" id="val-speed">2.2</span>
    </div>

    <div class="row">
      <label for="sel-sh">Lighting</label>
      <select id="sel-sh">
        <option value="3" selected>SH Degree 3 (View-Dependent)</option>
        <option value="0">SH Degree 0 (Diffuse Base)</option>
      </select>
      <select id="sel-bg" aria-label="Sky background color">
        <option value="0.07,0.09,0.12" selected>Night Slate</option>
        <option value="0.72,0.82,0.92">Day Sky</option>
        <option value="0.0,0.0,0.0">Pure Black</option>
      </select>
    </div>
  </aside>

  <div class="hint-bar">
    Left-drag: Orbit/Look · Right/Shift-drag: Pan · Wheel/Pinch: Zoom · Double-click: Focus · WASD/QE: Fly
  </div>

  <dialog id="shortcuts-dialog" aria-labelledby="shortcuts-title">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <strong id="shortcuts-title" style="font-size:15px;">Controls &amp; Shortcuts</strong>
      <button type="button" commandfor="shortcuts-dialog" command="close">Close</button>
    </div>
    <div class="shortcut-list">
      <span><kbd>Left Drag</kbd></span><span>Orbit around target (or look around in Free Fly)</span>
      <span><kbd>Right Drag</kbd> / <kbd>Shift+Drag</kbd></span><span>Pan camera &amp; target across scene</span>
      <span><kbd>Scroll</kbd> / <kbd>Pinch</kbd></span><span>Zoom smoothly toward/away from target</span>
      <span><kbd>Double Click</kbd></span><span>Center orbit focus along clicked 3D ray</span>
      <span><kbd>W A S D</kbd> + <kbd>Q E</kbd></span><span>Free-fly forward/strafe &amp; down/up (<kbd>Shift</kbd> sprint)</span>
      <span><kbd>[</kbd> / <kbd>]</kbd></span><span>Step through reconstructed flight cameras</span>
      <span><kbd>R</kbd> / <kbd>H</kbd> / <kbd>F</kbd></span><span>Reset view / Hide controls / Fullscreen</span>
    </div>
  </dialog>
</div>

<script type="module">
  // Feature detect Invoker Commands API on HTMLButtonElement.prototype per modern-web-guidance
  if (!('commandForElement' in HTMLButtonElement.prototype)) {
    (async () => {
      try {
        await import('https://esm.run/invokers-polyfill');
      } catch {
        // Offline LAN manual fallback for declarative dialog commands
        document.addEventListener('click', (ev) => {
          const btn = ev.composedPath().find((el) => el.matches?.('button[commandfor]'));
          if (!btn) return;
          const dlg = document.getElementById(btn.getAttribute('commandfor'));
          const cmd = btn.getAttribute('command');
          if (!dlg) return;
          if (cmd === 'show-modal' && typeof dlg.showModal === 'function') dlg.showModal();
          else if (cmd === 'close' && typeof dlg.close === 'function') dlg.close();
        });
      }
    })();
  }

  const canvas = document.getElementById('viewport');
  const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
  const textDecoder = new TextDecoder();

  let sceneInfo = null;
  let U = [0, -1, 0], H1 = [1, 0, 0], H2 = [0, 0, 1];

  const cam = {
    mode: 'orbit', // 'orbit' | 'fly'
    pos: [0, 0, -3],
    target: [0, 0, 0],
    yaw: 0,
    pitch: -0.5,
    dist: 3.0,
    fov: 75.0,
    scale: 1.0,
    shDegree: 3,
    bg: [0.07, 0.09, 0.12],
    speed: 2.2,
  };

  const TIERS = {
    balanced: { liveMax: 1280, stillMax: 1920, liveQ: 82, stillQ: 95 },
    ultra:    { liveMax: 1600, stillMax: 2560, liveQ: 86, stillQ: 96 },
    fast:     { liveMax: 960,  stillMax: 1920, liveQ: 76, stillQ: 94 },
  };
  let currentTier = TIERS.balanced;

  // Vector helpers
  const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  const cross = (a, b) => [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0],
  ];
  const norm = (a) => {
    const l = Math.hypot(a[0], a[1], a[2]) || 1e-8;
    return [a[0]/l, a[1]/l, a[2]/l];
  };
  const addScaled = (a, b, s) => [a[0] + b[0]*s, a[1] + b[1]*s, a[2] + b[2]*s];

  function computeBasis() {
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    const horiz = [cy*H1[0] + sy*H2[0], cy*H1[1] + sy*H2[1], cy*H1[2] + sy*H2[2]];
    const fwd = norm([cp*horiz[0] + sp*U[0], cp*horiz[1] + sp*U[1], cp*horiz[2] + sp*U[2]]);
    let right = cross(fwd, U);
    const rl = Math.hypot(right[0], right[1], right[2]);
    right = rl > 1e-6 ? [right[0]/rl, right[1]/rl, right[2]/rl] : cross(horiz, U);
    const down = norm(cross(fwd, right));
    return { fwd, right, down, horiz };
  }

  function syncPosFromOrbit() {
    const { fwd } = computeBasis();
    cam.pos = addScaled(cam.target, fwd, -cam.dist);
  }

  function syncTargetFromFly() {
    const { fwd } = computeBasis();
    cam.target = addScaled(cam.pos, fwd, cam.dist);
  }

  function applyBookmark(bm) {
    if (!bm) return;
    cam.pos = [...bm.pos];
    cam.target = [...bm.target];
    cam.yaw = bm.yaw;
    cam.pitch = bm.pitch;
    cam.dist = bm.dist;
    if (bm.fov) {
      cam.fov = bm.fov;
      document.getElementById('rng-fov').value = Math.round(cam.fov);
      document.getElementById('val-fov').textContent = `${Math.round(cam.fov)}°`;
    }
    if (cam.mode === 'orbit') syncPosFromOrbit();
    else syncTargetFromFly();
    markDirty(false);
  }

  // Streaming state with 1-in-flight backpressure coalescing
  let ws = null;
  let seqCounter = 0;
  let inFlight = false;
  let pendingInteractive = false;
  let pendingHighRes = false;
  let refineTimer = null;
  let lastFrameBlob = null;
  const frameTimes = [];

  function setConnStatus(state, label) {
    const icon = document.getElementById('conn-icon');
    const txt = document.getElementById('conn-text');
    icon.textContent = state === 'live' ? '●' : state === 'busy' ? '◉' : '✕';
    icon.style.color = state === 'live' ? '#34d399' : state === 'busy' ? '#4cc2ff' : '#f87171';
    txt.textContent = label;
  }

  function computeRenderDims(maxLongEdge) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rawW = Math.max(320, Math.round(canvas.clientWidth * dpr));
    const rawH = Math.max(240, Math.round(canvas.clientHeight * dpr));
    const scale = Math.min(1.0, maxLongEdge / Math.max(rawW, rawH));
    const w = Math.max(160, Math.round((rawW * scale) / 8) * 8);
    const h = Math.max(120, Math.round((rawH * scale) / 8) * 8);
    return { w, h };
  }

  function markDirty(highResOnly = false) {
    if (highResOnly) {
      pendingHighRes = true;
    } else {
      pendingInteractive = true;
      pendingHighRes = false;
      clearTimeout(refineTimer);
      refineTimer = setTimeout(() => {
        pendingHighRes = true;
        flushSend();
      }, 140);
    }
    flushSend();
  }

  function flushSend() {
    if (!ws || ws.readyState !== WebSocket.OPEN || inFlight) return;
    if (!pendingInteractive && !pendingHighRes) return;

    const isHigh = !pendingInteractive && pendingHighRes;
    pendingInteractive = false;
    if (isHigh) pendingHighRes = false;

    const { w, h } = computeRenderDims(isHigh ? currentTier.stillMax : currentTier.liveMax);
    const { fwd, right, down } = computeBasis();
    const payload = {
      seq: ++seqCounter,
      pos: cam.pos,
      R_cw: [right, down, fwd],
      width: w,
      height: h,
      fov_y: cam.fov,
      sh_degree: cam.shDegree,
      scale_modifier: cam.scale,
      bg_color: cam.bg,
      jpeg_quality: isHigh ? currentTier.stillQ : currentTier.liveQ,
      quality_tag: isHigh ? 'still' : 'live',
    };
    inFlight = true;
    ws.send(JSON.stringify(payload));
  }

  function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      inFlight = false;
      setConnStatus('live', 'Live eGPU');
      markDirty(false);
    };

    ws.onmessage = async (ev) => {
      if (!(ev.data instanceof ArrayBuffer)) return;
      const buf = ev.data;
      const dv = new DataView(buf);
      const hdrLen = dv.getUint32(0, true);
      const meta = JSON.parse(textDecoder.decode(new Uint8Array(buf, 4, hdrLen)));
      const jpgBytes = new Uint8Array(buf, 4 + hdrLen);
      const blob = new Blob([jpgBytes], { type: 'image/jpeg' });
      lastFrameBlob = blob;

      try {
        const bmp = await createImageBitmap(blob);
        if (canvas.width !== bmp.width || canvas.height !== bmp.height) {
          canvas.width = bmp.width;
          canvas.height = bmp.height;
        }
        ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
        bmp.close();
      } catch {}

      const now = performance.now();
      frameTimes.push(now);
      while (frameTimes.length > 20 && now - frameTimes[0] > 1500) frameTimes.shift();
      const fps = frameTimes.length > 1
        ? ((frameTimes.length - 1) * 1000) / (frameTimes[frameTimes.length - 1] - frameTimes[0])
        : (1000 / Math.max(meta.render_ms, 1));

      document.getElementById('stat-ms').textContent = meta.render_ms.toFixed(1);
      document.getElementById('stat-fps').textContent = Math.round(fps);
      document.getElementById('stat-splats').textContent = (meta.splats / 1000).toFixed(0) + 'k';
      const qBadge = document.getElementById('qual-badge');
      qBadge.textContent = `${meta.h}p ${meta.quality === 'still' ? 'HD REFINE' : 'LIVE'}`;
      if (meta.gpu_temp_c !== undefined) {
        document.getElementById('stat-gpu').innerHTML =
          `<strong>${meta.gpu_short}</strong> · ${Math.round(meta.gpu_temp_c)}°C · ${(meta.vram_mb / 1024).toFixed(1)} GB`;
      }

      inFlight = false;
      flushSend();
    };

    ws.onclose = () => {
      inFlight = false;
      setConnStatus('off', 'Reconnecting…');
      setTimeout(connectWS, 1000);
    };
  }

  // Pointer & Multi-Touch gestures
  const pointers = new Map();
  let lastPinchDist = 0;
  let lastMid = null;

  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, buttons: e.buttons });
    if (pointers.size === 2) {
      const pts = [...pointers.values()];
      lastPinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      lastMid = { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 };
    }
  });

  canvas.addEventListener('pointermove', (e) => {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    const dx = e.clientX - prev.x;
    const dy = e.clientY - prev.y;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, buttons: e.buttons });

    if (pointers.size === 2) {
      const pts = [...pointers.values()];
      const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      const mid = { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 };
      if (lastPinchDist > 10) {
        const ratio = lastPinchDist / Math.max(dist, 10);
        cam.dist = Math.max(0.08, Math.min(45.0, cam.dist * ratio));
      }
      if (lastMid) {
        const panScale = cam.dist * 0.0022;
        const { right, down } = computeBasis();
        const mdx = mid.x - lastMid.x, mdy = mid.y - lastMid.y;
        cam.target = addScaled(addScaled(cam.target, right, -mdx * panScale), down, -mdy * panScale);
      }
      syncPosFromOrbit();
      lastPinchDist = dist;
      lastMid = mid;
      markDirty(false);
      return;
    }

    const isPan = (e.buttons & 2) || (e.buttons & 4) || e.shiftKey;
    if (isPan) {
      const panScale = Math.max(cam.dist, 0.4) * 0.0018;
      const { right, down } = computeBasis();
      cam.target = addScaled(addScaled(cam.target, right, -dx * panScale), down, -dy * panScale);
      cam.pos = addScaled(addScaled(cam.pos, right, -dx * panScale), down, -dy * panScale);
    } else if (e.buttons & 1) {
      const sens = 0.0042;
      cam.yaw += dx * sens;
      cam.pitch = Math.max(-1.555, Math.min(1.555, cam.pitch - dy * sens));
      if (cam.mode === 'orbit') syncPosFromOrbit();
      else syncTargetFromFly();
    } else {
      return;
    }
    markDirty(false);
  });

  const endPtr = (e) => {
    pointers.delete(e.pointerId);
    lastPinchDist = 0;
    lastMid = null;
  };
  canvas.addEventListener('pointerup', endPtr);
  canvas.addEventListener('pointercancel', endPtr);

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = Math.exp(e.deltaY * 0.0011);
    if (cam.mode === 'orbit') {
      cam.dist = Math.max(0.08, Math.min(45.0, cam.dist * factor));
      syncPosFromOrbit();
    } else {
      const { fwd } = computeBasis();
      const step = -e.deltaY * 0.003 * cam.speed;
      cam.pos = addScaled(cam.pos, fwd, step);
      syncTargetFromFly();
    }
    markDirty(false);
  }, { passive: false });

  // Double-click to re-center orbit target along clicked screen ray
  canvas.addEventListener('dblclick', (e) => {
    const rect = canvas.getBoundingClientRect();
    const nx = ((e.clientX - rect.left) / rect.width - 0.5) * 2.0;
    const ny = ((e.clientY - rect.top) / rect.height - 0.5) * 2.0;
    const tanHalfY = Math.tan((cam.fov * Math.PI) / 360.0);
    const tanHalfX = tanHalfY * (rect.width / Math.max(rect.height, 1));
    const { fwd, right, down } = computeBasis();
    const ray = norm([
      fwd[0] + right[0] * nx * tanHalfX + down[0] * ny * tanHalfY,
      fwd[1] + right[1] * nx * tanHalfX + down[1] * ny * tanHalfY,
      fwd[2] + right[2] * nx * tanHalfX + down[2] * ny * tanHalfY,
    ]);
    // Intersect with plane passing through scene_center normal to U, fallback to 0.85 * dist
    let t = cam.dist * 0.85;
    const denom = dot(ray, U);
    if (Math.abs(denom) > 0.08 && sceneInfo?.navigation?.scene_center) {
      const planePt = sceneInfo.navigation.scene_center;
      const tPlane = dot([planePt[0] - cam.pos[0], planePt[1] - cam.pos[1], planePt[2] - cam.pos[2]], U) / denom;
      if (tPlane > 0.15 && tPlane < 35.0) t = tPlane;
    }
    cam.target = addScaled(cam.pos, ray, t);
    cam.yaw = Math.atan2(dot(ray, H2), dot(ray, H1));
    cam.pitch = Math.max(-1.555, Math.min(1.555, Math.asin(Math.max(-1, Math.min(1, dot(ray, U))))));
    cam.dist = t;
    cam.mode = 'orbit';
    updateModeUI();
    syncPosFromOrbit();
    markDirty(false);
  });

  // 60 Hz WASD + QE keyboard loop
  const keys = new Set();
  window.addEventListener('keydown', (e) => {
    if (e.target.matches('input, select, textarea')) return;
    const k = e.key.toLowerCase();
    if (['w', 'a', 's', 'd', 'q', 'e', 'c', ' ', 'shift'].includes(k)) {
      keys.add(k);
      if (k === ' ') e.preventDefault();
    } else if (k === 'r' && sceneInfo?.navigation?.presets?.[0]) {
      applyBookmark(sceneInfo.navigation.presets[0]);
    } else if (k === 'h') {
      toggleDock();
    } else if (k === 'f') {
      toggleFullscreen();
    } else if (k === '[' || k === ']') {
      stepCamera(k === ']' ? 1 : -1);
    }
  });
  window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
  window.addEventListener('blur', () => keys.clear());

  let lastTick = performance.now();
  function flyTick(now) {
    const dt = Math.min(0.05, (now - lastTick) / 1000);
    lastTick = now;
    if (keys.size > 0) {
      const mult = keys.has('shift') ? 2.8 : 1.0;
      const step = cam.speed * mult * dt;
      const { fwd, right } = computeBasis();
      let moved = false;
      let delta = [0, 0, 0];
      if (keys.has('w')) { delta = addScaled(delta, fwd, step); moved = true; }
      if (keys.has('s')) { delta = addScaled(delta, fwd, -step); moved = true; }
      if (keys.has('d')) { delta = addScaled(delta, right, step); moved = true; }
      if (keys.has('a')) { delta = addScaled(delta, right, -step); moved = true; }
      if (keys.has('e') || keys.has(' ')) { delta = addScaled(delta, U, step); moved = true; }
      if (keys.has('q') || keys.has('c')) { delta = addScaled(delta, U, -step); moved = true; }
      if (moved) {
        cam.pos = addScaled(cam.pos, delta, 1.0);
        cam.target = addScaled(cam.target, delta, 1.0);
        markDirty(false);
      }
    }
    requestAnimationFrame(flyTick);
  }
  requestAnimationFrame(flyTick);

  // UI Bindings
  function updateModeUI() {
    document.getElementById('mode-orbit').classList.toggle('active', cam.mode === 'orbit');
    document.getElementById('mode-fly').classList.toggle('active', cam.mode === 'fly');
  }
  document.getElementById('mode-orbit').onclick = () => { cam.mode = 'orbit'; syncTargetFromFly(); updateModeUI(); };
  document.getElementById('mode-fly').onclick = () => { cam.mode = 'fly'; syncTargetFromFly(); updateModeUI(); };
  document.getElementById('btn-level').onclick = () => {
    cam.pitch = Math.max(-0.6, Math.min(0.2, cam.pitch));
    if (cam.mode === 'orbit') syncPosFromOrbit(); else syncTargetFromFly();
    markDirty(false);
  };

  function toggleDock() {
    const dock = document.getElementById('control-dock');
    const btn = document.getElementById('btn-toggle-dock');
    const hidden = dock.classList.toggle('collapsed');
    btn.setAttribute('aria-expanded', String(!hidden));
  }
  document.getElementById('btn-toggle-dock').onclick = toggleDock;

  function toggleFullscreen() {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
    else document.exitFullscreen?.();
  }
  document.getElementById('btn-fullscreen').onclick = toggleFullscreen;

  document.getElementById('btn-snapshot').onclick = () => {
    if (!lastFrameBlob) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(lastFrameBlob);
    a.download = `splat_render_${Date.now()}.jpg`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  };

  document.getElementById('sel-quality').onchange = (e) => {
    currentTier = TIERS[e.target.value] || TIERS.balanced;
    markDirty(true);
  };
  document.getElementById('rng-fov').oninput = (e) => {
    cam.fov = Number(e.target.value);
    document.getElementById('val-fov').textContent = `${cam.fov}°`;
    markDirty(false);
  };
  document.getElementById('rng-scale').oninput = (e) => {
    cam.scale = Number(e.target.value);
    document.getElementById('val-scale').textContent = `${cam.scale.toFixed(2)}×`;
    markDirty(false);
  };
  document.getElementById('rng-speed').oninput = (e) => {
    cam.speed = Number(e.target.value);
    document.getElementById('val-speed').textContent = cam.speed.toFixed(1);
  };
  document.getElementById('sel-sh').onchange = (e) => {
    cam.shDegree = Number(e.target.value);
    markDirty(true);
  };
  document.getElementById('sel-bg').onchange = (e) => {
    cam.bg = e.target.value.split(',').map(Number);
    markDirty(true);
  };

  function stepCamera(delta) {
    const cams = sceneInfo?.navigation?.cameras;
    if (!cams?.length) return;
    const slider = document.getElementById('cam-slider');
    const next = (Number(slider.value) + delta + cams.length) % cams.length;
    slider.value = next;
    const c = cams[next];
    document.getElementById('cam-label').textContent = `${next + 1} / ${cams.length} · ${c.name}`;
    applyBookmark(c);
  }
  document.getElementById('btn-prev-cam').onclick = () => stepCamera(-1);
  document.getElementById('btn-next-cam').onclick = () => stepCamera(1);
  document.getElementById('cam-slider').oninput = (e) => {
    const cams = sceneInfo?.navigation?.cameras;
    if (!cams?.length) return;
    const idx = Number(e.target.value);
    const c = cams[idx];
    document.getElementById('cam-label').textContent = `${idx + 1} / ${cams.length} · ${c.name}`;
    applyBookmark(c);
  };

  new ResizeObserver(() => markDirty(false)).observe(canvas);

  // Initialize from server /api/info
  const res = await fetch('/api/info');
  sceneInfo = await res.json();
  U = sceneInfo.navigation.up_world;
  H1 = sceneInfo.navigation.H1;
  H2 = sceneInfo.navigation.H2;

  const pGrid = document.getElementById('preset-grid');
  for (const p of sceneInfo.navigation.presets) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = p.label;
    b.onclick = () => applyBookmark(p);
    pGrid.appendChild(b);
  }

  const cams = sceneInfo.navigation.cameras || [];
  if (cams.length > 1) {
    const slider = document.getElementById('cam-slider');
    slider.max = cams.length - 1;
    document.getElementById('cam-label').textContent = `1 / ${cams.length} · ${cams[0].name}`;
  } else {
    document.getElementById('flight-section').style.display = 'none';
  }

  if (sceneInfo.navigation.presets.length > 0) {
    applyBookmark(sceneInfo.navigation.presets[0]);
  }
  connectWS();
</script>
</body>
</html>
"""


def create_app(
    ply_path: Path,
    sparse_dir: Path | None = None,
    device: str = "cuda:0",
    port: int = 8080,
) -> FastAPI:
    """Create the FastAPI + WebSocket server with a GPU-resident 3DGS scene."""
    print(f"[insv-splat serve] Loading {ply_path} onto {device}...", flush=True)
    scene = GaussianSceneGPU(ply_path, device=device)
    scene.warmup()
    telemetry = GPUTelemetryCache(scene.device)
    nav = build_scene_navigation(scene, sparse_dir)
    gpu_lock = asyncio.Lock()
    lan_ips = detect_lan_ips()

    gpu_init = telemetry.get()
    gpu_short = gpu_init.get("gpu_name", "CUDA GPU").replace("NVIDIA GeForce ", "")
    print(
        f"[insv-splat serve] Ready: {scene.num_gaussians:,} active Gaussians "
        f"(SH{scene.sh_degree}) loaded in {scene.load_time_s}s on {gpu_init.get('gpu_name')}",
        flush=True,
    )
    for ip in lan_ips:
        print(f"  -> LAN Browser URL: http://{ip}:{port}", flush=True)

    app = FastAPI(title="insv-splat Server-Side eGPU 3DGS Viewer")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(VIEWER_HTML)

    @app.get("/api/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "num_gaussians": scene.num_gaussians})

    @app.get("/api/info")
    async def api_info() -> JSONResponse:
        return JSONResponse(
            {
                "ply_path": str(ply_path),
                "sparse_dir": str(sparse_dir) if sparse_dir else None,
                "num_gaussians": scene.num_gaussians,
                "sh_degree": scene.sh_degree,
                "load_time_s": scene.load_time_s,
                "gpu": telemetry.get(),
                "lan_urls": [f"http://{ip}:{port}" for ip in lan_ips],
                "navigation": nav,
            }
        )

    @app.websocket("/ws")
    async def websocket_stream(ws: WebSocket) -> None:
        await ws.accept()
        latest_req: dict[str, Any] | None = None
        wake_event = asyncio.Event()
        closed = False

        async def _receiver() -> None:
            nonlocal latest_req, closed
            try:
                while True:
                    raw = await ws.receive_text()
                    latest_req = json.loads(raw)
                    wake_event.set()
            except WebSocketDisconnect:
                closed = True
                wake_event.set()
            except Exception:
                closed = True
                wake_event.set()

        recv_task = asyncio.create_task(_receiver())
        try:
            while not closed:
                await wake_event.wait()
                wake_event.clear()
                if closed or latest_req is None:
                    break

                req = latest_req
                latest_req = None

                width = int(max(160, min(2560, req.get("width", 1280))))
                height = int(max(120, min(1600, req.get("height", 720))))
                fov_y = float(max(25.0, min(120.0, req.get("fov_y", 75.0))))
                sh_deg = int(max(0, min(3, req.get("sh_degree", 3))))
                scale_mod = float(max(0.1, min(3.0, req.get("scale_modifier", 1.0))))
                jpeg_q = int(max(40, min(98, req.get("jpeg_quality", 84))))
                bg = tuple(float(x) for x in req.get("bg_color", [0.07, 0.09, 0.12])[:3])
                cam_pos = np.asarray(req["pos"], dtype=np.float32)
                R_cw = np.asarray(req["R_cw"], dtype=np.float32)

                async with gpu_lock:
                    rgb_gpu, stats = scene.render(
                        cam_pos=cam_pos,
                        R_cw=R_cw,
                        width=width,
                        height=height,
                        fov_y_deg=fov_y,
                        sh_degree=sh_deg,
                        scale_modifier=scale_mod,
                        bg_color=bg,  # type: ignore[arg-type]
                    )
                    rgb_np = rgb_gpu.cpu().numpy()

                t_enc0 = time.perf_counter()
                bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                ok, jpg_buf = cv2.imencode(".jpg", bgr_np, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
                enc_ms = round((time.perf_counter() - t_enc0) * 1000.0, 2)
                if not ok:
                    continue

                gpu_stat = telemetry.get()
                hdr = json.dumps(
                    {
                        "seq": req.get("seq", 0),
                        "w": width,
                        "h": height,
                        "quality": req.get("quality_tag", "live"),
                        "render_ms": stats["render_ms"],
                        "encode_ms": enc_ms,
                        "splats": stats["visible_splats"],
                        "gpu_short": gpu_short,
                        "gpu_temp_c": gpu_stat.get("gpu_temp_c", 0),
                        "vram_mb": gpu_stat.get("gpu_mem_used_mb", gpu_stat.get("vram_reserved_mb", 0)),
                    },
                    separators=(",", ":"),
                ).encode("utf-8")

                packet = struct.pack("<I", len(hdr)) + hdr + jpg_buf.tobytes()
                await ws.send_bytes(packet)
        except WebSocketDisconnect:
            pass
        finally:
            recv_task.cancel()

    return app


def run_web_server(
    target_path: Path,
    host: str = "0.0.0.0",
    port: int = 8080,
    device: str = "cuda:0",
) -> int:
    """Resolve run/PLY path and start the Uvicorn server."""
    ply_path, sparse_dir = resolve_splat_and_sparse(target_path)
    app = create_app(ply_path=ply_path, sparse_dir=sparse_dir, device=device, port=port)
    uvicorn.run(app, host=host, port=port, log_level="info", ws_max_size=16 * 1024 * 1024)
    return 0
