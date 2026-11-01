"""Fisheye-to-Perspective (Mei omnidirectional / equidistant) virtual camera rig projection."""

from __future__ import annotations

import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .keyframes import KeyframeCandidate
from .metadata import InsvMetadata, MeiLensCalibration


@dataclass(frozen=True)
class VirtualViewSpec:
    """Specification of a single virtual pinhole camera inside a dual-fisheye sphere."""

    name: str
    stream_index: int  # 0 (nadir/front) or 1 (zenith/back)
    tilt_deg: float  # Tilt from Stream 0 optical axis when spherical_s0_frame=True, else from stream axis
    azim_deg: float
    fov_deg: float
    require_texture: bool = False
    spherical_s0_frame: bool = True  # Express camera orientation in Stream 0's right-side-up spherical frame
    in_primary_rig: bool = True  # True if camera shares Stream 0's exact optical center


@dataclass(frozen=True)
class ExtractedView:
    """Metadata for a single rendered distortion-free perspective image."""

    image_name: str
    keyframe_index: int
    timestamp_s: float
    view_name: str
    stream_index: int
    descriptor: list[float]


def compute_view_rotation_s0(spec: VirtualViewSpec) -> np.ndarray:
    """Return exact 3x3 rotation matrix mapping virtual camera rays into Stream 0's coordinate frame."""
    t = math.radians(spec.tilt_deg)
    a = math.radians(spec.azim_deg)
    rt = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(t), -math.sin(t)], [0.0, math.sin(t), math.cos(t)]],
        dtype=np.float64,
    )
    ra = np.array(
        [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return ra @ rt


def get_rig_specs(
    rig_type: str,
    fov_deg: float = 84.0,
    include_upper_views: bool = False,
    dense_azimuth: bool = True,
) -> list[VirtualViewSpec]:
    """Build virtual pinhole camera rig specs tailored to drone (nadir/zenith) or handheld 360."""
    if rig_type == "drone":
        # Analytical spherical aerial photogrammetry rig on Stream 0.
        # At 45.5 deg tilt + 84 deg FOV, max off-axis angle is 87.5 deg (zero rim/duct guard intrusion).
        az_list = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0) if dense_azimuth else (0.0, 90.0, 180.0, 270.0)
        specs = [VirtualViewSpec("nadir", 0, 0.0, 0.0, fov_deg, False, True, True)]
        for az in az_list:
            specs.append(
                VirtualViewSpec(f"obl{int(az):03d}", 0, 45.5, az, fov_deg, False, True, True)
            )
        if include_upper_views:
            # Upright Stream 1 horizon/upper-facade views (132.5 deg from nadir = 47.5 deg from zenith,
            # max off-axis on Lens 1 = 47.5 + 42.0 = 89.5 deg -> zero black border, right-side up!).
            for az in (0.0, 90.0, 180.0, 270.0):
                specs.append(
                    VirtualViewSpec(f"up{int(az):03d}", 1, 132.5, az, fov_deg, True, True, False)
                )
        return specs

    # Handheld 360 mode (Front = Stream 0, Back = Stream 1; avoids bottom selfie-stick nadir)
    return [
        VirtualViewSpec("front_c", 0, 0.0, 0.0, fov_deg, False, False, True),
        VirtualViewSpec("front_r", 0, 40.0, 0.0, fov_deg, False, False, True),
        VirtualViewSpec("front_l", 0, 40.0, 180.0, fov_deg, False, False, True),
        VirtualViewSpec("back_c", 1, 0.0, 0.0, fov_deg, False, False, False),
        VirtualViewSpec("back_r", 1, 40.0, 0.0, fov_deg, False, False, False),
        VirtualViewSpec("back_l", 1, 40.0, 180.0, fov_deg, False, False, False),
    ]


def pinhole_focal_px(out_size: int, fov_deg: float) -> float:
    """Compute exact pinhole focal length in pixels for square image of size `out_size`."""
    return (out_size / 2.0) / math.tan(math.radians(fov_deg / 2.0))


def _build_remap_grid(
    src_w: int,
    src_h: int,
    out_size: int,
    spec: VirtualViewSpec,
    mei: MeiLensCalibration | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute float32 (map_x, map_y) lookup arrays mapping virtual pinhole pixels to fisheye source."""
    f_pin = pinhole_focal_px(out_size, spec.fov_deg)
    u, v = np.meshgrid(
        np.arange(out_size, dtype=np.float32),
        np.arange(out_size, dtype=np.float32),
    )
    c_out = (out_size - 1) / 2.0
    x = (u - c_out) / f_pin
    y = (v - c_out) / f_pin
    z = np.ones_like(x)
    norm = np.sqrt(x * x + y * y + z * z)
    dirs = np.stack([x / norm, y / norm, z / norm], axis=-1)

    rot = compute_view_rotation_s0(spec).astype(np.float32)
    rays = dirs @ rot.T
    if spec.stream_index == 1 and spec.spherical_s0_frame:
        # Transform from Stream 0 (nadir +Z) to Stream 1 (zenith +Z = Ry(180 deg) from Stream 0)
        # so upper views render right-side up with sky at top and buildings/trees at bottom.
        rx, ry, rz = -rays[..., 0], rays[..., 1], -rays[..., 2]
    else:
        rx, ry, rz = rays[..., 0], rays[..., 1], rays[..., 2]

    if mei is not None:
        p = mei.scaled(src_w, src_h)
        denom = np.maximum(rz + p["xi"], 1e-6)
        xn = rx / denom
        yn = ry / denom
        r2 = xn * xn + yn * yn
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + p["k1"] * r2 + p["k2"] * r4 + p["k3"] * r6
        xd = xn * radial + 2.0 * p["p1"] * xn * yn + p["p2"] * (r2 + 2.0 * xn * xn)
        yd = yn * radial + p["p1"] * (r2 + 2.0 * yn * yn) + 2.0 * p["p2"] * xn * yn
        map_x = (p["fx"] * xd + p["cx"]).astype(np.float32)
        map_y = (p["fy"] * yd + p["cy"]).astype(np.float32)
    else:
        # Fallback equidistant fisheye (~193 deg full circle)
        max_theta = math.radians(193.0 / 2.0)
        theta = np.arccos(np.clip(rz, -1.0, 1.0))
        phi = np.arctan2(ry, rx)
        radius_px = 0.488 * min(src_w, src_h) * (theta / max_theta)
        map_x = (src_w / 2.0 + radius_px * np.cos(phi)).astype(np.float32)
        map_y = (src_h / 2.0 + radius_px * np.sin(phi)).astype(np.float32)

    return map_x, map_y


def _compute_global_descriptor(bgr: np.ndarray) -> list[float]:
    """Compute L2-normalized 256-D (8x8 spatial HSV + Sobel gradient) descriptor for loop-closure retrieval."""
    small = cv2.resize(bgr, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)

    # Reshape 128x128 into 8x8 spatial cells of 16x16 pixels across 4 channels = 256 floats
    stack = np.stack([hsv[..., 0], hsv[..., 1], hsv[..., 2], gmag], axis=-1)
    cells = stack.reshape(8, 16, 8, 16, 4).mean(axis=(1, 3))
    vec = cells.reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return vec.tolist()


def _has_sufficient_structure(bgr: np.ndarray, min_active_tile_ratio: float = 0.30) -> bool:
    """Reject views dominated by featureless overcast/blue sky."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(small, cv2.CV_32F)
    # Partition into 8x8 tiles and check how many tiles have meaningful edge energy
    patches = lap.reshape(8, 32, 8, 32)
    variances = patches.var(axis=(1, 3))
    active = int(np.sum(variances > 16.0))
    return (active / 64.0) >= min_active_tile_ratio


def _decode_full_frame(
    video_path: Path,
    stream_idx: int,
    ts: float,
    width: int,
    height: int,
    use_cuda: bool = True,
) -> np.ndarray:
    """Decode a full-resolution BGR24 fisheye frame at `ts` over stdout pipe."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if use_cuda:
        cmd += ["-hwaccel", "cuda"]
    cmd += [
        "-ss",
        f"{ts:.3f}",
        "-i",
        str(video_path),
        "-map",
        f"0:v:{stream_idx}",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    expected = width * height * 3
    res = subprocess.run(cmd, capture_output=True, check=False)
    if res.returncode != 0 or len(res.stdout) < expected:
        if use_cuda:
            return _decode_full_frame(video_path, stream_idx, ts, width, height, use_cuda=False)
        raise RuntimeError(
            f"FFmpeg failed decoding stream {stream_idx} at {ts:.2f}s from {video_path}: "
            f"{res.stderr.decode('utf-8', errors='replace')}"
        )
    return np.frombuffer(res.stdout[:expected], dtype=np.uint8).reshape((height, width, 3))


def extract_perspective_dataset(
    meta: InsvMetadata,
    keyframes: list[KeyframeCandidate],
    images_dir: Path,
    out_size: int = 1600,
    fov_deg: float = 84.0,
    include_upper_views: bool = True,
    dense_azimuth: bool = True,
    jpeg_quality: int = 98,
    workers: int = 8,
    use_cuda: bool = True,
) -> tuple[list[ExtractedView], list[VirtualViewSpec], dict]:
    """Decode selected keyframes and unproject distortion-free perspective rig images."""
    images_dir.mkdir(parents=True, exist_ok=True)
    specs = get_rig_specs(
        meta.rig_type,
        fov_deg=fov_deg,
        include_upper_views=include_upper_views,
        dense_azimuth=dense_azimuth,
    )
    for spec in specs:
        (images_dir / spec.name).mkdir(parents=True, exist_ok=True)

    needed_streams = sorted({s.stream_index for s in specs if s.stream_index < len(meta.streams)})

    stream_dims: dict[int, tuple[int, int]] = {}
    for st in meta.streams:
        stream_dims[st.stream_index] = (st.width, st.height)

    specs = [s for s in specs if s.stream_index in stream_dims]
    mei_by_lens = {m.lens_index: m for m in meta.mei_lenses}
    remap_tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spec in specs:
        w, h = stream_dims[spec.stream_index]
        remap_tables[spec.name] = _build_remap_grid(
            w, h, out_size, spec, mei_by_lens.get(spec.stream_index)
        )

    imwrite_flags = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR") and hasattr(
        cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444"
    ):
        imwrite_flags += [
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
        ]

    primary_p = Path(meta.primary_path)
    secondary_p = Path(meta.secondary_path) if meta.secondary_path else None

    def process_one(kf_idx: int, kf: KeyframeCandidate) -> list[ExtractedView]:
        frames: dict[int, np.ndarray] = {}
        for s_idx in needed_streams:
            w, h = stream_dims[s_idx]
            if s_idx == 1 and secondary_p is not None:
                frames[s_idx] = _decode_full_frame(secondary_p, 0, kf.timestamp_s, w, h, use_cuda)
            else:
                frames[s_idx] = _decode_full_frame(
                    primary_p, s_idx, kf.timestamp_s, w, h, use_cuda
                )

        out_views: list[ExtractedView] = []
        ts_ms = int(round(kf.timestamp_s * 1000))
        frame_file = f"kf{kf_idx:04d}_t{ts_ms:07d}.jpg"
        for spec in specs:
            mx, my = remap_tables[spec.name]
            rect = cv2.remap(
                frames[spec.stream_index],
                mx,
                my,
                interpolation=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
            )
            if spec.require_texture and not _has_sufficient_structure(rect):
                continue
            rel_name = f"{spec.name}/{frame_file}"
            cv2.imwrite(
                str(images_dir / rel_name),
                rect,
                imwrite_flags,
            )
            desc = _compute_global_descriptor(rect)
            out_views.append(
                ExtractedView(
                    image_name=rel_name,
                    keyframe_index=kf_idx,
                    timestamp_s=kf.timestamp_s,
                    view_name=spec.name,
                    stream_index=spec.stream_index,
                    descriptor=desc,
                )
            )
        return out_views

    all_views: list[ExtractedView] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(process_one, idx, kf): idx for idx, kf in enumerate(keyframes)
        }
        for fut in tqdm(
            as_completed(futs), total=len(futs), desc="Extracting Mei-calibrated Lanczos4 pinhole rig"
        ):
            all_views.extend(fut.result())

    # Remove any empty camera subdirectories (e.g., if an upper view never passed texture check)
    active_cam_names = {v.view_name for v in all_views}
    for spec in specs:
        sub = images_dir / spec.name
        if spec.name not in active_cam_names and sub.is_dir():
            try:
                sub.rmdir()
            except OSError:
                pass
    active_specs = [s for s in specs if s.name in active_cam_names]

    all_views.sort(key=lambda v: (v.keyframe_index, v.view_name))
    proj_info = {
        "rig_type": meta.rig_type,
        "calibration_model": "mei_omnidirectional_factory" if meta.mei_lenses else "equidistant_fallback",
        "interpolation": "lanczos4_jpeg98_444",
        "image_size": out_size,
        "fov_deg": fov_deg,
        "focal_px": round(pinhole_focal_px(out_size, fov_deg), 6),
        "views_per_keyframe_specs": [asdict(s) for s in active_specs],
        "total_extracted_images": len(all_views),
    }
    return all_views, active_specs, proj_info
