"""Command-line interface for the INSV -> 3D Gaussian Splatting pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

from .card import ensure_card_mounted, find_insv_files, resolve_input_insv
from .keyframes import select_keyframes
from .metadata import parse_insv_metadata
from .projection import extract_perspective_dataset, pinhole_focal_px
from .sfm import run_sfm
from .train import detect_egpu_env, launch_viewer, train_splats

PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "max_keyframes": 96,
        "probe_interval": 2.0,
        "image_size": 1024,
        "fov": 86.0,
        "include_upper_views": False,
        "sparse_azimuth": True,
        "max_features": 8192,
        "temporal_window": 5,
        "loop_topk": 16,
        "steps": 20000,
        "max_splats": 4000000,
        "growth_stop_iter": 14000,
        "growth_grad_threshold": 0.000035,
    },
    "balanced": {
        "max_keyframes": 140,
        "probe_interval": 1.0,
        "image_size": 1280,
        "fov": 84.0,
        "include_upper_views": True,
        "sparse_azimuth": False,
        "max_features": 10240,
        "temporal_window": 6,
        "loop_topk": 22,
        "steps": 30000,
        "max_splats": 6000000,
        "growth_stop_iter": 18000,
        "growth_grad_threshold": 0.00003,
    },
    "ultra": {
        "max_keyframes": 190,
        "probe_interval": 0.5,
        "image_size": 1600,
        "fov": 84.0,
        "include_upper_views": True,
        "sparse_azimuth": False,
        "max_features": 18432,
        "temporal_window": 8,
        "loop_topk": 28,
        "steps": 40000,
        "max_splats": 8000000,
        "growth_stop_iter": 22000,
        "growth_grad_threshold": 0.000025,
    },
}


def _parse_boost_windows(raw_windows: list[str] | None) -> list[tuple[float, float]]:
    if not raw_windows:
        return []
    out: list[tuple[float, float]] = []
    for item in raw_windows:
        for tok in item.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if ":" not in tok:
                raise ValueError(f"Invalid --boost-window '{tok}'; expected START:END in seconds (e.g. 480:595)")
            a_s, b_s = tok.split(":", 1)
            lo, hi = float(a_s), float(b_s)
            if hi <= lo:
                raise ValueError(f"Invalid --boost-window '{tok}'; END must be greater than START")
            out.append((lo, hi))
    return out


def _apply_preset_defaults(args: argparse.Namespace) -> None:
    preset_cfg = PRESETS[args.preset]
    for key, val in preset_cfg.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)


def _redact_metadata_dict(meta_dict: dict[str, Any], gpu_dict: dict[str, Any]) -> None:
    if meta_dict.get("serial_number"):
        meta_dict["serial_number"] = "REDACTED"
    meta_dict["primary_path"] = Path(meta_dict["primary_path"]).name
    if meta_dict.get("secondary_path"):
        meta_dict["secondary_path"] = Path(meta_dict["secondary_path"]).name
    if gpu_dict.get("pci_bus_id"):
        gpu_dict["pci_bus_id"] = "REDACTED"


def _cmd_inspect(args: argparse.Namespace) -> int:
    if args.input:
        src = resolve_input_insv(args.input, device=args.device)
        all_sources = [src]
    else:
        mounts = ensure_card_mounted(preferred_device=args.device)
        all_sources = find_insv_files(mounts)
        if not all_sources:
            raise FileNotFoundError(f"No .insv files found on mounts: {[str(m) for m in mounts]}")
        src = all_sources[0]

    meta = parse_insv_metadata(src.primary_path, src.secondary_path, rig_mode_override=args.rig_mode)
    _, gpu_info = detect_egpu_env(prefer_egpu=not args.no_egpu)
    meta_dict = meta.to_dict()
    if args.anonymize:
        _redact_metadata_dict(meta_dict, gpu_info)

    report = {
        "discovered_insv_count": len(all_sources),
        "selected_most_recent": {
            "primary_path": src.primary_path.name if args.anonymize else str(src.primary_path),
            "secondary_path": (
                (src.secondary_path.name if args.anonymize else str(src.secondary_path))
                if src.secondary_path
                else None
            ),
            "size_gb": round(src.size_bytes / (1024**3), 2),
            "modified_iso": dt.datetime.fromtimestamp(src.mtime).isoformat(),
        },
        "metadata": meta_dict,
        "egpu": gpu_info,
    }
    print(json.dumps(report, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    _apply_preset_defaults(args)
    if args.no_upper_views:
        args.include_upper_views = False

    print(f"=== [1/5] Resolving .insv input (preset={args.preset}) ===", flush=True)
    src = resolve_input_insv(args.input, device=args.device)
    meta = parse_insv_metadata(src.primary_path, src.secondary_path, rig_mode_override=args.rig_mode)
    if not meta.streams:
        raise RuntimeError(f"No video streams found in {src.primary_path}")

    stem = src.primary_path.stem
    if args.output_dir:
        run_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    else:
        default_root = Path(__file__).resolve().parents[2] / "runs"
        run_dir = default_root / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = run_dir / "colmap_dataset"
    images_dir = dataset_dir / "images"
    export_dir = run_dir / "exports"

    _, gpu_info = detect_egpu_env(prefer_egpu=not args.no_egpu)
    use_cuda_decode = bool(gpu_info.get("nvdec_ready", gpu_info.get("egpu_detected"))) and not args.no_egpu

    print(f"  Input INSV : {src.primary_path} ({src.size_bytes / (1024**3):.2f} GiB)", flush=True)
    print(f"  Camera     : {meta.camera_model} (rig={meta.rig_type}, streams={len(meta.streams)})", flush=True)
    print(
        f"  Stream 0   : {meta.streams[0].width}x{meta.streams[0].height} @ {meta.streams[0].fps:.2f} fps, "
        f"duration={meta.streams[0].duration_s:.1f}s",
        flush=True,
    )
    print(
        f"  Calibration: {'Factory Mei Omnidirectional (trailer v3)' if meta.mei_lenses else 'Equidistant fallback'}",
        flush=True,
    )
    print(f"  GPU        : {gpu_info}", flush=True)
    print(f"  Output Dir : {run_dir}", flush=True)

    print("\n=== [2/5] Motion-gated & sharpness-ranked keyframe selection ===", flush=True)
    boost_windows = _parse_boost_windows(args.boost_window)
    keyframes, kf_stats = select_keyframes(
        src.primary_path,
        duration_s=meta.streams[0].duration_s,
        max_keyframes=args.max_keyframes,
        probe_interval_s=args.probe_interval,
        min_motion_score=args.min_motion,
        workers=args.workers,
        use_cuda=use_cuda_decode,
        start_s=args.start_time,
        end_s=args.end_time,
        boost_windows=boost_windows,
    )
    print(f"  Keyframes  : {kf_stats}", flush=True)

    print("\n=== [3/5] Extracting distortion-free Lanczos4 PINHOLE camera rig ===", flush=True)
    if images_dir.exists():
        import shutil as _shutil

        _shutil.rmtree(images_dir)
    views, active_specs, proj_stats = extract_perspective_dataset(
        meta=meta,
        keyframes=keyframes,
        images_dir=images_dir,
        out_size=args.image_size,
        fov_deg=args.fov,
        include_upper_views=bool(args.include_upper_views),
        dense_azimuth=not bool(args.sparse_azimuth),
        jpeg_quality=args.jpeg_quality,
        workers=args.workers,
        use_cuda=use_cuda_decode,
    )
    print(f"  Extracted  : {len(views)} views ({args.image_size}x{args.image_size}, FOV={args.fov} deg)", flush=True)

    print("\n=== [4/5] Running SfM (SIFT + Analytical Spherical Rig + GLOMAP/COLMAP + BA + Dense Seeding) ===", flush=True)
    focal_px = pinhole_focal_px(args.image_size, args.fov)
    sfm_stats = run_sfm(
        dataset_dir=dataset_dir,
        views=views,
        active_specs=active_specs,
        image_size=args.image_size,
        focal_px=focal_px,
        mapper=args.mapper,
        temporal_window=args.temporal_window,
        loop_topk=args.loop_topk,
        max_num_features=args.max_features,
        use_spherical_rig=not args.no_spherical_rig,
        dense_surface_seeding=not args.no_dense_seeding,
    )
    print(f"  SfM Result : {sfm_stats}", flush=True)

    print("\n=== [5/5] Training 3D Gaussian Splatting on GPU ===", flush=True)
    refine_every = args.refine_every if args.refine_every > 0 else max(180, min(300, sfm_stats["registered_images"] // 5))
    train_stats = train_splats(
        dataset_dir=dataset_dir,
        export_dir=export_dir,
        total_steps=args.steps,
        export_every=args.export_every,
        eval_every=args.eval_every,
        eval_split_every=args.eval_split_every,
        max_resolution=args.image_size,
        max_splats=args.max_splats,
        refine_every=refine_every,
        growth_stop_iter=args.growth_stop_iter,
        growth_grad_threshold=args.growth_grad_threshold,
        growth_select_fraction=args.growth_select_fraction,
        opac_loss_weight=args.opac_loss_weight,
        scale_loss_weight=args.scale_loss_weight,
        with_viewer=args.with_viewer,
        prefer_egpu=not args.no_egpu,
        log_path=run_dir / "brush_train.log",
    )

    meta_dict = meta.to_dict()
    if args.anonymize:
        _redact_metadata_dict(meta_dict, train_stats.get("gpu", {}))

    summary = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "preset": args.preset,
        "input_insv": src.primary_path.name if args.anonymize else str(src.primary_path),
        "run_dir": run_dir.name if args.anonymize else str(run_dir),
        "metadata": meta_dict,
        "keyframes": kf_stats,
        "projection": proj_stats,
        "sfm": sfm_stats,
        "training": train_stats,
    }
    summary_path = run_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Pipeline Complete ===", flush=True)
    print(
        f"  Final Splat PLY : {train_stats['final_ply']} "
        f"({train_stats['final_ply_size_mb']} MB, {train_stats['num_gaussians']:,} Gaussians)",
        flush=True,
    )
    print(f"  Summary Report  : {summary_path}", flush=True)
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    target = Path(os.path.expanduser(args.path)).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")
    return launch_viewer(target, prefer_egpu=not args.no_egpu)


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import run_web_server

    target = Path(os.path.expanduser(args.path)).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")
    return run_web_server(target_path=target, host=args.host, port=args.port, device=args.cuda_device)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="insv-splat",
        description="Stitch-free Insta360 & Antigravity A1 (.insv) 360 video to 3D Gaussian Splatting pipeline",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_ins = sub.add_parser("inspect", help="Mount SD card if needed and inspect .insv calibration & GPU")
    p_ins.add_argument("--input", help="Explicit .insv file or folder (defaults to newest on SD card)")
    p_ins.add_argument("--device", help="Preferred block device to mount (e.g. /dev/mmcblk0p1)")
    p_ins.add_argument("--rig-mode", choices=["auto", "drone", "handheld"], default="auto")
    p_ins.add_argument("--anonymize", action="store_true", help="Redact device serial number and local paths")
    p_ins.add_argument("--no-egpu", action="store_true", help="Disable discrete NVIDIA GPU/eGPU preference")
    p_ins.set_defaults(func=_cmd_inspect)

    p_run = sub.add_parser("run", help="Run full .insv -> SfM -> 3D Gaussian Splatting pipeline")
    p_run.add_argument("--preset", choices=["fast", "balanced", "ultra"], default="ultra", help="Quality preset (default: ultra)")
    p_run.add_argument("--input", help="Explicit .insv file or folder (defaults to newest on SD card)")
    p_run.add_argument("--device", help="Preferred block device to mount (e.g. /dev/mmcblk0p1)")
    p_run.add_argument("--output-dir", help="Output run directory (defaults to ./runs/<video_stem>)")
    p_run.add_argument("--rig-mode", choices=["auto", "drone", "handheld"], default="auto")
    p_run.add_argument("--max-keyframes", type=int, default=None, help="Target active keyframes across video")
    p_run.add_argument("--probe-interval", type=float, default=None, help="Seconds between motion/sharpness probes")
    p_run.add_argument("--min-motion", type=float, default=3.5, help="Minimum grayscale L1 motion score to skip hovers")
    p_run.add_argument("--start-time", type=float, default=0.0, help="Start timestamp in seconds")
    p_run.add_argument("--end-time", type=float, default=None, help="End timestamp in seconds")
    p_run.add_argument(
        "--boost-window",
        action="append",
        default=None,
        help="Timestamp interval START:END in seconds for dense ~0.5s roof/detail sampling (repeatable)",
    )
    p_run.add_argument("--image-size", type=int, default=None, help="Square perspective cutout resolution")
    p_run.add_argument("--fov", type=float, default=None, help="Horizontal/vertical pinhole FOV in degrees")
    p_run.add_argument("--jpeg-quality", type=int, default=98, help="JPEG quality (98 + 4:4:4 chroma)")
    p_run.add_argument("--include-upper-views", action="store_true", default=None, help="Include upright sky-filtered Stream 1 upper views")
    p_run.add_argument("--no-upper-views", action="store_true", help="Disable Stream 1 upper views even in balanced/ultra presets")
    p_run.add_argument("--sparse-azimuth", action="store_true", default=None, help="Use 4 obliques (90 deg) instead of 6 obliques (60 deg)")
    p_run.add_argument("--no-spherical-rig", action="store_true", help="Disable analytical spherical COLMAP RigConfig")
    p_run.add_argument("--no-dense-seeding", action="store_true", help="Disable post-BA CUDA multi-view dense surface seeding")
    p_run.add_argument("--mapper", choices=["auto", "glomap", "colmap"], default="auto", help="SfM reconstruction backend")
    p_run.add_argument("--temporal-window", type=int, default=None, help="Sliding keyframe window for inter-frame matching")
    p_run.add_argument("--loop-topk", type=int, default=None, help="Visual retrieval top-K loop closure pairs per view")
    p_run.add_argument("--max-features", type=int, default=None, help="Maximum SIFT keypoints per view")
    p_run.add_argument("--steps", type=int, default=None, help="Total 3DGS training steps")
    p_run.add_argument("--export-every", type=int, default=10000, help="Save .ply checkpoint every N steps")
    p_run.add_argument("--eval-every", type=int, default=10000, help="Render held-out evaluation views every N steps")
    p_run.add_argument("--eval-split-every", type=int, default=25, help="Hold out every Nth image for evaluation")
    p_run.add_argument("--max-splats", type=int, default=None, help="Maximum Gaussian splat budget")
    p_run.add_argument("--refine-every", type=int, default=0, help="Gaussian densification interval (0 = auto)")
    p_run.add_argument("--growth-stop-iter", type=int, default=None, help="Iteration at which Gaussian splitting/cloning stops")
    p_run.add_argument("--growth-grad-threshold", type=float, default=None, help="Gradient threshold for splat densification")
    p_run.add_argument("--growth-select-fraction", type=float, default=0.15, help="Fraction of high-grad splats grown per refine step")
    p_run.add_argument("--opac-loss-weight", type=float, default=2e-9, help="Opacity regularization weight to suppress floaters")
    p_run.add_argument("--scale-loss-weight", type=float, default=2e-8, help="Scale regularization weight to suppress streak splats")
    p_run.add_argument("--workers", type=int, default=12, help="Parallel FFmpeg/remap worker threads")
    p_run.add_argument("--with-viewer", action="store_true", help="Open live Brush GUI during training")
    p_run.add_argument("--anonymize", action="store_true", help="Redact device serial number and local paths in summary JSON")
    p_run.add_argument("--no-egpu", action="store_true", help="Disable discrete NVIDIA GPU/eGPU preference")
    p_run.set_defaults(func=_cmd_run)

    p_view = sub.add_parser("view", help="Launch interactive desktop Vulkan 3DGS viewer on a trained .ply or run folder")
    p_view.add_argument("path", help="Path to .ply file or run directory")
    p_view.add_argument("--no-egpu", action="store_true", help="Disable discrete NVIDIA GPU/eGPU preference")
    p_view.set_defaults(func=_cmd_view)

    p_srv = sub.add_parser(
        "serve",
        help="Start server-side CUDA/Triton 3DGS web viewer accessible across your local LAN / Tailscale",
    )
    p_srv.add_argument("path", help="Path to .ply file or run directory")
    p_srv.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0 for LAN access)")
    p_srv.add_argument("--port", type=int, default=8080, help="HTTP/WebSocket port (default: 8080)")
    p_srv.add_argument("--cuda-device", default="cuda:0", help="PyTorch CUDA device (default: cuda:0)")
    p_srv.set_defaults(func=_cmd_serve)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
