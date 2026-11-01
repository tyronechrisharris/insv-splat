"""Unit tests for Mei trailer parsing, spherical pinhole geometry, and zero-baseline rig pairing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap

from insv_splat.cli import PRESETS, _apply_preset_defaults, build_parser
from insv_splat.metadata import _parse_mei_offset_v3
from insv_splat.projection import (
    ExtractedView,
    _build_remap_grid,
    compute_view_rotation_s0,
    get_rig_specs,
    pinhole_focal_px,
)
from insv_splat.sfm import _configure_spherical_rig, build_smart_pair_list


def test_parse_mei_offset_v3_type_155() -> None:
    # Synthetic Insta360 / Antigravity A1 type-155 dual-lens offset_v3 string
    synthetic = (
        "2_1.504000_3750.000_3750.500_2624.000_2624.000_-178.500_-89.700_90.200_"
        "0.000000_0.000000_0.000000_-0.04800000_0.16800000_-0.49800000_-0.00050000_-0.00050000_"
        "10496_5248_155_"
        "1.504000_3744.000_3744.500_7872.000_2614.000_-178.400_-89.800_89.800_"
        "-0.000200_0.000000_-0.071000_-0.04900000_0.17400000_-0.49900000_-0.00120000_-0.00010000_"
        "10496_5248_155_197632"
    )
    lenses = _parse_mei_offset_v3(synthetic)
    assert len(lenses) == 2
    assert lenses[0].ref_size == 5248
    assert np.isclose(lenses[0].xi, 1.504)
    # Lens 1 cx in side-by-side [0..10496] must be shifted into local [0..5248] coordinates
    assert np.isclose(lenses[1].cx, 7872.0 - 5248.0)
    assert np.isclose(lenses[1].translation_m[2], -0.071)

    scaled = lenses[0].scaled(3840, 3840)
    assert np.isclose(scaled["cx"], 1920.0)
    assert np.isclose(scaled["fx"], 3750.0 * (3840.0 / 5248.0))


def test_drone_rig_rotations_and_upright_stream1_remap() -> None:
    specs = get_rig_specs("drone", fov_deg=84.0, include_upper_views=True, dense_azimuth=True)
    assert len(specs) == 11  # 1 nadir + 6 obliques + 4 upper views
    primary = [s for s in specs if s.in_primary_rig]
    upper = [s for s in specs if not s.in_primary_rig]
    assert len(primary) == 7
    assert len(upper) == 4

    for s in specs:
        r = compute_view_rotation_s0(s)
        assert r.shape == (3, 3)
        assert np.allclose(r.T @ r, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(r), 1.0, atol=1e-6)

    # Check that upper views map inside valid fisheye bounds without crossing 90 deg off-axis on Lens 1
    mx, my = _build_remap_grid(3840, 3840, 256, upper[0], mei=None)
    assert mx.min() > 0.0 and mx.max() < 3840.0
    assert my.min() > 0.0 and my.max() < 3840.0


def test_smart_pair_list_excludes_intra_keyframe_zero_baseline(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    views: list[ExtractedView] = []
    for kf in range(5):
        for cam in ("nadir", "obl000", "obl120", "up000"):
            vec = rng.normal(size=256).astype(np.float32)
            vec /= np.linalg.norm(vec)
            views.append(
                ExtractedView(
                    image_name=f"{cam}/kf{kf:04d}_t{kf*1000:07d}.jpg",
                    keyframe_index=kf,
                    timestamp_s=float(kf),
                    view_name=cam,
                    stream_index=1 if cam.startswith("up") else 0,
                    descriptor=vec.tolist(),
                )
            )

    pairs_file = tmp_path / "pairs.txt"
    count = build_smart_pair_list(views, pairs_file, temporal_window=2, loop_topk=4)
    assert count > 0
    for line in pairs_file.read_text(encoding="utf-8").splitlines():
        a, b = line.split()
        kf_a = a.split("/")[1].split("_")[0]
        kf_b = b.split("/")[1].split("_")[0]
        assert kf_a != kf_b, f"Zero-baseline intra-keyframe pair generated: {a} vs {b}"


def test_spherical_rig_config_application(tmp_path: Path) -> None:
    specs = get_rig_specs("drone", fov_deg=84.0, include_upper_views=True, dense_azimuth=False)
    images_dir = tmp_path / "images"
    import cv2

    rng = np.random.default_rng(7)
    for kf in range(2):
        for s in specs:
            d = images_dir / s.name
            d.mkdir(parents=True, exist_ok=True)
            img = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(d / f"kf{kf:04d}_t{kf*1000:07d}.jpg"), img)

    db_path = tmp_path / "database.db"
    ropts = pycolmap.ImageReaderOptions()
    ropts.camera_model = "SIMPLE_PINHOLE"
    f = pinhole_focal_px(128, 84.0)
    ropts.camera_params = f"{f:.6f},64.0,64.0"
    pycolmap.extract_features(
        str(db_path),
        str(images_dir),
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
        reader_options=ropts,
    )
    n_sensors = _configure_spherical_rig(db_path, specs)
    assert n_sensors == 5  # nadir + 4 obliques


def test_cli_preset_defaults_and_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--preset", "fast", "--steps", "12345"])
    _apply_preset_defaults(args)
    assert args.max_keyframes == PRESETS["fast"]["max_keyframes"]
    assert args.image_size == PRESETS["fast"]["image_size"]
    assert args.steps == 12345
