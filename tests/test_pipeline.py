"""Unit tests for Mei trailer parsing, spherical pinhole geometry, and zero-baseline rig pairing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap

from insv_splat.cli import PRESETS, _apply_preset_defaults, _parse_boost_windows, build_parser
from insv_splat.metadata import _parse_mei_offset_v3
from insv_splat.projection import (
    ExtractedView,
    _build_remap_grid,
    _has_sufficient_structure,
    compute_view_rotation_s0,
    get_rig_specs,
    pinhole_focal_px,
)
from insv_splat.sfm import (
    _clean_and_flatten_reconstruction,
    _configure_spherical_rig,
    build_smart_pair_list,
)


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
    args = parser.parse_args(
        ["run", "--preset", "fast", "--steps", "12345", "--boost-window", "480:595"]
    )
    _apply_preset_defaults(args)
    assert args.max_keyframes == PRESETS["fast"]["max_keyframes"]
    assert args.image_size == PRESETS["fast"]["image_size"]
    assert args.steps == 12345
    assert _parse_boost_windows(args.boost_window) == [(480.0, 595.0)]


def test_sky_cloud_structure_filter() -> None:
    # Soft sinusoidal cloud waves should be rejected; crisp architectural checkerboard should pass
    yy, xx = np.meshgrid(np.linspace(0, 12, 256), np.linspace(0, 12, 256))
    clouds = np.clip(185.0 + 30.0 * np.sin(xx) * np.cos(yy), 0, 255).astype(np.uint8)
    cloud_bgr = np.stack([clouds, clouds, clouds], axis=-1)
    assert not _has_sufficient_structure(cloud_bgr)

    checker = (((np.arange(256)[:, None] // 8 + np.arange(256)[None, :] // 8) % 2) * 180 + 40).astype(
        np.uint8
    )
    facade_bgr = np.stack([checker, checker, checker], axis=-1)
    assert _has_sufficient_structure(facade_bgr)


def test_trajectory_jump_and_secondary_drift_pruning() -> None:
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(
        model="SIMPLE_PINHOLE", width=256, height=256, params=[140.0, 128.0, 128.0], camera_id=1
    )
    rec.add_camera_with_trivial_rig(cam)

    pts_world = [
        np.array([(p_i % 9 - 4) * 0.6, (p_i // 9 - 3) * 0.6, 12.0 + (p_i % 5) * 0.4], dtype=np.float64)
        for p_i in range(55)
    ]

    def _proj_kps(pos_w: np.ndarray) -> list[np.ndarray]:
        out = []
        for pw in pts_world:
            pc = pw - pos_w
            out.append(np.array([140.0 * pc[0] / pc[2] + 128.0, 140.0 * pc[1] / pc[2] + 128.0]))
        return out

    # 8 primary keyframes moving smoothly at 0.5 units/step along X, EXCEPT kf0004 which teleports +6.0 in Y
    next_iid = 1
    kf_to_iid: dict[int, int] = {}
    for kf in range(8):
        true_pos = np.array([kf * 0.5, 0.0, 0.0], dtype=np.float64)
        solved_pos = np.array([kf * 0.5, 6.0 if kf == 4 else 0.0, 0.0], dtype=np.float64)
        tcw = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), -solved_pos)
        im = pycolmap.Image(
            name=f"nadir/kf{kf:04d}_t{kf*1000:07d}.jpg",
            keypoints=_proj_kps(solved_pos if kf == 4 else true_pos),
            camera_id=1,
            image_id=next_iid,
        )
        rec.add_image_with_trivial_frame(im, tcw)
        kf_to_iid[kf] = next_iid
        next_iid += 1

    # Co-located upper view at kf0002 (offset 0.02 units -> kept)
    pos_good = np.array([1.02, 0.0, 0.0], dtype=np.float64)
    im_up_good = pycolmap.Image(
        name="up000/kf0002_t0002000.jpg", keypoints=_proj_kps(pos_good), camera_id=1, image_id=next_iid
    )
    rec.add_image_with_trivial_frame(
        im_up_good, pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), -pos_good)
    )
    iid_up_good = next_iid
    next_iid += 1

    # Drifted cloud-tracking upper view at kf0005 (offset 8.0 units -> pruned)
    pos_bad = np.array([2.5, 8.0, 0.0], dtype=np.float64)
    im_up_bad = pycolmap.Image(
        name="up000/kf0005_t0005000.jpg", keypoints=_proj_kps(pos_bad), camera_id=1, image_id=next_iid
    )
    rec.add_image_with_trivial_frame(
        im_up_bad, pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), -pos_bad)
    )
    iid_up_bad = next_iid

    for p_i, pw in enumerate(pts_world):
        tr = pycolmap.Track()
        for iid in list(kf_to_iid.values()) + [iid_up_good, iid_up_bad]:
            tr.add_element(iid, p_i)
        rec.add_point3D(pw, tr, np.array([120, 120, 120], dtype=np.uint8))

    flat, stats = _clean_and_flatten_reconstruction(rec, use_rig_constraints=False)
    surviving_names = {im.name for im in flat.images.values()}
    assert "nadir/kf0004_t0004000.jpg" not in surviving_names
    assert "up000/kf0005_t0005000.jpg" not in surviving_names
    assert "up000/kf0002_t0002000.jpg" in surviving_names
    assert stats["pruned_trajectory_jump_images"] == 1
    assert stats["pruned_drifted_secondary_images"] == 1
