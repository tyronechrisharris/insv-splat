"""Structure-from-Motion (SIFT + Analytical Spherical RigConfig + Loop-Closure + GLOMAP/BA)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pycolmap

from .projection import ExtractedView, VirtualViewSpec, compute_view_rotation_s0


def build_smart_pair_list(
    views: list[ExtractedView],
    pairs_path: Path,
    temporal_window: int = 8,
    loop_topk: int = 28,
    temporal_view_topk: int = 4,
) -> int:
    """Generate high-overlap inter-timestamp image pairs (excluding zero-baseline intra-timestamp pairs).

    Combines:
      1. Direction-aware temporal sliding window across keyframes (nadir anchors + top-M visually aligned views)
      2. Exhaustive nadir-to-nadir aerial anchor graph across the entire flight
      3. 256-D visual-descriptor cosine-similarity top-K loop closure for multi-pass / multi-altitude fusion
    """
    n = len(views)
    if n < 2:
        raise ValueError("Need at least 2 extracted views for SfM")

    pairs: set[tuple[str, str]] = set()

    def add_pair(a: str, b: str) -> None:
        if a == b:
            return
        pairs.add((a, b) if a < b else (b, a))

    mat = np.array([v.descriptor for v in views], dtype=np.float32)
    sim = mat @ mat.T  # (N, N) cosine similarity
    kf_arr = np.array([v.keyframe_index for v in views], dtype=np.int32)

    idx_by_kf: dict[int, list[int]] = {}
    for idx, v in enumerate(views):
        idx_by_kf.setdefault(v.keyframe_index, []).append(idx)
    kf_indices = sorted(idx_by_kf.keys())

    anchor_names = {"nadir", "front_c", "back_c"}

    # 1. Smart temporal sliding window across adjacent keyframes (strictly kf_a != kf_b -> non-zero baseline)
    for pos, kf_a in enumerate(kf_indices):
        idxs_a = idx_by_kf[kf_a]
        for step_offset, next_pos in enumerate(
            range(pos + 1, min(len(kf_indices), pos + 1 + temporal_window)), start=1
        ):
            idxs_b = idx_by_kf[kf_indices[next_pos]]
            # Always link anchor views to nearby views for strong vertical-to-oblique tie-ins
            for ia in idxs_a:
                va = views[ia]
                if va.view_name in anchor_names and step_offset <= 4:
                    for ib in idxs_b:
                        add_pair(va.image_name, views[ib].image_name)
                # Top-M visually overlapping views in kf_b (handles arbitrary yaw spins cleanly)
                sub_scores = sim[ia, idxs_b]
                k = min(temporal_view_topk, len(idxs_b))
                top_local = np.argsort(-sub_scores)[:k]
                for loc in top_local:
                    add_pair(va.image_name, views[idxs_b[int(loc)]].image_name)

            for ib in idxs_b:
                vb = views[ib]
                if vb.view_name in anchor_names and step_offset <= 4:
                    for ia in idxs_a:
                        add_pair(views[ia].image_name, vb.image_name)
                sub_scores = sim[ib, idxs_a]
                k = min(temporal_view_topk, len(idxs_a))
                top_local = np.argsort(-sub_scores)[:k]
                for loc in top_local:
                    add_pair(views[idxs_a[int(loc)]].image_name, vb.image_name)

    # 2. All-to-all nadir/center anchor pairs across the entire trajectory (guarantees global loop closure)
    anchor_views = [v for v in views if v.view_name in anchor_names]
    for i in range(len(anchor_views)):
        for j in range(i + 1, len(anchor_views)):
            if anchor_views[i].keyframe_index != anchor_views[j].keyframe_index:
                add_pair(anchor_views[i].image_name, anchor_views[j].image_name)

    # 3. Visual similarity top-K retrieval across non-adjacent keyframes (fuses low/mid/high passes & Stream 1)
    for i in range(n):
        row = sim[i].copy()
        mask_close = np.abs(kf_arr - kf_arr[i]) <= temporal_window
        row[mask_close] = -2.0
        if loop_topk > 0 and np.any(~mask_close):
            k = min(loop_topk, int(np.sum(~mask_close)))
            top_idx = np.argpartition(-row, k - 1)[:k]
            for j in top_idx:
                if row[j] > -1.0:
                    add_pair(views[i].image_name, views[int(j)].image_name)

    sorted_pairs = sorted(pairs)
    with pairs_path.open("w", encoding="utf-8") as f:
        for a, b in sorted_pairs:
            f.write(f"{a} {b}\n")

    return len(sorted_pairs)


def _configure_spherical_rig(db_path: Path, active_specs: list[VirtualViewSpec]) -> int:
    """Bind all primary-lens virtual views of each keyframe into an exact analytical spherical COLMAP Rig."""
    primary_specs = [s for s in active_specs if s.in_primary_rig]
    if len(primary_specs) < 2:
        return 0

    ref_spec = next((s for s in primary_specs if s.name in {"nadir", "front_c"}), primary_specs[0])
    r_ref = compute_view_rotation_s0(ref_spec)

    rig_cams: list[pycolmap.RigConfigCamera] = []
    for s in primary_specs:
        rc = pycolmap.RigConfigCamera()
        rc.image_prefix = f"{s.name}/"
        if s.name == ref_spec.name:
            rc.ref_sensor = True
        else:
            rc.ref_sensor = False
            r_s = compute_view_rotation_s0(s)
            # p_s0 = r_s @ p_cam = r_ref @ p_rig -> p_cam = (r_s.T @ r_ref) @ p_rig
            r_cam_from_rig = r_s.T @ r_ref
            rc.cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(r_cam_from_rig),
                np.zeros(3, dtype=np.float64),
            )
        rig_cams.append(rc)

    rig_cfg = pycolmap.RigConfig()
    rig_cfg.cameras = rig_cams
    with pycolmap.Database.open(str(db_path)) as db:
        pycolmap.apply_rig_config([rig_cfg], db)
    return len(primary_specs)


def _clean_and_flatten_reconstruction(
    rec: pycolmap.Reconstruction,
    use_rig_constraints: bool = True,
) -> tuple[pycolmap.Reconstruction, dict]:
    """Prune unanchored outlier cameras & high-error 3D points, run Ceres BA, and flatten to standard COLMAP."""
    raw_reg_images = int(rec.num_reg_images())
    raw_points = int(rec.num_points3D())

    def _prune_outlier_frames(min_single_pts: int = 35, min_rig_pts: int = 50) -> int:
        reg_ids = list(rec.reg_image_ids())
        if not reg_ids:
            return 0
        centers: dict[int, np.ndarray] = {}
        by_frame: dict[int, list[pycolmap.Image]] = {}
        for iid in reg_ids:
            im = rec.images[iid]
            t_cw = im.cam_from_world()
            centers[iid] = -t_cw.rotation.matrix().T @ t_cw.translation
            by_frame.setdefault(im.frame_id, []).append(im)

        all_c = np.array(list(centers.values()), dtype=np.float64)
        med_c = np.median(all_c, axis=0)
        dists = {iid: float(np.linalg.norm(c - med_c)) for iid, c in centers.items()}
        d_vals = list(dists.values())
        med_d = max(float(np.median(d_vals)), 1.0)
        p90_d = max(float(np.percentile(d_vals, 90)), 2.0)
        max_dist = max(p90_d * 3.0, med_d * 6.0, 20.0)

        removed_imgs = 0
        for fid, f_imgs in by_frame.items():
            f_pts = sum(int(im.num_points3D) for im in f_imgs)
            f_max_d = max(dists[im.image_id] for im in f_imgs)
            min_req = min_single_pts if len(f_imgs) == 1 else min_rig_pts
            if f_max_d > max_dist or f_pts < min_req:
                if fid in rec.frames and rec.frames[fid].has_pose():
                    rec.deregister_frame(fid)
                    removed_imgs += len(f_imgs)
        return removed_imgs

    removed_pre = _prune_outlier_frames(min_single_pts=35, min_rig_pts=50)

    rec.update_point_3d_errors()
    bad_pre = [
        pid
        for pid, pt in rec.points3D.items()
        if pt.error > 2.2 or pt.track.length() < 2 or (pt.track.length() == 2 and pt.error > 1.2)
    ]
    for pid in bad_pre:
        rec.delete_point3D(pid)

    # Global Ceres Bundle Adjustment locking exact analytical rig angles & Mei pinhole intrinsics
    ba_opts = pycolmap.BundleAdjustmentOptions()
    ba_opts.refine_sensor_from_rig = not use_rig_constraints
    ba_opts.refine_focal_length = False
    ba_opts.refine_principal_point = False
    ba_opts.print_summary = False
    if rec.num_reg_images() >= 3 and rec.num_points3D() >= 50:
        pycolmap.bundle_adjustment(rec, ba_opts)

    removed_post = _prune_outlier_frames(min_single_pts=30, min_rig_pts=45)
    rec.update_point_3d_errors()
    bad_post = [
        pid
        for pid, pt in rec.points3D.items()
        if pt.error > 1.85 or pt.track.length() < 2 or (pt.track.length() == 2 and pt.error > 1.05)
    ]
    for pid in bad_post:
        rec.delete_point3D(pid)

    # Flatten rig frames into universal trivial-frame COLMAP sparse/0 for Brush & Web Viewer
    flat = pycolmap.Reconstruction()
    for _, cam in sorted(rec.cameras.items()):
        flat.add_camera_with_trivial_rig(cam)
    for img_id in sorted(rec.reg_image_ids()):
        im = rec.images[img_id]
        t_cw = im.cam_from_world()
        im_new = pycolmap.Image(
            name=im.name,
            keypoints=[p.xy for p in im.points2D],
            camera_id=im.camera_id,
            image_id=im.image_id,
        )
        flat.add_image_with_trivial_frame(im_new, t_cw)

    for _, pt in rec.points3D.items():
        track = pycolmap.Track()
        for el in pt.track.elements:
            if flat.exists_image(el.image_id):
                track.add_element(el.image_id, el.point2D_idx)
        if track.length() >= 2:
            flat.add_point3D(pt.xyz, track, pt.color)

    flat.update_point_3d_errors()
    cleanup_stats = {
        "raw_registered_images": raw_reg_images,
        "pruned_outlier_images": removed_pre + removed_post,
        "raw_points3D": raw_points,
        "pruned_noisy_points3D": len(bad_pre) + len(bad_post),
    }
    return flat, cleanup_stats


def _select_best_model(recs: dict[int, pycolmap.Reconstruction]) -> tuple[int, pycolmap.Reconstruction] | None:
    if not recs:
        return None
    best_k = max(recs.keys(), key=lambda k: (recs[k].num_reg_images(), recs[k].num_points3D()))
    return best_k, recs[best_k]


def run_sfm(
    dataset_dir: Path,
    views: list[ExtractedView],
    active_specs: list[VirtualViewSpec],
    image_size: int,
    focal_px: float,
    mapper: str = "auto",
    temporal_window: int = 8,
    loop_topk: int = 28,
    max_num_features: int = 12288,
    use_spherical_rig: bool = True,
) -> dict:
    """Run SIFT feature extraction, analytical spherical rig binding, smart matching, and GLOMAP/COLMAP + BA."""
    images_dir = dataset_dir / "images"
    db_path = dataset_dir / "database.db"
    pairs_path = dataset_dir / "pairs.txt"
    sparse_dir = dataset_dir / "sparse"

    if db_path.exists():
        db_path.unlink()
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ropts = pycolmap.ImageReaderOptions()
    ropts.camera_model = "SIMPLE_PINHOLE"
    c = image_size / 2.0
    ropts.camera_params = f"{focal_px:.6f},{c:.1f},{c:.1f}"

    fopts = pycolmap.FeatureExtractionOptions()
    fopts.max_image_size = max(image_size, 2048)
    fopts.sift.max_num_features = max_num_features
    fopts.sift.peak_threshold = 0.005

    pycolmap.extract_features(
        str(db_path),
        str(images_dir),
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
        reader_options=ropts,
        extraction_options=fopts,
    )

    rig_sensor_count = 0
    if use_spherical_rig:
        rig_sensor_count = _configure_spherical_rig(db_path, active_specs)
    t_feat = time.time() - t0

    t1 = time.time()
    num_pairs = build_smart_pair_list(
        views,
        pairs_path=pairs_path,
        temporal_window=temporal_window,
        loop_topk=loop_topk,
    )
    popts = pycolmap.ImportedPairingOptions()
    popts.match_list_path = str(pairs_path)
    mopts = pycolmap.FeatureMatchingOptions()
    mopts.skip_image_pairs_in_same_frame = True
    pycolmap.match_image_pairs(str(db_path), matching_options=mopts, pairing_options=popts)
    t_match = time.time() - t1

    t2 = time.time()
    chosen_mapper = mapper
    best_rec: pycolmap.Reconstruction | None = None

    if mapper in {"auto", "glomap"}:
        glomap_dir = dataset_dir / "sparse_glomap"
        if glomap_dir.exists():
            shutil.rmtree(glomap_dir)
        glomap_dir.mkdir(parents=True, exist_ok=True)
        grecs = pycolmap.global_mapping(str(db_path), str(images_dir), str(glomap_dir))
        picked = _select_best_model(grecs)
        if picked is not None:
            _, best_rec = picked
            chosen_mapper = "glomap_spherical_rig" if rig_sensor_count >= 2 else "glomap"
        shutil.rmtree(glomap_dir, ignore_errors=True)

    # Fallback or explicit incremental COLMAP if GLOMAP registered < 75% of images
    min_acceptable = int(0.75 * len(views))
    if mapper == "colmap" or best_rec is None or (mapper == "auto" and best_rec.num_reg_images() < min_acceptable):
        inc_dir = dataset_dir / "sparse_colmap"
        if inc_dir.exists():
            shutil.rmtree(inc_dir)
        inc_dir.mkdir(parents=True, exist_ok=True)
        irecs = pycolmap.incremental_mapping(str(db_path), str(images_dir), str(inc_dir))
        ipicked = _select_best_model(irecs)
        if ipicked is not None:
            _, i_rec = ipicked
            if best_rec is None or i_rec.num_reg_images() > best_rec.num_reg_images():
                best_rec = i_rec
                chosen_mapper = "colmap_incremental_rig" if rig_sensor_count >= 2 else "colmap_incremental"
        shutil.rmtree(inc_dir, ignore_errors=True)

    if best_rec is None:
        raise RuntimeError("SfM failed to reconstruct any valid 3D sparse model.")

    clean_rec, cleanup_stats = _clean_and_flatten_reconstruction(
        best_rec,
        use_rig_constraints=(rig_sensor_count >= 2),
    )
    target0 = sparse_dir / "0"
    target0.mkdir(parents=True, exist_ok=True)
    clean_rec.write_binary(str(target0))
    t_map = time.time() - t2

    return {
        "mapper_used": chosen_mapper,
        "spherical_rig_sensors": rig_sensor_count,
        "total_input_images": len(views),
        "registered_images": int(clean_rec.num_reg_images()),
        "registration_rate": round(clean_rec.num_reg_images() / max(1, len(views)), 4),
        "num_points3D": int(clean_rec.num_points3D()),
        "mean_reprojection_error_px": round(float(clean_rec.compute_mean_reprojection_error()), 4),
        "matched_pairs": num_pairs,
        "cleanup": cleanup_stats,
        "feature_extraction_s": round(t_feat, 2),
        "matching_s": round(t_match, 2),
        "mapping_and_ba_s": round(t_map, 2),
        "sparse_model_path": str(target0),
    }
