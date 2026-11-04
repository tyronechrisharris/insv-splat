"""Structure-from-Motion (SIFT + Analytical Spherical RigConfig + Loop-Closure + GLOMAP/BA + CUDA MVS Seeding)."""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .projection import ExtractedView, VirtualViewSpec, compute_view_rotation_s0

_KF_NAME_RE = re.compile(r"^([^/]+)/kf(\d+)_t(\d+)\.[^.]+$")


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
    active_specs: list[VirtualViewSpec] | None = None,
    use_rig_constraints: bool = True,
) -> tuple[pycolmap.Reconstruction, dict]:
    """Prune outlier/teleported/drifted cameras & noisy 3D points, run Ceres BA, and flatten to standard COLMAP."""
    raw_reg_images = int(rec.num_reg_images())
    raw_points = int(rec.num_points3D())

    primary_names: set[str] | None = None
    if active_specs:
        primary_names = {s.name for s in active_specs if s.in_primary_rig}

    def _is_primary_view(vname: str) -> bool:
        if primary_names is not None:
            return vname in primary_names
        return not vname.startswith(("up", "back_"))

    def _deregister_fid(fid: int, by_frame: dict[int, list[pycolmap.Image]]) -> int:
        if fid in rec.frames and rec.frames[fid].has_pose():
            rec.deregister_frame(fid)
            return len(by_frame.get(fid, ()))
        return 0

    def _prune_outlier_frames(min_single_pts: int = 35, min_rig_pts: int = 50) -> dict[str, int]:
        reg_ids = list(rec.reg_image_ids())
        if not reg_ids:
            return {"global": 0, "trajectory_jump": 0, "secondary_drift": 0}

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

        removed_global = 0
        for fid, f_imgs in list(by_frame.items()):
            f_pts = sum(int(im.num_points3D) for im in f_imgs)
            f_max_d = max(dists[im.image_id] for im in f_imgs)
            min_req = min_single_pts if len(f_imgs) == 1 else min_rig_pts
            if f_max_d > max_dist or f_pts < min_req:
                removed_global += _deregister_fid(fid, by_frame)

        # Group surviving primary-rig frames by keyframe index to detect teleportation spikes
        rig_kf_entries: dict[int, tuple[float, set[int], list[np.ndarray]]] = {}
        secondary_entries: list[tuple[int, int, np.ndarray]] = []  # (kf_idx, frame_id, center)
        for iid in list(rec.reg_image_ids()):
            im = rec.images[iid]
            m = _KF_NAME_RE.match(im.name)
            if not m:
                continue
            vname, kf_idx, ts_ms = m.group(1), int(m.group(2)), int(m.group(3))
            t_cw = im.cam_from_world()
            c_im = -t_cw.rotation.matrix().T @ t_cw.translation
            if _is_primary_view(vname):
                ts_s, fids, clist = rig_kf_entries.setdefault(kf_idx, (ts_ms / 1000.0, set(), []))
                fids.add(im.frame_id)
                clist.append(c_im)
            else:
                secondary_entries.append((kf_idx, im.frame_id, c_im))

        removed_jump = 0
        ordered_kfs = sorted(rig_kf_entries.keys())
        if len(ordered_kfs) >= 6:
            kf_centers = {k: np.mean(rig_kf_entries[k][2], axis=0) for k in ordered_kfs}
            kf_times = {k: rig_kf_entries[k][0] for k in ordered_kfs}
            steps = [
                float(np.linalg.norm(kf_centers[ordered_kfs[i + 1]] - kf_centers[ordered_kfs[i]]))
                for i in range(len(ordered_kfs) - 1)
            ]
            med_step = max(float(np.median(steps)), 0.15)
            jump_thresh = max(3.5 * med_step, 1.85)
            dev_thresh = max(3.0 * med_step, 1.55)

            bad_kfs: set[int] = set()
            # Iteratively scan short round-trip excursions (1 to 4 keyframes) that spike off trajectory and back
            clean_kfs = list(ordered_kfs)
            for _pass in range(2):
                newly_flagged: set[int] = set()
                n_ck = len(clean_kfs)
                for exc_len in (1, 2, 3, 4):
                    for i in range(1, n_ck - exc_len):
                        ka = clean_kfs[i - 1]
                        kb = clean_kfs[i + exc_len]
                        span_kfs = clean_kfs[i : i + exc_len]
                        if any(k in newly_flagged for k in span_kfs) or ka in newly_flagged or kb in newly_flagged:
                            continue
                        ca, cb = kf_centers[ka], kf_centers[kb]
                        d_in = float(np.linalg.norm(kf_centers[span_kfs[0]] - ca))
                        d_out = float(np.linalg.norm(cb - kf_centers[span_kfs[-1]]))
                        d_bridge = float(np.linalg.norm(cb - ca))
                        if d_in <= jump_thresh or d_out <= jump_thresh:
                            continue
                        if (d_in + d_out) <= max(2.1 * d_bridge, 2.0 * jump_thresh):
                            continue
                        dt_ab = max(0.5, kf_times[kb] - kf_times[ka])
                        devs = []
                        for km in span_kfs:
                            alpha = np.clip((kf_times[km] - kf_times[ka]) / dt_ab, 0.0, 1.0)
                            c_interp = (1.0 - alpha) * ca + alpha * cb
                            devs.append(float(np.linalg.norm(kf_centers[km] - c_interp)))
                        if min(devs) > dev_thresh:
                            newly_flagged.update(span_kfs)
                if not newly_flagged:
                    break
                bad_kfs.update(newly_flagged)
                clean_kfs = [k for k in clean_kfs if k not in bad_kfs]

            for bk in bad_kfs:
                for fid in rig_kf_entries[bk][1]:
                    removed_jump += _deregister_fid(fid, by_frame)
                rig_kf_entries.pop(bk, None)

        # Physical dual-lens co-location gate: secondary fisheye views (up* / back_*) are ~7 cm from Stream 0
        # on the rigid camera body, so any secondary view solved far from the primary rig center drifted on sky/clouds.
        removed_drift = 0
        surviving_kfs = sorted(rig_kf_entries.keys())
        if len(surviving_kfs) >= 4 and secondary_entries:
            surv_centers = {k: np.mean(rig_kf_entries[k][2], axis=0) for k in surviving_kfs}
            surv_steps = [
                float(np.linalg.norm(surv_centers[surviving_kfs[i + 1]] - surv_centers[surviving_kfs[i]]))
                for i in range(len(surviving_kfs) - 1)
            ]
            med_step_s = max(float(np.median(surv_steps)), 0.15)
            max_lens_offset = max(0.40, 0.85 * med_step_s)
            seen_sec_fids: set[int] = set()
            for kf_idx, sec_fid, c_sec in secondary_entries:
                if sec_fid in seen_sec_fids:
                    continue
                if kf_idx not in surv_centers or float(np.linalg.norm(c_sec - surv_centers[kf_idx])) > max_lens_offset:
                    removed_drift += _deregister_fid(sec_fid, by_frame)
                    seen_sec_fids.add(sec_fid)

        return {
            "global": removed_global,
            "trajectory_jump": removed_jump,
            "secondary_drift": removed_drift,
        }

    pre_p = _prune_outlier_frames(min_single_pts=35, min_rig_pts=50)

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

    post_p = _prune_outlier_frames(min_single_pts=30, min_rig_pts=45)
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
    total_pruned_imgs = sum(pre_p.values()) + sum(post_p.values())
    cleanup_stats = {
        "raw_registered_images": raw_reg_images,
        "pruned_outlier_images": total_pruned_imgs,
        "pruned_trajectory_jump_images": pre_p["trajectory_jump"] + post_p["trajectory_jump"],
        "pruned_drifted_secondary_images": pre_p["secondary_drift"] + post_p["secondary_drift"],
        "raw_points3D": raw_points,
        "pruned_noisy_points3D": len(bad_pre) + len(bad_post),
    }
    return flat, cleanup_stats


def _zbuffer_delaunay_prior(
    pts2d: np.ndarray,
    pcam: np.ndarray,
    mvs_size: int,
    f_s: float,
    cx_s: float,
    cy_s: float,
    ys_grid: np.ndarray,
    xs_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Z-buffered 2D-3D Delaunay slanted-plane prior `(d_prior, n_prior)` on `(ys_grid, xs_grid)`."""
    gh, gw = len(ys_grid), len(xs_grid)
    d_prior = np.zeros((gh, gw), dtype=np.float32)
    n_prior = np.zeros((gh, gw, 3), dtype=np.float32)
    n_prior[:, :, 2] = -1.0
    if len(pts2d) < 6:
        return d_prior, n_prior

    # Coarse Z-buffer culling so occluded far-wall/back-yard points don't corrupt foreground roof triangles
    zc = 24
    zg = mvs_size // zc + 1
    zbuf = np.full((zg, zg), 1e9, dtype=np.float32)
    gx = np.clip((pts2d[:, 0] // zc).astype(int), 0, zg - 1)
    gy = np.clip((pts2d[:, 1] // zc).astype(int), 0, zg - 1)
    np.minimum.at(zbuf, (gy, gx), pcam[:, 2])
    zmin3 = zbuf.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            zmin3 = np.minimum(zmin3, np.roll(np.roll(zbuf, dy, axis=0), dx, axis=1))
    vis = pcam[:, 2] <= zmin3[gy, gx] * 1.18
    pts2d, pcam = pts2d[vis], pcam[vis]
    if len(pts2d) < 6:
        return d_prior, n_prior

    # Quantize to 8x8 px cells keeping front-biased tertile depth -> fast Subdiv2D (<450 vertices)
    q = 8
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx in range(len(pts2d)):
        key = (int(pts2d[idx, 0] // q), int(pts2d[idx, 1] // q))
        buckets.setdefault(key, []).append(idx)
    c2_list, c3_list = [], []
    for idxs in buckets.values():
        zs = pcam[idxs, 2]
        mi = idxs[int(np.argsort(zs)[len(zs) // 3])]
        c2_list.append(pts2d[mi])
        c3_list.append(pcam[mi])
    clean_2d = np.array(c2_list, dtype=np.float32)
    clean_3d = np.array(c3_list, dtype=np.float32)
    if len(clean_2d) < 4:
        return d_prior, n_prior

    subdiv = cv2.Subdiv2D((0, 0, mvs_size, mvs_size))
    for px, py in clean_2d:
        subdiv.insert((float(np.clip(px, 0.5, mvs_size - 1.5)), float(np.clip(py, 0.5, mvs_size - 1.5))))
    tris = subdiv.getTriangleList()

    for t_coords in tris:
        tri2d = t_coords.reshape(3, 2)
        if np.any(tri2d < 0.0) or np.any(tri2d >= mvs_size):
            continue
        v0 = tri2d[2] - tri2d[0]
        v1 = tri2d[1] - tri2d[0]
        area2 = abs(float(v0[0] * v1[1] - v0[1] * v1[0]))
        if area2 < 72.0:  # Skip tiny triangles inside already-dense regions
            continue
        i0 = int(np.argmin(np.sum((clean_2d - tri2d[0]) ** 2, axis=1)))
        i1 = int(np.argmin(np.sum((clean_2d - tri2d[1]) ** 2, axis=1)))
        i2 = int(np.argmin(np.sum((clean_2d - tri2d[2]) ** 2, axis=1)))
        if i0 == i1 or i1 == i2 or i0 == i2:
            continue
        p0, p1, p2 = clean_3d[i0], clean_3d[i1], clean_3d[i2]
        z_min, z_max = float(min(p0[2], p1[2], p2[2])), float(max(p0[2], p1[2], p2[2]))
        if z_max / max(z_min, 0.1) > 2.2:
            continue
        n_vec = np.cross(p1 - p0, p2 - p0)
        n_len = float(np.linalg.norm(n_vec))
        if n_len < 1e-6:
            continue
        n_vec /= n_len
        if n_vec[2] > 0:
            n_vec = -n_vec
        ray_c = (p0 + p1 + p2) / 3.0
        ray_c /= float(np.linalg.norm(ray_c)) + 1e-6
        if abs(float(np.dot(n_vec, ray_c))) < 0.20:
            continue
        plane_d = -float(np.dot(n_vec, p0))

        x_lo = int(np.searchsorted(xs_grid, np.min(tri2d[:, 0]), side="left"))
        x_hi = int(np.searchsorted(xs_grid, np.max(tri2d[:, 0]), side="right"))
        y_lo = int(np.searchsorted(ys_grid, np.min(tri2d[:, 1]), side="left"))
        y_hi = int(np.searchsorted(ys_grid, np.max(tri2d[:, 1]), side="right"))
        if x_lo >= x_hi or y_lo >= y_hi:
            continue

        sub_x, sub_y = np.meshgrid(xs_grid[x_lo:x_hi], ys_grid[y_lo:y_hi])
        v2_x = sub_x - tri2d[0, 0]
        v2_y = sub_y - tri2d[0, 1]
        dot00, dot01, dot11 = float(np.dot(v0, v0)), float(np.dot(v0, v1)), float(np.dot(v1, v1))
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-6:
            continue
        dot02 = v0[0] * v2_x + v0[1] * v2_y
        dot12 = v1[0] * v2_x + v1[1] * v2_y
        u_b = (dot11 * dot02 - dot01 * dot12) / denom
        v_b = (dot00 * dot12 - dot01 * dot02) / denom
        inside = (u_b >= -1e-3) & (v_b >= -1e-3) & (u_b + v_b <= 1.001)
        if not np.any(inside):
            continue

        vx = (sub_x[inside] - cx_s) / f_s
        vy = (sub_y[inside] - cy_s) / f_s
        ndotv = n_vec[0] * vx + n_vec[1] * vy + n_vec[2]
        z_interp = np.clip(-plane_d / np.minimum(ndotv, -0.15), z_min * 0.92, z_max * 1.08)
        d_prior[y_lo:y_hi, x_lo:x_hi][inside] = z_interp
        n_prior[y_lo:y_hi, x_lo:x_hi][inside] = n_vec

    return d_prior, n_prior


def _seed_sparse_starved_surfaces(
    flat_rec: pycolmap.Reconstruction,
    images_dir: Path,
    active_specs: list[VirtualViewSpec] | None = None,
    mvs_size: int = 640,
    max_seeded_points: int = 75000,
) -> dict:
    """Populate sparse-starved surfaces (e.g. dark shingle roofs & smooth gables) via 2-pass CUDA MVS."""
    import torch
    import torch.nn.functional as F

    t0 = time.time()
    if flat_rec.num_reg_images() < 8 or flat_rec.num_points3D() < 100:
        return {"enabled": False, "seeded_points3D": 0, "reason": "insufficient_sfm_base"}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    primary_names = {s.name for s in active_specs if s.in_primary_rig} if active_specs else None

    # Fast O(tracks) point-to-image index
    img_p3d_ids: dict[int, list[int]] = {}
    pts_xyz_list = []
    for idx, (_, pt) in enumerate(flat_rec.points3D.items()):
        pts_xyz_list.append(pt.xyz)
        for el in pt.track.elements:
            img_p3d_ids.setdefault(el.image_id, []).append(idx)
    pts_xyz_arr = np.array(pts_xyz_list, dtype=np.float32)

    views_info: dict[int, dict] = {}
    rig_centers_by_kf: dict[int, list[np.ndarray]] = {}
    for iid in sorted(flat_rec.reg_image_ids()):
        im = flat_rec.images[iid]
        m = _KF_NAME_RE.match(im.name)
        if not m:
            continue
        vname, kf_idx = m.group(1), int(m.group(2))
        if primary_names is not None and vname not in primary_names:
            continue
        if primary_names is None and vname.startswith(("up", "back_")):
            continue
        p_idxs = img_p3d_ids.get(iid, [])
        if len(p_idxs) < 65:
            continue
        cam = flat_rec.cameras[im.camera_id]
        tcw = im.cam_from_world()
        r_cw = tcw.rotation.matrix().astype(np.float32)
        t_cw = tcw.translation.astype(np.float32)
        c_w = (-r_cw.T @ t_cw).astype(np.float32)
        rig_centers_by_kf.setdefault(kf_idx, []).append(c_w)

        xyz = pts_xyz_arr[p_idxs]
        p_cam = xyz @ r_cw.T + t_cw[None, :]
        vz = p_cam[:, 2] > 0.15
        if int(np.sum(vz)) < 50:
            continue
        p_cam, xyz = p_cam[vz], xyz[vz]
        f, cx, cy = float(cam.params[0]), float(cam.params[1]), float(cam.params[2])
        xy_2d = np.stack([f * p_cam[:, 0] / p_cam[:, 2] + cx, f * p_cam[:, 1] / p_cam[:, 2] + cy], axis=-1)
        in_img = (xy_2d[:, 0] >= 0) & (xy_2d[:, 0] < cam.width) & (xy_2d[:, 1] >= 0) & (xy_2d[:, 1] < cam.height)
        if int(np.sum(in_img)) < 45:
            continue
        views_info[iid] = {
            "iid": iid,
            "name": im.name,
            "vname": vname,
            "kf": kf_idx,
            "W": cam.width,
            "H": cam.height,
            "f": f,
            "cx": cx,
            "cy": cy,
            "R": r_cw,
            "t": t_cw,
            "C": c_w,
            "p3d_set": set(p_idxs),
            "xyz": xyz[in_img],
            "p_cam": p_cam[in_img],
            "xy_2d": xy_2d[in_img],
            "d_p05": float(np.percentile(p_cam[in_img, 2], 5)),
            "d_med": float(np.median(p_cam[in_img, 2])),
            "d_p90": float(np.percentile(p_cam[in_img, 2], 90)),
        }

    if len(views_info) < 6:
        return {"enabled": False, "seeded_points3D": 0, "reason": "insufficient_primary_views"}

    ord_kfs = sorted(rig_centers_by_kf.keys())
    kf_c = [np.mean(rig_centers_by_kf[k], axis=0) for k in ord_kfs]
    steps = [float(np.linalg.norm(kf_c[i + 1] - kf_c[i])) for i in range(len(kf_c) - 1)] if len(kf_c) >= 2 else [0.5]
    med_step = max(float(np.median(steps)), 0.15)

    # Select well-spaced reference views that have sparse holes (prioritize oblique views that see pitched roofs/walls)
    cell_sz = 26
    g_dim = mvs_size // cell_sz + 1
    candidates_ranked = sorted(
        views_info.values(),
        key=lambda v: (0 if v["vname"] != "nadir" else 1, v["kf"], v["vname"]),
    )
    selected_refs: list[dict] = []
    accepted_pos_by_vname: dict[str, list[np.ndarray]] = {}
    min_ref_spacing = 0.28 * med_step
    for v in candidates_ranked:
        prev_pos = accepted_pos_by_vname.get(v["vname"], [])
        if any(float(np.linalg.norm(v["C"] - p)) < min_ref_spacing for p in prev_pos):
            continue
        sc = mvs_size / float(v["W"])
        occ = np.zeros((g_dim, g_dim), dtype=np.int32)
        p2 = v["xy_2d"] * sc
        np.add.at(
            occ,
            (np.clip((p2[:, 1] // cell_sz).astype(int), 0, g_dim - 1), np.clip((p2[:, 0] // cell_sz).astype(int), 0, g_dim - 1)),
            1,
        )
        empty_frac = float(np.mean(occ[1:-1, 1:-1] <= 1))
        if empty_frac >= 0.20:
            selected_refs.append(v)
            accepted_pos_by_vname.setdefault(v["vname"], []).append(v["C"])
    if len(selected_refs) > 260:
        idx_sub = np.linspace(0, len(selected_refs) - 1, 260, dtype=int)
        selected_refs = [selected_refs[i] for i in idx_sub]

    def _select_neighbors(ref_v: dict, max_k: int = 8) -> list[int]:
        w, h, f, cx, cy = ref_v["W"], ref_v["H"], ref_v["f"], ref_v["cx"], ref_v["cy"]
        ug, vg = np.meshgrid(np.linspace(0.2 * w, 0.8 * w, 5), np.linspace(0.2 * h, 0.8 * h, 5))
        dc = np.stack([(ug - cx) / f, (vg - cy) / f, np.ones_like(ug)], axis=-1).reshape(-1, 3).astype(np.float32)
        frust_pts = ref_v["C"][None, :] + ref_v["d_med"] * (dc @ ref_v["R"])
        sample_pts = np.vstack([ref_v["xyz"][:: max(1, len(ref_v["xyz"]) // 60)], frust_pts])
        v_ref = sample_pts - ref_v["C"][None, :]
        n_ref = np.linalg.norm(v_ref, axis=1) + 1e-6
        scored = []
        for sid, sv in views_info.items():
            if sv["kf"] == ref_v["kf"]:
                continue
            base = float(np.linalg.norm(sv["C"] - ref_v["C"]))
            if base < 0.035 * ref_v["d_med"] or base > 0.90 * ref_v["d_med"]:
                continue
            p_cam = sample_pts @ sv["R"].T + sv["t"][None, :]
            valid_z = p_cam[:, 2] > 0.2
            if float(np.mean(valid_z)) < 0.28:
                continue
            u = sv["f"] * p_cam[:, 0] / np.maximum(p_cam[:, 2], 1e-4) + sv["cx"]
            vy = sv["f"] * p_cam[:, 1] / np.maximum(p_cam[:, 2], 1e-4) + sv["cy"]
            in_fov = valid_z & (u >= 0.06 * sv["W"]) & (u <= 0.94 * sv["W"]) & (vy >= 0.06 * sv["H"]) & (vy <= 0.94 * sv["H"])
            if float(np.mean(in_fov)) < 0.24:
                continue
            v_src = sample_pts - sv["C"][None, :]
            n_src = np.linalg.norm(v_src, axis=1) + 1e-6
            cos_ang = np.clip(np.sum(v_ref * v_src, axis=1) / (n_ref * n_src), -1.0, 1.0)
            ang_deg = np.degrees(np.arccos(cos_ang))
            good_ang = in_fov & (ang_deg >= 1.6) & (ang_deg <= 28.0)
            frac_good = float(np.mean(good_ang))
            if frac_good < 0.18:
                continue
            shared = len(ref_v["p3d_set"] & sv["p3d_set"])
            t_bonus = 1.25 if abs(sv["kf"] - ref_v["kf"]) <= 10 else 1.0
            scored.append((t_bonus * (frac_good * 100.0 + min(shared, 250) * 0.4), sid))
        scored.sort(reverse=True)
        return [sid for _, sid in scored[:max_k]]

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    # Bounded LRU-style cache so CPU/GPU memory stays lightweight even on 2,500-image datasets
    img_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _load_tensors(v: dict) -> tuple[torch.Tensor, torch.Tensor] | None:
        iid = v["iid"]
        if iid in img_cache:
            return img_cache[iid]
        if len(img_cache) > 320:
            img_cache.clear()
        bgr = cv2.imread(str(images_dir / v["name"]))
        if bgr is None:
            return None
        small = cv2.resize(bgr, (mvs_size, mvs_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        cl = clahe.apply(gray).astype(np.float32) / 255.0
        cl = cv2.GaussianBlur(cl, (3, 3), 0.55)
        t_cl = torch.from_numpy(cl).to(device)[None, None]
        t_rgb = torch.from_numpy(rgb).permute(2, 0, 1).to(device)[None]
        img_cache[iid] = (t_cl, t_rgb)
        return t_cl, t_rgb

    patch_r = 3  # 7x7 patch on 640x640
    offsets = [(dy, dx) for dy in range(-patch_r, patch_r + 1) for dx in range(-patch_r, patch_r + 1)]
    p_len = len(offsets)
    off_t = torch.tensor(offsets, dtype=torch.float32, device=device)
    stride = 8
    ys_grid = np.arange(16, mvs_size - 16, stride)
    xs_grid = np.arange(16, mvs_size - 16, stride)
    yy, xx = np.meshgrid(ys_grid, xs_grid, indexing="ij")
    gh, gw = yy.shape

    nb_map = {rv["iid"]: _select_neighbors(rv, max_k=8) for rv in selected_refs}
    active_refs = [rv for rv in selected_refs if len(nb_map[rv["iid"]]) >= 3]

    pass1_xyz_list: list[np.ndarray] = []
    pass1_rgb_list: list[np.ndarray] = []

    # Pass 1: Fronto-parallel multi-view CLAHE-ZNCC + RGB sweep (captures dark-roof ridges, valleys, dormers, edges)
    with torch.no_grad():
        for rv in active_refs:
            loaded = _load_tensors(rv)
            if loaded is None:
                continue
            t_cl_ref, t_rgb_ref = loaded
            nb_ids = nb_map[rv["iid"]]
            scale = mvs_size / float(rv["W"])
            f_r, cx_r, cy_r = rv["f"] * scale, rv["cx"] * scale, rv["cy"] * scale

            occ = np.zeros((g_dim, g_dim), dtype=np.int32)
            p2 = rv["xy_2d"] * scale
            np.add.at(
                occ,
                (np.clip((p2[:, 1] // cell_sz).astype(int), 0, g_dim - 1), np.clip((p2[:, 0] // cell_sz).astype(int), 0, g_dim - 1)),
                1,
            )
            rgb_np = t_rgb_ref[0].permute(1, 2, 0).cpu().numpy()
            sky_like = (rgb_np[:, :, 2] > 0.68) & (rgb_np[:, :, 1] > 0.62) & (rgb_np[:, :, 0] > 0.55)
            cand_idx = np.flatnonzero((occ[yy // cell_sz, xx // cell_sz] <= 1) & (~sky_like[yy, xx]))
            if len(cand_idx) < 16:
                continue

            u_c = torch.from_numpy(xx.ravel()[cand_idx].astype(np.float32)).to(device)
            v_c = torch.from_numpy(yy.ravel()[cand_idx].astype(np.float32)).to(device)
            n_cand = len(u_c)
            pu = u_c[:, None] + off_t[None, :, 1]
            pv = v_c[:, None] + off_t[None, :, 0]
            grid_ref = torch.stack([(pu / (mvs_size - 1)) * 2.0 - 1.0, (pv / (mvs_size - 1)) * 2.0 - 1.0], dim=-1)[None]
            ref_cl_p = F.grid_sample(t_cl_ref, grid_ref, mode="bilinear", align_corners=True)[0, 0]
            ref_rgb_p = F.grid_sample(t_rgb_ref, grid_ref, mode="bilinear", align_corners=True)[0].permute(1, 2, 0)
            ref_zero = ref_cl_p - ref_cl_p.mean(dim=1, keepdim=True)
            ref_std = torch.sqrt((ref_zero**2).mean(dim=1) + 1e-6)

            dirs_cam = torch.stack([(pu - cx_r) / f_r, (pv - cy_r) / f_r, torch.ones_like(pu)], dim=-1)
            r_ref_t = torch.from_numpy(rv["R"]).to(device)
            c_ref_t = torch.from_numpy(rv["C"]).to(device)
            dirs_world = torch.einsum("npk,kj->npj", dirs_cam, r_ref_t)

            n_depths = 112
            d_min = max(0.30, 0.55 * rv["d_p05"])
            d_max = min(16.0, 1.25 * rv["d_p90"])
            if d_max <= d_min * 1.05:
                continue
            depths = 1.0 / torch.linspace(1.0 / d_max, 1.0 / d_min, n_depths, device=device)

            valid_nbs = []
            for sid in nb_ids:
                sl = _load_tensors(views_info[sid])
                if sl is not None:
                    valid_nbs.append((sid, sl[0], sl[1]))
            if len(valid_nbs) < 3:
                continue

            k_cnt = len(valid_nbs)
            zncc_all = torch.full((k_cnt, n_depths, n_cand), -1.0, device=device)
            rgb_all = torch.full((k_cnt, n_depths, n_cand), 1.0, device=device)

            for k_idx, (sid, t_cl_src, t_rgb_src) in enumerate(valid_nbs):
                sv = views_info[sid]
                s_scale = mvs_size / float(sv["W"])
                f_s, cx_s, cy_s = sv["f"] * s_scale, sv["cx"] * s_scale, sv["cy"] * s_scale
                r_src_t = torch.from_numpy(sv["R"]).to(device)
                t_src_t = torch.from_numpy(sv["t"]).to(device)
                base_cam = r_src_t @ c_ref_t + t_src_t
                ray_cam = torch.einsum("jk,npk->npj", r_src_t, dirs_world)

                for d0 in range(0, n_depths, 32):
                    dc = depths[d0 : d0 + 32]
                    xz = base_cam[2] + dc[:, None, None] * ray_cam[None, :, :, 2]
                    xx_c = base_cam[0] + dc[:, None, None] * ray_cam[None, :, :, 0]
                    xy_c = base_cam[1] + dc[:, None, None] * ray_cam[None, :, :, 1]
                    inv_z = 1.0 / torch.clamp(xz, min=1e-3)
                    su, sv_y = f_s * xx_c * inv_z + cx_s, f_s * xy_c * inv_z + cy_s
                    in_b = (
                        (xz[:, :, p_len // 2] > 0.2)
                        & (su[:, :, p_len // 2] >= 6)
                        & (su[:, :, p_len // 2] <= mvs_size - 7)
                        & (sv_y[:, :, p_len // 2] >= 6)
                        & (sv_y[:, :, p_len // 2] <= mvs_size - 7)
                    )
                    gs = torch.stack(
                        [(su / (mvs_size - 1)) * 2.0 - 1.0, (sv_y / (mvs_size - 1)) * 2.0 - 1.0],
                        dim=-1,
                    ).reshape(1, len(dc) * n_cand, p_len, 2)
                    scl = F.grid_sample(t_cl_src, gs, mode="bilinear", align_corners=True)[0, 0].reshape(len(dc), n_cand, p_len)
                    srgb = F.grid_sample(t_rgb_src, gs, mode="bilinear", align_corners=True)[0].permute(1, 2, 0).reshape(len(dc), n_cand, p_len, 3)
                    sz = scl - scl.mean(dim=2, keepdim=True)
                    zncc = (sz * ref_zero[None]).mean(dim=2) / (torch.sqrt((sz**2).mean(dim=2) + 1e-6) * ref_std[None] + 1e-6)
                    l1 = torch.abs(srgb - ref_rgb_p[None]).mean(dim=(2, 3))
                    zncc_all[k_idx, d0 : d0 + 32] = torch.where(in_b, zncc, torch.full_like(zncc, -1.0))
                    rgb_all[k_idx, d0 : d0 + 32] = torch.where(in_b, l1, torch.ones_like(l1))

            comb = zncc_all - 2.5 * torch.clamp(rgb_all - 0.035, min=0.0)
            n_votes = ((zncc_all >= 0.72) & (rgb_all <= 0.065)).sum(dim=0)
            score_dn = torch.where(
                n_votes >= 3,
                torch.topk(comb, k=min(3, k_cnt), dim=0).values.mean(dim=0),
                torch.full((n_depths, n_cand), -1.0, device=device),
            )
            best_sc, best_di = score_dn.max(dim=0)
            far = torch.abs(torch.arange(n_depths, device=device)[:, None] - best_di[None, :]) > 4
            sec_sc = torch.where(far, score_dn, torch.full_like(score_dn, -1.0)).max(dim=0).values
            keep = (ref_std > 0.035) & (best_sc >= 0.76) & ((best_sc - sec_sc) >= 0.035)

            d_map = np.zeros((gh, gw), dtype=np.float32)
            knp = keep.cpu().numpy()
            d_map.ravel()[cand_idx[knp]] = depths[best_di].cpu().numpy()[knp]
            agree = np.zeros((gh, gw), dtype=np.int32)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    sh = np.roll(np.roll(d_map, dy, axis=0), dx, axis=1)
                    agree += ((d_map > 0) & (sh > 0) & (np.abs(d_map - sh) / np.maximum(d_map, 1e-4) < 0.035)).astype(np.int32)
            agree[0, :] = agree[-1, :] = agree[:, 0] = agree[:, -1] = 0
            fy, fx = np.nonzero((d_map > 0) & (agree >= 3))
            if len(fy) > 0:
                ua, va, da = xx[fy, fx].astype(np.float32), yy[fy, fx].astype(np.float32), d_map[fy, fx]
                dca = np.stack([(ua - cx_r) / f_r, (va - cy_r) / f_r, np.ones_like(ua)], axis=-1)
                pass1_xyz_list.append(rv["C"][None, :] + da[:, None] * (dca @ rv["R"]))
                pass1_rgb_list.append((rgb_np[va.astype(int), ua.astype(int)] * 255.0).clip(0, 255).astype(np.uint8))

        p1_xyz = np.vstack(pass1_xyz_list) if pass1_xyz_list else np.zeros((0, 3), dtype=np.float32)
        p1_rgb = np.vstack(pass1_rgb_list) if pass1_rgb_list else np.zeros((0, 3), dtype=np.uint8)

        # Pass 2: Z-buffered Delaunay Slanted-Plane MVS to fill pitched dark roof facets & smooth gables
        pass2_xyz_list: list[np.ndarray] = []
        pass2_rgb_list: list[np.ndarray] = []
        for rv in active_refs:
            loaded = _load_tensors(rv)
            if loaded is None:
                continue
            t_cl_ref, t_rgb_ref = loaded
            nb_ids = nb_map[rv["iid"]]
            scale = mvs_size / float(rv["W"])
            f_r, cx_r, cy_r = rv["f"] * scale, rv["cx"] * scale, rv["cy"] * scale

            aug_xyz = np.vstack([rv["xyz"], p1_xyz]) if len(p1_xyz) else rv["xyz"]
            p_cam_aug = aug_xyz @ rv["R"].T + rv["t"][None, :]
            vz = (p_cam_aug[:, 2] > 0.25) & (p_cam_aug[:, 2] < 1.3 * rv["d_p90"])
            p_cam_aug = p_cam_aug[vz]
            if len(p_cam_aug) < 12:
                continue
            xy_aug = np.stack([f_r * p_cam_aug[:, 0] / p_cam_aug[:, 2] + cx_r, f_r * p_cam_aug[:, 1] / p_cam_aug[:, 2] + cy_r], axis=-1)
            in_b2 = (xy_aug[:, 0] >= 0) & (xy_aug[:, 0] < mvs_size) & (xy_aug[:, 1] >= 0) & (xy_aug[:, 1] < mvs_size)
            xy_aug, p_cam_aug = xy_aug[in_b2], p_cam_aug[in_b2]

            occ = np.zeros((g_dim, g_dim), dtype=np.int32)
            np.add.at(
                occ,
                (np.clip((xy_aug[:, 1] // cell_sz).astype(int), 0, g_dim - 1), np.clip((xy_aug[:, 0] // cell_sz).astype(int), 0, g_dim - 1)),
                1,
            )
            d_prior, n_prior = _zbuffer_delaunay_prior(xy_aug, p_cam_aug, mvs_size, f_r, cx_r, cy_r, ys_grid, xs_grid)
            rgb_np = t_rgb_ref[0].permute(1, 2, 0).cpu().numpy()
            sky_like = (rgb_np[:, :, 2] > 0.68) & (rgb_np[:, :, 1] > 0.62) & (rgb_np[:, :, 0] > 0.55)
            cand_idx = np.flatnonzero((occ[yy // cell_sz, xx // cell_sz] <= 1) & (d_prior > 0.25) & (~sky_like[yy, xx]))
            if len(cand_idx) < 10:
                continue

            u_c = torch.from_numpy(xx.ravel()[cand_idx].astype(np.float32)).to(device)
            v_c = torch.from_numpy(yy.ravel()[cand_idx].astype(np.float32)).to(device)
            dp_c = torch.from_numpy(d_prior.ravel()[cand_idx]).to(device)
            np_c = torch.from_numpy(n_prior.reshape(-1, 3)[cand_idx]).to(device)
            n_cand = len(u_c)

            pu = u_c[:, None] + off_t[None, :, 1]
            pv = v_c[:, None] + off_t[None, :, 0]
            grid_ref = torch.stack([(pu / (mvs_size - 1)) * 2.0 - 1.0, (pv / (mvs_size - 1)) * 2.0 - 1.0], dim=-1)[None]
            ref_cl_p = F.grid_sample(t_cl_ref, grid_ref, mode="bilinear", align_corners=True)[0, 0]
            ref_rgb_p = F.grid_sample(t_rgb_ref, grid_ref, mode="bilinear", align_corners=True)[0].permute(1, 2, 0)
            ref_zero = ref_cl_p - ref_cl_p.mean(dim=1, keepdim=True)
            ref_std = torch.sqrt((ref_zero**2).mean(dim=1) + 1e-6)

            v0_cam = torch.stack([(u_c - cx_r) / f_r, (v_c - cy_r) / f_r, torch.ones_like(u_c)], dim=-1)
            vp_cam = torch.stack([(pu - cx_r) / f_r, (pv - cy_r) / f_r, torch.ones_like(pu)], dim=-1)
            n_dot_v0 = torch.sum(np_c * v0_cam, dim=-1, keepdim=True)
            n_dot_vp = torch.sum(np_c[:, None, :] * vp_cam, dim=-1)
            slant_ratio = torch.clamp(n_dot_v0 / torch.clamp(n_dot_vp, max=-0.12), 0.75, 1.35)
            slant_dirs_cam = vp_cam * slant_ratio[:, :, None]

            r_ref_t = torch.from_numpy(rv["R"]).to(device)
            c_ref_t = torch.from_numpy(rv["C"]).to(device)
            dirs_world = torch.einsum("npk,kj->npj", slant_dirs_cam, r_ref_t)

            d_steps = 48
            mults = torch.linspace(0.78, 1.22, d_steps, device=device)
            d_cand = mults[:, None] * dp_c[None, :]

            valid_nbs = []
            for sid in nb_ids:
                sl = _load_tensors(views_info[sid])
                if sl is not None:
                    valid_nbs.append((sid, sl[0], sl[1]))
            if len(valid_nbs) < 3:
                continue

            k_cnt = len(valid_nbs)
            zncc_all = torch.full((k_cnt, d_steps, n_cand), -1.0, device=device)
            rgb_all = torch.full((k_cnt, d_steps, n_cand), 1.0, device=device)

            for k_idx, (sid, t_cl_src, t_rgb_src) in enumerate(valid_nbs):
                sv = views_info[sid]
                s_scale = mvs_size / float(sv["W"])
                f_s, cx_s, cy_s = sv["f"] * s_scale, sv["cx"] * s_scale, sv["cy"] * s_scale
                r_src_t = torch.from_numpy(sv["R"]).to(device)
                t_src_t = torch.from_numpy(sv["t"]).to(device)
                base_cam = r_src_t @ c_ref_t + t_src_t
                ray_cam = torch.einsum("jk,npk->npj", r_src_t, dirs_world)

                xz = base_cam[2] + d_cand[:, :, None] * ray_cam[None, :, :, 2]
                xx_c = base_cam[0] + d_cand[:, :, None] * ray_cam[None, :, :, 0]
                xy_c = base_cam[1] + d_cand[:, :, None] * ray_cam[None, :, :, 1]
                inv_z = 1.0 / torch.clamp(xz, min=1e-3)
                su, sv_y = f_s * xx_c * inv_z + cx_s, f_s * xy_c * inv_z + cy_s
                in_b = (
                    (xz[:, :, p_len // 2] > 0.2)
                    & (su[:, :, p_len // 2] >= 6)
                    & (su[:, :, p_len // 2] <= mvs_size - 7)
                    & (sv_y[:, :, p_len // 2] >= 6)
                    & (sv_y[:, :, p_len // 2] <= mvs_size - 7)
                )
                gs = torch.stack(
                    [(su / (mvs_size - 1)) * 2.0 - 1.0, (sv_y / (mvs_size - 1)) * 2.0 - 1.0],
                    dim=-1,
                ).reshape(1, d_steps * n_cand, p_len, 2)
                scl = F.grid_sample(t_cl_src, gs, mode="bilinear", align_corners=True)[0, 0].reshape(d_steps, n_cand, p_len)
                srgb = F.grid_sample(t_rgb_src, gs, mode="bilinear", align_corners=True)[0].permute(1, 2, 0).reshape(d_steps, n_cand, p_len, 3)
                sz = scl - scl.mean(dim=2, keepdim=True)
                zncc = (sz * ref_zero[None]).mean(dim=2) / (torch.sqrt((sz**2).mean(dim=2) + 1e-6) * ref_std[None] + 1e-6)
                l1 = torch.abs(srgb - ref_rgb_p[None]).mean(dim=(2, 3))
                zncc_all[k_idx] = torch.where(in_b, zncc, torch.full_like(zncc, -1.0))
                rgb_all[k_idx] = torch.where(in_b, l1, torch.ones_like(l1))

            comb = zncc_all - 2.5 * torch.clamp(rgb_all - 0.035, min=0.0)
            n_votes = ((zncc_all >= 0.68) & (rgb_all <= 0.060)).sum(dim=0)
            score_dn = torch.where(
                n_votes >= 3,
                torch.topk(comb, k=min(3, k_cnt), dim=0).values.mean(dim=0),
                torch.full((d_steps, n_cand), -1.0, device=device),
            )
            best_sc, best_di = score_dn.max(dim=0)
            far = torch.abs(torch.arange(d_steps, device=device)[:, None] - best_di[None, :]) > 3
            sec_sc = torch.where(far, score_dn, torch.full_like(score_dn, -1.0)).max(dim=0).values
            best_d = d_cand[best_di, torch.arange(n_cand, device=device)]
            keep = (ref_std > 0.030) & (best_sc >= 0.71) & ((best_sc - sec_sc) >= 0.025)

            d_map = np.zeros((gh, gw), dtype=np.float32)
            knp = keep.cpu().numpy()
            d_map.ravel()[cand_idx[knp]] = best_d.cpu().numpy()[knp]
            agree = np.zeros((gh, gw), dtype=np.int32)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    sh = np.roll(np.roll(d_map, dy, axis=0), dx, axis=1)
                    agree += ((d_map > 0) & (sh > 0) & (np.abs(d_map - sh) / np.maximum(d_map, 1e-4) < 0.035)).astype(np.int32)
            agree[0, :] = agree[-1, :] = agree[:, 0] = agree[:, -1] = 0
            fy, fx = np.nonzero((d_map > 0) & (agree >= 4))
            if len(fy) > 0:
                ua, va, da = xx[fy, fx].astype(np.float32), yy[fy, fx].astype(np.float32), d_map[fy, fx]
                dca = np.stack([(ua - cx_r) / f_r, (va - cy_r) / f_r, np.ones_like(ua)], axis=-1)
                pass2_xyz_list.append(rv["C"][None, :] + da[:, None] * (dca @ rv["R"]))
                pass2_rgb_list.append((rgb_np[va.astype(int), ua.astype(int)] * 255.0).clip(0, 255).astype(np.uint8))

    img_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    p2_xyz = np.vstack(pass2_xyz_list) if pass2_xyz_list else np.zeros((0, 3), dtype=np.float32)
    p2_rgb = np.vstack(pass2_rgb_list) if pass2_rgb_list else np.zeros((0, 3), dtype=np.uint8)
    raw_xyz = np.vstack([a for a in (p1_xyz, p2_xyz) if len(a)]) if (len(p1_xyz) or len(p2_xyz)) else np.zeros((0, 3), dtype=np.float32)
    raw_rgb = np.vstack([a for a in (p1_rgb, p2_rgb) if len(a)]) if (len(p1_rgb) or len(p2_rgb)) else np.zeros((0, 3), dtype=np.uint8)
    if len(raw_xyz) == 0:
        return {"enabled": True, "device": str(device), "seeded_points3D": 0, "seeding_time_s": round(time.time() - t0, 2)}

    # Scale-adaptive 3D voxel deduplication (avoids crowding existing SIFT points & guarantees non-zero spacing)
    voxel_sz = float(np.clip(0.035 * med_step, 0.015, 0.05))
    existing_vox: set[tuple[int, int, int]] = set()
    sift_vox = np.floor(pts_xyz_arr / voxel_sz).astype(np.int64)
    for vx, vy, vz in sift_vox:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    existing_vox.add((int(vx + dx), int(vy + dy), int(vz + dz)))

    new_vox_acc: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, int]] = {}
    cand_vox = np.floor(raw_xyz / voxel_sz).astype(np.int64)
    for idx in range(len(raw_xyz)):
        key = (int(cand_vox[idx, 0]), int(cand_vox[idx, 1]), int(cand_vox[idx, 2]))
        if key in existing_vox:
            continue
        if key in new_vox_acc:
            sx, sr, cnt = new_vox_acc[key]
            new_vox_acc[key] = (sx + raw_xyz[idx].astype(np.float64), sr + raw_rgb[idx].astype(np.float64), cnt + 1)
        else:
            new_vox_acc[key] = (raw_xyz[idx].astype(np.float64), raw_rgb[idx].astype(np.float64), 1)

    vox_items = list(new_vox_acc.values())
    if len(vox_items) > max_seeded_points:
        # Prioritize multi-view corroborated voxels first
        vox_items.sort(key=lambda item: item[2], reverse=True)
        vox_items = vox_items[:max_seeded_points]

    empty_track = pycolmap.Track()
    for sx, sr, cnt in vox_items:
        xyz_mean = (sx / cnt).astype(np.float64)
        rgb_mean = np.clip(np.round(sr / cnt), 0, 255).astype(np.uint8)
        flat_rec.add_point3D(xyz_mean, empty_track, rgb_mean)

    return {
        "enabled": True,
        "device": str(device),
        "reference_views_scanned": len(active_refs),
        "raw_pass1_points": int(len(p1_xyz)),
        "raw_pass2_slanted_points": int(len(p2_xyz)),
        "voxel_size": round(voxel_sz, 5),
        "seeded_points3D": len(vox_items),
        "seeding_time_s": round(time.time() - t0, 2),
    }


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
    max_num_features: int = 16384,
    use_spherical_rig: bool = True,
    dense_surface_seeding: bool = True,
) -> dict:
    """Run SIFT feature extraction, analytical spherical rig binding, smart matching, GLOMAP/BA, and CUDA MVS seeding."""
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
    fopts.sift.peak_threshold = 0.003

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
        active_specs=active_specs,
        use_rig_constraints=(rig_sensor_count >= 2),
    )
    sfm_points_count = int(clean_rec.num_points3D())
    mean_reproj_px = round(float(clean_rec.compute_mean_reprojection_error()), 4)

    seeding_stats: dict = {"enabled": False, "seeded_points3D": 0}
    if dense_surface_seeding:
        seeding_stats = _seed_sparse_starved_surfaces(
            clean_rec,
            images_dir=images_dir,
            active_specs=active_specs,
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
        "sfm_points3D": sfm_points_count,
        "mvs_seeded_points3D": int(seeding_stats.get("seeded_points3D", 0)),
        "num_points3D": int(clean_rec.num_points3D()),
        "mean_reprojection_error_px": mean_reproj_px,
        "matched_pairs": num_pairs,
        "cleanup": cleanup_stats,
        "dense_seeding": seeding_stats,
        "feature_extraction_s": round(t_feat, 2),
        "matching_s": round(t_match, 2),
        "mapping_and_ba_s": round(t_map, 2),
        "sparse_model_path": str(target0),
    }
