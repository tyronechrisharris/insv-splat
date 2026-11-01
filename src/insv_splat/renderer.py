"""Real-time server-side 3D Gaussian Splatting GPU rasterizer (PyTorch CUDA + Triton sm_120).

Matches Brush v0.3.0 forward projection, SH evaluation, StopThePop exact tile culling,
and front-to-back alpha compositing with fully vectorized 256-thread Triton kernels.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import numpy as np
import torch
import triton
import triton.language as tl
from plyfile import PlyData

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2_0 = 1.0925484305920792
SH_C2_1 = -1.0925484305920792
SH_C2_2 = 0.31539156525252005
SH_C2_3 = -1.0925484305920792
SH_C2_4 = 0.5462742152960396
SH_C3_0 = -0.5900435899266435
SH_C3_1 = 2.890611442640554
SH_C3_2 = -0.4570457994644658
SH_C3_3 = 0.3731763325901154
SH_C3_4 = -0.4570457994644658
SH_C3_5 = 1.445305721320277
SH_C3_6 = -0.5900435899266435

TILE_SIZE = 8
COV_BLUR = 0.3


@triton.jit
def _project_splats_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    s00_ptr,
    s01_ptr,
    s02_ptr,
    s11_ptr,
    s12_ptr,
    s22_ptr,
    pwr_ptr,
    cam_x,
    cam_y,
    cam_z,
    r00,
    r01,
    r02,
    r10,
    r11,
    r12,
    r20,
    r21,
    r22,
    fx,
    fy,
    cx_px,
    cy_px,
    lims_neg_x,
    lims_pos_x,
    lims_neg_y,
    lims_pos_y,
    scale_sq,
    img_w,
    img_h,
    num_tiles_x,
    num_tiles_y,
    n_total,
    out_keep_ptr,
    out_zv_ptr,
    out_ux_ptr,
    out_uy_ptr,
    out_ca_ptr,
    out_cb_ptr,
    out_cc_ptr,
    out_tmin_x_ptr,
    out_tmax_x_ptr,
    out_tmin_y_ptr,
    out_tmax_y_ptr,
    out_ntiles_ptr,
):
    gid = tl.program_id(0) * 256 + tl.arange(0, 256)
    in_bounds = gid < n_total

    px = tl.load(x_ptr + gid, mask=in_bounds, other=0.0) - cam_x
    py = tl.load(y_ptr + gid, mask=in_bounds, other=0.0) - cam_y
    pz = tl.load(z_ptr + gid, mask=in_bounds, other=0.0) - cam_z

    zc = r20 * px + r21 * py + r22 * pz
    in_front = in_bounds & (zc >= 0.01) & (zc <= 1e10)

    xc = r00 * px + r01 * py + r02 * pz
    yc = r10 * px + r11 * py + r12 * pz
    rz = tl.where(in_front, 1.0 / zc, 0.0)

    nx = xc * rz
    ny = yc * rz
    ux = fx * nx + cx_px
    uy = fy * ny + cy_px

    clip_x = tl.minimum(lims_pos_x, tl.maximum(lims_neg_x, nx))
    clip_y = tl.minimum(lims_pos_y, tl.maximum(lims_neg_y, ny))
    duv_x = fx * rz
    duv_y = fy * rz

    t00 = duv_x * (r00 - clip_x * r20)
    t01 = duv_x * (r01 - clip_x * r21)
    t02 = duv_x * (r02 - clip_x * r22)

    t10 = duv_y * (r10 - clip_y * r20)
    t11 = duv_y * (r11 - clip_y * r21)
    t12 = duv_y * (r12 - clip_y * r22)

    s00 = tl.load(s00_ptr + gid, mask=in_front, other=0.0)
    s01 = tl.load(s01_ptr + gid, mask=in_front, other=0.0)
    s02 = tl.load(s02_ptr + gid, mask=in_front, other=0.0)
    s11 = tl.load(s11_ptr + gid, mask=in_front, other=0.0)
    s12 = tl.load(s12_ptr + gid, mask=in_front, other=0.0)
    s22 = tl.load(s22_ptr + gid, mask=in_front, other=0.0)

    v0x = s00 * t00 + s01 * t01 + s02 * t02
    v0y = s01 * t00 + s11 * t01 + s12 * t02
    v0z = s02 * t00 + s12 * t01 + s22 * t02

    v1x = s00 * t10 + s01 * t11 + s02 * t12
    v1y = s01 * t10 + s11 * t11 + s12 * t12
    v1z = s02 * t10 + s12 * t11 + s22 * t12

    c_xx = (t00 * v0x + t01 * v0y + t02 * v0z) * scale_sq + 0.3
    c_xy = (t00 * v1x + t01 * v1y + t02 * v1z) * scale_sq
    c_yy = (t10 * v1x + t11 * v1y + t12 * v1z) * scale_sq + 0.3
    det = c_xx * c_yy - c_xy * c_xy

    pwr = tl.load(pwr_ptr + gid, mask=in_front, other=0.0)
    ext_x = tl.sqrt(tl.maximum(2.0 * pwr * c_xx, 0.0))
    ext_y = tl.sqrt(tl.maximum(2.0 * pwr * c_yy, 0.0))

    fw = img_w.to(tl.float32)
    fh = img_h.to(tl.float32)
    max_ext = tl.maximum(fw, fh) * 2.0

    valid = (
        in_front
        & (det > 1e-12)
        & (pwr > 0.0)
        & (ext_x <= max_ext)
        & (ext_y <= max_ext)
        & (ux + ext_x > 0.0)
        & (ux - ext_x < fw)
        & (uy + ext_y > 0.0)
        & (uy - ext_y < fh)
    )

    tmin_x = tl.minimum(num_tiles_x, tl.maximum(0, tl.floor((ux - ext_x) * 0.125).to(tl.int32)))
    tmax_x = tl.minimum(num_tiles_x, tl.maximum(0, tl.floor((ux + ext_x) * 0.125).to(tl.int32) + 1))
    tmin_y = tl.minimum(num_tiles_y, tl.maximum(0, tl.floor((uy - ext_y) * 0.125).to(tl.int32)))
    tmax_y = tl.minimum(num_tiles_y, tl.maximum(0, tl.floor((uy + ext_y) * 0.125).to(tl.int32) + 1))

    ntiles = (tmax_x - tmin_x) * (tmax_y - tmin_y)
    keep = valid & (ntiles > 0)

    tl.store(out_keep_ptr + gid, keep, mask=in_bounds)
    inv_det = tl.where(keep, 1.0 / det, 0.0)
    tl.store(out_zv_ptr + gid, zc, mask=keep)
    tl.store(out_ux_ptr + gid, ux, mask=keep)
    tl.store(out_uy_ptr + gid, uy, mask=keep)
    tl.store(out_ca_ptr + gid, c_yy * inv_det, mask=keep)
    tl.store(out_cb_ptr + gid, -c_xy * inv_det, mask=keep)
    tl.store(out_cc_ptr + gid, c_xx * inv_det, mask=keep)
    tl.store(out_tmin_x_ptr + gid, tmin_x, mask=keep)
    tl.store(out_tmax_x_ptr + gid, tmax_x, mask=keep)
    tl.store(out_tmin_y_ptr + gid, tmin_y, mask=keep)
    tl.store(out_tmax_y_ptr + gid, tmax_y, mask=keep)
    tl.store(out_ntiles_ptr + gid, ntiles, mask=keep)


@triton.jit
def _expand_tiles_kernel(
    tmin_x_ptr,
    tmax_x_ptr,
    tmin_y_ptr,
    tmax_y_ptr,
    ux_ptr,
    uy_ptr,
    ca_ptr,
    cb_ptr,
    cc_ptr,
    pwr_ptr,
    depths_bits_ptr,
    offsets_ptr,
    out_keys_ptr,
    out_vals_ptr,
    num_tiles_x,
    num_tiles_total,
    n_vis,
    total_isects,
):
    iid = (tl.program_id(0) * 256 + tl.arange(0, 256)).to(tl.int64)
    active = iid < total_isects

    # Load-balanced binary search: 1 thread per tile intersection (immune to giant sky splats)
    lo = tl.zeros([256], dtype=tl.int32)
    hi = tl.full([256], n_vis - 1, dtype=tl.int32)
    for _ in tl.static_range(23):
        mid = (lo + hi) >> 1
        end_m = tl.load(offsets_ptr + mid, mask=active, other=0)
        go_right = iid >= end_m
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_right, hi, mid)

    sid = lo
    x0 = tl.load(tmin_x_ptr + sid, mask=active, other=0)
    x1 = tl.load(tmax_x_ptr + sid, mask=active, other=1)
    y0 = tl.load(tmin_y_ptr + sid, mask=active, other=0)
    y1 = tl.load(tmax_y_ptr + sid, mask=active, other=1)
    w = tl.maximum(x1 - x0, 1)
    total = w * (y1 - y0)

    end_off = tl.load(offsets_ptr + sid, mask=active, other=0)
    start_off = end_off - total.to(tl.int64)
    k = (iid - start_off).to(tl.int32)
    dy = k // w
    dx = k - dy * w
    tx = x0 + dx
    ty = y0 + dy

    ux = tl.load(ux_ptr + sid, mask=active, other=0.0)
    uy = tl.load(uy_ptr + sid, mask=active, other=0.0)
    ca = tl.load(ca_ptr + sid, mask=active, other=1.0)
    cb = tl.load(cb_ptr + sid, mask=active, other=0.0)
    cc = tl.load(cc_ptr + sid, mask=active, other=1.0)
    pwr = tl.load(pwr_ptr + sid, mask=active, other=0.0)

    rx0 = (tx * 8).to(tl.float32)
    ry0 = (ty * 8).to(tl.float32)
    rx1 = rx0 + 8.0
    ry1 = ry0 + 8.0

    x_left = ux < rx0
    x_right = ux > rx1
    in_x = (~x_left) & (~x_right)
    y_above = uy < ry0
    y_below = uy > ry1
    in_y = (~y_above) & (~y_below)

    corner_x = tl.where(x_left, rx0, rx1)
    corner_y = tl.where(y_above, ry0, ry1)
    dx_edge = tl.where(x_left, 8.0, -8.0)
    dy_edge = tl.where(y_above, 8.0, -8.0)
    diff_x = ux - corner_x
    diff_y = uy - corner_y

    tx_opt = tl.where(
        in_y,
        0.0,
        tl.minimum(1.0, tl.maximum(0.0, (ca * diff_x + cb * diff_y) / tl.where(ca > 1e-12, ca * dx_edge, 1.0))),
    )
    ty_opt = tl.where(
        in_x,
        0.0,
        tl.minimum(1.0, tl.maximum(0.0, (cb * diff_x + cc * diff_y) / tl.where(cc > 1e-12, cc * dy_edge, 1.0))),
    )
    px_opt = corner_x + tx_opt * dx_edge
    py_opt = corner_y + ty_opt * dy_edge
    ddx = ux - px_opt
    ddy = uy - py_opt
    min_sigma = tl.where(in_x & in_y, 0.0, 0.5 * (ca * ddx * ddx + cc * ddy * ddy) + cb * ddx * ddy)
    hits = (total == 1) | (min_sigma <= pwr)

    depth_u32 = tl.load(depths_bits_ptr + sid, mask=active, other=0).to(tl.int64) & 0xFFFFFFFF
    sentinel_tile = num_tiles_total.to(tl.int64)
    tile_id = tl.where(hits, (ty * num_tiles_x + tx).to(tl.int64), sentinel_tile)
    key = (tile_id << 32) | depth_u32
    tl.store(out_keys_ptr + iid, key, mask=active)
    tl.store(out_vals_ptr + iid, sid, mask=active)


@triton.jit
def _rasterize_tiles_kernel(
    tile_starts_ptr,
    tile_ends_ptr,
    sorted_vals_ptr,
    uv_x_ptr,
    uv_y_ptr,
    conic_a_ptr,
    conic_b_ptr,
    conic_c_ptr,
    opac_ptr,
    rgb_r_ptr,
    rgb_g_ptr,
    rgb_b_ptr,
    out_img_ptr,
    img_w,
    img_h,
    num_tiles_x,
    bg_r,
    bg_g,
    bg_b,
):
    tile_id = tl.program_id(0)
    ty = tile_id // num_tiles_x
    tx = tile_id - ty * num_tiles_x

    lanes = tl.arange(0, 64)
    lx = lanes & 7
    ly = lanes >> 3
    px_i = tx * 8 + lx
    py_i = ty * 8 + ly
    inside = (px_i < img_w) & (py_i < img_h)

    px = px_i.to(tl.float32) + 0.5
    py = py_i.to(tl.float32) + 0.5

    start = tl.load(tile_starts_ptr + tile_id)
    raw_end = tl.load(tile_ends_ptr + tile_id)
    end = tl.minimum(raw_end, start + 4096)

    T = tl.full([64], 1.0, dtype=tl.float32)
    done = ~inside
    acc_r = tl.zeros([64], dtype=tl.float32)
    acc_g = tl.zeros([64], dtype=tl.float32)
    acc_b = tl.zeros([64], dtype=tl.float32)

    i = start
    while i < end:
        s = tl.load(sorted_vals_ptr + i)
        ux = tl.load(uv_x_ptr + s)
        uy = tl.load(uv_y_ptr + s)
        ca = tl.load(conic_a_ptr + s)
        cb = tl.load(conic_b_ptr + s)
        cc = tl.load(conic_c_ptr + s)
        op = tl.load(opac_ptr + s)

        dx = ux - px
        dy = uy - py
        sigma = 0.5 * (ca * dx * dx + cc * dy * dy) + cb * dx * dy
        raw_alpha = tl.minimum(0.999, op * tl.exp(-sigma))
        active = (~done) & (sigma >= 0.0) & (raw_alpha >= (1.0 / 255.0))
        alpha = tl.where(active, raw_alpha, 0.0)

        next_T = T * (1.0 - alpha)
        newly_done = active & (next_T <= 1e-3)
        contrib = active & (~newly_done)

        vis = tl.where(contrib, alpha * T, 0.0)
        r = tl.load(rgb_r_ptr + s)
        g = tl.load(rgb_g_ptr + s)
        b = tl.load(rgb_b_ptr + s)
        acc_r += vis * r
        acc_g += vis * g
        acc_b += vis * b
        T = tl.where(contrib, next_T, T)
        done = done | newly_done

        i += 1
        if ((i - start) & 15) == 0:
            if tl.min(done.to(tl.int32), axis=0) == 1:
                i = end

    final_r = tl.minimum(255.0, tl.maximum(0.0, (acc_r + T * bg_r) * 255.0 + 0.5)).to(tl.uint8)
    final_g = tl.minimum(255.0, tl.maximum(0.0, (acc_g + T * bg_g) * 255.0 + 0.5)).to(tl.uint8)
    final_b = tl.minimum(255.0, tl.maximum(0.0, (acc_b + T * bg_b) * 255.0 + 0.5)).to(tl.uint8)

    base = (py_i * img_w + px_i) * 3
    tl.store(out_img_ptr + base + 0, final_r, mask=inside)
    tl.store(out_img_ptr + base + 1, final_g, mask=inside)
    tl.store(out_img_ptr + base + 2, final_b, mask=inside)


class GaussianSceneGPU:
    """GPU-resident 3D Gaussian Splatting scene with SOA 3D covariances and SH bands."""

    def __init__(self, ply_path: Path, device: str = "cuda:0", min_opacity: float = 1.0 / 255.0):
        t0 = time.time()
        self.ply_path = Path(ply_path)
        self.device = torch.device(device)
        ply = PlyData.read(str(ply_path))
        self.comments = list(getattr(ply, "comments", []) or [])
        v = ply["vertex"].data
        total_raw = len(v)

        raw_opac = torch.from_numpy(np.asarray(v["opacity"], dtype=np.float32))
        opac = torch.sigmoid(raw_opac)

        rots_all = np.stack(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            axis=-1,
        ).astype(np.float32)
        rot_norm_sq = np.sum(rots_all * rots_all, axis=-1)

        xyz_all = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
        finite_mask = np.isfinite(xyz_all).all(axis=-1) & (rot_norm_sq >= 1e-6)
        keep_np = (opac.numpy() >= min_opacity) & finite_mask

        self.num_raw = total_raw
        self.opacities = opac[torch.from_numpy(keep_np)].to(self.device, non_blocking=True).contiguous()
        self.log_255_opac = torch.log(255.0 * self.opacities).contiguous()

        xyz = xyz_all[keep_np]
        self.xyz = torch.from_numpy(xyz).to(self.device, non_blocking=True).contiguous()
        self.x = self.xyz[:, 0].contiguous()
        self.y = self.xyz[:, 1].contiguous()
        self.z = self.xyz[:, 2].contiguous()
        self.num_gaussians = self.xyz.shape[0]
        n = self.num_gaussians

        scales = np.stack(
            [v["scale_0"][keep_np], v["scale_1"][keep_np], v["scale_2"][keep_np]], axis=-1
        ).astype(np.float32)
        rots = rots_all[keep_np]

        s_t = torch.exp(torch.from_numpy(scales).to(self.device))
        q_t = torch.from_numpy(rots).to(self.device)
        q_t = q_t * torch.rsqrt(torch.clamp(torch.sum(q_t * q_t, dim=-1, keepdim=True), min=1e-12))
        qw, qx, qy, qz = q_t[:, 0], q_t[:, 1], q_t[:, 2], q_t[:, 3]

        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        r01 = 2.0 * (qx * qy - qw * qz)
        r02 = 2.0 * (qx * qz + qw * qy)
        r10 = 2.0 * (qx * qy + qw * qz)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        r12 = 2.0 * (qy * qz - qw * qx)
        r20 = 2.0 * (qx * qz - qw * qy)
        r21 = 2.0 * (qy * qz + qw * qx)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

        R = torch.stack(
            [
                torch.stack([r00, r01, r02], dim=-1),
                torch.stack([r10, r11, r12], dim=-1),
                torch.stack([r20, r21, r22], dim=-1),
            ],
            dim=-2,
        )
        M = R * s_t.unsqueeze(-2)
        cov3d = M @ M.transpose(-1, -2)
        self.s00 = cov3d[:, 0, 0].contiguous()
        self.s01 = cov3d[:, 0, 1].contiguous()
        self.s02 = cov3d[:, 0, 2].contiguous()
        self.s11 = cov3d[:, 1, 1].contiguous()
        self.s12 = cov3d[:, 1, 2].contiguous()
        self.s22 = cov3d[:, 2, 2].contiguous()
        del cov3d, M, R

        # Preallocate reusable projection scratch buffers so per-frame projection does zero mallocs
        self._buf_keep = torch.empty((n,), dtype=torch.bool, device=self.device)
        self._buf_zv = torch.empty((n,), dtype=torch.float32, device=self.device)
        self._buf_ux = torch.empty((n,), dtype=torch.float32, device=self.device)
        self._buf_uy = torch.empty((n,), dtype=torch.float32, device=self.device)
        self._buf_ca = torch.empty((n,), dtype=torch.float32, device=self.device)
        self._buf_cb = torch.empty((n,), dtype=torch.float32, device=self.device)
        self._buf_cc = torch.empty((n,), dtype=torch.float32, device=self.device)
        self._buf_tmin_x = torch.empty((n,), dtype=torch.int32, device=self.device)
        self._buf_tmax_x = torch.empty((n,), dtype=torch.int32, device=self.device)
        self._buf_tmin_y = torch.empty((n,), dtype=torch.int32, device=self.device)
        self._buf_tmax_y = torch.empty((n,), dtype=torch.int32, device=self.device)
        self._buf_ntiles = torch.empty((n,), dtype=torch.int32, device=self.device)

        dc = np.stack(
            [v["f_dc_0"][keep_np], v["f_dc_1"][keep_np], v["f_dc_2"][keep_np]], axis=-1
        ).astype(np.float16)
        self.sh_dc = torch.from_numpy(dc).to(self.device).contiguous()

        names = set(v.dtype.names or ())
        if "f_rest_44" in names:
            self.sh_degree = 3
            rest = np.empty((n, 3, 15), dtype=np.float16)
            for c in range(3):
                for b in range(15):
                    rest[:, c, b] = v[f"f_rest_{c * 15 + b}"][keep_np]
            self.sh_rest = torch.from_numpy(np.transpose(rest, (0, 2, 1)).copy()).to(self.device).contiguous()
        elif "f_rest_23" in names:
            self.sh_degree = 2
            rest = np.empty((n, 3, 8), dtype=np.float16)
            for c in range(3):
                for b in range(8):
                    rest[:, c, b] = v[f"f_rest_{c * 8 + b}"][keep_np]
            self.sh_rest = torch.from_numpy(np.transpose(rest, (0, 2, 1)).copy()).to(self.device).contiguous()
        elif "f_rest_8" in names:
            self.sh_degree = 1
            rest = np.empty((n, 3, 3), dtype=np.float16)
            for c in range(3):
                for b in range(3):
                    rest[:, c, b] = v[f"f_rest_{c * 3 + b}"][keep_np]
            self.sh_rest = torch.from_numpy(np.transpose(rest, (0, 2, 1)).copy()).to(self.device).contiguous()
        else:
            self.sh_degree = 0
            self.sh_rest = None

        high_op = self.opacities > 0.35
        core_xyz = self.xyz[high_op] if int(high_op.sum()) > 1000 else self.xyz
        sub_core = core_xyz[:: max(1, len(core_xyz) // 100000)]
        self.bounds_q10 = torch.quantile(sub_core, 0.10, dim=0).cpu().numpy().tolist()
        self.bounds_median = torch.quantile(sub_core, 0.50, dim=0).cpu().numpy().tolist()
        self.bounds_q90 = torch.quantile(sub_core, 0.90, dim=0).cpu().numpy().tolist()
        self.load_time_s = round(time.time() - t0, 2)

    def warmup(self) -> None:
        """Trigger Triton kernel compilation upfront so the first browser frame is instant."""
        c = np.asarray(self.bounds_median, dtype=np.float32) - np.array([0.0, 0.0, 2.0], dtype=np.float32)
        eye = np.eye(3, dtype=np.float32)
        self.render(c, eye, 1280, 720, fov_y_deg=75.0)

    @torch.no_grad()
    def render(
        self,
        cam_pos: np.ndarray,
        R_cw: np.ndarray,
        width: int,
        height: int,
        fov_y_deg: float = 75.0,
        sh_degree: int = 3,
        scale_modifier: float = 1.0,
        bg_color: tuple[float, float, float] = (0.07, 0.08, 0.10),
        cx: float | None = None,
        cy: float | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Render an RGB uint8 frame (H, W, 3) on GPU and return (rgb_tensor, stats)."""
        t0 = time.perf_counter()
        fy = (height / 2.0) / math.tan(math.radians(fov_y_deg / 2.0))
        fx = fy
        cx_val = (width / 2.0) if cx is None else float(cx)
        cy_val = (height / 2.0) if cy is None else float(cy)

        lims_pos_x = (1.15 * width - cx_val) / fx
        lims_neg_x = (-0.15 * width - cx_val) / fx
        lims_pos_y = (1.15 * height - cy_val) / fy
        lims_neg_y = (-0.15 * height - cy_val) / fy

        num_tiles_x = (width + TILE_SIZE - 1) // TILE_SIZE
        num_tiles_y = (height + TILE_SIZE - 1) // TILE_SIZE
        num_tiles = num_tiles_x * num_tiles_y

        R_np = np.asarray(R_cw, dtype=np.float32)
        C_np = np.asarray(cam_pos, dtype=np.float32)
        n_total = self.num_gaussians

        _project_splats_kernel[(triton.cdiv(n_total, 256),)](
            self.x,
            self.y,
            self.z,
            self.s00,
            self.s01,
            self.s02,
            self.s11,
            self.s12,
            self.s22,
            self.log_255_opac,
            float(C_np[0]),
            float(C_np[1]),
            float(C_np[2]),
            float(R_np[0, 0]),
            float(R_np[0, 1]),
            float(R_np[0, 2]),
            float(R_np[1, 0]),
            float(R_np[1, 1]),
            float(R_np[1, 2]),
            float(R_np[2, 0]),
            float(R_np[2, 1]),
            float(R_np[2, 2]),
            float(fx),
            float(fy),
            float(cx_val),
            float(cy_val),
            float(lims_neg_x),
            float(lims_pos_x),
            float(lims_neg_y),
            float(lims_pos_y),
            float(scale_modifier * scale_modifier),
            int(width),
            int(height),
            int(num_tiles_x),
            int(num_tiles_y),
            int(n_total),
            self._buf_keep,
            self._buf_zv,
            self._buf_ux,
            self._buf_uy,
            self._buf_ca,
            self._buf_cb,
            self._buf_cc,
            self._buf_tmin_x,
            self._buf_tmax_x,
            self._buf_tmin_y,
            self._buf_tmax_y,
            self._buf_ntiles,
        )

        orig_idx = torch.nonzero(self._buf_keep, as_tuple=False).squeeze(-1)
        n_vis = int(orig_idx.numel())

        if n_vis == 0:
            blank = torch.empty((height, width, 3), dtype=torch.uint8, device=self.device)
            blank[..., 0] = int(round(bg_color[0] * 255))
            blank[..., 1] = int(round(bg_color[1] * 255))
            blank[..., 2] = int(round(bg_color[2] * 255))
            return blank, {"visible_splats": 0, "tile_intersections": 0, "render_ms": round((time.perf_counter() - t0) * 1000, 2)}

        zv = self._buf_zv[orig_idx].contiguous()
        uv_x = self._buf_ux[orig_idx].contiguous()
        uv_y = self._buf_uy[orig_idx].contiguous()
        conic_a = self._buf_ca[orig_idx].contiguous()
        conic_b = self._buf_cb[orig_idx].contiguous()
        conic_c = self._buf_cc[orig_idx].contiguous()
        tmin_x = self._buf_tmin_x[orig_idx].contiguous()
        tmax_x = self._buf_tmax_x[orig_idx].contiguous()
        tmin_y = self._buf_tmin_y[orig_idx].contiguous()
        tmax_y = self._buf_tmax_y[orig_idx].contiguous()
        ntiles = self._buf_ntiles[orig_idx].contiguous()
        opac_v = self.opacities[orig_idx].contiguous()
        pwr_v = self.log_255_opac[orig_idx].contiguous()

        # Evaluate Spherical Harmonics color on visible splats
        sh_dc = self.sh_dc[orig_idx].to(torch.float32)
        color = SH_C0 * sh_dc
        use_deg = min(sh_degree, self.sh_degree)
        if use_deg >= 1 and self.sh_rest is not None:
            C_t = torch.as_tensor(C_np, dtype=torch.float32, device=self.device)
            dirs = self.xyz[orig_idx] - C_t.unsqueeze(0)
            dirs = dirs * torch.rsqrt(torch.clamp(torch.sum(dirs * dirs, dim=-1, keepdim=True), min=1e-16))
            dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
            rest = self.sh_rest[orig_idx].to(torch.float32)

            color = color + (
                -SH_C1 * dy.unsqueeze(-1) * rest[:, 0]
                + SH_C1 * dz.unsqueeze(-1) * rest[:, 1]
                - SH_C1 * dx.unsqueeze(-1) * rest[:, 2]
            )
            if use_deg >= 2:
                xx, yy, zz = dx * dx, dy * dy, dz * dz
                xy, yz, xz = dx * dy, dy * dz, dx * dz
                color = color + (
                    SH_C2_0 * xy.unsqueeze(-1) * rest[:, 3]
                    + SH_C2_1 * yz.unsqueeze(-1) * rest[:, 4]
                    + SH_C2_2 * (2.0 * zz - xx - yy).unsqueeze(-1) * rest[:, 5]
                    + SH_C2_3 * xz.unsqueeze(-1) * rest[:, 6]
                    + SH_C2_4 * (xx - yy).unsqueeze(-1) * rest[:, 7]
                )
                if use_deg >= 3:
                    color = color + (
                        SH_C3_0 * (dy * (3.0 * xx - yy)).unsqueeze(-1) * rest[:, 8]
                        + SH_C3_1 * (xy * dz).unsqueeze(-1) * rest[:, 9]
                        + SH_C3_2 * (dy * (4.0 * zz - xx - yy)).unsqueeze(-1) * rest[:, 10]
                        + SH_C3_3 * (dz * (2.0 * zz - 3.0 * xx - 3.0 * yy)).unsqueeze(-1) * rest[:, 11]
                        + SH_C3_4 * (dx * (4.0 * zz - xx - yy)).unsqueeze(-1) * rest[:, 12]
                        + SH_C3_5 * (dz * (xx - yy)).unsqueeze(-1) * rest[:, 13]
                        + SH_C3_6 * (dx * (xx - 3.0 * yy)).unsqueeze(-1) * rest[:, 14]
                    )

        color = torch.clamp_min(color + 0.5, 0.0)
        rgb_r = color[:, 0].contiguous()
        rgb_g = color[:, 1].contiguous()
        rgb_b = color[:, 2].contiguous()

        offsets = torch.cumsum(ntiles.to(torch.int64), dim=0)
        total_isects = int(offsets[-1].item())
        out_keys = torch.empty((total_isects,), dtype=torch.int64, device=self.device)
        out_vals = torch.empty((total_isects,), dtype=torch.int32, device=self.device)
        depths_bits = zv.view(torch.int32)

        _expand_tiles_kernel[(triton.cdiv(total_isects, 256),)](
            tmin_x,
            tmax_x,
            tmin_y,
            tmax_y,
            uv_x,
            uv_y,
            conic_a,
            conic_b,
            conic_c,
            pwr_v,
            depths_bits,
            offsets,
            out_keys,
            out_vals,
            int(num_tiles_x),
            int(num_tiles),
            int(n_vis),
            int(total_isects),
        )

        order = torch.argsort(out_keys)
        sorted_keys = out_keys[order]
        sorted_vals = out_vals[order].contiguous()
        sorted_tiles = (sorted_keys >> 32).to(torch.int32).contiguous()

        tile_ids = torch.arange(num_tiles, dtype=torch.int32, device=self.device)
        tile_starts = torch.searchsorted(sorted_tiles, tile_ids, right=False).to(torch.int32).contiguous()
        tile_ends = torch.searchsorted(sorted_tiles, tile_ids, right=True).to(torch.int32).contiguous()

        out_img = torch.empty((height * width * 3,), dtype=torch.uint8, device=self.device)
        _rasterize_tiles_kernel[(num_tiles,)](
            tile_starts,
            tile_ends,
            sorted_vals,
            uv_x,
            uv_y,
            conic_a,
            conic_b,
            conic_c,
            opac_v,
            rgb_r,
            rgb_g,
            rgb_b,
            out_img,
            int(width),
            int(height),
            int(num_tiles_x),
            float(bg_color[0]),
            float(bg_color[1]),
            float(bg_color[2]),
        )
        torch.cuda.synchronize(self.device)
        render_ms = (time.perf_counter() - t0) * 1000.0
        return out_img.view(height, width, 3), {
            "visible_splats": n_vis,
            "tile_intersections": total_isects,
            "render_ms": round(render_ms, 2),
        }
