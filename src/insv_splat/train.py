"""3D Gaussian Splatting training & interactive viewing on discrete eGPU via Brush (Vulkan)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

VULKAN_ICD_CANDIDATES = (
    Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
    Path("/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json"),
    Path("/etc/vulkan/icd.d/nvidia_icd.json"),
    Path("/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json"),
)


def _find_nvidia_vulkan_icd() -> Path | None:
    for cand in VULKAN_ICD_CANDIDATES:
        if cand.is_file():
            return cand
    return None


def default_brush_binary() -> Path:
    """Locate Brush binary via BRUSH_BIN env var, bundled `bin/brush_app`, or PATH."""
    env_bin = os.environ.get("BRUSH_BIN")
    if env_bin:
        p = Path(os.path.expanduser(env_bin)).resolve()
        if p.is_file():
            return p
    bundled = Path(__file__).resolve().parents[2] / "bin" / "brush_app"
    if bundled.is_file():
        return bundled
    found = shutil.which("brush_app") or shutil.which("brush")
    if found:
        return Path(found)
    raise FileNotFoundError(
        f"Brush binary not found at {bundled}, $BRUSH_BIN, or on PATH. "
        "Run `./scripts/download_brush.sh` or install Brush from https://github.com/ArthurBrussee/brush/releases"
    )


def _vulkan_icd_usable(icd_path: Path) -> bool:
    """Verify that forcing `VK_ICD_FILENAMES=icd_path` actually enumerates a live Vulkan GPU."""
    test_env = os.environ.copy()
    test_env["VK_ICD_FILENAMES"] = str(icd_path)
    test_env["VK_DRIVER_FILES"] = str(icd_path)
    res = subprocess.run(
        ["vulkaninfo", "--summary"],
        env=test_env,
        capture_output=True,
        text=True,
        check=False,
    )
    return res.returncode == 0 and "GPU0" in res.stdout and "llvmpipe" not in res.stdout.split("GPU0:")[-1][:300]


def detect_egpu_env(prefer_egpu: bool = True) -> tuple[dict[str, str], dict]:
    """Configure Vulkan/wgpu environment variables to bind the discrete NVIDIA eGPU (with safe Vulkan fallback)."""
    env = os.environ.copy()
    gpu_info: dict = {
        "egpu_detected": False,
        "nvdec_ready": False,
        "vulkan_nvidia_ready": False,
        "gpu_name": "default",
        "vram_mib": 0,
    }

    if prefer_egpu:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            first = smi.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 4:
                gpu_info = {
                    "egpu_detected": True,
                    "nvdec_ready": True,
                    "vulkan_nvidia_ready": False,
                    "gpu_name": parts[0],
                    "vram_mib": int(parts[1]),
                    "driver_version": parts[2],
                    "pci_bus_id": parts[3],
                }
                env["WGPU_BACKEND"] = "vulkan"
                env["WGPU_POWER_PREF"] = "high"
                icd = _find_nvidia_vulkan_icd()
                if icd is not None and _vulkan_icd_usable(icd):
                    gpu_info["vulkan_nvidia_ready"] = True
                    env["WGPU_ADAPTER_NAME"] = parts[0]
                    env["VK_ICD_FILENAMES"] = str(icd)
                    env["VK_DRIVER_FILES"] = str(icd)
        else:
            # NVML/userspace driver mismatch (e.g. pending reboot after driver upgrade):
            # PyTorch CUDA forward-compat still runs on the NVIDIA eGPU while Brush uses active system Vulkan GPU.
            env["WGPU_BACKEND"] = "vulkan"
            env["WGPU_POWER_PREF"] = "high"
            vk_name = "system_vulkan"
            vk_res = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, check=False)
            if vk_res.returncode == 0:
                m_vk = re.search(r"deviceName\s*=\s*(.+)", vk_res.stdout)
                if m_vk:
                    vk_name = m_vk.group(1).strip()
            try:
                import torch

                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    gpu_info = {
                        "egpu_detected": True,
                        "nvdec_ready": False,
                        "vulkan_nvidia_ready": False,
                        "gpu_name": f"{props.name} (CUDA) + {vk_name} (Vulkan)",
                        "vram_mib": int(round(props.total_memory / (1024**2))),
                        "driver_version": "cuda_forward_compat_pending_reboot",
                        "pci_bus_id": f"{props.pci_bus_id:02x}:00.0" if hasattr(props, "pci_bus_id") else "cuda:0",
                    }
                else:
                    gpu_info["gpu_name"] = vk_name
            except Exception:
                gpu_info["gpu_name"] = vk_name

    return env, gpu_info


def _safe_float(val: str) -> float:
    try:
        return float(val)
    except ValueError:
        return 0.0


def query_egpu_live_usage() -> dict | None:
    """Sample live GPU utilization and VRAM consumption via nvidia-smi."""
    res = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return None
    parts = [p.strip() for p in res.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 4:
        return None
    return {
        "gpu_util_pct": _safe_float(parts[0]),
        "vram_used_mib": _safe_float(parts[1]),
        "power_w": _safe_float(parts[2]),
        "temp_c": _safe_float(parts[3]),
    }


def train_splats(
    dataset_dir: Path,
    export_dir: Path,
    total_steps: int = 30000,
    export_every: int = 5000,
    eval_every: int = 5000,
    eval_split_every: int = 20,
    max_resolution: int = 1600,
    max_splats: int = 8000000,
    refine_every: int = 250,
    growth_stop_iter: int = 20000,
    growth_grad_threshold: float = 0.000025,
    growth_select_fraction: float = 0.15,
    opac_loss_weight: float = 2e-9,
    scale_loss_weight: float = 2e-8,
    with_viewer: bool = False,
    prefer_egpu: bool = True,
    log_path: Path | None = None,
) -> dict:
    """Train a 3D Gaussian Splatting model using Brush on the eGPU and export standard .ply files."""
    brush_bin = default_brush_binary()
    export_dir.mkdir(parents=True, exist_ok=True)
    env, gpu_info = detect_egpu_env(prefer_egpu=prefer_egpu)

    cmd = [
        str(brush_bin),
        str(dataset_dir),
        "--total-steps",
        str(total_steps),
        "--export-every",
        str(export_every),
        "--export-path",
        str(export_dir),
        "--export-name",
        "splat_step_{iter}.ply",
        "--max-resolution",
        str(max_resolution),
        "--max-splats",
        str(max_splats),
        "--refine-every",
        str(refine_every),
        "--growth-stop-iter",
        str(growth_stop_iter),
        "--growth-grad-threshold",
        f"{growth_grad_threshold:.8f}",
        "--growth-select-fraction",
        str(growth_select_fraction),
        "--opac-loss-weight",
        f"{opac_loss_weight:.2e}",
        "--scale-loss-weight",
        f"{scale_loss_weight:.2e}",
        "--eval-split-every",
        str(eval_split_every),
        "--eval-every",
        str(eval_every),
        "--eval-save-to-disk",
    ]
    if with_viewer:
        cmd.append("--with-viewer")

    t0 = time.time()
    peak_vram_mib = 0.0
    peak_power_w = 0.0
    peak_util_pct = 0.0
    stop_evt = False

    import threading

    def _poll_gpu() -> None:
        nonlocal peak_vram_mib, peak_power_w, peak_util_pct
        while not stop_evt:
            live = query_egpu_live_usage()
            if live:
                peak_vram_mib = max(peak_vram_mib, live["vram_used_mib"])
                peak_power_w = max(peak_power_w, live["power_w"])
                peak_util_pct = max(peak_util_pct, live["gpu_util_pct"])
            time.sleep(2.0)

    if log_path is None:
        log_path = export_dir.parent / "brush_train.log"

    eval_metrics: dict[int, dict[str, float]] = {}
    eval_re = re.compile(r"Eval iter\s+(\d+):\s+PSNR\s+([0-9.]+),\s+ssim\s+([0-9.]+)")

    sampler = threading.Thread(target=_poll_gpu, daemon=True)
    sampler.start()
    try:
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            buf = ""
            while True:
                ch = proc.stdout.read(512)
                if not ch:
                    break
                log_f.write(ch)
                log_f.flush()
                buf += ch
                for m in eval_re.finditer(buf):
                    it_num = int(m.group(1))
                    psnr_v = float(m.group(2))
                    ssim_v = float(m.group(3))
                    if it_num not in eval_metrics:
                        eval_metrics[it_num] = {"psnr": round(psnr_v, 4), "ssim": round(ssim_v, 5)}
                        print(
                            f"  [Brush Eval @ step {it_num:,}] PSNR = {psnr_v:.2f} dB, SSIM = {ssim_v:.4f}",
                            flush=True,
                        )
                if len(buf) > 4096:
                    buf = buf[-2048:]
            rc = proc.wait()
    finally:
        stop_evt = True
        sampler.join(timeout=3.0)

    elapsed_s = time.time() - t0
    if rc != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:] if log_path.exists() else ""
        raise RuntimeError(f"Brush training exited with non-zero status {rc}:\n{tail}")

    ply_files = sorted(
        export_dir.glob("splat_step_*.ply"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)) if re.search(r"(\d+)", p.stem) else 0,
    )
    if not ply_files:
        raise RuntimeError(f"No exported .ply splats found in {export_dir}")

    latest_ply = ply_files[-1]
    final_ply = export_dir / "splat_final.ply"
    shutil.copy2(latest_ply, final_ply)

    # Count vertices (Gaussians) from PLY header
    num_gaussians = 0
    with final_ply.open("rb") as f:
        for _ in range(30):
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                num_gaussians = int(line.split()[-1])
                break
            if line == "end_header":
                break

    eval_renders = sorted(str(p) for p in export_dir.rglob("*.png"))
    final_eval = eval_metrics[max(eval_metrics.keys())] if eval_metrics else None
    return {
        "gpu": gpu_info,
        "peak_gpu_util_pct": round(peak_util_pct, 1),
        "peak_vram_used_mib": round(peak_vram_mib, 1),
        "peak_power_w": round(peak_power_w, 1),
        "total_steps": total_steps,
        "growth_stop_iter": growth_stop_iter,
        "growth_grad_threshold": growth_grad_threshold,
        "training_time_s": round(elapsed_s, 2),
        "eval_metrics_by_step": {str(k): v for k, v in sorted(eval_metrics.items())},
        "final_eval_psnr_db": final_eval["psnr"] if final_eval else None,
        "final_eval_ssim": final_eval["ssim"] if final_eval else None,
        "final_ply": str(final_ply),
        "final_ply_size_mb": round(final_ply.stat().st_size / (1024 * 1024), 2),
        "num_gaussians": num_gaussians,
        "checkpoints": [str(p) for p in ply_files],
        "eval_renders": eval_renders,
    }


def launch_viewer(target_path: Path, prefer_egpu: bool = True) -> int:
    """Open the interactive Brush 3DGS desktop viewer on the eGPU."""
    brush_bin = default_brush_binary()
    env, _ = detect_egpu_env(prefer_egpu=prefer_egpu)
    if target_path.is_dir():
        cand = target_path / "exports" / "splat_final.ply"
        if cand.is_file():
            target_path = cand
    res = subprocess.run([str(brush_bin), "--with-viewer", str(target_path)], env=env, check=False)
    return res.returncode
