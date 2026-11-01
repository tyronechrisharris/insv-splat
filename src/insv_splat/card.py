"""Removable SD/MMC card mounting and .insv file discovery."""

from __future__ import annotations

import getpass
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InsvSource:
    """Resolved .insv recording (single dual-stream file or paired _00_/_10_ files)."""

    primary_path: Path
    secondary_path: Path | None
    mtime: float
    size_bytes: int
    mount_point: Path | None


def ensure_card_mounted(preferred_device: str | None = None) -> list[Path]:
    """Ensure removable SD/MMC partitions are mounted and return active media mount points."""
    user = getpass.getuser()
    proc = subprocess.run(
        ["lsblk", "-J", "-o", "NAME,PATH,TYPE,RM,HOTPLUG,FSTYPE,MOUNTPOINTS"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    mount_points: list[Path] = []

    def visit(nodes: list[dict]) -> None:
        for node in nodes:
            dev_path = node.get("path") or f"/dev/{node['name']}"
            is_card = (
                bool(node.get("rm"))
                or bool(node.get("hotplug"))
                or str(node.get("name", "")).startswith("mmcblk")
            )
            fstype = (node.get("fstype") or "").lower()
            mps = [m for m in (node.get("mountpoints") or []) if m]

            if node.get("type") == "part" and fstype in {"exfat", "vfat", "fat32", "ntfs", "ext4"}:
                if preferred_device and dev_path != preferred_device:
                    pass
                elif is_card:
                    if not mps:
                        res = subprocess.run(
                            ["udisksctl", "mount", "-b", dev_path],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        if res.returncode == 0:
                            # Example: "Mounted /dev/mmcblk0p1 at /run/media/<user>/SD_CARD"
                            m = re.search(r" at (/[^\n\r]+?)\.?\s*$", res.stdout.strip())
                            if m:
                                mps = [m.group(1)]
                    for mp in mps:
                        mount_points.append(Path(mp))

            children = node.get("children") or []
            if children:
                visit(children)

    visit(data.get("blockdevices") or [])

    for root in (Path(f"/run/media/{user}"), Path(f"/media/{user}")):
        if root.is_dir():
            for child in root.iterdir():
                if child.is_dir() and child not in mount_points:
                    mount_points.append(child)

    return mount_points


def resolve_companion_insv(path: Path) -> tuple[Path, Path | None]:
    """Handle paired Insta360 files (VID_..._00_XXX.insv + VID_..._10_XXX.insv) if present."""
    name = path.name
    if "_10_" in name:
        cand0 = path.with_name(name.replace("_10_", "_00_", 1))
        if cand0.is_file():
            return cand0, path
    if "_00_" in name:
        cand1 = path.with_name(name.replace("_00_", "_10_", 1))
        if cand1.is_file():
            return path, cand1
    return path, None


def find_insv_files(search_roots: list[Path]) -> list[InsvSource]:
    """Find all .insv recordings under search_roots sorted newest-first."""
    seen: set[Path] = set()
    sources: list[InsvSource] = []

    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() == ".insv":
            candidates = [root]
            mp = None
        else:
            candidates = [
                p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".insv"
            ]
            mp = root

        for cand in candidates:
            primary, secondary = resolve_companion_insv(cand.resolve())
            if primary in seen:
                continue
            seen.add(primary)
            if secondary:
                seen.add(secondary)
            st = primary.stat()
            total_sz = st.st_size + (secondary.stat().st_size if secondary else 0)
            sources.append(
                InsvSource(
                    primary_path=primary,
                    secondary_path=secondary,
                    mtime=st.st_mtime,
                    size_bytes=total_sz,
                    mount_point=mp,
                )
            )

    sources.sort(key=lambda s: s.mtime, reverse=True)
    return sources


def resolve_input_insv(input_path: str | None = None, device: str | None = None) -> InsvSource:
    """Resolve an explicit .insv path/directory or auto-mount card and pick the newest .insv."""
    if input_path:
        p = Path(os.path.expanduser(input_path)).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input path does not exist: {p}")
        found = find_insv_files([p])
        if not found:
            raise FileNotFoundError(f"No .insv files found under: {p}")
        return found[0]

    mounts = ensure_card_mounted(preferred_device=device)
    if not mounts:
        raise RuntimeError(
            "No mounted removable SD/MMC media found. Insert the card or pass --input <path>."
        )
    found = find_insv_files(mounts)
    if not found:
        raise FileNotFoundError(
            f"No .insv files found on mounted media: {[str(m) for m in mounts]}"
        )
    return found[0]
