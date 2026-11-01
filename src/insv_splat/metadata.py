"""Insta360 / Antigravity .insv stream & binary trailer calibration parser."""

from __future__ import annotations

import json
import re
import struct
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

INSTA360_MAGIC = b"8db42d694ccc418790edff439fe026bf"


@dataclass(frozen=True)
class MeiLensCalibration:
    """Mei unified omnidirectional camera model parameters for a single fisheye lens."""

    lens_index: int
    xi: float
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float
    ref_size: int
    euler_deg: tuple[float, float, float]
    translation_m: tuple[float, float, float]

    def scaled(self, width: int, height: int) -> dict[str, float]:
        """Scale intrinsics from factory reference square resolution to actual video frame size."""
        sx = width / float(self.ref_size)
        sy = height / float(self.ref_size)
        return {
            "xi": self.xi,
            "fx": self.fx * sx,
            "fy": self.fy * sy,
            "cx": self.cx * sx,
            "cy": self.cy * sy,
            "k1": self.k1,
            "k2": self.k2,
            "k3": self.k3,
            "p1": self.p1,
            "p2": self.p2,
        }


@dataclass(frozen=True)
class VideoStreamInfo:
    """Metadata for a single video stream inside an .insv container."""

    stream_index: int
    codec_name: str
    width: int
    height: int
    fps: float
    duration_s: float
    nb_frames: int


@dataclass(frozen=True)
class InsvMetadata:
    """Complete parsed metadata and factory lens calibration for an .insv recording."""

    primary_path: str
    secondary_path: str | None
    camera_model: str
    serial_number: str
    firmware: str
    trailer_version: int
    trailer_size_bytes: int
    rig_type: str  # "drone" (nadir/zenith) or "handheld" (front/back)
    streams: list[VideoStreamInfo]
    mei_lenses: list[MeiLensCalibration]
    raw_offset_v3: str | None
    raw_offset_v2: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_fps(rate_str: str) -> float:
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 30.0
    return float(rate_str or 30.0)


def probe_streams(path: Path) -> list[VideoStreamInfo]:
    """Probe video streams inside an .insv container using ffprobe."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(proc.stdout)
    fmt_dur = float((info.get("format") or {}).get("duration") or 0.0)
    streams: list[VideoStreamInfo] = []
    v_idx = 0
    for st in info.get("streams") or []:
        if st.get("codec_type") != "video":
            continue
        dur = float(st.get("duration") or fmt_dur)
        fps = _parse_fps(st.get("avg_frame_rate") or st.get("r_frame_rate") or "30/1")
        nbf = int(st.get("nb_frames") or round(dur * fps))
        streams.append(
            VideoStreamInfo(
                stream_index=v_idx,
                codec_name=str(st.get("codec_name") or "hevc"),
                width=int(st.get("width") or 0),
                height=int(st.get("height") or 0),
                fps=fps,
                duration_s=dur,
                nb_frames=nbf,
            )
        )
        v_idx += 1
    return streams


def _parse_mei_offset_v3(offset_str: str) -> list[MeiLensCalibration]:
    """Parse Insta360 type-155 Mei unified omnidirectional dual-lens calibration string."""
    parts = offset_str.strip().split("_")
    # Format:
    # 2 _ xi0 _ fx0 _ fy0 _ cx0 _ cy0 _ rx0 _ ry0 _ rz0 _ tx0 _ ty0 _ tz0 _ k1 _ k2 _ k3 _ p1 _ p2 _ refW _ refH _ 155 _
    #     xi1 _ fx1 _ fy1 _ cx1 _ cy1 _ rx1 _ ry1 _ rz1 _ tx1 _ ty1 _ tz1 _ k1 _ k2 _ k3 _ p1 _ p2 _ refW _ refH _ 155 ...
    if len(parts) < 39 or parts[0] != "2" or parts[19] != "155":
        return []
    try:
        vals0 = [float(x) for x in parts[1:19]]
        ref_w0, ref_h0 = int(round(vals0[16])), int(round(vals0[17]))
        lens0 = MeiLensCalibration(
            lens_index=0,
            xi=vals0[0],
            fx=vals0[1],
            fy=vals0[2],
            cx=vals0[3],
            cy=vals0[4],
            euler_deg=(vals0[5], vals0[6], vals0[7]),
            translation_m=(vals0[8], vals0[9], vals0[10]),
            k1=vals0[11],
            k2=vals0[12],
            k3=vals0[13],
            p1=vals0[14],
            p2=vals0[15],
            ref_size=ref_h0,
        )

        vals1 = [float(x) for x in parts[20:38]]
        ref_h1 = int(round(vals1[17]))
        # Lens 1 cx is stored in side-by-side [0..2*ref_h] coordinates; shift to [0..ref_h]
        cx1_local = vals1[3] - ref_h1 if vals1[3] > ref_h1 else vals1[3]
        lens1 = MeiLensCalibration(
            lens_index=1,
            xi=vals1[0],
            fx=vals1[1],
            fy=vals1[2],
            cx=cx1_local,
            cy=vals1[4],
            euler_deg=(vals1[5], vals1[6], vals1[7]),
            translation_m=(vals1[8], vals1[9], vals1[10]),
            k1=vals1[11],
            k2=vals1[12],
            k3=vals1[13],
            p1=vals1[14],
            p2=vals1[15],
            ref_size=ref_h1,
        )
        return [lens0, lens1]
    except (ValueError, IndexError):
        return []


def parse_insv_metadata(
    primary_path: Path,
    secondary_path: Path | None = None,
    rig_mode_override: str = "auto",
) -> InsvMetadata:
    """Extract video stream info and Insta360 trailer lens calibration from an .insv file."""
    streams = probe_streams(primary_path)
    if secondary_path is not None:
        sec_streams = probe_streams(secondary_path)
        if sec_streams:
            s = sec_streams[0]
            streams.append(
                VideoStreamInfo(
                    stream_index=1,
                    codec_name=s.codec_name,
                    width=s.width,
                    height=s.height,
                    fps=s.fps,
                    duration_s=s.duration_s,
                    nb_frames=s.nb_frames,
                )
            )

    file_size = primary_path.stat().st_size
    tail_bytes = min(file_size, 262144)
    with primary_path.open("rb") as f:
        f.seek(file_size - tail_bytes)
        tail = f.read(tail_bytes)

    trailer_size = 0
    trailer_ver = 0
    if len(tail) >= 40 and tail[-32:] == INSTA360_MAGIC:
        trailer_size, trailer_ver = struct.unpack("<II", tail[-40:-32])

    # Extract printable strings from the final camera-info protobuf block
    info_window = tail[-4096:]
    ascii_strings = [s.decode("latin1") for s in re.findall(b"[ -~]{4,400}", info_window)]

    camera_model = "Insta360 Dual-Fisheye"
    serial_number = ""
    firmware = ""
    raw_v3: str | None = None
    raw_v2: str | None = None

    for s in ascii_strings:
        if any(k in s for k in ("Antigravity", "Insta360", "OneX", "ONEX", "X3", "X4", "X5")):
            camera_model = s.strip()
        elif re.fullmatch(r"[A-Z0-9]{12,24}", s.strip()):
            serial_number = s.strip()
        if "*s2_" in s:
            firmware = s.split("*s", 1)[0].strip()
        m_v3 = re.search(r"(2_[0-9.\-_]+_155_[0-9.\-_]+_155(?:_[0-9]+)?)", s)
        if m_v3:
            raw_v3 = m_v3.group(1)
        m_v2 = re.search(r"(2_[0-9.\-_]+_112_[0-9.\-_]+_112(?:_[0-9]+)?)", s)
        if m_v2:
            raw_v2 = m_v2.group(1)

    mei_lenses = _parse_mei_offset_v3(raw_v3) if raw_v3 else []

    if rig_mode_override in {"drone", "handheld"}:
        rig_type = rig_mode_override
    else:
        rig_type = "drone" if "antigravity" in camera_model.lower() else "handheld"

    return InsvMetadata(
        primary_path=str(primary_path),
        secondary_path=str(secondary_path) if secondary_path else None,
        camera_model=camera_model,
        serial_number=serial_number,
        firmware=firmware,
        trailer_version=trailer_ver,
        trailer_size_bytes=trailer_size,
        rig_type=rig_type,
        streams=streams,
        mei_lenses=mei_lenses,
        raw_offset_v3=raw_v3,
        raw_offset_v2=raw_v2,
    )
