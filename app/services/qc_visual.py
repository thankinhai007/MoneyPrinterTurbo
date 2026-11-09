"""Technical image checks for material clips, using only ffmpeg filters.

One ffmpeg pass per clip runs ``blackdetect`` (black frames), ``freezedetect``
(frozen picture) and ``blurdetect`` (out-of-focus picture) on a 480p / 10 fps
proxy, so a clip takes ~0.3 s regardless of its resolution. Calibration on that
proxy: a sharp test pattern scores ~4, a 1080p clip with sigma=6 gaussian blur
~7.2, a 720p clip with sigma=6 blur ~9.7. Shallow depth-of-field footage
(blurred background) scores higher than flat footage; raise ``max_blur`` if
such clips are rejected.
"""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass

from loguru import logger

ANALYZE_SECONDS = 30  # only the beginning of long clips is ever rendered
DEFAULT_MAX_BLACK_RATIO = 0.2
DEFAULT_MAX_FROZEN_RATIO = 0.5
DEFAULT_MAX_BLUR = 7.0

_BLACK = re.compile(r"black_duration:\s*([0-9.]+)")
_FREEZE_START = re.compile(r"freeze_start:\s*([0-9.]+)")
_FREEZE_END = re.compile(r"freeze_end:\s*([0-9.]+)")
_BLUR = re.compile(r"blur mean:\s*(-?nan|[0-9.]+)", re.I)


@dataclass
class VisualStats:
    analyzed_seconds: float
    black_seconds: float
    frozen_seconds: float
    blur_mean: float | None  # None when the picture has no edges at all

    @property
    def black_ratio(self) -> float:
        return self.black_seconds / self.analyzed_seconds if self.analyzed_seconds else 0.0

    @property
    def frozen_ratio(self) -> float:
        return self.frozen_seconds / self.analyzed_seconds if self.analyzed_seconds else 0.0


def analyze(ffmpeg: str, file_path: str, duration: float) -> VisualStats | None:
    analyzed = min(float(duration or 0), ANALYZE_SECONDS)
    if analyzed <= 0:
        return None
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostats", "-i", file_path,
             "-t", str(ANALYZE_SECONDS), "-an",
             "-vf", "fps=10,scale=-2:480,"
                    "blackdetect=d=0.3:pic_th=0.98:pix_th=0.10,"
                    "freezedetect=n=-60dB:d=1,"
                    "blurdetect=block_width=32:block_height=32",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=300, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug(f"visual analysis failed for {file_path}: {exc}")
        return None
    if result.returncode != 0:
        return None
    log = result.stderr or ""

    black = sum(float(v) for v in _BLACK.findall(log))
    starts = [float(v) for v in _FREEZE_START.findall(log)]
    ends = [float(v) for v in _FREEZE_END.findall(log)]
    frozen = 0.0
    for index, start in enumerate(starts):
        # A freeze that lasts until the end of the clip has no freeze_end.
        end = ends[index] if index < len(ends) else analyzed
        frozen += max(0.0, min(end, analyzed) - start)

    blur_values = _BLUR.findall(log)
    blur = None
    if blur_values:
        try:
            value = float(blur_values[-1])
            blur = None if math.isnan(value) else value
        except ValueError:
            blur = None
    return VisualStats(analyzed, min(black, analyzed), min(frozen, analyzed), blur)
