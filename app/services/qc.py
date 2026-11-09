"""Quality-control gate that runs between material preparation and rendering.

Every inspected item (the script, each search term, the narration audio, each
subtitle cue, each material clip) becomes one check. The pass rate is the share
of passed checks; rendering only starts when the rate is strictly above the
configured threshold (default 98%) and no *critical* check failed. Critical
checks are those that would make the render pointless (missing audio, no
subtitles although they were requested, footage too short to cover narration).

The module is deterministic, offline and free: it only uses ffprobe/ffmpeg and
the files already produced by the earlier pipeline stages.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from loguru import logger

from app.config import config
from app.models.schema import VideoAspect, VideoConcatMode, VideoParams
from app.utils import utils
from app.utils.subtitle_reader import file_to_subtitles

DEFAULT_PASS_THRESHOLD = 0.98
QC_REPORT_FILE = "qc_report.json"

# Thresholds. Kept as module constants so tests and config can override them.
MIN_CLIP_SECONDS = 1.0
MIN_DIMENSION = 470  # mirrors video.is_material_resolution_acceptable (480 - 10)
MAX_SILENCE_RATIO = 0.35
SILENCE_NOISE_DB = -40
SILENCE_MIN_SECONDS = 1.0
WORDS_PER_SECOND_RANGE = (1.0, 5.0)  # space-delimited languages (vi, en, ...)
CHARS_PER_SECOND_RANGE = (2.0, 10.0)  # CJK scripts without spaces
SUBTITLE_TEXT_MATCH_RATIO = 0.8
SUBTITLE_TAIL_TOLERANCE_SECONDS = 2.0
SUBTITLE_OVERRUN_TOLERANCE_SECONDS = 0.5

_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")
_MARKDOWN_LEFTOVER = re.compile(r"(\*\*|```|^#{1,6}\s|\[[^\]]*\]\([^)]*\))", re.M)


@dataclass
class QCCheck:
    category: str
    name: str
    passed: bool
    critical: bool = False
    detail: str = ""
    clip: str = ""  # material file the check is about (used to drop bad clips)


@dataclass
class QCReport:
    threshold: float
    checks: list[QCCheck] = field(default_factory=list)

    threshold_note: str = ""
    character_specs: list[dict] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def add(self, category, name, passed, *, critical=False, detail="", clip=""):
        self.checks.append(
            QCCheck(category, name, bool(passed), bool(critical), str(detail or ""), clip)
        )

    def failed_clips(self) -> set[str]:
        return {c.clip for c in self.failures if c.clip}

    def only_clip_failures(self) -> bool:
        failures = self.failures
        return bool(failures) and all(c.clip for c in failures)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total if self.total else 0.0

    @property
    def critical_failures(self) -> list[QCCheck]:
        return [c for c in self.checks if c.critical and not c.passed]

    @property
    def failures(self) -> list[QCCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def passed(self) -> bool:
        return (
            self.total > 0
            and not self.critical_failures
            and self.pass_rate > self.threshold
        )

    def summary(self, max_failures: int = 20) -> dict[str, Any]:
        by_category: dict[str, dict[str, int]] = {}
        for check in self.checks:
            bucket = by_category.setdefault(check.category, {"passed": 0, "total": 0})
            bucket["total"] += 1
            bucket["passed"] += int(check.passed)
        return {
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "threshold": self.threshold,
            "passed_checks": self.passed_count,
            "total_checks": self.total,
            "critical_failures": [asdict(c) for c in self.critical_failures],
            "failures": [asdict(c) for c in self.failures[:max_failures]],
            "by_category": by_category,
            "character_specs": self.character_specs,
            "skipped": self.skipped,
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.summary(max_failures=len(self.checks))
        data["checks"] = [asdict(c) for c in self.checks]
        return data

    def error_message(self) -> str:
        head = (
            f"QC pass rate {self.pass_rate:.2%} "
            f"({self.passed_count}/{self.total}) "
            f"{'<=' if self.pass_rate <= self.threshold else '>'} "
            f"threshold {self.threshold:.2%}"
        )
        if self.critical_failures:
            names = ", ".join(c.name for c in self.critical_failures[:5])
            head += f"; critical failures: {names}"
        elif self.failures:
            names = ", ".join(c.name for c in self.failures[:5])
            head += f"; failed: {names}"
        return head


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
def _qc_config() -> dict:
    return dict(getattr(config, "qc", {}) or {})


def is_enabled(params: VideoParams) -> bool:
    override = getattr(params, "qc_enabled", None)
    if override is not None:
        return bool(override)
    return bool(_qc_config().get("enabled", True))


def get_threshold(params: VideoParams) -> float:
    value = getattr(params, "qc_pass_threshold", None)
    if value is None:
        value = _qc_config().get("pass_threshold", DEFAULT_PASS_THRESHOLD)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = DEFAULT_PASS_THRESHOLD
    # Accept both 0.98 and 98.
    if value > 1:
        value /= 100.0
    return min(max(value, 0.0), 1.0)


def get_max_retries(params: VideoParams) -> int:
    value = getattr(params, "qc_max_retries", None)
    if value is None:
        value = _qc_config().get("max_retries", 1)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 1


def _cfg_float(key: str, default: float) -> float:
    try:
        return float(_qc_config().get(key, default))
    except (TypeError, ValueError):
        return default


def _cfg_bool(key: str, default: bool) -> bool:
    value = _qc_config().get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


# --------------------------------------------------------------------------- #
# media probing
# --------------------------------------------------------------------------- #
def _ffmpeg_binary() -> str:
    try:
        from app.services import video as video_service

        return video_service.get_ffmpeg_binary()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"


def _ffprobe_binary() -> str | None:
    ffmpeg = _ffmpeg_binary()
    directory = os.path.dirname(ffmpeg)
    if directory:
        name = "ffprobe.exe" if ffmpeg.lower().endswith(".exe") else "ffprobe"
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("ffprobe")


def probe_media(file_path: str) -> dict[str, Any]:
    """Return {'duration', 'width', 'height', 'has_video', 'has_audio'} or {}."""
    if not file_path or not os.path.isfile(file_path):
        return {}
    ffprobe = _ffprobe_binary()
    if ffprobe:
        try:
            result = subprocess.run(
                [ffprobe, "-v", "error", "-print_format", "json",
                 "-show_format", "-show_streams", file_path],
                capture_output=True, text=True, timeout=60, check=False,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout or "{}")
                streams = data.get("streams") or []
                video_stream = next(
                    (s for s in streams if s.get("codec_type") == "video"), None
                )
                duration = float((data.get("format") or {}).get("duration") or 0)
                if not duration and video_stream:
                    duration = float(video_stream.get("duration") or 0)
                return {
                    "duration": duration,
                    "width": int((video_stream or {}).get("width") or 0),
                    "height": int((video_stream or {}).get("height") or 0),
                    "has_video": video_stream is not None,
                    "has_audio": any(s.get("codec_type") == "audio" for s in streams),
                }
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            logger.debug(f"ffprobe failed for {file_path}: {exc}")
    # Fallback: MoviePy (slower but always installed with the project).
    try:
        from app.services import video as video_service

        clip = video_service._open_video_clip_quietly(file_path)
        try:
            width, height = clip.size
            return {
                "duration": float(clip.duration or 0),
                "width": int(width), "height": int(height),
                "has_video": True, "has_audio": False,
            }
        finally:
            video_service.close_clip(clip)
    except Exception as exc:
        logger.debug(f"moviepy probe failed for {file_path}: {exc}")
        return {}


def measure_silence(audio_file: str) -> float | None:
    """Total seconds of silence detected by ffmpeg, or None if unmeasurable."""
    try:
        result = subprocess.run(
            [_ffmpeg_binary(), "-hide_banner", "-nostats", "-i", audio_file,
             "-af", f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_SECONDS}",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=300, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug(f"silencedetect failed: {exc}")
        return None
    if result.returncode != 0:
        return None
    durations = re.findall(r"silence_duration:\s*([0-9.]+)", result.stderr or "")
    return sum(float(d) for d in durations)


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def _normalize_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", (text or "").lower())
    return re.findall(r"\w+", text, flags=re.UNICODE)


def _is_cjk_heavy(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    cjk = sum(1 for ch in letters if "぀" <= ch <= "鿿" or "가" <= ch <= "힯")
    return cjk / len(letters) > 0.5


def _srt_seconds(value: str) -> float:
    match = _SRT_TIME.search(value or "")
    if not match:
        return -1.0
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000


def _parse_cue_times(times_line: str) -> tuple[float, float]:
    parts = (times_line or "").split("-->")
    if len(parts) != 2:
        return -1.0, -1.0
    return _srt_seconds(parts[0]), _srt_seconds(parts[1])


# --------------------------------------------------------------------------- #
# check groups
# --------------------------------------------------------------------------- #
def _check_script(report: QCReport, video_script: str) -> None:
    script = (video_script or "").strip()
    report.add("script", "script_not_empty", bool(script), critical=True)
    report.add(
        "script", "script_no_provider_error",
        not script.startswith("Error: "), critical=True,
    )
    leftover = _MARKDOWN_LEFTOVER.search(script)
    report.add(
        "script", "script_no_markdown",
        leftover is None,
        detail=f"found {leftover.group(0)!r}" if leftover else "",
    )


def _check_terms(report: QCReport, params: VideoParams, video_terms) -> None:
    if params.video_source == "local":
        return
    terms = [t for t in (video_terms or []) if isinstance(t, str) and t.strip()]
    report.add(
        "terms", "terms_count", len(terms) >= 3, critical=not terms,
        detail=f"{len(terms)} terms",
    )
    for term in terms:
        words = term.split()
        ascii_ok = all(ord(ch) < 128 for ch in term)
        report.add(
            "terms", f"term:{term}",
            ascii_ok and 1 <= len(words) <= 4,
            detail="" if ascii_ok else "non-English characters",
        )


def _check_audio(
    report: QCReport, params: VideoParams, video_script: str,
    audio_file: str, audio_duration: float,
) -> float:
    info = probe_media(audio_file) if audio_file else {}
    measured = float(info.get("duration") or 0) or float(audio_duration or 0)
    report.add(
        "audio", "audio_readable", bool(info), critical=True,
        detail="" if info else f"cannot read {audio_file!r}",
    )
    report.add(
        "audio", "audio_duration_positive", measured > 0, critical=True,
        detail=f"{measured:.2f}s",
    )
    if measured <= 0:
        return measured

    from app.services import voice

    is_silent_voice = voice.is_no_voice(getattr(params, "voice_name", ""))
    is_custom_audio = bool(getattr(params, "custom_audio_file", None))
    if is_silent_voice:
        return measured

    if not is_custom_audio:
        spoken = utils.remove_pause_tags(video_script or "")
        if _is_cjk_heavy(spoken):
            units = len(re.sub(r"\s+", "", spoken))
            low, high = CHARS_PER_SECOND_RANGE
            unit_name = "chars/s"
        else:
            units = len(_normalize_words(spoken))
            low, high = WORDS_PER_SECOND_RANGE
            unit_name = "words/s"
        rate = units / measured
        report.add(
            "audio", "audio_speech_rate", low <= rate <= high,
            detail=f"{rate:.2f} {unit_name} (expected {low}-{high})",
        )

    silence = measure_silence(audio_file) if info else None
    if silence is not None:
        ratio = silence / measured
        report.add(
            "audio", "audio_silence_ratio", ratio <= MAX_SILENCE_RATIO,
            detail=f"{ratio:.0%} silent (max {MAX_SILENCE_RATIO:.0%})",
        )
    return measured


def _check_subtitles(
    report: QCReport, params: VideoParams, video_script: str,
    subtitle_path: str, audio_duration: float,
) -> None:
    if not params.subtitle_enabled:
        return
    cues = file_to_subtitles(subtitle_path) if subtitle_path else []
    report.add(
        "subtitle", "subtitle_generated", bool(cues), critical=True,
        detail="" if cues else "subtitles enabled but no usable subtitle file",
    )
    if not cues:
        return

    script_words = set(_normalize_words(utils.remove_pause_tags(video_script or "")))
    check_text = bool(script_words) and not getattr(params, "custom_audio_file", None)
    previous_end = 0.0
    last_end = 0.0
    for index, times_line, text in cues:
        start, end = _parse_cue_times(times_line)
        problems = []
        if start < 0 or end <= start:
            problems.append(f"bad timing {times_line!r}")
        elif start + 0.05 < previous_end:
            problems.append("overlaps previous cue")
        if not (text or "").strip():
            problems.append("empty text")
        elif check_text:
            words = _normalize_words(text)
            if words:
                hit = sum(1 for w in words if w in script_words) / len(words)
                if hit < SUBTITLE_TEXT_MATCH_RATIO:
                    problems.append(f"only {hit:.0%} of words match script")
        report.add(
            "subtitle", f"cue:{index}", not problems, detail="; ".join(problems)
        )
        previous_end = max(previous_end, end)
        last_end = max(last_end, end)

    if audio_duration > 0:
        covers = (
            last_end >= audio_duration - SUBTITLE_TAIL_TOLERANCE_SECONDS
            and last_end <= audio_duration + SUBTITLE_OVERRUN_TOLERANCE_SECONDS
        )
        report.add(
            "subtitle", "subtitle_matches_audio_length", covers,
            detail=f"last cue ends {last_end:.2f}s, audio {audio_duration:.2f}s",
        )


def _orientation_matches(width: int, height: int, aspect: VideoAspect) -> bool:
    if aspect == VideoAspect.portrait:
        return height > width
    if aspect == VideoAspect.landscape:
        return width > height
    return abs(width - height) <= max(width, height) * 0.05


def _check_materials(
    report: QCReport, params: VideoParams, materials: list[str], audio_duration: float,
) -> None:
    report.add(
        "materials", "materials_present", bool(materials), critical=True,
        detail=f"{len(materials or [])} files",
    )
    try:
        aspect = VideoAspect(params.video_aspect)
    except ValueError:
        aspect = VideoAspect.portrait
    concat_mode = getattr(params.video_concat_mode, "value", params.video_concat_mode)
    sequential = (
        params.match_materials_to_script
        or concat_mode == VideoConcatMode.sequential.value
    )
    speed = utils.normalize_clip_speed(getattr(params, "video_clip_speed", 1.0))
    clip_seconds = float(params.video_clip_duration or 5)

    usable_seconds = 0.0
    seen: set[str] = set()
    for file_path in materials or []:
        name = os.path.basename(file_path)
        info = probe_media(file_path)
        problems = []
        if not info or not info.get("has_video"):
            problems.append("unreadable or no video stream")
        else:
            duration = float(info.get("duration") or 0)
            width, height = info.get("width", 0), info.get("height", 0)
            if duration < MIN_CLIP_SECONDS:
                problems.append(f"too short ({duration:.2f}s)")
            if min(width, height) < MIN_DIMENSION:
                problems.append(f"low resolution {width}x{height}")
            if width and height and not _orientation_matches(width, height, aspect):
                problems.append(f"orientation {width}x{height} != {aspect.value}")
            real = os.path.realpath(file_path)
            if real in seen:
                problems.append("duplicate file")
            seen.add(real)
            if not problems or problems == [f"orientation {width}x{height} != {aspect.value}"]:
                # Wrong orientation still renders (cropped/letterboxed), so it
                # counts toward coverage while failing its own check.
                playable = duration / speed
                usable_seconds += min(playable, clip_seconds) if sequential else playable
        report.add(
            "materials", f"clip:{name}", not problems,
            detail="; ".join(problems), clip=name,
        )

    if audio_duration > 0 and materials:
        report.add(
            "materials", "materials_cover_audio",
            usable_seconds >= audio_duration, critical=True,
            detail=f"usable {usable_seconds:.1f}s vs narration {audio_duration:.1f}s",
        )


def _readable_clips(materials: list[str]) -> list[tuple[str, dict]]:
    clips = []
    for file_path in materials or []:
        info = probe_media(file_path)
        if info and info.get("has_video") and float(info.get("duration") or 0) > 0:
            clips.append((file_path, info))
    return clips


def _check_visual(
    report: QCReport, params: VideoParams, clips: list[tuple[str, dict]]
) -> None:
    """Black frames, frozen picture and blur, via ffmpeg filters."""
    from app.services import qc_visual

    if not _cfg_bool("visual_check", True):
        report.skipped.append("visual: disabled in config")
        return
    max_black = _cfg_float("max_black_ratio", qc_visual.DEFAULT_MAX_BLACK_RATIO)
    max_frozen = _cfg_float("max_frozen_ratio", qc_visual.DEFAULT_MAX_FROZEN_RATIO)
    max_blur = _cfg_float("max_blur", qc_visual.DEFAULT_MAX_BLUR)
    ffmpeg = _ffmpeg_binary()
    for file_path, info in clips:
        name = os.path.basename(file_path)
        stats = qc_visual.analyze(ffmpeg, file_path, info.get("duration") or 0)
        if stats is None:
            report.add("visual", f"clip:{name}:analyzable", False,
                       detail="ffmpeg could not analyse the picture", clip=name)
            continue
        report.add(
            "visual", f"clip:{name}:black", stats.black_ratio <= max_black,
            detail=f"{stats.black_ratio:.0%} black (max {max_black:.0%})", clip=name,
        )
        # Still images rendered without motion are frozen by design.
        if getattr(getattr(params, "image_motion", None), "value", None) != "static":
            report.add(
                "visual", f"clip:{name}:frozen", stats.frozen_ratio <= max_frozen,
                detail=f"{stats.frozen_ratio:.0%} frozen (max {max_frozen:.0%})",
                clip=name,
            )
        if stats.blur_mean is not None:
            report.add(
                "visual", f"clip:{name}:blur", stats.blur_mean <= max_blur,
                detail=f"blur {stats.blur_mean:.1f} (max {max_blur:g})", clip=name,
            )


def _resolve_character_specs(params: VideoParams, video_script: str):
    from app.services import qc_character

    explicit = [
        spec for spec in (
            qc_character.CharacterSpec.from_any(item)
            for item in (getattr(params, "qc_characters", None) or [])
        ) if spec
    ]
    if explicit:
        return explicit, "request"
    if _cfg_bool("auto_character_spec", True) and (video_script or "").strip():
        extracted = qc_character.extract_specs_with_llm(params.video_subject, video_script)
        if extracted:
            return extracted, "script"
    return [], ""


def _check_characters(
    report: QCReport, params: VideoParams, video_script: str,
    clips: list[tuple[str, dict]],
) -> None:
    """Gender / age / skin tone of the person on screen must match the character."""
    from app.services import qc_character as qcc

    mode = str(_qc_config().get("character_check", "auto")).strip().lower()
    if mode in {"false", "off", "0", "no"} or getattr(params, "qc_character_check", None) is False:
        report.skipped.append("character: disabled")
        return
    if not clips:
        return

    specs, spec_source = _resolve_character_specs(params, video_script)
    expects_one_character = params.video_source in qcc.CHARACTER_SOURCES
    if not specs and not expects_one_character:
        report.skipped.append(
            "character: no character described in script and stock footage shows "
            "unrelated people"
        )
        return
    report.character_specs = [
        {"source": spec_source, **spec.__dict__} for spec in specs
    ]

    try:
        qcc.get_model()
    except qcc.CharacterModelUnavailable as exc:
        report.add("character", "character_model_available", False,
                   critical=True, detail=str(exc))
        return

    age_tolerance = int(_cfg_float("age_tolerance", qcc.DEFAULT_AGE_TOLERANCE))
    frames = int(_cfg_float("frames_per_clip", qcc.DEFAULT_FRAMES_PER_CLIP))
    people: list[qcc.ClipPerson] = []
    for file_path, _info in clips:
        try:
            person = qcc.analyze_clip(file_path, frames)
        except Exception as exc:
            name = os.path.basename(file_path)
            report.add("character", f"clip:{name}:person", False,
                       detail=f"face analysis failed: {exc}", clip=name)
            continue
        if person:
            people.append(person)
    if not people:
        report.skipped.append("character: no person visible in any clip")
        return

    if not specs:
        # One recurring character expected but not described: clips must agree.
        specs = [qcc.consensus_spec(people)]
        age_tolerance = max(age_tolerance * 2, 8)
        report.character_specs = [{"source": "consensus", **specs[0].__dict__}]

    for person in people:
        spec, problems = qcc.best_spec(person, specs, age_tolerance)
        for attribute in ("gender", "age", "skin_tone"):
            constrained = {
                "gender": spec.gender,
                "age": spec.age_min or spec.age_max,
                "skin_tone": spec.skin_tone and person.ita is not None,
            }[attribute]
            if not constrained:
                continue
            report.add(
                "character", f"clip:{person.clip}:{attribute}",
                attribute not in problems,
                detail=problems.get(attribute) or f"{person.describe()} ~ {spec.label()}",
                clip=person.clip,
            )

    identity = str(_qc_config().get("identity_check", "auto")).strip().lower()
    run_identity = identity in {"true", "on", "1", "yes"} or (
        identity == "auto" and expects_one_character and len(specs) == 1
    )
    with_embeddings = [p for p in people if p.embedding is not None]
    if run_identity and len(with_embeddings) >= 2:
        threshold = _cfg_float("identity_threshold", qcc.DEFAULT_IDENTITY_THRESHOLD)
        medoid = max(
            with_embeddings,
            key=lambda p: sum(qcc.cosine(p.embedding, o.embedding) for o in with_embeddings),
        )
        for person in with_embeddings:
            if person is medoid:
                continue
            score = qcc.cosine(person.embedding, medoid.embedding)
            report.add(
                "character", f"clip:{person.clip}:same_person", score >= threshold,
                detail=f"face similarity {score:.2f} to {medoid.clip} (min {threshold:.2f})",
                clip=person.clip,
            )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def run(
    task_id: str,
    params: VideoParams,
    *,
    video_script: str,
    video_terms,
    audio_file: str,
    audio_duration: float,
    subtitle_path: str,
    materials: list[str],
    threshold: float | None = None,
) -> QCReport:
    report = QCReport(threshold=get_threshold(params) if threshold is None else threshold)
    _check_script(report, video_script)
    _check_terms(report, params, video_terms)
    measured = _check_audio(report, params, video_script, audio_file, audio_duration)
    _check_subtitles(report, params, video_script, subtitle_path, measured)
    _check_materials(report, params, materials, measured)
    clips = _readable_clips(materials)
    _check_visual(report, params, clips)
    _check_characters(report, params, video_script, clips)

    try:
        report_path = os.path.join(utils.task_dir(task_id), QC_REPORT_FILE)
        with open(report_path, "w", encoding="utf-8") as fp:
            json.dump(report.to_dict(), fp, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning(f"failed to write QC report: {exc}")

    level = logger.success if report.passed else logger.warning
    level(
        f"QC task {task_id}: {report.pass_rate:.2%} "
        f"({report.passed_count}/{report.total}), threshold {report.threshold:.2%}, "
        f"passed={report.passed}"
    )
    return report
