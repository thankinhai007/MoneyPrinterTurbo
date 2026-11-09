"""Character-consistency checks: gender, age and skin tone of people on screen.

How it works
------------
1. Each material clip is sampled at a few evenly spaced frames (OpenCV).
2. InsightFace ``buffalo_l`` (offline ONNX model, ~280 MB, downloaded once from
   the InsightFace GitHub release into ./models) detects faces and estimates
   gender and age. The largest confident face in a frame is the clip's subject.
3. Skin tone is measured, not guessed: the Individual Typology Angle (ITA, a
   dermatology colour measure in CIELAB) of the cheek/nose area of that face.
   ITA > 41 -> light, 10..41 -> medium, < 10 -> dark.
4. Per clip, the median age / ITA and the majority gender across frames are
   compared with the expected character spec(s). Clips without a person are
   not scored.

Where the spec comes from (first match wins)
--------------------------------------------
* ``VideoParams.qc_characters`` - explicit, e.g.
  ``[{"name": "anh trai", "gender": "male", "age_min": 35, "age_max": 44,
  "skin_tone": "light"}]``
* the configured LLM extracting characters from the script (text-only, cheap),
  enabled by ``[qc] auto_character_spec``.
* no spec: for local / AI-generated footage the clips are checked against each
  other (majority gender, median age +/- tolerance, majority skin tone), because
  those clips are expected to show one recurring character. Stock footage
  without a spec is not scored (different stock clips show different people).

Optionally (``identity_check``) face embeddings verify that local / AI clips
show the same person throughout.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from loguru import logger

from app.utils import utils

SKIN_TONES = ("light", "medium", "dark")
ITA_LIGHT_MIN = 41.0
ITA_DARK_MAX = 10.0
# Lighting shifts ITA a few degrees; values this close to a boundary count for
# both neighbouring classes.
ITA_BOUNDARY_TOLERANCE = 5.0

DEFAULT_AGE_TOLERANCE = 5
DEFAULT_FRAMES_PER_CLIP = 4
DEFAULT_IDENTITY_THRESHOLD = 0.30
MIN_FACE_PIXELS = 60
MIN_DET_SCORE = 0.6

# Footage that is expected to show one recurring character (user-provided or
# AI-generated). Stock libraries are excluded on purpose.
CHARACTER_SOURCES = frozenset({
    "local", "wavespeed", "volcengine_seedance", "jalapeno", "ofox", "muapi",
    "metaso_minimax", "infistar", "loomloom", "openai_image", "codex_image",
})

_model_lock = threading.Lock()
_model = None
_model_error: str | None = None


class CharacterModelUnavailable(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# spec
# --------------------------------------------------------------------------- #
@dataclass
class CharacterSpec:
    name: str = ""
    gender: str | None = None  # "male" / "female"
    age_min: int | None = None
    age_max: int | None = None
    skin_tone: str | None = None  # light / medium / dark

    @classmethod
    def from_any(cls, value: Any) -> "CharacterSpec | None":
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        if not isinstance(value, dict):
            return None
        gender = str(value.get("gender") or "").strip().lower() or None
        gender = {"m": "male", "man": "male", "nam": "male",
                  "f": "female", "woman": "female", "nữ": "female", "nu": "female"}.get(gender, gender)
        if gender not in {"male", "female"}:
            gender = None
        tone = str(value.get("skin_tone") or "").strip().lower() or None
        tone = {"white": "light", "fair": "light", "trắng": "light",
                "tan": "medium", "olive": "medium", "ngăm": "medium",
                "black": "dark", "brown": "dark", "đen": "dark"}.get(tone, tone)
        if tone not in SKIN_TONES:
            tone = None

        def as_int(key):
            try:
                number = int(float(value.get(key)))
                return number if 0 < number < 120 else None
            except (TypeError, ValueError):
                return None

        age_min, age_max = as_int("age_min"), as_int("age_max")
        if age_min and age_max and age_min > age_max:
            age_min, age_max = age_max, age_min
        spec = cls(str(value.get("name") or "").strip(), gender, age_min, age_max, tone)
        return spec if spec.has_constraints() else None

    def has_constraints(self) -> bool:
        return any([self.gender, self.age_min, self.age_max, self.skin_tone])

    def label(self) -> str:
        parts = [self.name or "character"]
        if self.gender:
            parts.append(self.gender)
        if self.age_min or self.age_max:
            parts.append(f"{self.age_min or '?'}-{self.age_max or '?'}y")
        if self.skin_tone:
            parts.append(f"{self.skin_tone} skin")
        return " ".join(parts)


_SPEC_PROMPT = """
# Role: Casting continuity assistant

Read the video script and list the recurring human characters whose look is
described or clearly implied. For each, give only what the script states or
makes unambiguous:
- gender: "male" or "female"
- age_min / age_max: integers. Vietnamese "U<N>" (U30, U40...) means about N
  years old: use N-5 to N+4. "trẻ"/"young" without a number: 18-30.
- skin_tone: "light", "medium" or "dark" (e.g. "da trắng" -> light,
  "da ngăm" -> medium, "da đen" -> dark)
Use null for anything not stated. Do not invent attributes.
If the script has no specific human character, return [].

Return only a JSON array, e.g.
[{"name": "anh trai", "gender": "male", "age_min": 35, "age_max": 44, "skin_tone": "light"}]

## Video subject
{subject}

## Video script
{script}
""".strip()


_spec_cache: dict[tuple[str, str], list[CharacterSpec]] = {}
_clip_cache: dict[tuple[str, float, int, int], "ClipPerson | None"] = {}
_cache_lock = threading.Lock()


def extract_specs_with_llm(video_subject: str, video_script: str) -> list[CharacterSpec]:
    """LLM extraction, cached per (subject, script) so QC re-runs don't re-bill."""
    key = (video_subject or "", video_script or "")
    with _cache_lock:
        if key in _spec_cache:
            return list(_spec_cache[key])
    specs = _extract_specs_with_llm(*key)
    with _cache_lock:
        if len(_spec_cache) > 256:
            _spec_cache.clear()
        _spec_cache[key] = list(specs)
    return specs


def _extract_specs_with_llm(video_subject: str, video_script: str) -> list[CharacterSpec]:
    from app.services import llm

    prompt = _SPEC_PROMPT.replace("{subject}", video_subject or "").replace(
        "{script}", utils.remove_pause_tags(video_script or "")[:6000]
    )
    try:
        response = llm._generate_response(prompt)
        if not response or response.startswith("Error: "):
            raise ValueError(response or "empty response")
        text = llm._strip_code_fence(response)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = llm._extract_embedded_json(text, "[")
    except Exception as exc:
        logger.warning(f"QC: could not extract character spec from script: {exc}")
        return []
    specs = [CharacterSpec.from_any(item) for item in (data if isinstance(data, list) else [])]
    return [spec for spec in specs if spec]


# --------------------------------------------------------------------------- #
# model + measurements
# --------------------------------------------------------------------------- #
def _model_root() -> str:
    return os.path.join(utils.root_dir(), "models", "insightface")


def get_model():
    """Load InsightFace once per process. Raises CharacterModelUnavailable."""
    global _model, _model_error
    with _model_lock:
        if _model is not None:
            return _model
        if _model_error:
            raise CharacterModelUnavailable(_model_error)
        try:
            from insightface.app import FaceAnalysis

            modules = ["detection", "genderage", "recognition"]
            model = FaceAnalysis(
                name="buffalo_l", root=_model_root(), allowed_modules=modules,
                providers=["CPUExecutionProvider"],
            )
            model.prepare(ctx_id=-1, det_size=(640, 640))
            _model = model
            return _model
        except ImportError:
            _model_error = (
                "character QC needs insightface + onnxruntime + opencv "
                "(pip install insightface onnxruntime opencv-python-headless)"
            )
        except Exception as exc:  # download / ONNX errors
            _model_error = f"cannot load face model buffalo_l: {exc}"
        raise CharacterModelUnavailable(_model_error)


def ita_degrees(face_bgr) -> float | None:
    """Median Individual Typology Angle of the cheek/nose region of a face crop."""
    import cv2
    import numpy as np

    if face_bgr is None or face_bgr.size == 0:
        return None
    h, w = face_bgr.shape[:2]
    region = face_bgr[int(h * 0.40):int(h * 0.80), int(w * 0.25):int(w * 0.75)]
    if region.size == 0:
        return None
    ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 180, 135)) > 0
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype("float32")
    pixels = lab[mask] if mask.sum() >= 0.15 * mask.size else lab.reshape(-1, 3)
    if len(pixels) == 0:
        return None
    lightness = float(np.median(pixels[:, 0])) * 100.0 / 255.0
    b_star = float(np.median(pixels[:, 2])) - 128.0
    return math.degrees(math.atan2(lightness - 50.0, b_star if abs(b_star) > 1e-6 else 1e-6))


def ita_classes(ita: float) -> set[str]:
    classes = set()
    if ita > ITA_LIGHT_MIN - ITA_BOUNDARY_TOLERANCE:
        classes.add("light")
    if ITA_DARK_MAX - ITA_BOUNDARY_TOLERANCE < ita <= ITA_LIGHT_MIN + ITA_BOUNDARY_TOLERANCE:
        classes.add("medium")
    if ita <= ITA_DARK_MAX + ITA_BOUNDARY_TOLERANCE:
        classes.add("dark")
    return classes


def ita_class(ita: float) -> str:
    if ita > ITA_LIGHT_MIN:
        return "light"
    if ita > ITA_DARK_MAX:
        return "medium"
    return "dark"


@dataclass
class ClipPerson:
    clip: str
    frames_with_face: int
    gender: str
    age: float
    ita: float | None
    embedding: Any = field(default=None, repr=False)

    @property
    def skin_tone(self) -> str | None:
        return ita_class(self.ita) if self.ita is not None else None

    def describe(self) -> str:
        tone = f", {self.skin_tone} skin (ITA {self.ita:.0f}°)" if self.ita is not None else ""
        return f"{self.gender}, ~{self.age:.0f}y{tone}"


def _sample_frames(file_path: str, count: int):
    import cv2

    capture = cv2.VideoCapture(file_path)
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return []
        positions = sorted({int(total * (i + 0.5) / count) for i in range(count)})
        frames = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(position, total - 1))
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(frame)
        return frames
    finally:
        capture.release()


def analyze_clip(file_path: str, frames_per_clip: int = DEFAULT_FRAMES_PER_CLIP) -> ClipPerson | None:
    """Return the main on-screen person of a clip, or None if nobody is visible."""
    stat = os.stat(file_path)
    key = (os.path.realpath(file_path), stat.st_mtime, stat.st_size, frames_per_clip)
    with _cache_lock:
        if key in _clip_cache:
            return _clip_cache[key]
    person = _analyze_clip(file_path, frames_per_clip)
    with _cache_lock:
        if len(_clip_cache) > 2048:
            _clip_cache.clear()
        _clip_cache[key] = person
    return person


def _analyze_clip(file_path: str, frames_per_clip: int) -> ClipPerson | None:
    import numpy as np

    model = get_model()
    genders, ages, itas, embeddings = [], [], [], []
    for frame in _sample_frames(file_path, frames_per_clip):
        faces = [
            f for f in model.get(frame)
            if float(f.det_score) >= MIN_DET_SCORE
            and min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= MIN_FACE_PIXELS
        ]
        if not faces:
            continue
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = [int(max(0, v)) for v in face.bbox]
        genders.append("male" if face.sex == "M" else "female")
        ages.append(float(face.age))
        ita = ita_degrees(frame[y1:y2, x1:x2])
        if ita is not None:
            itas.append(ita)
        if getattr(face, "normed_embedding", None) is not None:
            embeddings.append(face.normed_embedding)
    if not genders:
        return None
    embedding = None
    if embeddings:
        mean = np.mean(embeddings, axis=0)
        embedding = mean / (np.linalg.norm(mean) or 1.0)
    return ClipPerson(
        clip=os.path.basename(file_path),
        frames_with_face=len(genders),
        gender=Counter(genders).most_common(1)[0][0],
        age=float(median(ages)),
        ita=float(median(itas)) if itas else None,
        embedding=embedding,
    )


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def mismatches(person: ClipPerson, spec: CharacterSpec, age_tolerance: int) -> dict[str, str]:
    """Return {attribute: reason} for every attribute that contradicts the spec."""
    problems = {}
    if spec.gender and person.gender != spec.gender:
        problems["gender"] = f"looks {person.gender}, expected {spec.gender}"
    low = (spec.age_min - age_tolerance) if spec.age_min else None
    high = (spec.age_max + age_tolerance) if spec.age_max else None
    if (low is not None and person.age < low) or (high is not None and person.age > high):
        problems["age"] = (
            f"looks ~{person.age:.0f}y, expected {spec.age_min or '?'}-{spec.age_max or '?'}y"
        )
    if spec.skin_tone and person.ita is not None and spec.skin_tone not in ita_classes(person.ita):
        problems["skin_tone"] = (
            f"{person.skin_tone} skin (ITA {person.ita:.0f}°), expected {spec.skin_tone}"
        )
    return problems


def best_spec(person: ClipPerson, specs: list[CharacterSpec], age_tolerance: int):
    scored = [(len(mismatches(person, s, age_tolerance)), i, s) for i, s in enumerate(specs)]
    _, _, spec = min(scored)
    return spec, mismatches(person, spec, age_tolerance)


def consensus_spec(people: list[ClipPerson]) -> CharacterSpec:
    """Majority look across clips, used when no explicit spec exists."""
    gender = Counter(p.gender for p in people).most_common(1)[0][0]
    age = median(p.age for p in people)
    tones = [p.skin_tone for p in people if p.skin_tone]
    tone = Counter(tones).most_common(1)[0][0] if tones else None
    return CharacterSpec("consensus", gender, int(round(age)), int(round(age)), tone)


def cosine(a, b) -> float:
    import numpy as np

    return float(np.dot(a, b))
