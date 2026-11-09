import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models import const
from app.models.schema import VideoParams
from app.services import qc
from app.services import task as tm
from app.utils import utils

FFMPEG = shutil.which("ffmpeg")
SCRIPT = "Tóc giả cao cấp giúp bạn tự tin mỗi ngày với kiểu dáng tự nhiên"


def _ffmpeg(*args):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *args], check=True)


@unittest.skipUnless(FFMPEG, "ffmpeg is required")
class TestQCGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="qc-test-")
        d = cls.tmp

        def clip(name, size, seconds):
            out = os.path.join(d, name)
            _ffmpeg("-f", "lavfi", "-i", f"testsrc=size={size}:rate=25:duration={seconds}",
                    "-pix_fmt", "yuv420p", out)
            return out

        cls.portrait = [clip(f"p{i}.mp4", "1080x1920", 3) for i in range(4)]
        cls.landscape = clip("land.mp4", "1920x1080", 3)
        cls.tiny = clip("tiny.mp4", "240x426", 3)
        cls.corrupt = os.path.join(d, "corrupt.mp4")
        Path(cls.corrupt).write_bytes(b"not a video")

        cls.audio = os.path.join(d, "audio.mp3")
        _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=5", cls.audio)
        cls.silent_audio = os.path.join(d, "silent.mp3")
        _ffmpeg("-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "5", cls.silent_audio)

        cls.srt = os.path.join(d, "subtitle.srt")
        Path(cls.srt).write_text(
            "1\n00:00:00,000 --> 00:00:02,400\nTóc giả cao cấp giúp bạn tự tin\n\n"
            "2\n00:00:02,400 --> 00:00:04,900\nmỗi ngày với kiểu dáng tự nhiên\n",
            encoding="utf-8",
        )
        cls.bad_srt = os.path.join(d, "bad.srt")
        Path(cls.bad_srt).write_text(
            "1\n00:00:00,000 --> 00:00:02,400\nTóc giả cao cấp giúp bạn tự tin\n\n"
            "2\n00:00:02,000 --> 00:00:04,900\nlorem ipsum dolor sit amet\n",
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        patcher = patch(
            "app.services.qc_character.extract_specs_with_llm", return_value=[]
        )
        self.extract = patcher.start()
        self.addCleanup(patcher.stop)

    def _params(self, **kwargs):
        defaults = dict(
            video_subject="tóc giả", video_aspect="9:16", video_clip_duration=3,
            video_source="pexels", voice_name="vi-VN-HoaiMyNeural-Female",
            subtitle_enabled=True, qc_enabled=True,
        )
        defaults.update(kwargs)
        return VideoParams(**defaults)

    def _run(self, params=None, **overrides):
        args = dict(
            video_script=SCRIPT,
            video_terms=["wig", "hair salon", "woman hair"],
            audio_file=self.audio,
            audio_duration=5,
            subtitle_path=self.srt,
            materials=list(self.portrait),
        )
        args.update(overrides)
        return qc.run("qc-test", params or self._params(), **args)

    def test_clean_inputs_pass_with_full_rate(self):
        report = self._run()
        self.assertTrue(report.passed, report.to_dict())
        self.assertEqual(report.pass_rate, 1.0)
        saved = json.loads(
            Path(utils.task_dir("qc-test"), qc.QC_REPORT_FILE).read_text("utf-8")
        )
        self.assertTrue(saved["passed"])
        self.assertEqual(saved["total_checks"], report.total)

    def test_rate_must_be_strictly_above_threshold(self):
        report = self._run(materials=self.portrait + [self.landscape])
        self.assertFalse(report.critical_failures)
        self.assertLess(report.pass_rate, 1.0)
        failed = [c.name for c in report.failures]
        self.assertEqual(failed, ["clip:land.mp4"])
        self.assertFalse(report.passed)  # e.g. 14/15 = 93% < 98%
        lenient = self._run(params=self._params(qc_pass_threshold=90),
                            materials=self.portrait + [self.landscape])
        self.assertTrue(lenient.passed)
        exact = qc.QCReport(threshold=0.5)
        exact.add("x", "a", True)
        exact.add("x", "b", False)
        self.assertFalse(exact.passed)  # 50% is not > 50%

    def test_detects_bad_clips(self):
        report = self._run(materials=self.portrait + [self.tiny, self.corrupt])
        details = {c.name: c.detail for c in report.failures}
        self.assertIn("low resolution", details["clip:tiny.mp4"])
        self.assertIn("unreadable", details["clip:corrupt.mp4"])

    def test_missing_subtitles_is_critical(self):
        report = self._run(subtitle_path="")
        self.assertFalse(report.passed)
        self.assertEqual([c.name for c in report.critical_failures], ["subtitle_generated"])

    def test_subtitle_cue_problems(self):
        report = self._run(subtitle_path=self.bad_srt)
        detail = {c.name: c.detail for c in report.failures}["cue:2"]
        self.assertIn("overlaps", detail)
        self.assertIn("match script", detail)

    def test_footage_shorter_than_narration_is_critical(self):
        report = self._run(materials=self.portrait[:1])
        self.assertIn("materials_cover_audio", [c.name for c in report.critical_failures])

    def test_silent_and_unreadable_audio(self):
        silent = self._run(audio_file=self.silent_audio)
        self.assertIn("audio_silence_ratio", [c.name for c in silent.failures])
        missing = self._run(audio_file=os.path.join(self.tmp, "nope.mp3"))
        self.assertIn("audio_readable", [c.name for c in missing.critical_failures])

    def test_threshold_accepts_percent_and_config(self):
        self.assertEqual(qc.get_threshold(self._params(qc_pass_threshold=98)), 0.98)
        self.assertEqual(qc.get_threshold(self._params(qc_pass_threshold=0.9)), 0.9)
        with patch.dict(qc.config.qc, {"pass_threshold": 0.95}):
            self.assertEqual(qc.get_threshold(self._params()), 0.95)

    def test_gate_prunes_bad_clips_when_coverage_allows(self):
        params = self._params()
        with patch.object(tm, "get_video_materials") as refetch:
            result = tm._run_qc_gate(
                "qc-gate-prune", params, video_script=SCRIPT,
                video_terms=["wig", "hair salon", "woman hair"],
                audio_file=self.audio, audio_duration=5, subtitle_path=self.srt,
                downloaded_videos=self.portrait + [self.corrupt],
            )
        refetch.assert_not_called()
        self.assertEqual(result["materials"], self.portrait)
        self.assertTrue(result["qc"]["passed"])
        self.assertEqual(result["qc"]["pruned_materials"], ["corrupt.mp4"])

    def test_gate_refetches_free_footage_then_fails_task(self):
        params = self._params(qc_max_retries=1)
        with patch.object(tm, "get_video_materials", return_value=self.portrait[:1]) as refetch:
            result = tm._run_qc_gate(
                "qc-gate-fail", params, video_script=SCRIPT,
                video_terms=["wig", "hair salon", "woman hair"],
                audio_file=self.audio, audio_duration=5, subtitle_path=self.srt,
                downloaded_videos=self.portrait[:1],
            )
        refetch.assert_called_once()
        self.assertEqual(result["state"], const.TASK_STATE_FAILED)
        self.assertEqual(result["failed_stage"], "qc")
        self.assertIn("QC pass rate", result["error"])
        self.assertFalse(result["qc"]["passed"])

    def test_paid_sources_are_never_refetched(self):
        params = self._params(video_source="ofox", qc_max_retries=3)
        with patch.object(tm, "get_video_materials") as refetch:
            result = tm._run_qc_gate(
                "qc-gate-paid", params, video_script=SCRIPT,
                video_terms=["wig", "hair salon", "woman hair"],
                audio_file=self.audio, audio_duration=5, subtitle_path=self.srt,
                downloaded_videos=self.portrait[:1],
            )
        refetch.assert_not_called()
        self.assertEqual(result["failed_stage"], "qc")

    def test_pipeline_stops_before_rendering_when_qc_fails(self):
        params = self._params(video_script=SCRIPT)
        with (
            patch.object(tm, "generate_terms", return_value=["wig", "hair", "salon"]),
            patch.object(tm, "generate_audio", return_value=(self.audio, 5, None)),
            patch.object(tm, "generate_subtitle", return_value=""),
            patch.object(tm, "get_video_materials", return_value=list(self.portrait)),
            patch.object(tm, "generate_final_videos") as render,
        ):
            result = tm.start("qc-pipeline", params)
        render.assert_not_called()
        self.assertEqual(result["failed_stage"], "qc")
        self.assertIn("subtitle_generated", result["error"])


    def test_visual_black_frozen_blur(self):
        d = self.tmp
        black = os.path.join(d, "black.mp4")
        _ffmpeg("-f", "lavfi", "-i", "color=black:size=1080x1920:rate=25:duration=3",
                "-pix_fmt", "yuv420p", black)
        frozen = os.path.join(d, "frozen.mp4")
        _ffmpeg("-i", self.portrait[0], "-vf", "trim=end_frame=1,loop=loop=74:size=1",
                "-pix_fmt", "yuv420p", frozen)
        blurry = os.path.join(d, "blurry.mp4")
        _ffmpeg("-i", self.portrait[1], "-vf", "gblur=sigma=8", "-pix_fmt", "yuv420p", blurry)
        report = self._run(materials=self.portrait + [black, frozen, blurry])
        failed = {c.name for c in report.failures}
        self.assertIn("clip:black.mp4:black", failed)
        self.assertIn("clip:frozen.mp4:frozen", failed)
        self.assertIn("clip:blurry.mp4:blur", failed)
        self.assertFalse(any(name.startswith("clip:p") for name in failed), failed)
        self.assertEqual(report.failed_clips(), {"black.mp4", "frozen.mp4", "blurry.mp4"})


def _face_clips(directory):
    """Build short clips of real faces from InsightFace's bundled sample photo."""
    import cv2
    import insightface
    import numpy as np

    image = insightface.data.get_image("t1")
    boxes = {"man41": (466, 268, 573, 415), "woman25": (268, 145, 372, 267),
             "man32": (903, 62, 1013, 205)}
    crops = {}
    for name, (x1, y1, x2, y2) in boxes.items():
        crops[name] = image[max(0, y1 - 80):y2 + 80, max(0, x1 - 80):x2 + 80]
    # Same man, skin rendered much darker (simulates a different-looking actor).
    crops["man41_dark"] = (
        crops["man41"].astype(np.float32) * np.array([0.45, 0.5, 0.62])
    ).clip(0, 255).astype(np.uint8)
    clips = {}
    for name, crop in crops.items():
        height = int(540 * crop.shape[0] / crop.shape[1]) // 2 * 2
        canvas = np.full((960, 540, 3), 40, np.uint8)
        top = (960 - height) // 2
        canvas[top:top + height] = cv2.resize(crop, (540, height))
        png = os.path.join(directory, f"{name}.png")
        cv2.imwrite(png, canvas)
        out = os.path.join(directory, f"{name}.mp4")
        _ffmpeg("-loop", "1", "-i", png, "-t", "3", "-r", "25",
                "-vf", "zoompan=z='1+0.003*on':d=1:s=540x960:fps=25",
                "-pix_fmt", "yuv420p", out)
        clips[name] = out
    return clips


def _face_model_available():
    try:
        from app.services import qc_character

        qc_character.get_model()
        return True
    except Exception:
        return False


@unittest.skipUnless(FFMPEG and _face_model_available(), "face model unavailable")
class TestCharacterConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="qc-face-")
        cls.clips = _face_clips(cls.tmp)
        cls.audio = os.path.join(cls.tmp, "audio.mp3")
        _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=5", cls.audio)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _copies(self, name, count):
        paths = []
        for i in range(count):
            target = os.path.join(self.tmp, f"{name}_{i}.mp4")
            shutil.copyfile(self.clips[name], target)
            paths.append(target)
        return paths

    def _run(self, materials, characters=None, source="local", llm_specs=None):
        params = VideoParams(
            video_subject="tóc giả nam", video_aspect="9:16", video_clip_duration=3,
            video_source=source, subtitle_enabled=False, qc_enabled=True,
            voice_name="vi-VN-NamMinhNeural-Male", qc_characters=characters,
        )
        with patch("app.services.qc_character.extract_specs_with_llm",
                   return_value=llm_specs or []) as extract:
            report = qc.run(
                "qc-face", params, video_script="Anh trai U40 thử tóc giả mới",
                video_terms=["wig"], audio_file=self.audio, audio_duration=5,
                subtitle_path="", materials=materials,
            )
        return report, extract

    def _character_failures(self, report):
        return {c.name: c.detail for c in report.failures if c.category == "character"}

    def test_u40_man_spec(self):
        spec = [{"name": "anh trai", "gender": "male", "age_min": 35, "age_max": 44}]
        report, _ = self._run(
            [self.clips["man41"], self.clips["man32"], self.clips["woman25"]], spec
        )
        failures = self._character_failures(report)
        self.assertNotIn("clip:man41.mp4:age", failures)
        self.assertNotIn("clip:man41.mp4:gender", failures)
        self.assertIn("clip:man32.mp4:age", failures)       # looks ~27, too young
        self.assertIn("clip:woman25.mp4:gender", failures)
        self.assertIn("clip:woman25.mp4:age", failures)
        self.assertEqual(report.failed_clips(), {"man32.mp4", "woman25.mp4"})

    def test_light_skin_spec_rejects_dark_skin(self):
        spec = [{"gender": "male", "skin_tone": "light"}]
        report, _ = self._run([self.clips["woman25"], self.clips["man41_dark"]], spec)
        failures = self._character_failures(report)
        self.assertIn("clip:man41_dark.mp4:skin_tone", failures)
        self.assertIn("dark skin", failures["clip:man41_dark.mp4:skin_tone"])
        self.assertNotIn("clip:woman25.mp4:skin_tone", failures)

    def test_spec_extracted_from_script_when_not_given(self):
        from app.services.qc_character import CharacterSpec

        llm_spec = [CharacterSpec("anh trai", "male", 35, 44, None)]
        report, extract = self._run([self.clips["man32"]], llm_specs=llm_spec,
                                    source="pexels")
        extract.assert_called_once()
        self.assertEqual(report.character_specs[0]["source"], "script")
        self.assertIn("clip:man32.mp4:age", self._character_failures(report))

    def test_stock_footage_without_spec_is_not_scored(self):
        report, _ = self._run([self.clips["man41"], self.clips["woman25"]], source="pexels")
        self.assertFalse([c for c in report.checks if c.category == "character"])
        self.assertTrue(any("character" in s for s in report.skipped))

    def test_local_footage_without_spec_must_be_consistent(self):
        report, _ = self._run(self._copies("man41", 3) + [self.clips["woman25"]])
        self.assertEqual(report.character_specs[0]["source"], "consensus")
        failures = self._character_failures(report)
        self.assertIn("clip:woman25.mp4:gender", failures)
        self.assertIn("clip:woman25.mp4:same_person", failures)
        self.assertFalse([n for n in failures if n.startswith("clip:man41_")])

    def test_gate_drops_off_character_clips(self):
        params = VideoParams(
            video_subject="tóc giả nam", video_aspect="9:16", video_clip_duration=3,
            video_source="local", subtitle_enabled=False, qc_enabled=True,
            voice_name="vi-VN-NamMinhNeural-Male",
            qc_characters=[{"gender": "male", "age_min": 35, "age_max": 44}],
        )
        good = self._copies("man41", 2)
        with patch("app.services.qc_character.extract_specs_with_llm", return_value=[]):
            result = tm._run_qc_gate(
                "qc-face-gate", params, video_script="Anh trai U40 thử tóc giả mới",
                video_terms=[], audio_file=self.audio, audio_duration=5,
                subtitle_path="", downloaded_videos=good + [self.clips["woman25"]],
            )
        self.assertEqual(result["materials"], good)
        self.assertEqual(result["qc"]["pruned_materials"], ["woman25.mp4"])


class TestCharacterSpecParsing(unittest.TestCase):
    def test_vietnamese_aliases(self):
        from app.services.qc_character import CharacterSpec

        spec = CharacterSpec.from_any({"gender": "nam", "skin_tone": "trắng",
                                       "age_min": 44, "age_max": 35})
        self.assertEqual((spec.gender, spec.skin_tone, spec.age_min, spec.age_max),
                         ("male", "light", 35, 44))
        self.assertIsNone(CharacterSpec.from_any({"name": "x"}))

    def test_llm_response_is_parsed(self):
        from app.services import qc_character

        qc_character._spec_cache.clear()
        reply = '```json\n[{"name": "anh trai", "gender": "male", "age_min": 35, ' \
                '"age_max": 44, "skin_tone": "light"}]\n```'
        with patch("app.services.llm._generate_response", return_value=reply) as llm:
            specs = qc_character.extract_specs_with_llm("s", "Anh trai da trắng U40")
            again = qc_character.extract_specs_with_llm("s", "Anh trai da trắng U40")
        llm.assert_called_once()  # cached
        self.assertEqual(specs, again)
        self.assertEqual(specs[0].skin_tone, "light")
        self.assertEqual(specs[0].age_min, 35)

    def test_skin_tone_boundaries(self):
        from app.services.qc_character import ita_class, ita_classes

        self.assertEqual(ita_class(59), "light")
        self.assertEqual(ita_class(34), "medium")
        self.assertEqual(ita_class(-20), "dark")
        self.assertEqual(ita_classes(39), {"light", "medium"})  # near boundary
        self.assertNotIn("light", ita_classes(-20))


if __name__ == "__main__":
    unittest.main()
