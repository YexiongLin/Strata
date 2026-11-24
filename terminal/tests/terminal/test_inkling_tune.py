"""Inkling-2-Mini: the tuner's own class I arms (prompt chunk, DSpark off) instead of the refusal it had, and a saved
tuning applied at every start."""
from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[2]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from strata_control import autotune, presets, tune_knobs, tuning  # noqa: E402

INK_ARGS = ["--native", "/m/Inkling-2-Mini-SQ-20.strata", "--max-context", "131072",
            "--vision", "/m/Inkling-2-Mini-Vision.strata", "--vram-experts-gb", "15.68", "--ram-experts-gb", "0.00"]
INK_CFG = {"exe": "component:strata-inkling", "arch": "inkling2_loop", "model_name": "inkling-2-mini-sq-20", "args": list(INK_ARGS)}


class TestInklingTuning(unittest.TestCase):
    def ctx(self, **kw):
        return tune_knobs.Context(backend="cuda", family="inkling-2-mini", vram_mib=24564, ram_gib=62, model_gib=18.0,
                                  context_tokens=131072, base_args=list(INK_ARGS), **kw)

    def test_class_i_only(self):
        self.assertEqual(tune_knobs.machine_classes(self.ctx()), ["I"])

    def test_plans_only_inkling_arms(self):
        planned, _ = tune_knobs.plan_arms(self.ctx(allow_placement=True), mode="thorough")
        ids = [p.arm.id for p in planned]
        self.assertTrue(ids)
        self.assertTrue(all(i.startswith("ink_") for i in ids), ids)
        self.assertIn("ink_draft_off", ids)

    def test_qwen_plans_no_inkling_arm(self):
        ctx = tune_knobs.Context(backend="cuda", family="flash-next", vram_mib=24564, ram_gib=62, model_gib=47.0,
                                 context_tokens=262144, base_args=["--pack", "P", "--native", "m.gguf"], allow_placement=True)
        planned, _ = tune_knobs.plan_arms(ctx, mode="thorough")
        self.assertFalse([p.arm.id for p in planned if p.arm.id.startswith("ink_")])

    def test_no_per_request_rows_for_inkling(self):
        rows = [r for r in tune_knobs.PER_REQUEST if "inkling-2-mini" in r.get("families", ())]
        self.assertEqual(rows, [])          # strata-inkling takes none of the per-request tuning keys

    def test_arm_values_are_inside_the_bounds(self):
        ctx = self.ctx(allow_placement=True)
        for arm in tune_knobs.ARMS:
            if "inkling-2-mini" in arm.families:
                self.assertEqual(tune_knobs.check_setting(dict(arm.set_args), {}, ctx), [], arm.id)

    def test_context_family_from_the_tune_command(self):
        ctx = autotune.context_from_config(INK_CFG, engine_backend="cuda", opts=autotune.Options(),
                                           system={"memory": {"total_mib": 63000}, "gpus": [], "cpu": {}}, family="inkling-2-mini")
        self.assertEqual(ctx.family, "inkling-2-mini")
        self.assertEqual(ctx.context_tokens, 131072)

    def test_preset_choices_of_an_inkling_config(self):
        ch = presets.choices_from_config(INK_CFG)
        self.assertEqual(ch["family"], "inkling-2-mini")
        self.assertEqual(ch["kv"], "fp16")           # strata-inkling's own default, not the Qwen int8

    def test_saved_tuning_reaches_an_inkling_start(self):
        from serve import server
        self.assertTrue(server.takes_saved_tuning("strata", INK_CFG))       # the preset a tune saved is applied at every start
        hy4 = {"exe": "component:strata-hy4", "arch": "hy4_flash", "model_name": "hy4-flash", "args": []}
        if server.hy4_kind.is_hy4(hy4):
            self.assertFalse(server.takes_saved_tuning("strata", hy4))      # Hy4-Flash has no tuning
        self.assertFalse(server.takes_saved_tuning("llama", INK_CFG))

    @unittest.skipIf(os.name == "nt", "a shell script stands in for the engine")
    def test_flag_probe_reads_strata_inklings_words(self):
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d) / "strata-inkling"
            eng.write_text('#!/bin/sh\ncase "$1" in --chunk) echo "missing value after --chunk" >&2; exit 2;; '
                           '*) echo "unknown option: $1" >&2; exit 2;; esac\n')
            eng.chmod(eng.stat().st_mode | stat.S_IEXEC)
            self.assertTrue(tuning.engine_accepts_flag(eng, "--chunk"))
            self.assertFalse(tuning.engine_accepts_flag(eng, "--kv-resident"))


if __name__ == "__main__":
    unittest.main()
