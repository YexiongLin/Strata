"""A setting must not buy its speed with conversation memory the measurement cannot see.

The tuner measures decode on short prompts, which always sit in graphics memory. On strata-inkling a larger prompt chunk
takes VRAM workspace, so the global KV keeps fewer tokens in VRAM and long conversations decode slower (RTX 4090, SQ-20,
128K: --chunk 1024 kept 13568 instead of 15360 tokens in VRAM, prompts read 1-6% faster, decode at 19K-38K tokens ~8%
slower). The engine's log says where its KV went; a setting that moves conversation out of VRAM, or shortens the
conversation the engine can hold, is not taken."""

from __future__ import annotations

import unittest

from strata_control import autotune, tune_knobs, tune_report
from strata_control.tune_session import parse_facts

from tests.tuning.stub_session import FakeClock, StubSession, flag

INK_BASE = ["--native", "/m/Inkling-2-Mini-SQ-20.strata", "--max-context", "131072", "--kv", "k8v8",
            "--vision", "/m/Inkling-2-Mini-Vision.strata", "--vram-experts-gb", "15.68", "--ram-experts-gb", "0.00"]
AUTO = ("[InkCUDA] Auto Host KV Offload (kv_fmt=2, 4.17 GiB free VRAM): global KV = {v} VRAM + {h} mapped host tokens "
        "(5593.5 MiB host RAM, max_ctx=131072)")


def ink_facts(vram: int, host: int) -> dict:
    return parse_facts(AUTO.format(v=vram, h=host))


class ParseFactsTest(unittest.TestCase):
    def test_auto_offload_line(self):
        f = ink_facts(15360, 115712)
        self.assertEqual((f["kv_vram_tokens"], f["kv_host_tokens"]), (15360, 115712))

    def test_offload_on_line(self):
        f = parse_facts("[InkCUDA] Host KV Offload ON (kv_fmt=0): 8 global visits = 0 VRAM + 131072 mapped host tokens (4096.0 MiB host RAM)")
        self.assertEqual((f["kv_vram_tokens"], f["kv_host_tokens"]), (0, 131072))

    def test_a_later_spill_wins(self):
        f = parse_facts(AUTO.format(v=15360, h=115712) + "\n"
                        "[InkCUDA] Elastic global KV spilled to mapped host RAM: 9000 VRAM + 3000 host = 12000 tokens\n")
        self.assertEqual((f["kv_vram_tokens"], f["kv_host_tokens"]), (9000, 3000))

    def test_a_shortened_context(self):
        f = parse_facts("[InkCUDA] Auto-sizing max_ctx 131072 -> 65536 for parallel=1 (4.17 GiB free VRAM)")
        self.assertEqual(f["ctx_auto_sized"], 65536)

    def test_other_engines_say_nothing_about_it(self):
        f = parse_facts("expert cache 3000 slots\n6 pool workers\nprompt chunk auto: 512 tokens\n")
        for key in ("kv_vram_tokens", "kv_host_tokens", "ctx_auto_sized"):
            self.assertNotIn(key, f)


class RuleTest(unittest.TestCase):
    def test_fewer_vram_tokens_with_host_kv_is_a_cost(self):
        why = autotune.kv_residency_cost(ink_facts(13568, 117504), ink_facts(15360, 115712))
        self.assertIn("1792 fewer", why)
        self.assertIn("13568 instead of 15360", why)

    def test_within_one_rounding_step_is_not(self):
        self.assertIsNone(autotune.kv_residency_cost(ink_facts(15104, 115968), ink_facts(15360, 115712)))

    def test_more_vram_tokens_is_not(self):
        self.assertIsNone(autotune.kv_residency_cost(ink_facts(16384, 114688), ink_facts(15360, 115712)))

    def test_all_in_vram_is_not(self):
        self.assertIsNone(autotune.kv_residency_cost({}, {}))
        self.assertIsNone(autotune.kv_residency_cost(ink_facts(4096, 0), ink_facts(8192, 0)))

    def test_moving_a_fitting_conversation_to_host_is_a_cost(self):
        self.assertIn("out of graphics memory", autotune.kv_residency_cost(ink_facts(98304, 32768), {}))

    def test_a_shorter_conversation_is_a_cost(self):
        self.assertIn("65536", autotune.kv_residency_cost({"ctx_auto_sized": 65536}, {}))
        self.assertIn("65536", autotune.kv_residency_cost({"ctx_auto_sized": 65536}, {"ctx_auto_sized": 98304}))
        self.assertIsNone(autotune.kv_residency_cost({"ctx_auto_sized": 65536}, {"ctx_auto_sized": 65536}))


def ink_ctx() -> tune_knobs.Context:
    return tune_knobs.Context(backend="cuda", family="inkling-2-mini", vram_mib=24564, ram_gib=62, model_gib=18.0,
                              context_tokens=131072, base_args=list(INK_BASE), allow_placement=True)


def chunk_pp(args, env):
    return 3450.0 * {"1024": 1.09, "2048": 1.08}.get(flag(args, "--chunk"), 1.0)


def run(facts_for):
    clock = FakeClock()
    sess = StubSession(model=lambda a, e, tune=None: 108.6, pp=chunk_pp, clock=clock, noise=0.001, facts_for=facts_for)
    t = autotune.Tuner(sess, list(INK_BASE), {}, ink_ctx(), autotune.Options(budget_s=3600, allow_placement=True), clock=clock,
                       sleep=clock.sleep, system=sess.system, total_ram_mib=64000, total_vram_mib=24564, quiet=lambda: None)
    return t.run()


class TunerTest(unittest.TestCase):
    def test_a_chunk_that_takes_kv_out_of_vram_is_not_taken(self):
        vram = {"1024": 13568, "2048": 12800}

        def facts_for(args):
            v = vram.get(flag(args, "--chunk"), 15360)
            return ink_facts(v, 131072 - v)
        r = run(facts_for)
        self.assertNotIn("--chunk", r.settings.get("args") or {})
        arms = {a["id"]: a for a in r.arms}
        for arm in ("ink_chunk_1024", "ink_chunk_2048"):
            self.assertEqual(arms[arm]["status"], "rejected", arms[arm])
            self.assertIn("fewer tokens of the conversation in graphics memory", arms[arm]["reason"])
            self.assertTrue(tune_report._verdict(arms[arm]).startswith("not taken: "))

    def test_the_same_chunk_wins_when_the_kv_stays_put(self):
        r = run(lambda args: ink_facts(15360, 115712))
        self.assertEqual((r.settings.get("args") or {}).get("--chunk"), 1024, r.notes)


if __name__ == "__main__":
    unittest.main()
