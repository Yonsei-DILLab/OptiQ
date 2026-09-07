import json
import unittest

from scripts.parameter_sweep import DEFAULT_SPEC, build_runs


class SweepTest(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(DEFAULT_SPEC.read_text())

    def test_full_matrix(self):
        runs = list(build_runs(self.spec))
        self.assertEqual(len(runs), 432)
        self.assertEqual(len({r["run_name"] for r in runs}), 432)
        for run in runs:
            cfg = run["overrides"]
            self.assertEqual(cfg["alg.actor.proposal_clip"], 2.5 * cfg["alg.actor.proposal_std"])
            anchor = cfg["alg.actor.include_anchor"]
            self.assertEqual(run["candidate_count"], 16 * cfg["alg.actor.proposals_per_policy_sample"])
            self.assertEqual(run["candidate_count"] - 16 * anchor, 64)

    def test_subset(self):
        self.assertEqual(len(list(build_runs(self.spec, ["dog-run"], [0]))), 36)

    def test_invalid_task(self):
        with self.assertRaises(ValueError):
            list(build_runs(self.spec, ["nonexistent"]))

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            list(build_runs(self.spec, ["dog-run"], [0, 0]))


if __name__ == "__main__":
    unittest.main()
