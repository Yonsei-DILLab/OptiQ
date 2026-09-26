"""Differential CPU check against the pinned, unmodified DIPO replay class."""
import ast
import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


class UpstreamMemoryTest(unittest.TestCase):
    def test_vector_append_and_replace_match_upstream(self):
        source = Path(os.environ.get("DIPO_UPSTREAM_SOURCE", ROOT))
        spec = importlib.util.spec_from_file_location(
            "pinned_dipo_replay", source / "gmm40-baseline/DIPO/agent/replay_memory.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        upstream = module.DiffusionMemory
        # Load the actual nested adapter without constructing a CUDA learner.
        tree = ast.parse((ROOT / "maze_benchmarks/agents.py").read_text())
        dipo = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DIPO")
        adapter = next(n for n in ast.walk(dipo)
                       if isinstance(n, ast.ClassDef) and n.name == "VectorDiffusionMemory")
        namespace = {"DiffusionMemory": upstream, "np": np}
        exec(compile(ast.Module(body=[adapter], type_ignores=[]), "agents.py", "exec"), namespace)
        vector_class = namespace["VectorDiffusionMemory"]
        self.assertIs(vector_class.replace, upstream.replace)
        vector, scalar = [cls(3, 2, 8, "cpu") for cls in (vector_class, upstream)]
        for offset in (0, 6):
            states = np.arange(offset, offset + 18, dtype=np.float32).reshape(6, 3)
            actions = np.arange(offset, offset + 12, dtype=np.float32).reshape(6, 2) / 30
            vector.append(states, actions)
            for state, action in zip(states, actions):
                scalar.append(state, action)
        for name in ("states", "best_actions"):
            np.testing.assert_array_equal(getattr(vector, name), getattr(scalar, name))
        self.assertEqual((vector.idx, vector.full), (scalar.idx, scalar.full))
        before = vector.best_actions.copy()
        for memory in (vector, scalar):
            np.random.seed(4)
            _, sampled, indices = memory.sample(16)
            with torch.no_grad():
                sampled.add_(0.25)
            # Within-batch action improvement is retained, upstream replay is not rewritten.
            np.testing.assert_allclose(sampled.detach().numpy(), before[indices] + 0.25)
            memory.replace(indices, sampled.detach().numpy())
            np.testing.assert_array_equal(memory.best_actions, before)


if __name__ == "__main__":
    unittest.main()
