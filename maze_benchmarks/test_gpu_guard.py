"""An unhealthy peer GPU must not prevent querying a healthy locked slot."""
import unittest
from unittest.mock import patch
from . import gpu_guard


class IsolatedGpuQueryTest(unittest.TestCase):
    def test_queries_are_limited_to_selected_device(self):
        def query(*args):
            self.assertIn(args[0], ("--id=2", "--id=GPU-healthy"))
            if "--query-gpu=index,uuid" in args:
                return ["2, GPU-healthy"]
            return ["GPU-healthy, 123", "GPU-unrelated, 456"]
        with patch.object(gpu_guard, "nvidia_query", side_effect=query):
            self.assertEqual(gpu_guard.gpu_uuid(2), "GPU-healthy")
            self.assertEqual(gpu_guard.compute_pids("GPU-healthy"), {123})


if __name__ == "__main__":
    unittest.main()
