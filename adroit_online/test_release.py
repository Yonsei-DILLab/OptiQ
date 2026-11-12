"""A partial, unreachable, failed or busy instance never passes the release gate."""
import unittest
from .release_monitor import ready


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        self.reports = [dict(instance_id=i, completed=32, total=32,
                             ready_for_final_backup=True, failures=[], gpu_processes='')
                        for i in (60326900, 60327673)]

    def test_both_exact_instances_must_finish(self):
        self.assertTrue(ready(self.reports))
        self.assertFalse(ready(self.reports[:1]))
        self.reports[1]['instance_id'] = 123
        self.assertFalse(ready(self.reports))

    def test_pending_failed_unreachable_or_other_gpu_work_blocks(self):
        for patch in [dict(completed=31), dict(total=33), dict(failures=['failed-run']),
                      dict(ready_for_final_backup=False, error='unreachable'),
                      dict(gpu_processes='unrelated pid')]:
            reports = [dict(r) for r in self.reports]
            reports[0].update(patch)
            self.assertFalse(ready(reports))


if __name__ == '__main__':
    unittest.main()
