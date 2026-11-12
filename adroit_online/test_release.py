"""A partial, unreachable, failed or busy instance never passes the release gate."""
import unittest
from .release_monitor import ready
from .release_audit import compare
from .archive_direct import constraint_args, REMOTE


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

    def test_missing_or_corrupted_archive_cannot_pass(self):
        source = dict(files={'checkpoints/final': dict(bytes=123, sha256='abc')}, missing=[])
        self.assertTrue(compare(source, source)['passed'])
        for stored in [dict(files={}, missing=['checkpoints/final']),
                       dict(files={'checkpoints/final': dict(bytes=123, sha256='bad')}, missing=[])]:
            with self.assertRaises(AssertionError):
                compare(source, stored)

    def test_archive_agent_is_scoped_to_explicit_hop_and_storage(self):
        self.assertEqual(constraint_args(['[175.155.64.205]:19159']), [
            '-h', 'heejoonorm@[175.155.64.205]:19159',
            '-h', '[175.155.64.205]:19159>heejoonorm@[dil.yonsei.ac.kr]:2200'])
        compile(REMOTE, 'remote-archive', 'exec')


if __name__ == '__main__':
    unittest.main()
