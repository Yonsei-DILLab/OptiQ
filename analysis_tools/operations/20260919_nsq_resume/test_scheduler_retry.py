import subprocess,unittest
from unittest.mock import patch
import worker

class SchedulerRetry(unittest.TestCase):
    def setUp(self):worker.STOP=False
    def test_failure_and_partial_response_do_not_release_live_leases(self):
        replies=[subprocess.CalledProcessError(1,['squeue'],stderr='controller unavailable'),subprocess.CompletedProcess(['squeue'],0,stdout=''),subprocess.CompletedProcess(['squeue'],0,stdout='123_0\n456_0\n')]
        with patch.object(worker.subprocess,'run',side_effect=replies) as query,patch.object(worker.time,'sleep') as sleep:
            self.assertEqual(worker.scheduler_snapshot('123_0'),{'123_0','456_0'})
            self.assertEqual(query.call_count,3)
            self.assertEqual([x.args[0] for x in sleep.call_args_list],[10,20])
    def test_stop_after_timeout_does_not_return_empty_active_set(self):
        def stopped(_):worker.STOP=True
        with patch.object(worker.subprocess,'run',side_effect=subprocess.TimeoutExpired(['squeue'],30)),patch.object(worker.time,'sleep',side_effect=stopped):
            self.assertIsNone(worker.scheduler_snapshot('123_0'))

if __name__=='__main__':unittest.main()
