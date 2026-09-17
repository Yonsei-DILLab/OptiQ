"""Check ordering, exact overrides, and fail-closed dependency behavior."""
from pathlib import Path
import tempfile
import unittest
from controller import read, write, tasks_for, command, predecessor_state


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.plan = read(Path(__file__).with_name('plan.json'))

    def test_twenty_unique_runs_in_requested_order(self):
        tasks = tasks_for(self.plan)
        self.assertEqual(len(tasks), 20)
        self.assertEqual(len({t['name'] for t in tasks}), 20)
        self.assertEqual([t['env'] for t in tasks],
                         [env for env in ['ant', 'humanoid', 'hopper', 'walker2d', 'halfcheetah'] for _ in range(4)])
        for env in self.plan['order']:
            group = [t for t in tasks if t['env'] == env]
            self.assertEqual([t['seed'] for t in group], [0, 1, 2, 3])
            self.assertEqual([t['gpu'] for t in group], [0, 1, 2, 3])
        cmd = command(self.plan, tasks[0], 'a' * 40)
        for arg in ['alg.actor.num_policy_samples=64', 'alg.actor.proposals_per_policy_sample=4',
                    'alg.actor.temperature=0.5', 'total_steps=1000000',
                    'wandb.project=DirectGMM_heejoon', 'wandb.mode=online']:
            self.assertIn(arg, cmd)

    def test_requires_full_predecessor_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(predecessor_state(root), 'waiting_for_predecessor')
            tasks = [dict(state='complete') for _ in range(20)]
            write(root / 'queue.json', dict(tasks=tasks, state='complete'))
            self.assertEqual(predecessor_state(root), 'waiting_for_predecessor')
            write(root / 'ALL_COMPLETE.json', dict(runs=20))
            self.assertEqual(predecessor_state(root), 'ready')
            tasks[0]['state'] = 'running'
            write(root / 'queue.json', dict(tasks=tasks, state='complete'))
            self.assertEqual(predecessor_state(root), 'waiting_for_predecessor')
            tasks[0]['state'] = 'failed'
            write(root / 'queue.json', dict(tasks=tasks, state='complete'))
            self.assertEqual(predecessor_state(root), 'blocked_predecessor_failure')


if __name__ == '__main__':
    unittest.main()
