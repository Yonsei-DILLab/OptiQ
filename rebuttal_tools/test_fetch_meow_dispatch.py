import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rebuttal_tools.fetch_entropy_campaign import job
from rebuttal_tools.fetch_meow_dispatch import build_controller,mark_delegated,tick,finish_ready,mirror


class DispatchTests(unittest.TestCase):
    def test_scheduler_clone_gates_delegated_jobs(self):
        original=Path(__file__).with_name('fetch_entropy_campaign.py').read_text()
        source=build_controller(original)
        self.assertIn("if j.get('delegation'):continue",source)
        self.assertIn('finish_ready(plan)',source)
        self.assertIn('waiting_for_cuda_runtime',source)
        self.assertIn("if not j.get('delegation') and j['status'] in",source)

    def test_remote_pid_is_not_treated_as_a_local_worker(self):
        plan=dict(dispatch_assignments={'pilot':2})
        j=dict(id='pilot',phase='pilot',pid=123,start_ticks=55,gpu=0,learner_cpus=[0],worker_cpus=[[1],[2]])
        mark_delegated(plan,j)
        self.assertEqual(j['delegation']['host'],2)
        self.assertNotIn('pid',j)
        self.assertNotIn('learner_cpus',j)

    def test_shards_wait_for_selection_and_start_disjoint_finals_once(self):
        with tempfile.TemporaryDirectory() as directory:
            all_seeds={}
            for host,seeds in [(2,[0,1]),(3,[2,3,4])]:
                plan=dict(dispatch_role='shard',dispatch_final_seeds=seeds,jobs=[],campaign='test',
                    source='/source',source_manifest='/manifest',torch_python='/python',folder=directory,
                    ibolt_reference_config='/reference',dispatch_peers=[{}])
                self.assertFalse(finish_ready(plan))
                selections={task+'/meow':dict(status='selected',best=dict(value=.2)) for task in ['slide','reach']}
                with patch('rebuttal_tools.fetch_meow_dispatch.read_peer',return_value=dict(selections=selections)):
                    tick(plan,None,job);plan['last_dispatch_poll']=0;tick(plan,None,job)
                self.assertEqual(len(plan['jobs']),2*len(seeds))
                self.assertEqual({j['phase'] for j in plan['jobs']},{'final'})
                self.assertTrue(finish_ready(plan))
                all_seeds[host]={j['seed'] for j in plan['jobs']}
            self.assertFalse(all_seeds[2]&all_seeds[3])
            self.assertEqual(all_seeds[2]|all_seeds[3],{0,1,2,3,4})

    def test_result_mirror_retains_execution_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            j=dict(id='slide-meow-g1-s6-pilot',method='meow',phase='pilot',seed=6,task='slide',value=.2,output=directory)
            r=dict(j,status='completed',output='/remote/run',progress=dict(steps=5000192,updates=623424),
                verified=dict(completed=True,steps=5000192,updates=623424,backend='gpu',seed=6,
                    entropy_search=dict(value=.2),finished_unix=1.))
            mirror({},j,r,3)
            saved=json.loads((Path(directory)/'verified-completion.json').read_text())
            self.assertEqual(saved['remote_origin']['host'],3)
            self.assertEqual(saved['remote_origin']['output'],'/remote/run')
            self.assertNotIn('pid',j)


if __name__=='__main__':unittest.main()
