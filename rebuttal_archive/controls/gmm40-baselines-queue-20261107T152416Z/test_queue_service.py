import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('queue_service',Path(__file__).with_name('queue_service.py'))
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)

class QueueTests(unittest.TestCase):
 def test_dependency_waits_for_all_reserved_jobs_and_coordinator(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'upstream.json';jobs=[dict(status='completed') for _ in range(4)]+[dict(status='queued')]
   p.write_text(json.dumps(dict(jobs=jobs,finished_unix=None)))
   cfg=dict(prerequisites=[dict(program='priority-job',manifest=str(p),expected_jobs=5)],blocking_program_prefixes=['priority-'])
   self.assertFalse(q.prerequisites_ready(cfg,{'priority-job':'RUNNING'})[0])
   self.assertFalse(q.prerequisites_ready(cfg,{'priority-job':'EXITED'})[0])
   jobs[-1]['status']='completed';p.write_text(json.dumps(dict(jobs=jobs,finished_unix=1)))
   self.assertTrue(q.prerequisites_ready(cfg,{'priority-job':'EXITED'})[0])
   self.assertFalse(q.prerequisites_ready(cfg,{'priority-job':'EXITED','priority-new':'STARTING'})[0])
   jobs[-1]['status']='failed';p.write_text(json.dumps(dict(jobs=jobs,finished_unix=1)))
   self.assertFalse(q.prerequisites_ready(cfg,{'priority-job':'EXITED'})[0])
 def test_gpu_quiet_utilization_does_not_hide_process_or_memory(self):
  cfg=dict(max_idle_memory_mib=256,max_idle_utilization=5)
  self.assertTrue(q.gpu_idle(dict(pids=[],memory_mib=2,utilization=0),cfg))
  self.assertFalse(q.gpu_idle(dict(pids=['123'],memory_mib=2,utilization=0),cfg))
  self.assertFalse(q.gpu_idle(dict(pids=[],memory_mib=1066,utilization=0),cfg))
  self.assertFalse(q.gpu_idle(dict(pids=[],memory_mib=2,utilization=9),cfg))
 def test_dry_run_never_queries_gpu_or_starts_jobs_when_dependency_active(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);config=root/'config.json';state=root/'state.json'
   cfg=dict(campaign='test',host='test-host',state_file=str(state),revision='abc',poll_seconds=1,
            jobs=[dict(id='meow-s0',baseline='meow',seed=0,gpu=0,program='queued-job',output=str(root/'output'),actor_updates=100000)],
            prerequisites=[],blocking_program_prefixes=['priority-'])
   config.write_text(json.dumps(cfg))
   with patch.object(sys,'argv',['queue','--config',str(config),'--once','--dry-run']),patch.object(q,'supervisor_states',return_value={'priority-active':'RUNNING','queued-job':'STOPPED'}),patch.object(q,'gpu_inventory',side_effect=AssertionError('must not query GPUs')),patch.object(q.subprocess,'run',side_effect=AssertionError('must not launch processes')):
    q.main()
   result=json.loads((root/'state.dry-run.json').read_text())
   self.assertEqual(result['status'],'waiting');self.assertFalse(result['prerequisites_ready'])
   self.assertEqual(result['jobs'][0]['status'],'queued');self.assertFalse((root/'output').exists())
 def test_completed_marker_must_match_method_seed_budget_and_revision(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);job=dict(output=str(p),actor_updates=100000,baseline='td3',seed=4)
   (p/'completed.json').write_text(json.dumps(dict(actor_updates=100000,completed=True)))
   cfg=dict(baseline='td3',seed=4,steps=100000,runtime=dict(revision='abc'))
   (p/'config.json').write_text(json.dumps(cfg));self.assertTrue(q.completed(job,'abc'))
   self.assertFalse(q.completed(job,'different'))
   cfg['seed']=3;(p/'config.json').write_text(json.dumps(cfg));self.assertFalse(q.completed(job,'abc'))

if __name__=='__main__':unittest.main()
