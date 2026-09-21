import json, tempfile, time, unittest
from pathlib import Path
import worker

class Lanes(unittest.TestCase):
    def test_all_registered_tasks_partition_without_legacy_duplicates(self):
        plan=json.loads((Path(__file__).resolve().parents[2]/'studies/20260918_nonstationary_nd/tasks.json').read_text())
        selected=[(revision,t['name'],worker.lane_for(t)) for revision in worker.REVISIONS for t in plan if worker.revision_for(t)==revision]
        self.assertEqual(len(selected),8528)
        self.assertEqual(len({name for _,name,_ in selected}),8528)
        self.assertEqual(sum(lane=='exact' for _,_,lane in selected),756)
        self.assertEqual(sum(rev=='91cb9c3_legacy_td' for rev,_,_ in selected),64)
        self.assertTrue(all(worker.lane_for(t)=='fast' for t in plan if t['dim']==1 or t['n']==16))

    def test_dependencies_live_leases_and_completed_or_failed_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);t=dict(name='trial',parent='parent',q_source=None)
            self.assertFalse(worker.selectable(root,t,set()))
            p=root/'runs/parent/COMPLETE.json';p.parent.mkdir(parents=True);p.write_text('{}')
            self.assertTrue(worker.selectable(root,t,set()))
            lease=root/'queue/trial.json';lease.parent.mkdir();lease.write_text(json.dumps(dict(job_id='12_0',time=0)))
            self.assertFalse(worker.selectable(root,t,{'12_0'}))
            self.assertTrue(worker.selectable(root,t,set()))
            lease.write_text(json.dumps(dict(job_id='12_0',time=time.time())))
            self.assertFalse(worker.selectable(root,t,set()))
            lease.unlink();out=root/'runs/trial';out.mkdir()
            for marker in ['COMPLETE.json','FAILED.json']:
                (out/marker).write_text('{}');self.assertFalse(worker.selectable(root,t,set()));(out/marker).unlink()

    def test_ready_change_precedes_new_prefix_and_resume_precedes_new_work(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            base=dict(priority=100,dim=2,seed=0,name='trial')
            self.assertLess(worker.rank(dict(base,stage='mass'),out),worker.rank(dict(base,stage='prefix'),out))
            new=worker.rank(dict(base,stage='mass'),out)
            (out/'progress.json').write_text('{}')
            self.assertLess(worker.rank(dict(base,stage='prefix'),out),new)

if __name__=='__main__':unittest.main()
