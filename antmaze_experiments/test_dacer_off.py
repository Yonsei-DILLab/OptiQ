"""Ensure the DACER-off ablation does not alter other T=1 settings."""
import unittest
from .register_dacer_off import campaign_manifest
from .register_dense_t1 import campaign_manifest as fixed_t1


class DacerOffTests(unittest.TestCase):
    def test_four_fresh_policies_change_only_behavior_regulator(self):
        jobs=[]
        for shard in (0,1):
            controls={j['task']:j for j in fixed_t1('/unused','test',shard)['jobs']}
            manifest=campaign_manifest('/unused','test',shard)
            self.assertEqual(manifest['wandb_project'],'antmaze')
            self.assertFalse(manifest['dacer_enabled'])
            self.assertEqual(len(manifest['jobs']),2)
            for entry in manifest['jobs']:
                self.assertEqual(entry['dacer'],'off')
                comparable={k:v for k,v in entry.items() if k!='dacer'}
                comparable['id']=controls[entry['task']]['id']
                self.assertEqual(comparable,controls[entry['task']])
                jobs.append(entry)
        self.assertEqual({j['task'] for j in jobs},{'v1','v2','v3','v4'})
        self.assertEqual(len({j['id'] for j in jobs}),4)


if __name__=='__main__':unittest.main()
