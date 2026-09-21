"""Final source validation before freezing the long-running campaign."""
import argparse,json,unittest
from pathlib import Path
import numpy as np
from . import tests
from .landscape import run as landscape
from .plots import run as plots
from .campaign import tasks_for,cross_tasks
from .shared import config
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--smoke-root',required=True); ap.add_argument('--frozen-root',required=True); a=ap.parse_args(); out=Path(a.out)
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(tests)); assert result.wasSuccessful()
    cfg=config(); assert cfg.alg.actor.temperature==.25 and cfg.alg.actor.density_beta==1 and cfg.alg.actor.num_policy_samples==16 and cfg.alg.actor.proposals_per_policy_sample==4
    assert cfg.alg.actor.td_noise_std==.2 and cfg.alg.actor.source_q_eval=='mean' and cfg.alg.policy_tau==1 and cfg.alg.tau==.005
    assert len(tasks_for(out,range(5)))==175 and len(tasks_for(out,range(5,20)))==525 and len(cross_tasks(out,range(5)))==5
    for command_group in tasks_for(out,range(1))+cross_tasks(out,range(1)):
        for command in command_group: assert '--out' in command
    smoke=Path(a.smoke_root)
    landscape(argparse.Namespace(out=str(out/'cross_td3_sd3'),method='td3',end_method='sd3',seed=0,start=str(smoke/'td3/checkpoint_12.msgpack'),end=str(smoke/'sd3/checkpoint_12.msgpack'),critics=['td3:'+str(smoke/'td3/checkpoint_12.msgpack'),'sd3:'+str(smoke/'sd3/checkpoint_12.msgpack')],directions=4))
    plots(a.frozen_root,out/'frozen_figures'); plots(out,out/'cross_figures')
    for f in Path(a.frozen_root).rglob('local_modes*.npz'):
        d=np.load(f); np.testing.assert_allclose(d['mode_mass'].sum(),1,atol=1e-6)
    (out/'COMPLETE').write_text('Final release checks passed\n')
