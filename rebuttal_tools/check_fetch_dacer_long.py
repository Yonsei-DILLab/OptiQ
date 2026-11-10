"""Check grouping invariance and the prespecified long-study selection rule."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json
from pathlib import Path
import tempfile
import numpy as np
from rebuttal_tools.check_fetch_vector import check_replay,check_density
from rebuttal_tools.fetch_dacer_long_campaign import choose


def check_grouping():
    from rebuttal_tools.fetch_vector_env import GroupedFetchVecEnv
    a=GroupedFetchVecEnv(8,[[],[]]);b=GroupedFetchVecEnv(8,[[],[],[],[]])
    try:
        a.seed(91700);b.seed(91700);assert np.array_equal(a.reset(),b.reset())
        for t in range(50):
            actions=np.tile(np.array([.1,-.2,.05,-.3],np.float32),(8,1))
            x=a.step(actions);y=b.step(actions)
            for i in range(3):assert np.array_equal(x[i],y[i]),(t,i)
            assert np.array_equal(np.stack([v['fetch_raw_next_observation'] for v in x[3]]),
                                  np.stack([v['fetch_raw_next_observation'] for v in y[3]]))
        return dict(passed=True,checks=['Two and four process groups produce identical paired-seed trajectories, rewards, dones, terminal states and autoresets'])
    finally:a.close();b.close()


def check_selection():
    with tempfile.TemporaryDirectory() as tmp:
        jobs=[]
        for name,five,ten,early in [('late',.6,1.,.5),('balanced',.8,.85,.7),('slow',.8,.85,.2)]:
            for seed in (0,1):
                out=Path(tmp)/f'{name}-{seed}';out.mkdir()
                rows=[dict(steps=t,stochastic_z=dict(success_rate=v,stages=dict(air_over5cm=dict(episodes=100,final_successes=0))))
                      for t,v in [(0,0),(2500096,early),(5000192,five),(10000128,ten)]]
                (out/'evaluations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                jobs.append(dict(profile=name,seed=seed,phase='search',output=str(out),target_entropy_per_dim=-.9,noise_scale=2.))
        assert choose(jobs)['selected_profile']=='balanced'
        assert choose(list(reversed(jobs)))['selected_profile']=='balanced'
        try:choose([j for j in jobs if j['seed']==0])
        except AssertionError:pass
        else:raise AssertionError('Selection accepted a single seed')
        return dict(passed=True,checks=['Reject a high-10M but weak-5M candidate','Use AUC for tied endpoints','Independent of job ordering','Require seeds0 and1'])


if __name__=='__main__':
    print(json.dumps(dict(replay=check_replay(),density=check_density(),grouping=check_grouping(),selection=check_selection())),flush=True)
