from pathlib import Path
import os
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
import importlib.util
import json
import sys
import hashlib
import numpy as np

ROOT=Path('/root/OptiQ-v5')
OUT=Path(__file__).resolve().parent
BASE=Path('/root/optiq-experiments/ant_v4_default_256x2_20260912T113914Z')
MEAN=Path('/root/optiq-experiments/ant_v4_mean_ot_T025_20260912T235059Z')
sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
from flax import serialization
from optiq_dime.algorithm import OptiQDIME
from run_optiq_dime import create_algorithm
from scripts.verify_v5 import verify

def load_algorithm(name, root):
    spec=importlib.util.spec_from_file_location('optiq_dime.'+name,root/'source/optiq_dime/algorithm.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.OptiQDIME

references={'sample':load_algorithm('frozen_v4',BASE),'mean':load_algorithm('frozen_mean_ot',MEAN)}
old=json.loads((BASE/'manifest.json').read_text())
rows=[]
hashes={}
for seed in range(4):
    cfg=verify(['benchmark=ant',f'seed={seed}',f'output_root={OUT/"validation_models"/str(seed)}'])
    model,callbacks=create_algorithm(cfg)
    job=next(j for j in old['jobs'] if j['seed']==seed and j['temperature']==.25)
    directory=next(Path(job['output_root']).glob('*/config.json')).parent
    try:
        def restore(kind, state):
            path=next(directory.rglob(f'{kind}_state_50000.msgpack'))
            data=path.read_bytes();hashes[str(path)]=hashlib.sha256(data).hexdigest()
            return serialization.from_bytes(state,data)
        actor=restore('actor',model.policy.actor_state)
        critic=restore('critic',model.policy.qf_state)
        with np.load(next(directory.rglob('landscape_probe_batch.npz'))) as data:
            obs=jnp.asarray(data['observations'][:16])
        for mode,reference in references.items():
            args=(actor,critic,obs,jax.random.PRNGKey(77+seed),jnp.array([-3600.]),
                16,4,'exact',.05,.5,False,True,1.,False,16.,257,.25,.25,100,
                'mean','argmax',True,False,'conditional_ot_nll','conditional_mixture',0.,False)
            expected=reference.update_actor(*args)
            actual=OptiQDIME.update_actor(*args,ot_student_action=mode)
            diffs=[float(np.max(np.abs(np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))))
                   for a,b in zip(jax.tree_util.tree_leaves(expected),jax.tree_util.tree_leaves(actual))]
            equal=all(np.array_equal(a,b) for a,b in zip(jax.tree_util.tree_leaves(expected),jax.tree_util.tree_leaves(actual)))
            assert max(diffs)<5e-7,(seed,mode,max(diffs))
            np.testing.assert_array_equal(expected[2],actual[2])
            row={'seed':seed,'mode':mode,'max_absolute_difference':max(diffs),'bitwise_equal':equal,
                 'checked':'actor params, optimizer state, loss, next RNG and every metric'}
            rows.append(row);print(json.dumps(row),flush=True)
    finally:
        callbacks.callbacks[0].eval_env.close();model.get_env().close();model.logger.close()
(OUT/'parity.json').write_text(json.dumps({'passed':True,'records':rows,'checkpoint_hashes':hashes,
    'current_algorithm_sha256':hashlib.sha256((ROOT/'optiq_dime/algorithm.py').read_bytes()).hexdigest(),
    'baseline_source':str(BASE/'source'),'mean_ot_source':str(MEAN/'source'),
    'reference_source_sha256':{mode:hashlib.sha256((root/'source/optiq_dime/algorithm.py').read_bytes()).hexdigest()
        for mode,root in [('sample',BASE),('mean',MEAN)]}},indent=2)+'\n')
