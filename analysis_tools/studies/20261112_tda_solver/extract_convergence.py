"""Nested latent banks, gated on all six unchanged-N/M validation results."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import numpy as np
from checkpoint_actor import load_actor,locate,evaluate


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoints','core','validation_root','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--commit',required=True);a=p.parse_args()
    src=Path(__file__).resolve().parent
    cfg=json.loads((src/'convergence.json').read_text());cases=json.loads((src/'phase_a.json').read_text())['cases']
    proofs=[]
    for case in cases:
        stem=f'{case["env"]}_s{case["seed"]}_state{case["state"]}_n32768_verified'
        found=list(a.validation_root.glob(f'validation-final-gpu*/{stem}/result.json'))
        if len(found)!=1:raise ValueError(('Validation result missing or ambiguous',case))
        result=json.loads(found[0].read_text())
        if not result['validation_passed'] or result['n']!=32768 or result['m']!=32768:
            raise ValueError(('Validation gate failed',case))
        proofs.append(dict(file=str(found[0]),sha256=hashlib.sha256(found[0].read_bytes()).hexdigest()))
    a.out.mkdir(parents=True,exist_ok=False)
    manifest=dict(source_commit=a.commit,gate_proofs=proofs,cases=[],config=cfg)
    for case in cases:
        env,seed,state=case['env'],case['seed'],case['state']
        path=locate(a.checkpoints,env,seed);params,stats=load_actor(path)
        with np.load(a.core/'states'/f'{env}_s{seed}_states.npz') as f:obs=f['states'][state:state+1]
        dim=params['mu']['bias'].shape[0]
        z=np.asarray(jax.random.normal(jax.random.PRNGKey(cfg['latent_seed']),(max(cfg['n_values']),dim)),np.float32)
        mu,ls=evaluate(params,stats,obs,z)
        for n in cfg['n_values']:
            file=a.out/f'{env}_s{seed}_state{state}_n{n}.npz'
            np.savez(file,mu=mu[:,:n],log_std=ls[:,:n],states=obs,z=z[:n],original_state=state,seed=seed,env=env)
            manifest['cases'].append(dict(case,n=n,file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                                         checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print('EXTRACTED',file.name,flush=True)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
