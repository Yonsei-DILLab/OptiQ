"""Reevaluate all frozen100K actors with one million IID actual actions each."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import flax.serialization
import jax
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,value):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--parent',type=Path,required=True)
    args=ap.parse_args();cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    import experiments
    experiments.__path__.append(str(args.parent/'source/experiments'))
    from experiments.kl_mode_missing_search_1d.core import Experiment
    from experiments.kl_mode_missing_search_1d.evaluate import reference
    assert jax.default_backend()=='gpu'
    assert not jax.config.x64_enabled
    manifest=json.loads((args.root/'SOURCE_MANIFEST.json').read_text())
    for method in cfg['methods']:
        for seed in cfg['seeds']:
            name=f'{method}_s{seed}';src=args.parent/'runtime/confirm/f05'/name;out=args.root/'runtime/results'/name
            out.mkdir(parents=True,exist_ok=True)
            if (out/'COMPLETE.json').exists():continue
            run=json.loads((src/'RUN.json').read_text());assert run['source_commit']==cfg['parent_commit']
            cp=src/'checkpoint.msgpack';digest=sha(cp)
            exp=Experiment(run['config'],method,run['L'],seed);exp.restore(cp)
            assert int(exp.state.step)==cfg['checkpoint_step']
            psha=hashlib.sha256(flax.serialization.to_bytes(exp.state.params)).hexdigest()
            meta=dict(name=name,method=method,seed=seed,analysis_commit=manifest['commit'],
                      parent_commit=run['source_commit'],checkpoint=str(cp),checkpoint_sha256=digest,
                      parent_config=run['config'],config=cfg,parameter_sha256=psha,
                      job_id=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename,
                      device=str(jax.devices()[0]),started=time.time())
            write(out/'RUN.json',meta)
            # Same external RNG stream per pairedseed/method; independent of training.
            key=jax.random.PRNGKey(cfg['random_seed']+seed)
            chunks=[]
            for chunk in range(cfg['sample_count']//cfg['sample_chunk']):
                a,_,_=exp.sample_fn(exp.state.params,jax.random.fold_in(key,chunk),cfg['sample_chunk'])
                a=np.asarray(a).ravel();assert a.dtype==np.float32
                assert np.all(np.isfinite(a)) and np.all(np.abs(a)<=cfg['action_bound'])
                chunks.append(a)
            actions=np.concatenate(chunks);assert len(actions)==cfg['sample_count']
            edges=np.linspace(-cfg['action_bound'],cfg['action_bound'],cfg['histogram_bins']+1)
            counts=np.histogram(actions,edges)[0];assert counts.sum()==cfg['sample_count']
            mass=counts/len(actions);target=np.diff(reference(run['config'],edges)[1])
            assert abs(target.sum()-1)<1e-12
            tv=float(.5*np.abs(mass-target).sum())
            prefix=np.histogram(actions[:32768],edges)[0]/32768
            centers=np.array(run['config']['target_centers']);bounds=np.r_[-10.,(centers[:-1]+centers[1:])/2,10.]
            mode_mass=np.histogram(actions,bounds)[0]/len(actions)
            previous=json.loads((src/'metrics_100000.json').read_text())['histogram_TV']
            metric=dict(step=cfg['checkpoint_step'],samples=len(actions),bins=cfg['histogram_bins'],
                        histogram_TV=tv,old_32768_TV=previous,
                        new_prefix_32768_TV=float(.5*np.abs(prefix-target).sum()),
                        mode_mass=mode_mass.tolist(),sample_min=float(actions.min()),sample_max=float(actions.max()))
            np.savez_compressed(out/'histogram.npz',edges=edges,counts=counts,mass=mass,target_mass=target)
            np.savez_compressed(out/'actions_1048576.npz',actions=actions)
            write(out/'METRICS.json',metric)
            assert sha(cp)==digest and hashlib.sha256(flax.serialization.to_bytes(exp.state.params)).hexdigest()==psha
            write(out/'COMPLETE.json',dict(time=time.time(),seconds=time.time()-meta['started'],analysis_commit=manifest['commit'],checkpoint_sha256=digest))
            print(json.dumps(dict(name=name,**metric)),flush=True)
            jax.clear_caches()


if __name__=='__main__':main()
