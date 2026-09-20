"""Portable CPU/GPU preflight. This validates implementation, not GMM recovery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = (
    'gmm40/train_v8.py','gmm40/validate_v8.py','gmm40/validate_v7.py','gmm40/evaluation.py',
    'gmm40/v8.py','gmm40/v7.py','gmm40/target.py','gmm40/target_definition.json',
    'optiq_dime/conditional_sac_v8.py','optiq_dime/gaussian_transport.py',
    'optiq_dime/conditional_sac.py','optiq_dime/semi_implicit.py',
    'optiq_dime/policy.py','optiq_dime/optimizers.py','optiq_dime/latent.py',
    'optiq_dime/transport.py','optiq_dime/latent_transport.py','models/utils.py',
)


def source_hashes(root=ROOT):
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in DEPENDENCIES}


def scientific_config(seed=0, *, batch=256, num_students=4096, proposal_components=256,
                      teacher_resample_count=16, max_iterations=500, min_iterations=10,
                      relative_tolerance=1e-3, actor_max_grad_norm=None):
    return dict(seed=seed,batch=batch,num_students=num_students,proposal_components=proposal_components,
        teacher_resample_count=teacher_resample_count,actor_samples=teacher_resample_count,
        proposal_std=.05,temperature=1.,latent_seed=seed,hidden_dims=[256,256],
        max_iterations=max_iterations,min_iterations=min_iterations,
        relative_tolerance=relative_tolerance,actor_max_grad_norm=actor_max_grad_norm)


def make_agent(config):
    from .v8 import GMM40V8
    return GMM40V8(**config)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--platform',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--quick',action='store_true',help='Small CPU-scale implementation check; never a full-shape GPU benchmark')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--max-iterations',type=int,default=500)
    parser.add_argument('--min-iterations',type=int,default=10)
    parser.add_argument('--relative-tolerance',type=float,default=1e-3)
    parser.add_argument('--actor-max-grad-norm',type=float)
    args=parser.parse_args()
    os.environ['JAX_PLATFORMS']=args.platform
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    import jax
    import numpy as np
    from .evaluation import atomic_json
    from .validate_v7 import checkpoint_audit
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',scope='implementation_validation',requested_platform=args.platform,
                quick=args.quick,source_sha256=source_hashes())
    try:
        report['backend']=jax.default_backend()
        cfg=scientific_config(args.seed,max_iterations=args.max_iterations,min_iterations=args.min_iterations,
            relative_tolerance=args.relative_tolerance,actor_max_grad_norm=args.actor_max_grad_norm)
        if args.quick:
            cfg.update(batch=2,num_students=32,proposal_components=16,teacher_resample_count=4,actor_samples=4)
        report['config']=cfg
        agent=make_agent(cfg)
        data=agent.prepare(); jax.block_until_ready(data)
        assert np.all(data['ot']['converged'])
        expected=np.broadcast_to(np.arange(cfg['proposal_components']),data['teacher_component_indices'].shape)
        np.testing.assert_array_equal(data['teacher_component_indices'],expected)
        np.testing.assert_allclose(data['source_importance'],1.)
        times=[]
        for _ in range(3):
            started=time.monotonic(); metrics=agent.advance(2); times.append((time.monotonic()-started)/2)
        report['first_block_seconds_per_update']=times[0]
        report['warm_seconds_per_update']=float(np.median(times[1:]))
        checkpoint=out/'step_0000006.bin';agent.save(checkpoint)
        report['checkpoint_audit']=checkpoint_audit(checkpoint,6)
        restored=make_agent(cfg);restored.restore(checkpoint)
        agent.advance(1);restored.advance(1)
        for left,right in zip(jax.tree.leaves(agent.checkpoint()),jax.tree.leaves(restored.checkpoint())):
            np.testing.assert_array_equal(left,right)
        before=np.asarray(agent.key).copy(); samples,_=agent.evaluate_samples(128,900000+args.seed)
        assert np.isfinite(samples).all();np.testing.assert_array_equal(before,agent.key)
        report.update(status='passed',teacher_component_ids_each_once=True,teacher_draws_per_component=1,
            exact_resume_verified=True,evaluation_rng_isolated=True,metrics=metrics,
            full_shape=not args.quick,actual_updates=agent.updates)
        atomic_json(out/'preflight.json',report)
        print(json.dumps(report,indent=2))
    except Exception as error:
        report.update(status='failed',error=repr(error))
        atomic_json(out/'preflight.json',report)
        raise


if __name__=='__main__':
    main()
