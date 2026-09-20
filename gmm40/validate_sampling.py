"""Confirm saved rollout samplers match native policies and isolate train RNG."""
import argparse
import json
import numpy as np
from .target import Target,RESULTS
from .evaluation import atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--framework',choices=['torch','jax'],required=True)
    args=parser.parse_args();target=Target();results={};n=64;seed=9123
    if args.framework=='torch':
        import torch
        from .torch_agents import make_agent
        for method in ('sac','dipo','meow'):
            agent=make_agent(method,target,0,256)
            path=RESULTS/({'sac':'sac_seed0','dipo':'dipo_native_seed0','meow':'validation_meow_100'}[method])/'checkpoints'/('step_0000100.bin' if method=='meow' else 'step_0010000.bin')
            agent.restore(path)
            cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
            saved,_,_=agent.evaluate_samples(n,seed)
            assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
            with torch.random.fork_rng(devices=[0]),torch.no_grad():
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                obs=torch.zeros((n,1),device='cuda')
                if method=='sac':native=40*agent.actor(obs)[0].tanh()
                elif method=='dipo':native=40*agent.actor.sample(obs,eval=False)
                else:native=40*agent.actor.sample(n,obs)[0]
            error=float(np.max(np.abs(saved-native.cpu().numpy())))
            assert error<2e-5,(method,error)
            results[method]={'native_max_absolute_error':error,'training_rng_unchanged':True,'checkpoint':str(path)}
    else:
        import jax
        import jax.numpy as jnp
        from .mfpo import MFPO,action_sampler
        agent=MFPO(target,0,256)
        path=RESULTS/'validation_mfpo_native_100/checkpoints/step_0000100.bin';agent.restore(path)
        key=np.asarray(agent.key).copy()
        native_fn=jax.jit(lambda params,key:40*action_sampler(agent.state.apply_fn,params,2,jax.random.normal(key,(n,2)),jnp.zeros((n,1)),True))
        saved,_,_=agent.evaluate_samples(n,seed)
        native=native_fn(agent.state.params,jax.random.PRNGKey(seed))
        default_error=float(np.max(np.abs(saved-np.asarray(native))))
        # On this GPU, Python-unrolled versus lax.scan graphs select different TF32
        # matmul kernels. Compare equations at full precision instead of attributing
        # compiler roundoff to a policy change; retain the default discrepancy.
        with jax.default_matmul_precision('highest'):
            saved,_,_=agent.evaluate_samples(n,seed)
            native=native_fn(agent.state.params,jax.random.PRNGKey(seed))
            error=float(np.max(np.abs(saved-np.asarray(native))))
        assert np.array_equal(key,np.asarray(agent.key)) and error<2e-5,error
        results['mfpo']={'native_max_absolute_error_highest_precision':error,
                         'native_max_absolute_error_default_precision':default_error,
                         'numerical_note':'Unrolled and scan graphs agree at highest precision; default TF32 graph roundoff recorded separately. Training precision unchanged.',
                         'training_rng_unchanged':True,'checkpoint':str(path)}
    atomic_json(RESULTS/f'sampling_validation_{args.framework}.json',dict(status='passed',results=results))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
