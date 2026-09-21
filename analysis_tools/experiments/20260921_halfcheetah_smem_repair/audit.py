"""Diagnose the frozen 50K-step policy on saved replay observations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import train
from optiq_dime import smem_tr as smem
from optiq_dime.projection_repair import distill_actor as repaired_distill


original_distill = smem.distill_actor


def traced_distill(state, observations, latents, old, fitted, actions, weights, cfg):
    batch, components, dims = old[0].shape
    obs = jnp.broadcast_to(observations[:, None], (batch, components, observations.shape[-1])).reshape(-1, observations.shape[-1])
    z = latents.reshape(-1, dims)

    def outputs(params):
        return tuple(x.reshape(batch, components, dims) for x in state.apply_fn({'params': params}, obs, z, return_raw=True))

    def regression(params):
        mu, _, raw_scale = outputs(params)
        return (.5*((mu-fitted[0])*jnp.exp(-old[1]))**2 + (raw_scale-fitted[1])**2).mean()

    baseline = smem.batch_nll(old, actions, weights).mean()

    def trace(start):
        def step(trial, _):
            loss, grad = jax.value_and_grad(regression)(trial.params)
            trial = trial.apply_gradients(grads=grad)
            mu, scale, _ = outputs(trial.params)
            gain = baseline-smem.batch_nll((mu, scale), actions, weights).mean()
            kl = smem.box_kl(mu, scale, *old).mean()
            return trial, jnp.stack((loss, gain, kl, ((gain >= 0) & (kl <= cfg['tr_kl'])).astype(float)))
        return jax.lax.scan(step, start, None, length=10)[1]

    result, metrics = original_distill(state, observations, latents, old, fitted, actions, weights, cfg)
    metrics['trace_adam'] = trace(state)
    metrics['trace_reset_adam'] = trace(state.replace(opt_state=state.tx.init(state.params)))
    return result, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    smem.distill_actor = traced_distill
    cfg = train.compose_config(['benchmark=halfcheetah', 'alg.buffer_size=32', f'output_root={tempfile.mkdtemp(prefix="smem-audit-")}'], 'smem_tr')
    model, callbacks = train.runner.create_algorithm(cfg)
    actor0, critic0 = model.policy.actor_state, model.policy.qf_state
    records = []
    for variant in ['original', 'parameter_regression', 'em_auxiliary']:
        smem.distill_actor = traced_distill if variant == 'original' else repaired_distill
        smem.DEFAULTS.update(projection_objective=variant, actor_backtracks=10,
                             projection_steps=10 if variant == 'original' else 1)
        update = jax.jit(lambda a,c,o,k: smem.update_actor(a,c,o,k,jnp.array([-3600.]),64,1,'exact',.006737946999085467,True,1.,.25,'mean'))
        for seed in range(3):
            run = next(args.campaign.glob(f'outputs/halfcheetah-smem_tr-seed{seed}_*'))
            ckpt = next((run/'checkpoints').iterdir())
            actor = flax.serialization.from_bytes(actor0, (ckpt/'actor_state_50000.msgpack').read_bytes())
            critic = flax.serialization.from_bytes(critic0, (ckpt/'critic_state_50000.msgpack').read_bytes())
            obs = jnp.asarray(np.load(ckpt/'landscape_probe_batch.npz')['observations'][:32])
            for key_seed in range(2):
                state, metrics, _ = update(actor, critic, obs, jax.random.PRNGKey(key_seed))
                jax.block_until_ready(state)
                assert all(np.isfinite(x).all() for x in jax.tree.leaves((state,metrics)))
                assert float(metrics['actor_nll_gain']) >= -2e-6
                assert float(metrics['actor_kl_bound']) <= .050002
                record = {'variant':variant, 'seed': seed, 'key_seed': key_seed, 'actor_step': int(actor.step), 'metrics': {k:np.asarray(v).tolist() for k,v in metrics.items()}}
                records.append(record)
                print(json.dumps(record), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'job_id':os.environ.get('SLURM_JOB_ID'), 'records':records},indent=2)+'\n')
    model.get_env().close()
    callbacks.callbacks[0].eval_env.close()


if __name__ == '__main__':
    main()
