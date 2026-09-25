"""IID latent-bank convergence with frozen original actor and fixed actions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import flax.serialization
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(p)


def make_accumulator(chunk, dtype=jnp.float64):
    def acc(mu, ls, actions, start, end, carry):
        mu = mu.astype(dtype).reshape(-1, chunk, 1)
        ls = ls.astype(dtype).reshape(-1, chunk, 1)
        a = actions.astype(dtype).reshape(1, -1)
        offsets = jnp.arange(chunk)
        carry = tuple(x.astype(dtype) for x in carry)

        def body(b, old):
            peak, total, numerator, squares = old
            m, l = mu[b], ls[b]
            inv = jnp.exp(-l)
            norm = jsp.special.ndtr((10-m)*inv)-jsp.special.ndtr((-10-m)*inv)
            logk = -.5*((a-m)*inv)**2-l-.5*jnp.log(2*jnp.pi)-jnp.log(norm)
            valid = (b*chunk+offsets >= start) & (b*chunk+offsets < end)
            logk = jnp.where(valid[:, None], logk, -jnp.inf)
            p = jnp.max(logk, axis=0)
            w = jnp.exp(logk-p)
            new_peak = jnp.maximum(peak, p)
            old_factor, new_factor = jnp.exp(peak-new_peak), jnp.exp(p-new_peak)
            return (new_peak, old_factor*total+new_factor*w.sum(0),
                    old_factor*numerator+new_factor*(w*(m-a)*inv**2).sum(0),
                    old_factor**2*squares+new_factor**2*(w*w).sum(0))

        return jax.lax.fori_loop(start//chunk, (end+chunk-1)//chunk, body, carry)
    return jax.jit(acc)


def initial(count):
    return (jnp.full(count, -jnp.inf, dtype=jnp.float64),
            jnp.zeros(count, dtype=jnp.float64),
            jnp.zeros(count, dtype=jnp.float64),
            jnp.zeros(count, dtype=jnp.float64))


def unpack(carry, L):
    peak, total, numerator, squares = map(np.asarray, carry)
    return numerator/total, peak+np.log(total)-np.log(L), total**2/squares


def validate(accumulator):
    """Includes a near-boundary Gaussian so truncation normalization is checked."""
    rng = np.random.default_rng(338)
    mu = jnp.array(rng.uniform(-9.9, 9.9, (8192, 1)), dtype=jnp.float32)
    ls = jnp.array(rng.uniform(-5, -1, (8192, 1)), dtype=jnp.float32)
    a = jnp.array([-9.8, -4.25, -.3, 0., .7, 4.25, 9.8], dtype=jnp.float64)
    m, l = mu.astype(jnp.float64), ls.astype(jnp.float64)

    def log_density(x, size):
        mm, ll = m[:size], l[:size]
        inv = jnp.exp(-ll)
        norm = jsp.special.ndtr((10-mm)*inv)-jsp.special.ndtr((-10-mm)*inv)
        ell = -.5*((x-mm[:, 0])*inv[:, 0])**2-ll[:, 0]-.5*jnp.log(2*jnp.pi)-jnp.log(norm[:, 0])
        return jsp.special.logsumexp(ell)-jnp.log(size)

    carry, prev, errors = initial(len(a)), 0, []
    for L in [128, 256, 4096, 8192]:
        carry = accumulator(mu, ls, a, prev, L, carry)
        observed, _, _ = unpack(carry, L)
        expected = np.asarray(jax.vmap(jax.grad(log_density), in_axes=(0, None))(a, L))
        error = float(np.max(np.abs(observed-expected)))
        assert np.allclose(observed, expected, atol=1e-8, rtol=1e-8), error
        errors.append(dict(L=L, max_abs_score_error=error))
        prev = L
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--index', type=int, required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    # Reuse original numerical source without copying or modifying the training snapshot.
    import experiments
    experiments.__path__.append(str(args.parent/'source/experiments'))
    from experiments.kl_six_highL_1d.core import implementation

    case = cfg['tasks'][args.index]['case']
    cfg['fixed_target_probes'] = cfg['target_probes'][case]
    method = 'reverse'
    seed = cfg['tasks'][args.index]['seed']
    name = f'{case}/{method}_s{seed}'
    src = args.parent/'runtime/confirm'/name
    assert json.loads((src/'COMPLETE.json').read_text())['step'] == 100000
    out = args.root/'runtime/results'/name
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        return
    assert jax.default_backend() == 'gpu', 'Do not run analysis on the login node'
    run = json.loads((src/'RUN.json').read_text())
    assert run['source_commit'] == cfg['parent_source_commit']
    assert run['L'] == cfg['training_L']
    cp = src/'checkpoint.msgpack'
    digest = sha(cp)
    # Templates are restored, not trained. Preserve float32 parameter dtypes.
    with jax.experimental.enable_x64(False):
        exp = implementation(run['config'])(run['config'], method, run['L'], seed)
        exp.restore(cp)
    assert int(exp.state.step) == cfg['checkpoint_step']
    assert all(v.dtype == jnp.float32 for v in jax.tree_util.tree_leaves(exp.state.params))
    actor_hash = hashlib.sha256(flax.serialization.to_bytes(exp.state.params)).hexdigest()
    saved = np.load(src/'samples_100000.npz')['actions'].ravel()
    indices = np.linspace(0, len(saved)-1, cfg['fixed_policy_probes']).astype(int)
    policy = saved[indices]
    quantiles = np.quantile(saved, cfg['policy_quantiles'])
    actions = np.concatenate([policy, quantiles, cfg['fixed_target_probes']]).astype(np.float32)
    np.save(out/'fixed_actions.npy', actions)
    meta = dict(name=name, case=case, method=method, seed=seed, checkpoint_step=int(exp.state.step),
                checkpoint_path=str(cp), checkpoint_sha256=digest, actor_sha256=actor_hash,
                parent_source_commit=run['source_commit'],
                analysis_commit=json.loads((args.root/'SOURCE_MANIFEST.json').read_text())['commit'],
                config=cfg, device=str(jax.devices()[0]), device_kind=jax.devices()[0].device_kind,
                job_id=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename,
                action_sha256=sha(out/'fixed_actions.npy'), started=time.time())
    write(out/'RUN.json', meta)
    acc = make_accumulator(cfg['kernel_chunk'])
    write(out/'VALIDATION.json', dict(autodiff_parity=validate(acc)))

    def bank_impl(z):
        mu, ls = jax.lax.map(lambda zz: exp.density_components(exp.state.params, zz),
                             z.reshape(-1, cfg['actor_chunk'], 1))
        return mu.reshape(-1, 1), ls.reshape(-1, 1)
    bank = jax.jit(bank_impl)
    Ls = np.array([2**p for p in cfg['L_powers']])
    reference_scores, reference_logs = [], []
    scores, logps, esses = [], [], []
    reference_sd = None
    precision = {}
    total_repeats = cfg['repeats']+cfg['independent_reference_repeats']
    for rep in range(total_repeats):
        t0 = time.perf_counter()
        rng = np.random.default_rng(cfg['random_seed']+1000*args.index+rep)
        z = rng.standard_normal((int(max(Ls)), 1), dtype=np.float32)
        mu, ls = bank(jnp.asarray(z))
        jax.block_until_ready(mu)
        assert mu.dtype == jnp.float32 and ls.dtype == jnp.float32
        carry, prev, ss, ll, ee = initial(len(actions)), 0, [], [], []
        ref = rep >= cfg['repeats']
        for L in [cfg['reference_L']] if ref else Ls:
            carry = acc(mu, ls, jnp.asarray(actions), prev, int(L), carry)
            s, lp, ess = unpack(carry, L)
            assert np.all(np.isfinite(s)) and np.all(np.isfinite(lp))
            ss.append(s); ll.append(lp); ee.append(ess)
            if rep == 0 and L == cfg['training_L']:
                small_mu, small_ls = mu[:L], ls[:L]
                carry32 = make_accumulator(cfg['kernel_chunk'], jnp.float32)(
                    small_mu, small_ls, jnp.asarray(actions), 0, int(L), initial(len(actions)))
                s32, _, _ = unpack(carry32, L)
                precision = dict(L=int(L), policy_max_abs_difference=float(np.max(np.abs(s32[:len(policy)]-s[:len(policy)]))),
                                 all_actions_max_abs_difference=float(np.max(np.abs(s32-s))))
            prev = int(L)
        if ref:
            reference_scores.append(ss[-1]); reference_logs.append(ll[-1])
        else:
            scores.append(ss); logps.append(ll); esses.append(ee)
        np.savez_compressed(out/'scores.npz', Ls=Ls, actions=actions,
                            scores=np.asarray(scores), log_density=np.asarray(logps),
                            ess=np.asarray(esses), reference_scores=np.asarray(reference_scores),
                            reference_log_density=np.asarray(reference_logs))
        status = dict(phase='independent_reference' if ref else 'MC', completed=rep+1,
                      total=total_repeats, last_seconds=time.perf_counter()-t0, time=time.time())
        write(out/'STATUS.json', status)
        print(json.dumps(dict(name=name, **status)), flush=True)

    scores = np.asarray(scores)
    reference = np.mean(reference_scores, axis=0)
    reference_sd = np.std(reference_scores, axis=0, ddof=1)
    n = len(policy)
    scale = float(np.sqrt(np.mean(reference[:n]**2)))
    reference_se_rms = float(np.sqrt(np.mean((reference_sd[:n]/np.sqrt(len(reference_scores)))**2)))
    rows = []
    for i, L in enumerate(Ls):
        errors = scores[:, i, :n]-reference[None, :n]
        rmse = np.sqrt(np.mean(errors**2, axis=1))
        rows.append(dict(L=int(L), policy_RMSE_mean=float(rmse.mean()),
                         policy_RMSE_max=float(rmse.max()),
                         policy_relative_RMSE_mean=float(rmse.mean()/scale),
                         policy_relative_RMSE_max=float(rmse.max()/scale),
                         policy_repeat_SD_RMS=float(np.sqrt(np.mean(scores[:, i, :n].std(0, ddof=1)**2))),
                         target_probe_max_abs_bias=float(np.max(np.abs(scores[:, i, -9:].mean(0)-reference[-9:])))))
    summary = dict(name=name, reference_label='mean of 4 independent IID banks, each L=2^24; not exact',
                   reference_policy_score_RMS=scale, reference_policy_SE_RMS=reference_se_rms,
                   fixed_quantile_actions=quantiles.tolist(), precision=precision, results=rows)
    write(out/'SUMMARY.json', summary)
    assert sha(cp) == digest
    assert hashlib.sha256(flax.serialization.to_bytes(exp.state.params)).hexdigest() == actor_hash
    write(out/'COMPLETE.json', dict(time=time.time(), seconds=time.time()-meta['started'],
                                   analysis_commit=meta['analysis_commit'], checkpoint_sha256=digest))
    write(out/'STATUS.json', dict(phase='complete', time=time.time()))


if __name__ == '__main__':
    main()
