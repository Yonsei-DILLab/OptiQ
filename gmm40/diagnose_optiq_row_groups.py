"""Read-only decomposition of OT-row variance into nearest-component groups."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .evaluation import atomic_json
from .optiq import OptiQ
from .target import RESULTS, Target
from gmm40._v5.optiq_dime.semi_implicit import ConditionalGaussianProposal
from gmm40._v5.optiq_dime.transport import sinkhorn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--sinkhorn-iterations', type=int, default=1000)
    args = parser.parse_args()
    if args.sinkhorn_iterations <= 0:
        raise ValueError('sinkhorn iterations must be positive')
    before = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    cfg = json.loads((args.checkpoint.parent.parent/'config.json').read_text())
    target = Target()
    agent = OptiQ(target, hidden_dims=(cfg.get('width', 256),)*cfg.get('depth', 2))
    agent.restore(args.checkpoint)
    out = RESULTS/'diagnostics'/args.name
    out.mkdir(parents=True, exist_ok=False)
    key = jax.random.PRNGKey(87063)
    z = jax.random.normal(key, (64, 2))[:16]
    mu, ls = agent.actor.apply({'params': agent.state.params}, jnp.zeros((16, 1)), z)
    mu = jnp.broadcast_to(mu, (128, 16, 2))
    ls = jnp.broadcast_to(ls, mu.shape)
    proposal = ConditionalGaussianProposal(mu, ls, .05)
    actions, u, _ = proposal.sample(jax.random.fold_in(key, 64), 4, 'exact')
    q = target.jax_log_prob(40*actions)
    weights = jax.nn.softmax(q-proposal.log_prob(u)+2*math.log(40), axis=-1)
    costs = ((jnp.tanh(mu)[:, :, None]-actions[:, None])**2).sum(-1)
    distances = (((40*actions[:, :, None]-jnp.asarray(target.means)) /
                  jnp.asarray(target.std)[None, None, :, None])**2).sum(-1)
    groups = jax.nn.one_hot(distances.argmin(-1), 40)

    @jax.jit
    def decompose_float32(row):
        mean = jnp.einsum('bnm,bmd->bnd', row, u)
        second = jnp.einsum('bnm,bmd->bnd', row, u*u)
        variance = jnp.maximum(second-mean*mean, 0)
        mass = jnp.einsum('bnm,bmc->bnc', row, groups)
        grouped_mean = jnp.einsum('bnm,bmc,bmd->bncd', row, groups, u) / jnp.maximum(mass[..., None], 1e-30)
        grouped_second = jnp.einsum('bnm,bmc,bmd->bncd', row, groups, u*u) / jnp.maximum(mass[..., None], 1e-30)
        within = (mass[..., None]*jnp.maximum(grouped_second-grouped_mean**2, 0)).sum(-2)
        between = (mass[..., None]*(grouped_mean-mean[:, :, None])**2).sum(-2)
        return dict(within_nearest_component=within.mean(),
                    between_nearest_components=between.mean(),
                    total_within_row=variance.mean(),
                    across_teacher_clouds=mean.var(0).mean(),
                    squared_mean_bias=((mu[0]-mean.mean(0))**2).mean(),
                    identity_max_abs=jnp.abs(variance-within-between).max(),
                    row_largest_component_mass=mass.max(-1).mean())

    def decompose(row):
        # Use double-precision normalized probabilities for the diagnostic law
        # of total variance. Float32 moment subtraction can violate that identity
        # at large pre-tanh means, despite finite unchanged training tensors.
        probability = np.asarray(row, dtype=np.float64)
        normalization_error = float(np.abs(probability.sum(-1)-1).max())
        probability = probability/probability.sum(-1, keepdims=True)
        values = np.asarray(u, dtype=np.float64)
        labels = np.asarray(groups, dtype=np.float64)
        mean = np.einsum('bnm,bmd->bnd', probability, values)
        second = np.einsum('bnm,bmd->bnd', probability, values*values)
        variance = np.maximum(second-mean*mean, 0)
        mass = np.einsum('bnm,bmc->bnc', probability, labels)
        grouped_mean = np.einsum('bnm,bmc,bmd->bncd', probability, labels, values)/np.maximum(mass[...,None], 1e-30)
        grouped_second = np.einsum('bnm,bmc,bmd->bncd', probability, labels, values*values)/np.maximum(mass[...,None], 1e-30)
        within = (mass[...,None]*np.maximum(grouped_second-grouped_mean**2, 0)).sum(-2)
        between = (mass[...,None]*(grouped_mean-mean[:,:,None])**2).sum(-2)
        return dict(within_nearest_component=within.mean(), between_nearest_components=between.mean(),
                    total_within_row=variance.mean(), across_teacher_clouds=mean.var(0).mean(),
                    squared_mean_bias=((np.asarray(mu[0])-mean.mean(0))**2).mean(),
                    identity_max_abs=np.abs(variance-within-between).max(),
                    row_largest_component_mass=mass.max(-1).mean(),
                    row_sum_max_error_before_float64_normalization=normalization_error,
                    float32_identity_max_abs=float(decompose_float32(row)['identity_max_abs']))

    rows = []
    saved = dict(z=np.asarray(z), mu=np.asarray(mu), log_std=np.asarray(ls),
                 teacher_u=np.asarray(u), teacher_weights=np.asarray(weights),
                 nearest_component=np.asarray(distances.argmin(-1)))
    for epsilon in [.1, .03, .01]:
        plan = jax.jit(lambda c, w: sinkhorn(c, w, epsilon, args.sinkhorn_iterations))(costs, weights)
        row = plan/jnp.maximum(plan.sum(-1, keepdims=True), 1e-30)
        values = {k: float(v) for k, v in decompose(row).items()}
        assert all(np.isfinite(v) for v in values.values())
        assert values['identity_max_abs'] < 2e-5
        values.update(epsilon=epsilon, n=16, m=64, sinkhorn_iterations=args.sinkhorn_iterations,
                      actor_variance=float(jnp.exp(2*ls).mean()),
                      row_error_relative_max=float(16*jnp.abs(plan.sum(-1)-1/16).max()),
                      between_fraction_of_row=values['between_nearest_components']/values['total_within_row'])
        rows.append(values)
        saved[f'row_eps_{epsilon}'] = np.asarray(row)
        print(json.dumps(values), flush=True)
    np.savez_compressed(out/'supervision.npz', **saved)
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == before
    result = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=before,
                  checkpoint_unchanged=True, optimizer_updates_performed=0,
                  training_updates_of_checkpoint=agent.updates,
                  temperature=1, fixed_latents=16, teacher_clouds=128,
                  units='Per-coordinate pre-tanh variance',
                  diagnostic_precision='NumPy float64 moments after row renormalization; original float32 row and its normalization/identity errors are retained. Training tensors and checkpoints are unchanged.',
                  group_semantics='Nearest target-component centers; not true latent component labels. Labels are diagnostic only.',
                  backend=jax.default_backend(), rows=rows)
    atomic_json(out/'results.json', result)
    lines = ['# OT 행 분산의 성분별 분해', '',
             '저장된 actor를 읽기만 한 진단이다. 정책·optimizer 업데이트는 수행하지 않았다.',
             '같은 16개 latent에서 teacher 64개를 128회 새로 표집했다. T=1이다.',
             'Teacher를 가장 가까운 정답 성분 중심으로 분류하되, 이 분류는 학습에 사용하지 않는다.',
             '분산은 pre-tanh 좌표의 좌표당 값이다. 정답 GMM의 실제 latent 성분을 추정한 것이 아니라',
             '최근접 중심 그룹에 대한 정확한 분산 분해이므로 성분이 겹치는 구간에서는 해석에 주의한다.', '',
             '| ε | 같은 중심 그룹 내부 | 서로 다른 그룹 평균 사이 | 행 내부 합계 | 그룹 간 비중 | Teacher cloud 간 평균 변동 | μ bias² |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['epsilon']} | {r['within_nearest_component']:.5f} | {r['between_nearest_components']:.5f} | {r['total_within_row']:.5f} | {r['between_fraction_of_row']:.1%} | {r['across_teacher_clouds']:.5f} | {r['squared_mean_bias']:.5f} |")
    lines.extend(['', 'NLL 분산 목표는 행 내부 합계 + cloud 간 평균 변동 + 현재 μ의 bias²다.',
                  '이 분해는 현재 체크포인트의 국소 지도 신호를 설명하며 새 하이퍼파라미터의 최종 성능을 예측하지 않는다.',
                  '더 작은 ε에서의 유한 Sinkhorn 오차도 JSON에 보존했다.', '',
                  '[원자료](results.json) · [표본·가중치·row 배열](supervision.npz)'])
    (out/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
