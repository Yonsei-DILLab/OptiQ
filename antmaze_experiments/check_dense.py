"""Regression checks for terminal C51 overflow and dense/NovelD-off semantics."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .numerics import (stable_projection, guarded_binary_cross_entropy,
                       guarded_optimizer_step, DisabledIntrinsic)


def probability_checks(device):
    from ddiffpg.utils.distl_util import projection
    torch.manual_seed(0)
    batch, atoms = 4096, 51
    p = torch.randn(batch, atoms, device=device).softmax(-1)
    support = torch.linspace(0, 5, atoms, device=device)
    reward, done = torch.full((batch, 1), 10., device=device), torch.ones(batch, 1, device=device)
    original = projection(p, reward, done, .99, 0, 5, atoms, support, device)
    # Never invoke the broken BCE on CUDA: its device assertion poisons the context.
    original_failure = None
    if device == 'cpu':
        try:
            F.binary_cross_entropy(p, original)
        except RuntimeError as error:
            original_failure = str(error)
        assert original_failure and float(original.max()) > 1
    cases = []
    for lower, upper in [(0., 5.), (-6000., 5.)]:
        support = torch.linspace(lower, upper, atoms, device=device)
        for terminal in [0., 1.]:
            for scale in [0., 10., 20., -60., -1e6]:
                reward = torch.full((batch, 1), scale, device=device)
                done.fill_(terminal)
                result = stable_projection(p, reward, done, .99, lower, upper, atoms, support)
                assert torch.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
                error = float((result.sum(-1)-1).abs().max())
                assert error < 2e-6
                expectation = ((reward.double()+(1-done.double())*.99*support.double()).clamp(lower, upper)
                               * (p.double()/p.double().sum(-1,keepdim=True))).sum(-1)
                actual = (result.double()*support.double()).sum(-1)
                assert torch.allclose(actual, expectation, atol=.002, rtol=1e-6)
                logits = torch.randn(batch, atoms, device=device, requires_grad=True)
                loss = guarded_binary_cross_entropy(logits.softmax(-1), result)
                loss.backward()
                assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
                cases.append(error)
    # Saturated softmax and rounded boundary inputs remain finite.
    x = torch.tensor([0., 1., -.5*torch.finfo(torch.float32).eps, 1.+torch.finfo(torch.float32).eps],
                     device=device, requires_grad=True)
    loss = guarded_binary_cross_entropy(x, torch.tensor([1., 0., 0., 1.], device=device))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    for invalid in [float('nan'), float('inf'), -.1, 1.1]:
        try:
            guarded_binary_cross_entropy(torch.tensor([.5],device=device),
                                         torch.tensor([invalid],device=device))
        except FloatingPointError:
            pass
        else:
            raise AssertionError(f'Invalid target {invalid} was hidden')
    # Large finite gradients are clipped; nonfinite gradients cannot alter weights.
    parameter = torch.nn.Parameter(torch.ones(8, device=device))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    guarded_optimizer_step(optimizer, parameter.sum()*1e8, 1.)
    assert float(parameter.grad.norm()) <= 1.00001
    before = parameter.detach().clone()
    hook = parameter.register_hook(lambda g: g * float('inf'))
    try:
        guarded_optimizer_step(optimizer, parameter.sum(), 1.)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Nonfinite gradients were accepted')
    hook.remove()
    assert torch.equal(before, parameter)
    disabled = DisabledIntrinsic()
    assert disabled.compute_reward(torch.ones(8,29,device=device)).eq(0).all()
    assert disabled.update(None) == (0.,0.) and disabled.update_step == 0
    return dict(device=device, original_max=float(original.max()),
        original_overflow_rows=int((original.max(-1).values > 1).sum()),
        original_cpu_bce_failure=original_failure, checked_cases=len(cases),
        max_mass_error=max(cases), finite_bce_and_gradients=True,
        nonfinite_gradient_rejected_before_update=True, intrinsic_zero_without_updates=True)


def environment_checks():
    from .envs import make_one
    results = {}
    for task in ('v1','v2','v3','v4'):
        sparse = make_one(task,123,reward_profile='sparse')
        dense = make_one(task,123,reward_profile='dense')
        try:
            assert np.array_equal(sparse.reset(),dense.reset())
            rng=np.random.default_rng(321)
            for _ in range(30):
                action=rng.uniform(-1,1,8)
                s,rs,ds,is_=sparse.step(action)
                d,rd,dd,id_=dense.step(action)
                assert np.array_equal(s,d) and ds == dd and is_.get('success',0)==id_.get('success',0)
                goals=np.asarray(dense.physics_env.target_goal).reshape(-1,2)
                assert np.isclose(rd,-np.linalg.norm(d[:2]-goals,axis=1).min(),atol=1e-5)
                assert id_['upstream_sparse_reward']==rs
                if ds: assert np.array_equal(sparse.reset(),dense.reset())
            # Arrival must still terminate, with dense distance and no goal bonus.
            state=dense.state();state['qpos'][:2]=goals[0];state['qvel'][:]=0
            dense.restore(state)
            obs,reward,done,info=dense.step(np.zeros(8))
            assert done and info['success'] > 0 and -.5 <= reward <= 0
            results[task]=dict(physics_reset_terminal_parity=True,
                dense_goal_reward=reward,sparse_goal_reward=info['upstream_sparse_reward'])
        finally:
            sparse.close();dense.close()
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'antmaze'))
    # Physics workers/tests are done before creating the CUDA context.
    result=dict(environments=environment_checks(),
                probabilities=[probability_checks('cpu'),probability_checks('cuda')])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
