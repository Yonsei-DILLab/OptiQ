"""Recompute DIPO losses from a frozen navigation checkpoint, without updates.

The native trainer does not return losses. These independent replay probes are
not the losses observed by the live optimizer and never change its RNG/state.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .evaluation import atomic_json
from .navigation import GMM40Navigation
from .navigation_agents import DIPOOnline
from .target import RESULTS


def diagnose(checkpoint, samples=4096, seed=93802):
    checkpoint = Path(checkpoint).resolve()
    folder = checkpoint.parent.parent
    config = json.loads((folder/'config.json').read_text())
    if config.get('method') != 'dipo' or config.get('Q') != 'learned, not oracle':
        raise ValueError('Expected a native DIPO navigation checkpoint')
    if samples < 1:
        raise ValueError('Probe sample count must be positive')
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    env = GMM40Navigation()
    wrapper = DIPOOnline(env, config['seed'], folder)
    agent = wrapper.agent
    for name in ('actor', 'actor_target', 'critic', 'critic_target'):
        module = getattr(agent, name)
        module.load_state_dict(saved[name])
        module.eval()
        if not all(torch.isfinite(parameter).all().item() for parameter in module.parameters()):
            raise FloatingPointError(f'Nonfinite saved parameters: {name}')
    memory, diffusion = saved['memory'], saved['diffusion_memory']
    valid = memory['capacity'] if memory['full'] else memory['idx']
    diffusion_valid = diffusion['capacity'] if diffusion['full'] else diffusion['idx']
    if valid == 0 or valid != diffusion_valid or memory['idx'] != diffusion['idx']:
        raise ValueError('Expected nonempty aligned original/diffusion replay buffers')
    indices = np.random.default_rng(seed).integers(valid, size=samples)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    predicted, targets, q_disagreement, actor_losses = [], [], [], []
    with torch.no_grad():
        for start in range(0, samples, 256):
            idx = indices[start:start+256]
            batch = {key: torch.as_tensor(memory[key][idx], device=wrapper.device)
                     for key in ('states', 'actions', 'rewards', 'next_states', 'masks')}
            if not np.array_equal(memory['states'][idx], diffusion['states'][idx]):
                raise ValueError('Replay state alignment mismatch')
            q1, q2 = agent.critic(batch['states'], batch['actions'])
            next_actions = agent.actor_target(batch['next_states'], eval=False)
            tq1, tq2 = agent.critic_target(batch['next_states'], next_actions)
            target = batch['rewards'] + batch['masks'] * torch.minimum(tq1, tq2)
            best = torch.as_tensor(diffusion['best_actions'][idx], device=wrapper.device)
            loss = agent.actor.loss(best, batch['states'])
            actor_losses.append((float(loss.cpu()), len(idx)))
            predicted.append(torch.cat([q1, q2], dim=-1).cpu().numpy())
            targets.append(target.cpu().numpy())
            q_disagreement.append((q1-q2).abs().cpu().numpy())
    predicted, target = np.concatenate(predicted), np.concatenate(targets)
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise FloatingPointError('Nonfinite checkpoint probe Q values')
    residual = predicted-target
    original = memory['actions'][indices]
    best = diffusion['best_actions'][indices]
    result = dict(
        checkpoint=str(checkpoint), checkpoint_sha256=before,
        diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        updates=int(saved['updates']), replay_size=int(valid), probe_samples=samples,
        probe_seed=seed, training_performed=False, live_process_modified=False,
        parameter_finiteness_passed=True,
        critic_loss_sum_of_twin_mse=float((residual**2).mean(0).sum()),
        twin_td_rmse=[float(value) for value in np.sqrt((residual**2).mean(0))],
        twin_td_bias=[float(value) for value in residual.mean(0)],
        mean_q=float(predicted.mean()), mean_td_target=float(target.mean()),
        twin_q_absolute_difference=float(np.concatenate(q_disagreement).mean()),
        actor_denoising_loss=sum(loss*n for loss,n in actor_losses)/samples,
        replay_reward_mean=float(memory['rewards'][indices].mean()),
        original_action_saturation_fraction=float((np.abs(original)>.99).mean()),
        refined_action_saturation_fraction=float((np.abs(best)>.99).mean()),
        refined_action_mean_l2_change=float(np.linalg.norm(best-original,axis=-1).mean()),
        caveat='Independent replay probe, not a live-training loss log or true-Q calibration. Target actor sampling and denoising randomness use a separate fixed seed. Masks already contain gamma. No action refinement or optimizer steps are performed.')
    if not np.isfinite(result['actor_denoising_loss']):
        raise FloatingPointError('Nonfinite denoising loss')
    after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError('Input checkpoint changed during probe')
    result['input_checkpoint_unchanged'] = True
    out = RESULTS/'diagnostics/navigation_dipo'/folder.name
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/f'{checkpoint.stem}.json', result)
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--samples', type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.checkpoint, args.samples), indent=2))


if __name__ == '__main__':
    main()
