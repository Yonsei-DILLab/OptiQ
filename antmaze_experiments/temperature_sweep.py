"""Pure, host-independent manifest for the requested 27-run dense sweep."""
import argparse
import json
from pathlib import Path
import subprocess

from .settings import total_budget, WANDB_ENTITY, WANDB_PROJECT

CAMPAIGN = 'antmaze-dense-T10-T5-T3-3seed-20260924'


def campaign_manifest(source, sha):
    jobs = []
    # Rotate mazes within each temperature/seed to compare settings early.
    for seed in range(3):
        for temperature in (10., 5., 3.):
            for task in ('v1', 'v3', 'v4'):
                jobs.append(dict(
                    id=f'{task}-optiq-T{temperature:g}-s{seed}', method='optiq',
                    task=task, seed=seed, temperature=temperature,
                    reward_profile='dense', noveld='off', eval_starts='random',
                    steps=total_budget(task), interim_eval_episodes=40,
                    final_eval_episodes=100, save_intermediate_policy=True,
                    optiq_profile=dict(actor_hidden_dims=[256, 256, 256],
                                       critic_hidden_dims=[256, 256, 256],
                                       actor_lr=3e-4, critic_lr=5e-4)))
    return dict(campaign=CAMPAIGN, source=str(source), source_commit=sha,
                jobs=jobs, wandb_entity=WANDB_ENTITY, wandb_project=WANDB_PROJECT,
                num_envs=256, batch_size=4096, updates_per_vector_step=8,
                protocol='antmaze_experiments/TEMPERATURE_SWEEP_PROTOCOL.md',
                fresh_training=True, preserve_existing_jobs=True,
                automatic_restart=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source,
                                  text=True).strip()
    print(json.dumps(campaign_manifest(args.source, sha), indent=2))


if __name__ == '__main__':
    main()
