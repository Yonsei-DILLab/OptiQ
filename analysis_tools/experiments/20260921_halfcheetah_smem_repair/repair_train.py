"""Train a named, explicitly configured SMEM actor projection repair."""
import sys
import subprocess
import csv
import json
import wandb
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback

import train
from optiq_dime import smem_tr
from optiq_dime.projection_repair import distill_actor


class LearningGate(BaseCallback):
    """Stop the tuning seed if its recent evaluation remains near the failed run."""
    def __init__(self, output_root):
        super().__init__()
        self.output_root = Path(output_root)
        self.checked = False

    def _on_step(self):
        # Evaluation callbacks are before this one; at 50,001 the 50K CSV row
        # has been flushed and no callback ordering can drop the gate point.
        if self.checked or self.num_timesteps < 50_001:
            return True
        self.checked = True
        with (self.output_root/'logs/progress.csv').open() as stream:
            rows = [r for r in csv.DictReader(stream) if r.get('eval/mean_reward')]
        recent = [float(r['eval/mean_reward']) for r in rows[-3:]]
        mean = sum(recent)/len(recent) if recent else float('-inf')
        passed = len(recent) == 3 and mean >= 1000.
        (self.output_root/'learning_gate.json').write_text(json.dumps({
            'step':self.num_timesteps, 'recent_eval_returns':recent,
            'threshold':1000., 'passed':passed,
            'purpose':'tuning gate, not an unbiased performance estimate',
        },indent=2)+'\n')
        print(f'LEARNING_GATE passed={passed} recent_mean={mean}', flush=True)
        return passed


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('direct', 'smem_regression', 'smem_aux'):
        raise SystemExit('usage: repair_train.py {direct|smem_regression|smem_aux} [Hydra overrides]')
    method = sys.argv[1]
    is_direct = method == 'direct'
    objective = 'parameter_regression' if method == 'smem_regression' else 'em_auxiliary'
    label = {'direct':'Direct GMM + TRG', 'smem_regression':'SMEM + TR (backtracked regression)', 'smem_aux':'SMEM + TR (EM auxiliary)'}[method]
    overrides = [
        'benchmark=halfcheetah',
        f'++algorithm="{label}"',
        '++experiment_revision=projection_repair_v2',
        '++comparison_phase=pilot',
        '++learning_gate=false',
        f'++experiment.method={method}',
        'wandb.group=HalfCheetah_projection_repair_v2',
    ]
    if not is_direct:
        overrides += [
            f'++experiment.projection_objective={objective}',
            '++experiment.actor_backtracks=10', 'experiment.projection_steps=1',
            '++experiment.momentum_reset_on_rejection=true',
        ]
    cfg = train.compose_config(overrides+sys.argv[2:], 'direct' if is_direct else 'smem_tr')
    if not is_direct:
        # JIT closes over these constants once per fresh process. Record all
        # values in the resolved run config; never hot-change an active run.
        for name in tuple(smem_tr.DEFAULTS)+('projection_objective','actor_backtracks'):
            if name in cfg.experiment:
                smem_tr.DEFAULTS[name] = cfg.experiment[name]
        smem_tr.distill_actor = distill_actor
    original_provenance = train.runner.provenance

    def provenance():
        info = original_provenance()
        info['archived_source_manifest'] = info.pop('source_manifest', None)
        info.update(
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=train.REPO,text=True).strip(),
            git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=train.REPO,text=True).strip()),
            algorithm_source=str(train.HERE.relative_to(train.REPO)),
            experiment_revision='projection_repair_v2',
        )
        return info

    train.runner.provenance = provenance
    if cfg.learning_gate:
        original_create = train.runner.create_algorithm
        def create_with_gate(config):
            model, callbacks = original_create(config)
            callbacks.callbacks.append(LearningGate(config.output_root))
            return model, callbacks
        train.runner.create_algorithm = create_with_gate
    try:
        train.runner.initialize_and_run(cfg)
    except BaseException:
        wandb.finish(exit_code=1)
        raise
    else:
        wandb.finish(exit_code=0)


if __name__ == '__main__':
    main()
