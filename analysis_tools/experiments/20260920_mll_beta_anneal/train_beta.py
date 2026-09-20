"""Direct MLL with beta(t)=clip(t/100000,0,1), clocked by env steps."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'20260920_direct_mll64'
sys.path.insert(0,str(BASE))
import train as base
from optiq_dime.algorithm import OptiQDIME

ORIGINAL_COMPOSE=base.compose_config
EXTRA=[
    'wandb.project=heejoon-direct-mll-beta-anneal',
    'wandb.job_type=direct-mll-beta-anneal',
    'run_name=${task}-directMLL-N64-M64-T025-beta0to1-100K-s${seed}',
    'wandb.group=${env_name}_directMLL_beta0to1_100K_T025',
    '+experiment.density_beta_schedule={initial:0.0,final:1.0,end_step:100000,clock:environment_steps}',
]


def beta_at_step(step):
    return min(1.0,max(0.0,float(step)/100000.0))


class BetaAnnealedMLL(OptiQDIME):
    def train(self,batch_size,gradient_steps):
        actor=self.cfg.alg.actor
        previous=actor.density_correction_beta
        beta=beta_at_step(self.num_timesteps)
        # density_beta is an OmegaConf interpolation to this field. Pass a
        # dynamic scalar to the unchanged jitted learner (not a static arg).
        actor.density_correction_beta=beta
        try:
            result=super().train(batch_size,gradient_steps)
        finally:
            actor.density_correction_beta=previous
        self.logger.record('train/density_beta',beta)
        return result


def compose_config(overrides=()):
    return ORIGINAL_COMPOSE(EXTRA+list(overrides))


def main():
    base.compose_config=compose_config
    base.runner.OptiQDIME=BetaAnnealedMLL
    base.main()


if __name__=='__main__':main()
