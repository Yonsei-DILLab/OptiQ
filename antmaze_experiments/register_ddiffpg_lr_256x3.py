"""Launch four fresh OptiQ AntMaze jobs with DDiffPG LRs and 256x3 networks."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .settings import total_budget, WANDB_ENTITY, WANDB_PROJECT

CAMPAIGN = 'antmaze-optiq-ddiffpg-lr-256x3-s0-20260923-r3'


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--host',choices=['vast-heechan-180'],required=True)
    args=parser.parse_args()
    source=Path(__file__).resolve().parents[1]
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert source==Path('/home/heechan/OptiQ-ops/sources')/sha
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip()
    root=Path('/home/heechan/optiq-experiments')/CAMPAIGN
    assert not root.exists(),root
    conf=Path('/home/heechan/OptiQ-ops/supervisor/jobs')/(CAMPAIGN+'.conf')
    assert not conf.exists(),conf
    jobs=[dict(id=f'{task}-optiq-s0',task=task,method='optiq',temperature=.01,
               reward_profile='sparse',noveld='on',eval_starts='upstream',
               steps=total_budget(task),interim_eval_episodes=40,final_eval_episodes=100,
               save_intermediate_policy=True,
               optiq_profile=dict(actor_hidden_dims=[256,256,256],critic_hidden_dims=[256,256,256],
                                  actor_lr=3e-4,critic_lr=5e-4)) for task in ('v1','v2','v3','v4')]
    manifest=dict(campaign=CAMPAIGN,source=str(source),source_commit=sha,
                  host=args.host,wandb_mode='online',wandb_entity=WANDB_ENTITY,
                  wandb_project=WANDB_PROJECT,jobs=jobs,seed=0,
                  num_envs=256,batch_size=4096,updates_per_vector_step=8,
                  reward_profile='sparse',noveld_enabled=True,noveld_coefficient=.01,
                  temperature=.01,log_std_min=-5.,log_std_max=-1.,initial_log_std=-1.,
                  protocol='antmaze_experiments/DDIFFPG_LR_256X3_PROTOCOL.md',
                  actor_hidden_dims=[256,256,256],critic_hidden_dims=[256,256,256],
                  actor_lr=3e-4,critic_lr=5e-4,rnd_lr=1e-4,tau=.005,
                  previous_campaign='antmaze-sparse-noveld-t001-optiq-s0-20260923',
                  changed_learning_fields=['critic_lr','actor_hidden_dims','critic_hidden_dims'],
                  no_additional_methods_or_seeds=True)
    root.mkdir()
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    conf.write_text(f'''[program:{CAMPAIGN}]
command=/home/heechan/.venv-ddiffpg-native/bin/python -m antmaze_experiments.controller --root {root}
directory={source}
environment=PYTHONDONTWRITEBYTECODE="1",WANDB_MODE="online",OPTIQ_CAMPAIGN="{CAMPAIGN}"
autostart=false
autorestart=false
startsecs=2
stopasgroup=true
killasgroup=true
stopwaitsecs=30
stdout_logfile={root}/controller.log
stderr_logfile={root}/controller.err
stdout_logfile_maxbytes=0
stderr_logfile_maxbytes=0
''')
    ctl=['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    for command in (['reread'],['update',CAMPAIGN],['start',CAMPAIGN]):
        subprocess.run(ctl+command,check=True)
    (root/'registration.json').write_text(json.dumps(dict(time=time.time(),supervisor=CAMPAIGN,
        manifest=manifest,config=str(conf)),indent=2)+'\n')
    print(json.dumps(dict(root=str(root),jobs=jobs,source_commit=sha)))


if __name__=='__main__':
    main()
