"""Finite ten-run campaign, atomic claims; failures recorded, never auto retried."""
import json
import os
from pathlib import Path
import subprocess

root=Path('/home/heechan/optiq-experiments/antmaze-v1-stepcost-entropy-250k-20260925')
root.mkdir(exist_ok=True,parents=True)
for temperature in (0.5,1,3,5,10):
    for coefficient in (0.,.1):
        name=f'T{temperature:g}-xyH{coefficient:g}-s0'
        job=root/name
        try: job.mkdir()
        except FileExistsError: continue
        env=dict(os.environ,OPTIQ_CAMPAIGN='antmaze-v1-stepcost-entropy-250k-20260925',
                 OPTIQ_WANDB_PROJECT='jaehun-antmaze')
        args=['bash','antmaze_experiments/launch.sh','--method','optiq','--task','v1',
              '--temperature',str(temperature),'--budget-steps','250000',
              '--reward-profile','progress100_euclidean_no_bonus','--noveld','off','--dacer','off',
              '--eval-starts','fixed','--collection-profile','single-update1',
              '--optiq-config-profile','basic','--eval-interval','5000',
              '--interim-eval-episodes','100','--final-eval-episodes','100',
              '--save-intermediate-policy','--center-traces-only',
              '--xy-entropy-coefficient',str(coefficient)]
        (job/'launch.json').write_text(json.dumps(dict(args=args,gpu=os.environ['CAMPAIGN_GPU'],
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2))
        with (job/'stdout.log').open('w') as out, (job/'stderr.log').open('w') as err:
            code=subprocess.call(args+['--preflight','--output',str(job/'preflight')],env=env,stdout=out,stderr=err)
            if code==0:
                code=subprocess.call(args+['--output',str(job/'run')],env=env,stdout=out,stderr=err)
        (job/'exit.json').write_text(json.dumps(dict(exit_code=code)))
