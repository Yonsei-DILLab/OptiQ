"""Submit the fixed-method seeds only after the actual 50K tuning gate passes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


COMMIT = 'a76b98ce124333030a904a3ef27f7ca3e85e3833'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args()
    root=args.campaign.resolve()
    gates=list(root.glob('outputs/hc-smem_aux-tuning-s0_*/learning_gate.json'))
    if len(gates)!=1 or not json.loads(gates[0].read_text())['passed']:
        raise SystemExit('No unique passed tuning gate; refusing confirmation submission')
    snapshot=root/'snapshots/a76b98c'
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=snapshot,text=True).strip()
    assert actual==COMMIT
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=snapshot,text=True).strip()
    diagnostics={}
    for name,cmd in [
        ('user',['whoami']),
        ('partitions',['sinfo','-o','%20P %8a %10l %6D %12T %G']),
        ('jobs',['squeue','-u','gsmin2024','-o','%.18i %.18P %.12q %.24j %.2t %.10M %.4D %R']),
    ]:
        diagnostics[name]=subprocess.check_output(cmd,text=True)
        print(diagnostics[name],flush=True)
    destination=snapshot/'analysis_tools/experiments/20260921_halfcheetah_smem_repair'
    manifest=root/'submissions.json'
    records=json.loads(manifest.read_text())
    seen={(x.get('method'),x.get('seed')) for x in records if x.get('phase')=='confirmation'}
    # Five independent single-GPU jobs; long confirmation runs use big_qos.
    for method,seed in [('smem_aux',1),('smem_aux',2),('direct',0),('direct',1),('direct',2)]:
        if (method,seed) in seen:
            continue
        cmd=['sbatch','--parsable','--partition=dell_rtx3090,base_suma_rtx3090,big_suma_rtx3090,suma_rtx4090',
             '--qos=big_qos','--exclude=cs-gpu-01,node24',f'--job-name=hc-{method}-s{seed}',
             'run.sbatch',method,str(seed),'1000000','confirmation',COMMIT,'false']
        job_id=subprocess.check_output(cmd,cwd=destination,text=True).strip().split(';')[0]
        record=dict(phase='confirmation',job_id=job_id,method=method,seed=seed,
                    env='HalfCheetah-v4',git_commit=COMMIT,source_snapshot=str(snapshot),
                    command=cmd,cwd=str(destination),qos='big_qos',gpus=1,cpus=4,
                    total_steps=1000000,output_root=str(root/'outputs'),
                    wandb_project='OptiQ/optiq-direct-gmm-trg-vs-smem-tr',
                    gate=str(gates[0]),submitted_utc=datetime.now(timezone.utc).isoformat(),
                    live_state=diagnostics)
        records.append(record)
        manifest.write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps({k:record[k] for k in ['job_id','method','seed']}),flush=True)


if __name__=='__main__':
    main()
