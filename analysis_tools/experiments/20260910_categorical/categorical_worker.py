import argparse,json,os
from pathlib import Path
from .shared import config

def configure_categorical(seed):
    # lru_cache distinguishes config() and config(0). The production wrapper
    # uses the no-argument call, while manifests/initializers use config(seed).
    for cfg in (config(),config(0),config(seed)):
        cfg.alg.actor.transport_target_mode='categorical'
    assert config().alg.actor.transport_target_mode=='categorical'
    assert config(seed).alg.actor.transport_target_mode=='categorical'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);a=ap.parse_args()
    p=Path(a.campaign);task=json.loads((p/'tasks.json').read_text())[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    out=Path(task['out'])
    if (out/'COMPLETE').exists():return
    assert not out.exists(),f'Partial run must be reviewed before retry: {out}'
    configure_categorical(task['seed'])
    print(json.dumps({'actual_update_target_mode':config().alg.actor.transport_target_mode,
                      'manifest_target_mode':config(task['seed']).alg.actor.transport_target_mode,
                      'seed':task['seed'],'case':task['case']}),flush=True)
    from .frozen import run
    args=argparse.Namespace(case=task['case'],seed=task['seed'],initialization=task['initialization'],
        initial_checkpoint=task['initial_checkpoint'],updates=20000,repetitions=2000,
        ks=[1,8,16,50,64,256,1024],out=str(out),variant='categorical')
    run(args)
    from .checkpoint_integrity import verify_initialization
    verify_initialization(task,mark=True)

if __name__=='__main__':main()
