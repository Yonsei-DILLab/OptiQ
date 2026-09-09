"""Export independent direct g(z) draws for the paper's auxiliary evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import flax.serialization
import jax
import numpy as np
import wandb

from optiq_dime.runtime import load_environment, provenance
from .sampler import initialize, draw
from .train import write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=20260930)
    parser.add_argument('--batch-size', type=int, default=4096)
    args = parser.parse_args()
    config = json.loads((args.checkpoint.parent/'config.json').read_text())
    load_environment()
    args.output.mkdir(parents=True, exist_ok=False)
    record = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    record.update(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  parent_config=config, direct_generator_samples=True,
                  source='g(z) only, without proposal noise, reweighting, OT postprocessing or target samples')
    run = wandb.init(project=os.environ.get('WANDB_PROJECT','optiq_dime_no_anchor'),
        entity=os.environ.get('WANDB_ENTITY'),mode='online',dir=str(args.output),
        group='gmm40-idem-protocol',job_type='evaluation-sampling',
        name=f"gmm40-export-seed{config['seed']}-{args.count}",config=record)
    try:
        record['runtime'] = provenance()
        run.config.update({'runtime':record['runtime']})
        actor, _, _ = initialize(config['seed'],config['hidden_dims'],config['learning_rate'],config['coordinate_scale'])
        actor = flax.serialization.from_bytes(actor,args.checkpoint.read_bytes())
        started = time.monotonic()
        key = jax.random.PRNGKey(args.seed)
        batches = []
        for start in range(0,args.count,args.batch_size):
            key, draw_key = jax.random.split(key)
            batch = np.asarray(draw(actor,draw_key,args.batch_size,config['coordinate_scale'],config['unbounded_actions']))
            batches.append(batch[:min(args.batch_size,args.count-start)])
        samples = np.concatenate(batches,axis=0)
        if samples.shape != (args.count,2) or not np.isfinite(samples).all():
            raise FloatingPointError('Invalid generator samples')
        np.save(args.output/'samples.npy',samples)
        record.update(wandb_url=run.url,actor_update=int(actor.step),elapsed_seconds=time.monotonic()-started,
                      samples_sha256=hashlib.sha256((args.output/'samples.npy').read_bytes()).hexdigest())
        write_json(args.output/'summary.json',record)
        run.summary.update({'actor_update':int(actor.step),'exported_samples':len(samples)})
        artifact=wandb.Artifact('gmm40-direct-export-'+run.id,type='gmm40-samples')
        artifact.add_file(str(args.output/'samples.npy'));artifact.add_file(str(args.output/'summary.json'))
        run.log_artifact(artifact);run.finish()
        print(json.dumps({'samples':str(args.output/'samples.npy'),'wandb_url':record['wandb_url']}),flush=True)
    except BaseException:
        run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
