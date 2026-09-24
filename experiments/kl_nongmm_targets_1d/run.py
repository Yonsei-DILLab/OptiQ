import argparse,json
from pathlib import Path
import jax
from ..kl_diverse_targets_1d.run import train
from .core import Experiment
from .evaluate import evaluate

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--stage',choices=['screen','validate_seeds'],default='screen');args=p.parse_args()
 cfg=json.loads(Path(__file__).with_name('config.json').read_text());assert not cfg['allow_large_L'] and cfg['reverse_L']==1024 and jax.default_backend()=='gpu'
 seeds=cfg['seeds'] if args.stage=='screen' else cfg['validation_seeds'];case=cfg['cases'][args.index//len(seeds)];seed=seeds[args.index%len(seeds)]
 for method in ['forward','reverse']:
  train(args.root,cfg,case,seed,method,100000,1024 if method=='reverse' else 0,args.stage,experiment_cls=Experiment,evaluate_fn=evaluate)
if __name__=='__main__':main()
