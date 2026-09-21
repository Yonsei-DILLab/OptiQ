import argparse
from pathlib import Path
import jax
from .movecar import run as movecar
from .landscape import run as landscape
from .plots import run as plots

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); args=ap.parse_args()
    root=Path(args.out)
    for method in ('optiq','ddpg','sd2','td3','sd3'):
        movecar(argparse.Namespace(out=str(root/method),method=method,seed=0,importance='no_is',steps=12,warmup=8,interval=12,value_horizon=32))
        jax.clear_caches()
    landscape(argparse.Namespace(out=str(root/'landscape'),method='optiq',seed=0,start=str(root/'optiq/checkpoint_9.msgpack'),end=str(root/'optiq/checkpoint_12.msgpack'),critics=None,directions=4))
    plots(root,root/'figures')
    (root/'COMPLETE').write_text('smoke passed\n')
