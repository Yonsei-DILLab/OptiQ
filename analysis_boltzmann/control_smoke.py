import argparse
from pathlib import Path
import jax
from .control import run
from .learned_backup import run as learned
from .frozen import run as frozen
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--checkpoint',required=True); a=p.parse_args(); root=Path(a.out)
    for backup in ('max','boltzmann','actor','local'):
        run(argparse.Namespace(out=str(root/backup),backup=backup,seed=0,updates=2,grid=17)); jax.clear_caches()
    learned(argparse.Namespace(out=str(root/'learned'),checkpoint=a.checkpoint,seed=0,states=2,repetitions=8,ks=[1,50]))
    frozen(argparse.Namespace(out=str(root/'coverage'),case='modes2d_4',seed=0,initialization='coverage',updates=2,repetitions=8,ks=[1,50]))
    (root/'COMPLETE').write_text('control and learned-Q smoke passed\n')
