"""Numerical identities and actual four-arm two-update control smoke."""
import argparse,json
from pathlib import Path
import numpy as np
import jax
from .control_precision import moments
from .control import run
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();p=Path(a.out);p.mkdir(parents=True,exist_ok=False)
    q=np.full((2,4096),197.25);assert np.array_equal(moments(q),[197.25]*2)
    x=-1+(np.arange(16384)+.5)*2/16384;true=1/np.tanh(4)-.25
    assert abs(moments(x[None,:])[0]-true)<1e-8
    assert abs(moments((x+200)[None,:])[0]-(true+200))<1e-8
    results=[]
    for arm in ['max','boltzmann','actor','local']:
        run(argparse.Namespace(out=str(p/arm),backup=arm,seed=0,updates=2,grid=2056));g=np.load(p/arm/'grid_convergence.npz');err=float(g['max_error']);assert err<1e-3
        results.append({'arm':arm,'grid_error':err,'reference_n':int(g['reference_n'])});jax.clear_caches()
    (p/'summary.json').write_text(json.dumps(results,indent=2));(p/'COMPLETE').write_text('analytic checks and all four control arms passed\n')
