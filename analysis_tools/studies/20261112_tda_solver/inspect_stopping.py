"""Read-only diagnosis of the stopping rule at stored legacy endpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from legacy_diagnose import legacy
from verified_solver import Mixture,ascend


def main():
    p=argparse.ArgumentParser()
    for name in ('inputs','runs','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--commit',required=True);a=p.parse_args()
    torch.set_num_threads(4);src=Path(__file__).resolve().parent
    cases=json.loads((src/'phase_a.json').read_text())['cases'];rows=[]
    L=legacy(src/'upstream_basin_ph.py','cpu',0)
    for case in cases:
        name=f'{case["env"]}_s{case["seed"]}_state{case["state"]}_n32768'
        old=next(a.runs.glob(f'validation-ca3f7e1-gpu*/{name}_legacy'))
        new=next(a.runs.glob(f'validation-final-gpu*/{name}_verified'))
        with np.load(a.inputs/(name+'.npz')) as f:mix=Mixture(f['mu'][0],f['log_std'][0])
        m32=L['Mix'](mix.mu.float(),mix.ls.float())
        xc=torch.from_numpy(np.load(old/'legacy_endpoints.npy'))
        delta32=[];delta64=[];grad=[]
        for chunk in xc.split(mix.chunk):
            step32=m32.step(chunk);step64,res=mix.em(chunk.double())
            delta32.append((step32-chunk).abs().amax(-1));delta64.append((step64-chunk.double()).abs().amax(-1));grad.append(res)
        d32=torch.cat(delta32).numpy();d64=torch.cat(delta64).numpy();g=torch.cat(grad).numpy()
        with np.load(old/'legacy_graph.npz') as f,np.load(new/'verified_graph.npz') as fnew:
            far=f['assignment_distance']>.01
            modes=fnew['modes'];labels=fnew['labels']
            true_distance=np.max(abs(xc.numpy()-modes[labels]),axis=1)
            indices=np.unique(np.r_[np.argsort(-f['endpoint_height_violation'])[:16],np.flatnonzero(far)[:16],np.arange(16)])
            continued,_,info=ascend(mix,xc[indices].double())
            dist=np.max(abs(continued.numpy()[:,None]-modes[None]),axis=-1);continued_label=dist.argmin(1)
            old_map=np.max(abs(f['modes'][:,None]-modes[None]),axis=-1).argmin(1)
            selected=dict(indices=indices.tolist(),convergence=info,
                          labels_changed_by_continuing=int(np.sum(continued_label!=old_map[f['labels'][indices]])),
                          continued_vs_full_fp64_labels=int(np.sum(continued_label!=labels[indices])),
                          max_continued_to_mode_distance=float(dist.min(1).max()))
        def stats(mask):
            if not np.any(mask):return None
            return dict(count=int(mask.sum()),step32_under_1e4=int(np.sum(d32[mask]<1e-4)),
                        step64_under_1e4=int(np.sum(d64[mask]<1e-4)),
                        scaled_gradient_quantiles=np.quantile(g[mask],[0,.5,.95,1]).tolist(),
                        remaining_distance_quantiles=np.quantile(true_distance[mask],[0,.5,.95,1]).tolist())
        row=dict(case=case,all=stats(np.ones(len(xc),bool)),far=stats(far),continued_selected=selected)
        rows.append(row);print(json.dumps(row),flush=True)
        a.out.write_text(json.dumps(dict(source_commit=a.commit,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
