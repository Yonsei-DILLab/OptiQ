"""Post-training evidence; raw GD endpoints remain immutable."""
import argparse,json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import root
from .core import load_config,cases,quadrature,loss,adaptive,initial_means


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=load_config();rows=[]
    for case in cases(cfg):
        run=a.root/'runs'/case['id'];cp=run/'COMPLETE.json'
        if not cp.exists():continue
        done=json.loads(cp.read_text());x,w=quadrature(case['centers'],cfg['sigma'],16385)
        fn=lambda mu:loss(mu,jnp.asarray(x),jnp.asarray(w),cfg['sigma'])
        vg=jax.jit(jax.value_and_grad(fn));hf=jax.jit(jax.hessian(fn))
        truth=float(fn(jnp.array(case['centers'])))
        starts,meta=initial_means(case,cfg)
        for idx,(mu,info) in enumerate(zip(done['means'],done['meta'])):
            mu=np.asarray(mu);v,g=vg(jnp.asarray(mu));h=np.asarray(hf(jnp.asarray(mu)))
            row=dict(case=case['id'],centers=case['centers'],**info,step=done['step'],commit=done['commit'],initial_means=starts[idx].tolist(),raw_means=mu.tolist(),raw_loss=float(v),raw_kl=float(v)-truth,raw_gradient_norm=float(np.linalg.norm(g)),raw_hessian_eigenvalues=np.linalg.eigvalsh(h).tolist(),strict_bad_minimum=False)
            if float(v)-truth>.01:
                # This does NOT update any training checkpoint. It estimates the
                # stationary point near the observed GD endpoint for diagnosis.
                sol=root(lambda m:np.asarray(vg(jnp.asarray(m))[1]),mu,jac=lambda m:np.asarray(hf(jnp.asarray(m))),tol=1e-11)
                m=sol.x if np.linalg.norm(sol.x-mu)<.25 else mu
                va,ga,ha,err=adaptive(m,case['centers'],cfg['sigma'])
                vf,gf=vg(jnp.asarray(m));hh=np.asarray(hf(jnp.asarray(m)))
                eig,vec=np.linalg.eigh(ha)
                disagreement=max(abs(va-float(vf)),np.max(np.abs(ga-gf)),np.max(np.abs(ha-hh)))
                strict=bool(np.linalg.norm(ga)<1e-9 and eig[0]>max(1e-7,100*disagreement) and va-truth>.01)
                rng=np.random.default_rng(20260925)
                directions=np.r_[vec.T,rng.normal(size=(24,3))];directions/=np.linalg.norm(directions,axis=1,keepdims=True)
                perturb=[]
                for radius in [.001,.01,.05]:
                    delta=np.asarray([float(fn(jnp.asarray(m+sgn*radius*d)))-float(vf) for d in directions for sgn in [-1,1]])
                    perturb.append(dict(radius=radius,min_loss_increase=float(delta.min()),max_loss_increase=float(delta.max())))
                row.update(diagnostic_means=m.tolist(),polishing_displacement=float(np.linalg.norm(m-mu)),root_solver_success=bool(sol.success),diagnostic_kl=va-truth,diagnostic_gradient_norm=float(np.linalg.norm(ga)),diagnostic_hessian_eigenvalues=eig.tolist(),quadrature_disagreement=float(disagreement),adaptive_error_estimate=err,perturbations=perturb,strict_bad_minimum=strict)
            rows.append(row)
        print(case['id'],[(r['seed'],round(r['raw_kl'],5),r['strict_bad_minimum']) for r in rows if r['case']==case['id'] and r['kind']=='bad_structure'],flush=True)
    result=dict(rows=rows,complete_cases=len(set(r['case'] for r in rows)),total_cases=len(cases(cfg)),config=cfg)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
