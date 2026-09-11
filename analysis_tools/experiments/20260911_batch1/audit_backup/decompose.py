"""Exact mode-bin decomposition of existing actor backup bias; no fitting."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'boltzmann_analysis_work'))
from analysis_boltzmann.problems import make_problem


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True)
    parser.add_argument('--out',required=True);args=parser.parse_args()
    root=Path(args.data);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[];fig,axes=plt.subplots(2,2,figsize=(10,6))
    for case in ('separable_4','separable_8','modes2d_8'):
        p=make_problem(case);ref=np.load(root/'references'/(case+'.npz'))
        grid,mass=ref['grid'],ref['mass'];b=ref['mode_mass'];truth=float(ref['truth'])
        if p.separable:
            base=p.base();labels=base.labels(grid);q=base.q(grid)
            bin_mass=np.bincount(labels,weights=mass,minlength=2)
            conditional=np.bincount(labels,weights=mass*q,minlength=2)/bin_mass
            ids=np.arange(len(b));mu_b=np.zeros(len(b))
            for d in range(p.dim):mu_b+=conditional[(ids//(2**d))%2]
        else:
            labels=p.labels(grid);q=p.q(grid)
            mu_b=np.bincount(labels,weights=mass*q,minlength=len(b))/b
        np.testing.assert_allclose(b@mu_b,truth,atol=1e-10)
        for version in ('ver1','ver2'):
            folder=root/(case+'_'+version+'_seed0')
            samples=np.load(folder/'distribution_20000.npz')['samples'].astype(np.float64)
            values=p.q(samples);labels=p.labels(samples)
            counts=np.bincount(labels,minlength=len(b));a=counts/len(samples)
            mu_a=np.divide(np.bincount(labels,weights=values,minlength=len(b)),counts,
                           out=np.zeros(len(b)),where=counts>0)
            mode_term=float((a-b)@mu_b)
            within_term=float(a@(mu_a-mu_b))
            bias=float(values.mean()-truth)
            np.testing.assert_allclose(mode_term+within_term,bias,atol=1e-9)
            row=dict(case=case,version=version,seed=0,actor_mean=float(values.mean()),
                reference_mean=truth,bias=bias,mode_mass_term=mode_term,within_mode_term=within_term,
                within_mode_fraction=within_term/bias,mean_with_actor_modes_and_reference_conditionals=float(a@mu_b),
                mode_bin_tv=float(.5*abs(a-b).sum()),mean_q_se=float(values.std(ddof=1)/np.sqrt(len(values))))
            if p.separable:
                scalar_samples=samples.reshape(-1);scalar_labels=base.labels(scalar_samples[:,None]);moments=[]
                for mode in range(2):
                    ix=base.labels(grid)==mode;w=mass[ix]/bin_mass[mode];x=grid[ix,0]
                    mean=float(w@x);std=float(np.sqrt(w@((x-mean)**2)))
                    chosen=scalar_samples[scalar_labels==mode]
                    moments.append(dict(mode=mode,reference_mean=mean,reference_std=std,
                        actor_mean=float(chosen.mean()),actor_std=float(chosen.std()),
                        reference_valley_mass=float(mass[abs(grid[:,0])<.2].sum()),
                        actor_valley_mass=float((abs(scalar_samples)<.2).mean())))
                row['pooled_coordinate_conditionals']=moments
                i=0 if p.dim==4 else 1
                axes[i,0].hist(scalar_samples,bins=np.linspace(-1,1,101),density=True,histtype='step',label=version)
                axes[i,1].hist(values,bins=100,density=True,histtype='step',label=version)
            rows.append(row);print(json.dumps(row),flush=True)
        if p.separable:
            i=0 if p.dim==4 else 1
            dx=grid[1,0]-grid[0,0]
            axes[i,0].plot(grid[:,0],mass/dx,color='black',ls='--',label='Boltzmann T=1')
            axes[i,1].hist(p.q(ref['samples']),bins=100,density=True,histtype='step',color='black',ls='--',label='Boltzmann T=1')
            axes[i,0].set_title(str(p.dim)+'D: pooled coordinate density')
            axes[i,1].set_title(str(p.dim)+'D: distribution of Q(a)')
            axes[i,0].set_xlabel('Action coordinate');axes[i,1].set_xlabel('Q(a)')
            axes[i,0].legend();axes[i,1].legend()
    fig.suptitle('Seed 0, saved 20k actors; no retraining');fig.tight_layout()
    fig.savefig(out/'within_mode.png',dpi=170);fig.savefig(out/'within_mode.pdf')
    (out/'decomposition.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
