"""Report every predeclared N/M condition and cutoff, with complete-case checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();src=Path(__file__).resolve().parent
    cfg=json.loads((src/'convergence.json').read_text());cases=json.loads((src/'phase_a.json').read_text())['cases']
    rows=[];graphs={}
    for ci,case in enumerate(cases):
        for n in cfg['n_values']:
            for m in cfg['m_values']:
                name=f'{case["env"]}_s{case["seed"]}_state{case["state"]}_n{n}_m{m}'
                paths=list(a.runs.glob(f'convergence-215d04b-gpu*/{name}/result.json'))
                if len(paths)!=1:raise ValueError(('Missing or duplicated pilot condition',name,paths))
                r=json.loads(paths[0].read_text());e=r['events'][-1]
                if not r['validation_passed'] or e['far'] or e['negative_pairs']:raise ValueError(('Failed condition',name))
                row=dict(case,n=n,m=m,source_commit=r['source_commit'],data_sha256=r['data_sha256'],samples_sha256=r['samples_sha256'],
                         modes=e['modes'],components=e['n_components'],counts=e['counts'],far=e['far'],negative_pairs=e['negative_pairs'],
                         maximum_mode_residual=e['max_mode_residual'],seconds=e['seconds'],
                         reused_endpoints=bool(r['events'][0].get('reused_endpoint_sha256')))
                rows.append(row)
                with np.load(paths[0].with_name('verified_graph.npz')) as f:
                    graphs[ci,n,m]={key:f[key].copy() for key in ('pairs','pers','roots','births')}
    a.out.mkdir(parents=True,exist_ok=False)
    summary=dict(scope='Six preselected states; one nested latent bank; not a five-seed aggregate.',
                 density_bank='N=32768 is the prefix of an N=65536 bank; differs from the original table bank.',
                 approximation='Verified sampled modes and k=15 midpoint graph saddles; no exhaustive continuous-mode guarantee.',
                 configs=cfg,results=rows)
    (a.out/'pilot_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=[summary['scope'],summary['density_bank'],summary['approximation'],'',
           'env seed state N M cutoff K(0.5) K(1) K(2) modes components seconds cached']
    for row in rows:
        for cutoff in cfg['cutoffs']:
            k=row['counts'][str(cutoff)]
            lines.append(f'{row["env"]} {row["seed"]} {row["state"]} {row["n"]} {row["m"]} {cutoff:g} '
                         f'{k["0.5"]} {k["1.0"]} {k["2.0"]} {row["modes"]} {row["components"]} {row["seconds"]:.3f} {row["reused_endpoints"]}')
    (a.out/'pilot_summary.txt').write_text('\n'.join(lines)+'\n')
    taus=np.unique(np.r_[np.linspace(.25,2.5,450),.5,1.,2.])
    colors={32768:'#2465a8',65536:'#cf651d'}
    for cutoff,filename,title in [(0.01,'curves_1pct.png','1% peak-height cutoff'),(0.,'curves_no_cutoff.png','No peak-height cutoff')]:
        fig,axes=plt.subplots(2,3,figsize=(14,7.7),sharex=True,layout='constrained')
        handles=[];labels=[]
        for ci,(ax,case) in enumerate(zip(axes.flat,cases)):
            for n in cfg['n_values']:
                for m in cfg['m_values']:
                    g=graphs[ci,n,m];floor=g['births'].max()+np.log(cutoff) if cutoff else -np.inf
                    valid=g['pairs'][:,0]>=floor
                    y=np.array([np.sum((g['pers']>=tau)&valid)+np.sum(g['births'][g['roots']]>=floor) for tau in taus])
                    h,=ax.step(taus,y,where='post',color=colors[n],linestyle='--' if m==32768 else '-',linewidth=1.9,
                              alpha=.9,label=f'N={n//1024}k, M={m//1024}k')
                    if ci==0:handles.append(h);labels.append(h.get_label())
            ax.set_title(f'{case["env"].capitalize()} | seed {case["seed"]}, state {case["state"]}',fontsize=11)
            ax.set_xlim(.25,2.5);ax.set_ylim(bottom=0);ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=.18);ax.spines[['right','top']].set_visible(False)
            for tau in (.5,1,2):ax.axvline(tau,color='#777777',linestyle=':',linewidth=.65,alpha=.5)
        fig.suptitle('iBOLT N/M sensitivity — '+title,fontsize=16)
        fig.supxlabel('Persistence threshold τ (log-density units)',fontsize=11)
        fig.supylabel('K(τ)',fontsize=12)
        fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
        fig.savefig(a.out/filename,dpi=160);plt.close(fig)
    print(json.dumps(dict(conditions=len(rows),all_far_zero=all(r['far']==0 for r in rows),
                          all_negative_zero=all(r['negative_pairs']==0 for r in rows),
                          max_components=max(r['components'] for r in rows))))
    for ci,case in enumerate(cases):
        selected=[r for r in rows if (r['env'],r['seed'],r['state'])==(case['env'],case['seed'],case['state'])]
        print(case['env'],case['seed'],case['state'],'1%',[r['counts']['0.01']['0.5'] for r in selected],
              'none',[r['counts']['0.0']['0.5'] for r in selected])


if __name__=='__main__':main()
