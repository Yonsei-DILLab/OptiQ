"""Four-seed GMM40 summaries and a portable final-results archive."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .campaign import read,write,verified

METRICS=[('mode_coverage','Covered components ↑'),('high_density_fraction','Within GT 3σ ↑'),
         ('mmd2','MMD² ↓'),('sliced_wasserstein2','Sliced Wasserstein ↓'),
         ('mode_mass_tv','Mode mass TV ↓'),('mean_log_target','Mean target log density ↑')]
LABELS=dict(optiq_trg='OptiQ Direct GMM/TRG',optiq='OptiQ v5 OT',sac='SAC',dipo='DIPO',meow='MEOW',mfpo='MFPO',sql='SQL (JAX SVGD)')


def build(root):
    manifest=read(root/'manifest.json');plan=manifest['plan'];jobs=manifest['jobs']
    if not all(verified(root,j) for j in jobs):raise RuntimeError('Every approved run must finish and pass update-count auditing')
    out=root/'summary';out.mkdir(exist_ok=True)
    methods=plan['methods'];seeds=plan['seeds'];runs={};rows=[];groups={}
    for job in jobs:
        folder=root/'results'/job['name']
        history=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()]
        by_step={r['step']:r for r in history};history=[by_step[s] for s in sorted(by_step)]
        latest=read(folder/'latest.json')
        assert latest['step']==plan['steps']
        runs[(job['method'],job['seed'])]=dict(folder=folder,history=history,latest=latest)
        rows.append(dict(method=job['method'],seed=job['seed'],updates=latest['step'],
            **{key:latest[key] for key,_ in METRICS},
            Q_evaluations=latest['training'].get('Q_evaluations'),
            train_seconds=latest['training'].get('train_seconds'),
            parameters=read(folder/'model_sizes.json')['total']))
    for method in methods:
        subset=[r for r in rows if r['method']==method]
        assert sorted(r['seed'] for r in subset)==sorted(seeds)
        groups[method]={key:dict(mean=float(np.mean([r[key] for r in subset])),
                                seed_std=float(np.std([r[key] for r in subset],ddof=1)),
                                per_seed=[r[key] for r in subset]) for key,_ in METRICS}
    write(out/'results.json',dict(source_commit=manifest['source_commit'],plan=plan,per_seed=rows,
                                 aggregate=groups,dispersion='Sample standard deviation across four training seeds, ddof=1'))
    with (out/'per_seed.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'summary.csv').open('w') as f:
        names=['method']+[key+suffix for key,_ in METRICS for suffix in ('_mean','_seed_std')]
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for method in methods:
            writer.writerow(dict(method=method,**{key+suffix:groups[method][key][field]
                for key,_ in METRICS for suffix,field in [('_mean','mean'),('_seed_std','seed_std')]}))
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    colors=plt.cm.tab10(np.linspace(0,.9,len(methods)))
    for ax,(key,label) in zip(axes.ravel(),METRICS):
        for color,method in zip(colors,methods):
            hs=[runs[(method,s)]['history'] for s in seeds]
            steps=[r['step'] for r in hs[0]]
            assert all([r['step'] for r in h]==steps for h in hs)
            data=np.array([[r[key] for r in h] for h in hs]);mean=data.mean(0);sd=data.std(0,ddof=1)
            ax.plot(steps,mean,color=color,label=LABELS[method])
            ax.fill_between(steps,mean-sd,mean+sd,color=color,alpha=.12)
        ax.set(xlabel='Actor updates',ylabel=label);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('GMM40 fixed-Q · T=1 · 4 training seeds · mean ± seed SD')
    for ext in ('png','pdf'):fig.savefig(out/f'learning_curves.{ext}',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,10),constrained_layout=True)
    for ax,(key,label) in zip(axes.ravel(),METRICS):
        ax.bar(range(len(methods)),[groups[m][key]['mean'] for m in methods],
               yerr=[groups[m][key]['seed_std'] for m in methods],capsize=3,color=colors)
        ax.set_xticks(range(len(methods)),[LABELS[m] for m in methods],rotation=25,ha='right',fontsize=8)
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Final 100k · mean ± seed SD (4 seeds per method)')
    for ext in ('png','pdf'):fig.savefig(out/f'final_metrics.{ext}',dpi=150)
    plt.close(fig)
    target=read(root/'results/target/definition.json');means=np.array(target['means']);std=np.array(target['std'])
    fig,axes=plt.subplots(len(seeds),len(methods),figsize=(5*len(methods),4.8*len(seeds)),squeeze=False,constrained_layout=True)
    theta=np.linspace(0,2*np.pi,80)
    for r,seed in enumerate(seeds):
        for c,method in enumerate(methods):
            ax=axes[r,c];run=runs[(method,seed)];latest=run['latest']
            samples=np.load(run['folder']/'evaluations'/f"step_{plan['steps']:07d}"/'samples.npy')
            assert len(samples)==plan['eval_samples'] and np.isfinite(samples).all()
            ax.scatter(*samples[:5000].T,s=1,alpha=.25,color='#dd7932')
            ax.scatter(*means.T,marker='+',color='black',s=15)
            for center,sigma in zip(means,std):ax.plot(center[0]+3*sigma*np.cos(theta),center[1]+3*sigma*np.sin(theta),color='gray',lw=.4,ls='--')
            ax.set(xlim=(-42,42),ylim=(-42,42),aspect='equal',
                title=f"{LABELS[method]} · seed {seed}\ncoverage {latest['mode_coverage']}/40 · near {latest['high_density_fraction']:.1%}")
    fig.suptitle('Final stochastic policy samples · first 5,000 shown, all 10,000 saved per run')
    for ext in ('png','pdf'):fig.savefig(out/f'final_distributions.{ext}',dpi=140)
    plt.close(fig)
    lines=['# GMM40 100k · 4-seed 비교','',f"소스: `{manifest['source_commit']}`",'',
           '고정 Q = 원본 GMM40 log density, T=1. 각 방법 seed 0~3, 100,000 actor updates, 평가당 10,000 samples.',
           '표의 ±는 학습 시드 간 표준편차(ddof=1)입니다. Target 샘플은 평가에서만 사용합니다.','',
           '| Method | Coverage /40 | Within 3σ | MMD² ↓ | SW ↓ | Mass TV ↓ |',
           '|---|---:|---:|---:|---:|---:|']
    for method in methods:
        cells=[LABELS[method]]
        for key in ('mode_coverage','high_density_fraction','mmd2','sliced_wasserstein2','mode_mass_tv'):
            metric=groups[method][key];cells.append(f"{metric['mean']:.5g} ± {metric['seed_std']:.3g}")
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', 'OptiQ는 현재 box-truncated Direct GMM NLL, N=M64, random latent, 256×2, log σ[-5,-1]입니다.',
              '다른 baseline은 기존 native architecture와 optimizer를 유지했습니다. 모든 방법의 batch는 256입니다.',
              '동일 update 수 비교이며 Q 질의량과 계산량은 다릅니다. `per_seed.csv`의 Q_evaluations, train_seconds, parameters를 함께 보세요.',
              'GMM component coverage는 density의 실제 local maxima 개수와 같지 않습니다.','',
              '![학습곡선](learning_curves.png)','![최종지표](final_metrics.png)','![최종분포](final_distributions.png)']
    (out/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    archive=root/'final-results.tar.gz';temp=root/'final-results.tar.gz.tmp'
    with tarfile.open(temp,'w:gz') as tar:
        for p in [root/'manifest.json',root/'preflight.json',out,root/'results/target']:
            tar.add(p,arcname=str(p.relative_to(root)))
        for job in jobs:
            folder=root/'results'/job['name'];final=folder/'evaluations'/f"step_{plan['steps']:07d}"
            for name in ['config.json','status.json','latest.json','metrics.jsonl','model_sizes.json','update_count_audit.json','wandb_status.json']:
                p=folder/name
                if p.exists():tar.add(p,arcname=str(p.relative_to(root)))
            for p in [final,folder/'checkpoints'/f"step_{plan['steps']:07d}.bin"]:
                tar.add(p,arcname=str(p.relative_to(root)))
    temp.replace(archive)
    digest=hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    write(root/'result.json',dict(status='completed',runs=len(jobs),source_commit=manifest['source_commit'],
           summary=str(out),archive=str(archive),archive_sha256=digest.hexdigest(),archive_bytes=archive.stat().st_size,
           all_intermediate_checkpoints_preserved_in=str(root/'results')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    build(parser.parse_args().root.resolve())
