"""Actual action histograms at saved optimizer steps, with unchanged paper styling."""
import argparse, csv, json, hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .prepare import CASES, original_folder
from ..kl_diverse_targets_1d.target import reference as gm_ref
from ..kl_nongmm_targets_1d.target import reference as ng_ref

BLUE,ORANGE='#1f77b4','#e67e22'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def draw(data,case,steps,out,wide=False):
    nr,nc=(2,len(steps)) if wide else (len(steps),2)
    pw,ph,gx,gy=3.4,2.35,.60,.42
    left,right,bottom,top=1.4,.12,.65,.6
    width=left+nc*pw+(nc-1)*gx+right;height=bottom+nr*ph+(nr-1)*gy+top
    fig=plt.figure(figsize=(width,height))
    for r in range(nr):
        for c in range(nc):
            step=steps[c if wide else r];method=['forward','reverse'][r if wide else c]
            d=data[case,method,step];color=BLUE if method=='forward' else ORANGE
            y=bottom+(nr-r-1)*(ph+gy);x=left+c*(pw+gx)
            ax=fig.add_axes([x/width,y/height,pw/width,ph/height])
            for v in d['values']:ax.stairs(v,d['edges'],color=color,alpha=.18,lw=.6)
            mean=d['values'].mean(0)
            ax.stairs(mean,d['edges'],color=color,fill=True,alpha=.20,lw=0)
            ax.stairs(mean,d['edges'],color=color,lw=1.55)
            ax.plot(d['x'],d['target'],'k--',lw=1.45)
            ymax=max(max(float(data[case,method,s]['values'].max()),
                         float(data[case,method,s]['target'].max())) for s in steps)
            ax.set(xlim=(-10,10),ylim=(0,1.05*ymax));ax.set_xticks([-10,-5,0,5,10])
            ax.tick_params(direction='out',length=4,width=.8)
            if r==nr-1:ax.set_xlabel(r'$a$')
            if r==0:ax.set_title(f'{step//1000}K' if wide else method.capitalize()+' KL',pad=10)
            if c==0:
                ax.set_ylabel('Density')
                fig.text(.06/width,(y+ph/2)/height,method.capitalize()+' KL' if wide else f'{step//1000}K',
                    va='center',ha='left',rotation=90 if wide else 0,fontsize=16)
    name=case+('_wide' if wide else '')
    for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=250,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workspace',type=Path);p.add_argument('--original',action='store_true')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    steps=[10000,25000,50000,75000,100000] if a.original else list(range(10000,100001,10000))
    tasks=read(a.root/'data/TASKS.json');data={};inputs=[];metrics=[]
    for task in tasks:
        case,method,seed=task['case'],task['method'],task['seed'];cfg=task['config']
        ref=ng_ref if cfg['target_kind']=='nongmm' else gm_ref;x=np.linspace(-10,10,16385)
        for step in steps:
            folder=original_folder(a.workspace,case,method,seed,step) if a.original else a.root/'runtime/replay'/case/f'{method}_s{seed}'
            metric=read(folder/f'metrics_{step:06d}.json');file=folder/f'samples_{step:06d}.npz'
            with np.load(file) as z:
                edges,mass=z['edges'].copy(),z['histogram_mass'].copy();count=len(z['actions'])
                assert len(mass)==512 and count==metric['sample_count']
                assert np.allclose(np.histogram(z['actions'],edges)[0]/count,mass)
                assert np.isclose(.5*np.abs(mass-z['target_mass']).sum(),metric['histogram_TV'])
                if not a.original:assert count==2**20
            key=case,method,step
            if key not in data:data[key]=dict(values=[],edges=edges,x=x,target=ref(cfg,x)[0])
            data[key]['values'].append(mass/np.diff(edges))
            inputs.append(dict(case=case,method=method,seed=seed,step=step,file=str(file),sha256=sha(file),count=count))
            metrics.append(dict(case=case,method=method,seed=seed,step=step,samples=count,TV=metric['histogram_TV']))
    for d in data.values():d['values']=np.asarray(d['values'])
    a.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'mathtext.fontset':'dejavusans','font.size':12,'axes.titlesize':18,'axes.labelsize':16,
        'xtick.labelsize':11,'ytick.labelsize':11,'axes.edgecolor':'#666666','axes.linewidth':.8,
        'pdf.fonttype':42,'svg.fonttype':'none'})
    for case in CASES:
        draw(data,case,steps,a.out)
        if not a.original:draw(data,case,steps,a.out,wide=True)
    with (a.out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    provenance=dict(commit=a.commit,original_saved=a.original,steps=steps,reverse_L=2**20,
        seeds=[0,1,2,3],inputs=inputs,reconstructed_trajectory=not a.original)
    (a.out/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    note=('Original saved times only. Missing 10K multiples are NOT interpolated. Intermediate counts 32768/seed; final counts 1048576/seed.'
          if a.original else 'Reconstructed runs with retained 10K checkpoints. Every time point uses 1048576 real actions/seed. Original and replayed trajectories are not spliced.')
    (a.out/'README.md').write_text('# Five-environment training progression\n\n'+note+'\n\n'
        'N=M=128, batch=32, reverse L=2^20; seeds 0–3. The bold colored curve is the mean of four 512-bin histograms. Thin curves show individual seeds. No KDE or interpolation. Black dashed: exact target. Y limits are fixed over time within each environment and method.\n')
    html='<html><meta charset="utf-8"><title>Five-target training progression</title><body style="font-family:sans-serif;max-width:1100px;margin:auto">'
    html+='<h1>Five-target training progression</h1><p>'+note+'</p>'
    for case in CASES:html+=f'<h2>{case}</h2><p><a href="{case}.pdf">Vector PDF</a></p><img style="width:100%" src="{case}.png">'
    (a.out/'report.html').write_text(html+'</body></html>')
    (a.out/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in a.out.iterdir() if p.is_file() and p.name!='SHA256.json'},indent=2)+'\n')
    print(a.out)
if __name__=='__main__':main()
