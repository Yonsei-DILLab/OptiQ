"""Six-timepoint density comparisons, with explicit early/late segment provenance."""
import argparse,csv,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ..kl_five_progress.figure import BLUE,ORANGE,gm_ref,ng_ref
from ..kl_five_progress.prepare import CASES

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def label(step):return 'Initialization' if step==0 else (f'{step//1000}K steps' if step>=1000 else f'{step} steps')

def load_hist(folder,step):
    file=folder/f'samples_{step:06d}.npz';metric=read(folder/f'metrics_{step:06d}.json')
    with np.load(file) as z:
        edges=z['edges'].copy();mass=z['histogram_mass'].copy();count=len(z['actions'])
        assert count==metric['sample_count']==2**20 and len(mass)==512
        assert np.allclose(np.histogram(z['actions'],edges)[0]/count,mass)
        probe={name:z[name].ravel()[:128].copy() for name in ['actions','mu','sigma']}
    return edges,mass,probe,metric,dict(file=str(file),sha256=sha(file),samples=count)

def draw(data,case,steps,out,wide):
    nr,nc=(2,len(steps)) if wide else (len(steps),2)
    pw,ph,gx,gy=3.4,2.35,.6,.42;left,right,bottom,top=(1.6 if wide else 2.0),.12,1.0,.6
    w=left+nc*pw+(nc-1)*gx+right;h=bottom+nr*ph+(nr-1)*gy+top
    fig=plt.figure(figsize=(w,h))
    for r in range(nr):
        for c in range(nc):
            step=steps[c if wide else r];method=['forward','reverse'][r if wide else c]
            d=data[case,method,step];color=BLUE if method=='forward' else ORANGE
            x=left+c*(pw+gx);y=bottom+(nr-r-1)*(ph+gy)
            ax=fig.add_axes([x/w,y/h,pw/w,ph/h])
            for values in d['values']:ax.stairs(values,d['edges'],color=color,lw=.6,alpha=.18)
            avg=np.mean(d['values'],axis=0)
            ax.stairs(avg,d['edges'],color=color,lw=0,fill=True,alpha=.20)
            ax.stairs(avg,d['edges'],color=color,lw=1.55)
            ax.plot(d['x'],d['target'],'k--',lw=1.45)
            ymax=max(max(float(np.max(data[case,method,s]['values'])),float(np.max(data[case,method,s]['target']))) for s in steps)
            ax.set(xlim=(-10,10),ylim=(0,1.05*ymax));ax.set_xticks([-10,-5,0,5,10])
            ax.tick_params(length=4,width=.8)
            if r==nr-1:ax.set_xlabel(r'$a$')
            if r==0:ax.set_title(label(step) if wide else method.capitalize()+' KL',pad=10)
            if c==0:
                ax.set_ylabel('Density')
                fig.text(.08/w,(y+ph/2)/h,method.capitalize()+' KL' if wide else label(step),
                    rotation=90 if wide else 0,ha='left',va='center',fontsize=14 if not wide else 16)
    if len(steps)==6:
        # Do not disguise separately replayed early observations as recovered
        # checkpoints from the uninterrupted long run.
        note='0–1K: independently replayed early segment; 10K/100K: ongoing 100K campaign. Four-seed means.'
        if wide:
            xx=(left+4*(pw+gx)-gx/2)/w
            fig.add_artist(plt.Line2D([xx,xx],[bottom/h,(h-top)/h],transform=fig.transFigure,color='.55',lw=.8,ls=':'))
        fig.text(.5,.025,note,ha='center',va='bottom',fontsize=10 if not wide else 12)
    name=case+('_2x6' if wide and len(steps)==6 else '_6x2' if len(steps)==6 else '_early_2x4')
    for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=250,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--early-only',action='store_true');a=p.parse_args()
    steps=[0,10,100,1000] if a.early_only else [0,10,100,1000,10000,100000]
    data={};inputs=[];comparisons=[];metrics=[]
    for task in read(a.root/'data/TASKS.json'):
        case,method,seed=task['case'],task['method'],task['seed'];cfg=task['config']
        early=a.root/'runtime/early'/case/f'{method}_s{seed}'
        late=a.parent/'runtime/replay'/case/f'{method}_s{seed}'
        run=read(early/'RUN.json');parent=read(late/'RUN.json')
        assert run['L']==parent['L']==(2**20 if method=='reverse' else 0)
        assert run['initial_parameter_sha256']==parent['initial_parameter_sha256']
        for step in steps:
            folder=early if step<=1000 else late
            edges,mass,_,metric,source=load_hist(folder,step)
            x=np.linspace(-10,10,16385);ref=ng_ref if cfg['target_kind']=='nongmm' else gm_ref
            key=case,method,step
            if key not in data:data[key]=dict(edges=edges,values=[],x=x,target=ref(cfg,x)[0])
            data[key]['values'].append(mass/np.diff(edges))
            inputs.append(dict(case=case,method=method,seed=seed,step=step,segment='early' if step<=1000 else 'main',**source))
            metrics.append(dict(case=case,method=method,seed=seed,step=step,TV=metric['histogram_TV'],samples=metric['sample_count']))
        # Cross-run agreement is measured at the overlap, never assumed.
        if (late/'metrics_001000.json').exists():
            _,em,ep,_,_=load_hist(early,1000);_,lm,lp,_,_=load_hist(late,1000)
            entry=dict(case=case,method=method,seed=seed,overlap_step=1000,
                histogram_TV_between_runs=float(.5*np.abs(em-lm).sum()))
            for name in ep:entry[name+'_max_abs_difference']=float(np.max(np.abs(ep[name]-lp[name])))
            entry['visible_disagreement_flag']=entry['histogram_TV_between_runs']>.02
            comparisons.append(entry)
        elif not a.early_only:raise FileNotFoundError(late/'metrics_001000.json')
    a.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'mathtext.fontset':'dejavusans',
        'font.size':12,'axes.titlesize':18,'axes.labelsize':16,'xtick.labelsize':11,'ytick.labelsize':11,
        'axes.edgecolor':'#666666','axes.linewidth':.8,'pdf.fonttype':42,'svg.fonttype':'none'})
    for case in CASES:
        draw(data,case,steps,a.out,True)
        if not a.early_only:draw(data,case,steps,a.out,False)
    for name,rows in [('metrics.csv',metrics),('overlap_1k.csv',comparisons)]:
        if rows:
            with (a.out/name).open('w') as f:
                wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    sha_commit=read(a.root/'SOURCE_MANIFEST.json')['commit']
    prov=dict(source_commit=sha_commit,steps=steps,early_only=a.early_only,seeds=[0,1,2,3],reverse_L=2**20,
        samples_per_seed=2**20,bins=512,inputs=inputs,overlap_comparisons=comparisons,
        note='Early segment is independently replayed, not a byte-identical segment recovered from the later run.')
    (a.out/'PROVENANCE.json').write_text(json.dumps(prov,indent=2)+'\n')
    note='0–1K는 동일 설정·초기화로 별도 재현한 구간입니다. 10K·100K는 기존 진행 중인 재현 실행의 결과입니다. 두 구간은 bitwise-identical한 단일 학습 궤적으로 주장하지 않습니다. 1K의 겹치는 시점에서 차이를 별도로 기록합니다.'
    text='# Initial → 10 → 100 → 1K → 10K → 100K\n\n'+note+'\n\n'
    text+='N=M=128, batch=32, reverse L=2^20; four seeds (0–3). Each histogram uses 2^20 real actions per seed, 512 bins, no KDE. Thick curves: four-seed means; thin curves: seeds; black dashed: exact target. Fixed y range over time within each method/environment.\n'
    (a.out/'README.md').write_text(text)
    html='<html><meta charset="utf-8"><title>Early-to-final density progression</title><body style="font-family:sans-serif;margin:24px">'
    html+='<h1>Initialization → 10 → 100 → 1K → 10K → 100K</h1><p>'+note+'</p>'
    if a.early_only:html+='<p><b>초기 1K까지의 미리보기입니다. 10K·100K를 포함한 버전은 본 실행 완료 뒤 생성됩니다.</b></p>'
    flagged=[x for x in comparisons if x['visible_disagreement_flag']]
    if flagged:html+=f'<p><b>주의: {len(flagged)}개 seed에서 두 1K 분포의 TV가 0.02를 넘습니다. 구간 사이 변화를 단일 궤적으로 해석하면 안 됩니다.</b></p>'
    for case in CASES:
        stem=case+('_early_2x4' if a.early_only else '_2x6')
        html+=f'<h2>{case}</h2><p><a href="{stem}.pdf">PDF</a> · <a href="{stem}.png">PNG</a></p><img style="width:100%" src="{stem}.png">'
    (a.out/'report.html').write_text(html+'</body></html>')
    (a.out/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in a.out.iterdir() if p.is_file() and p.name!='SHA256.json'},indent=2)+'\n')
    print(a.out)
if __name__=='__main__':main()
