"""Render the user's first three selected targets from completed L1024 runs."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..kl_diverse_targets_1d.target import reference as gmref
from .target import reference as ngref

CASES=[('t00_reference','Three Gaussian modes','20260925_kl_diverse_targets',gmref),
       ('n00_spike_ramp','Narrow spike + rising shoulder','20260925_kl_nongmm_targets',ngref),
       ('n07_spike_flat_ramp','Spike + plateau + ramp','20260925_kl_nongmm_targets',ngref)]

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
 out=a.workspace/'reports/20260925_kl_three_targets';out.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(3,2,figsize=(11.4,8.6))
 provenance={'stage':'completed L1024 screening; not high-L confirmation','updates':100000,'N':128,'M':128,'batch':32,'reverse_L':1024,'seeds':[0,1,2,3],'histogram_bins':512,'evaluation_samples_per_seed':262144,'runs':[]}
 x=np.linspace(-10,10,16385)
 for row,(ident,title,study,ref) in enumerate(CASES):
  root=a.workspace/'studies'/study;cfg=json.loads((root/'config.json').read_text());case=next(c for c in cfg['cases'] if c['id']==ident);target=ref(dict(cfg,**case),x)[0]
  for col,method in enumerate(['forward','reverse']):
   ax=axs[row,col];ys=[];tvs=[]
   for seed in range(4):
    stage='screen' if seed<2 else 'validate_seeds';run=root/'runtime'/stage/ident/f'{method}_s{seed}'
    assert json.loads((run/'COMPLETE.json').read_text())['step']==100000
    metric=json.loads((run/'metrics_100000.json').read_text());assert metric['sample_count']==262144
    with np.load(run/'samples_100000.npz') as z:
     edges=z['edges'];mass=z['histogram_mass'];assert len(mass)==512 and abs(mass.sum()-1)<1e-6
     ys.append(mass/np.diff(edges))
     assert abs(.5*np.abs(mass-z['target_mass']).sum()-metric['histogram_TV'])<1e-8
    tvs.append(metric['histogram_TV'])
    provenance['runs'].append({'case':ident,'method':method,'seed':seed,'folder':str(run),'sample_file_sha256':hashlib.sha256((run/'samples_100000.npz').read_bytes()).hexdigest(),'metric':metric})
   color=['#1f77b4','#e67e22'][col]
   for y in ys:ax.stairs(y,edges,color=color,alpha=.22,lw=.6)
   ax.stairs(np.mean(ys,axis=0),edges,fill=True,alpha=.20,color=color,lw=0)
   ax.stairs(np.mean(ys,axis=0),edges,color=color,lw=1.5,label='Mean of 4 seeds')
   ax.plot(x,target,'k--',lw=1.3,label='True target')
   ax.set(xlim=(-10,10),ylim=(0,None),xlabel='Action',ylabel='Density')
   ax.set_xticks([-10,-5,0,5,10])
   ax.set_title(f'{title} | {method.capitalize()} KL\nMean seed TV = {np.mean(tvs):.3f}')
   if row==0:ax.legend(loc='upper right',fontsize=8.5,frameon=False)
 fig.tight_layout(rect=(0,.035,1,1),h_pad=1.65,w_pad=2.5)
 fig.text(.5,.012,r'$N=M=128$ | 100K updates | Reverse $L=2^{10}$ | 512-bin sample histograms',ha='center',fontsize=9,color='#444444')
 for ext in ['png','pdf','svg']:fig.savefig(out/f'forward_vs_reverse_three.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
 (out/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
 (out/'caption.md').write_text('''# Forward KL vs. Reverse KL — three selected targets

Completed screening results: 100K updates, N=M=128, batch 32, Reverse L=2^10; seeds 0–3. These are not the L=2^20 confirmation results.

Black dashed lines show the exact target density. Blue (Forward) and orange (Reverse) curves show the mean of four 512-bin action histograms, with 262,144 sampled actions per seed and no KDE smoothing. Light fills show the learned density; thin curves show individual seeds. Mean seed TV is the mean of the four per-seed TV errors, not the TV of the averaged density. Each subplot has its own density scale.

The three targets are the first three previously selected illustrative examples: three Gaussian modes; a narrow non-Gaussian spike plus a rising shoulder; and a spike, plateau, and ramp. They were selected during screening to illustrate the observed difference and do not establish universal superiority across targets.
''')
 print(out)
if __name__=='__main__':main()
