"""Analysis/figures from a frozen report-input snapshot. No training imports."""
from pathlib import Path
import argparse,json,collections,csv,functools,math,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import logsumexp,ndtr

DIMS=[1,2,4,8]
EPS=[.0001,.0005,.001,.005,.01,.05,.1,.5,1.,5.,10.,50.]
METHODS=['gmm_learned','exact_learned','exact_fixed05','exact_fixed01','sinkhorn_fixed05','sinkhorn_fixed01','argmax_truncated']+['sinkhorn_e'+format(x,'g') for x in EPS]
LABEL={'gmm_learned':'Direct GMM','exact_learned':'Exact OT / learned sigma','exact_fixed05':'Exact OT / sigma=.5','exact_fixed01':'Exact OT / sigma=.1','sinkhorn_fixed05':'Sinkhorn / sigma=.5','sinkhorn_fixed01':'Sinkhorn / sigma=.1','argmax_truncated':'Legacy row-argmax'}
LABEL.update({'sinkhorn_e'+format(x,'g'):f'Sinkhorn eps={x:g}' for x in EPS})
ANCHORS=['gmm_learned','exact_learned','sinkhorn_e0.001','sinkhorn_e0.01','sinkhorn_e0.1','sinkhorn_e1','argmax_truncated']
COLORS=dict(zip(ANCHORS,['#dd5c37','#7952ac','#ad9140','#009877','#07899e','#a27b65','#29445b']))
GROUPS=[('double','mass'),('tri','mass'),('tri','split'),('tri','replay'),('tri','closed')]
GLABEL={('double','mass'):'Two modes: mass reversal',('tri','mass'):'Three modes: mass reversal',('tri','split'):'Three / six / three modes',('tri','replay'):'Shared learned-Q replay',('tri','closed'):'Independent actor-critic'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'savefig.facecolor':'white'})

def finite_mean(x):
 a=np.asarray(x,float);a=a[np.isfinite(a)];return float(a.mean()) if len(a) else float('nan')
def stable(x,y,limit):
 for i in range(len(x)-2):
  if np.all(np.isfinite(y[i:i+3])) and np.all(y[i:i+3]<=limit):return float(x[i])
 return float('nan')
def auc(x,y,a,b):
 good=np.isfinite(y);x,y=x[good],y[good]
 if len(x)<2 or x[0]>a or x[-1]<b:return float('nan')
 inside=(x>a)&(x<b);xx=np.r_[a,x[inside],b];yy=np.r_[np.interp(a,x,y),y[inside],np.interp(b,x,y)]
 return float(np.trapz(yy,xx)/(b-a))
def writecsv(path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

class Report:
 def __init__(self,src,out,n=16,m=64):
  self.src=Path(src);self.out=Path(out);self.fig=self.out/'figures';self.fig.mkdir(parents=True,exist_ok=True)
  inv=json.loads((self.src/'inventory.json').read_text());self.inv=inv;self.all=inv['tasks'];self.ts=[t for t in self.all if t['status']=='complete' and t['n']==n and t['m']==m and t['stage'] not in ['prefix','source']]
  self.lookup={(t['family'],t['stage'],t['dim'],t['method'],t['seed']):t for t in self.ts}
  self.saved=[];self.rows=[];self.events=[];self.aggs=[]
 def folder(self,t):return self.src/'runs'/t['name']
 @functools.lru_cache(None)
 def series(self,name):
  with np.load(self.src/'runs'/name/'series.npz') as z:return {k:z[k] for k in z.files}
 def density(self,t,step='final_independent'):
  path=self.folder(t)/'density'/(str(step)+'.npz')
  if str(step)=='020000' and t.get('parent'):path=self.src/'runs'/t['parent']/'density'/'020000.npz'
  if not path.exists():return None
  with np.load(path) as z:return {k:z[k] for k in z.files}
 def save(self,fig,name):
  fig.savefig(self.fig/(name+'.png'),dpi=155,bbox_inches='tight');plt.close(fig);self.saved.append(name+'.png');print('FIGURE',name,flush=True);return f'figures/{name}.png'
 def summarize(self):
  for t in self.ts:
   s=self.series(t['name']);x=s['step'];d=self.density(t);stats=json.loads((self.folder(t)/'stats.json').read_text());p=json.loads((self.folder(t)/'progress.json').read_text())
   r={k:t[k] for k in ['name','family','stage','dim','method','seed','host']};r.update(final_tv=float(d['first_axis_tv']),final_basin_tv=float(d['basin_mass_tv']),final_sliced_w1=float(d['sliced_w1']),final_backup_bias=float(d['backup_bias']),final_sigma=float(d['sigma_mean']),reference_reliable_fraction=float(np.mean(s.get('reference_reliable',np.ones(len(x))))),elapsed_seconds=p['seconds'],train_seconds=p['train_seconds'],**stats)
   reliable=s.get('reference_reliable',np.ones(len(x))).astype(bool)
   r['response_measure']=('Mean TV over updates1..1000 after each shift' if t['stage']=='mass' else 'Mean TV over updates20001..35000' if t['stage']=='split' else 'Mean TV over reliable saved evaluations')
   if t['stage']=='mass':
    pre=self.src/'runs'/t['parent']/'density'/'final_independent.npz'
    with np.load(pre) as z:r.update(initial_tv=float(z['first_axis_tv']),initial_basin_tv=float(z['basin_mass_tv']))
    ev=[]
    for boundary in [20000,25000,30000]:
     sel=(x>boundary)&(x<=boundary+5000);xx=x[sel]-boundary;tv=s['first_axis_tv'][sel];btv=s['basin_mass_tv'][sel]
     e=dict(name=t['name'],family=t['family'],dim=t['dim'],method=t['method'],seed=t['seed'],boundary=boundary,
      tv_auc100=auc(xx,tv,1,100),tv_auc1000=auc(xx,tv,1,1000),tv_auc5000=auc(xx,tv,1,5000),mass_auc1000=auc(xx,btv,1,1000),
      tv_reach01=stable(xx,tv,.1),mass_reach005=stable(xx,btv,.05))
     self.events.append(e);ev.append(e)
    r.update(tv_auc1000=finite_mean([e['tv_auc1000'] for e in ev]),mass_auc1000=finite_mean([e['mass_auc1000'] for e in ev]),tv_success_rate=float(np.mean([np.isfinite(e['tv_reach01']) for e in ev])),mass_success_rate=float(np.mean([np.isfinite(e['mass_reach005']) for e in ev])),mass_recovery_conditional=finite_mean([e['mass_reach005'] for e in ev]))
   elif t['stage']=='split':
    sel=x>20000;r['tv_auc1000']=auc(x[sel],s['first_axis_tv'][sel],20001,35000);r['mass_auc1000']=auc(x[sel],s['basin_mass_tv'][sel],20001,35000)
   else:
    yy=np.where(reliable,s['first_axis_tv'],np.nan);r['tv_auc1000']=finite_mean(yy);r['mass_auc1000']=finite_mean(np.where(reliable,s['basin_mass_tv'],np.nan))
   if 'stochastic_return_mean' in s:r['final_return']=float(s['stochastic_return_mean'][-1])
   self.rows.append(r)
  groups=collections.defaultdict(list)
  for r in self.rows:groups[(r['family'],r['stage'],r['dim'],r['method'])].append(r)
  for (family,stage,dim,method),rs in groups.items():
   a=dict(family=family,stage=stage,dim=dim,method=method,seeds=len(rs))
   keys=[k for k,v in rs[0].items() if isinstance(v,(float,int)) and k not in ['seed','dim']]
   for k in keys:a[k]=finite_mean([r.get(k,np.nan) for r in rs])
   a['tv_auc_sd']=float(np.std([r['tv_auc1000'] for r in rs],ddof=1)) if len(rs)>1 else float('nan')
   self.aggs.append(a)
  writecsv(self.out/'run_summary.csv',self.rows);writecsv(self.out/'shift_events.csv',self.events);writecsv(self.out/'method_summary.csv',self.aggs)
  (self.out/'findings.json').write_text(json.dumps(self.aggs,indent=2))
 def curves(self,ax,group,dim,key,methods=ANCHORS,xrange=None,reliable=False):
  for method in methods:
   ts=[self.lookup.get((*group,dim,method,s)) for s in range(4)];ts=[t for t in ts if t]
   ss=[self.series(t['name']) for t in ts]
   if not ss or any(key not in s for s in ss):continue
   common=sorted(set.intersection(*[set(s['step']) for s in ss]));xx=np.array(common);ys=[]
   for s in ss:
    ix=np.searchsorted(s['step'],xx);yy=s[key][ix].copy()
    if reliable:yy=np.where(s['reference_reliable'][ix],yy,np.nan)
    ys.append(yy)
   y=np.array(ys);valid=np.all(np.isfinite(y),axis=0)
   if not valid.any():continue
   mean=np.full(len(xx),np.nan);sd=mean.copy();mean[valid]=y[:,valid].mean(0);sd[valid]=y[:,valid].std(0,ddof=1) if len(y)>1 else 0
   ax.plot(xx/1000,mean,label=LABEL[method]+f' (n={len(ts)})',color=COLORS.get(method),lw=1.45)
   ax.fill_between(xx/1000,mean-sd,mean+sd,color=COLORS.get(method),alpha=.1,linewidth=0)
  ax.set_title(f'{dim}D')
  if xrange:ax.set_xlim(np.array(xrange)/1000)
 def tracking(self,group):
  fig,axes=plt.subplots(2,4,figsize=(18,8),sharex=True)
  for col,d in enumerate(DIMS):
   self.curves(axes[0,col],group,d,'first_axis_tv',xrange=(20000,35000) if group[1] in ['mass','split'] else (0,35000),reliable=group[1] in ['closed','replay'])
   key='stochastic_return_mean' if group[1]=='closed' else 'basin_mass_tv'
   self.curves(axes[1,col],group,d,key,reliable=group[1]=='replay')
   axes[0,col].set_ylim(0,1)
   if key!='stochastic_return_mean':axes[1,col].set_ylim(0,.85)
   for row in axes[:,col]:
    if group[1]=='mass':
     for x in [20,25,30]:row.axvline(x,c='gray',ls=':',lw=1)
    elif group[1]=='split':row.axvspan(20,22,color='gray',alpha=.08);row.axvspan(25,27,color='gray',alpha=.08)
  axes[0,0].set_ylabel('First-coordinate histogram TV');axes[1,0].set_ylabel('Return / 200 steps' if group[1]=='closed' else 'Basin-mass TV')
  fig.supxlabel('Actor updates (thousands)');handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8,bbox_to_anchor=(.5,-.06));fig.suptitle(GLABEL[group]+' | seed mean ± SD; matched evaluation steps');fig.tight_layout()
  self.save(fig,'tracking_'+'_'.join(group))
 def mass_flow(self,group):
  # Direct histogram basin probabilities from seed0, alongside exact changing mass.
  fig,axes=plt.subplots(1,4,figsize=(18,4),sharey=True)
  for ax,dim in zip(axes,DIMS):
   targetdrawn=False
   for method in ANCHORS:
    t=self.lookup.get((*group,dim,method,0))
    if not t:continue
    xs=[];ys=[];targets=[]
    for step in [20000,20001,20020,20200,21000,25000,25001,25020,25200,26000,30000,30001,30020,30200,35000]:
     z=self.density(t,f'{step:06d}')
     if z is None:continue
     xs.append(step/1000);ys.append(z['basin_mass'][0]);targets.append(z['target_basin_mass'][0])
    ax.plot(xs,ys,color=COLORS.get(method),label=LABEL[method],lw=1.5)
    if not targetdrawn:ax.plot(xs,targets,'k--',lw=2,label='Exact target left-basin mass');targetdrawn=True
   ax.set(title=f'{dim}D',xlabel='Actor updates (k)',ylim=(-.03,1.03),xlim=(20,35))
  axes[0].set_ylabel('Probability in leftmost basin');fig.suptitle(GLABEL[group]+' | seed0: mass movement is not density recovery');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.18),ncol=4,fontsize=8);fig.tight_layout();self.save(fig,'massflow_'+'_'.join(group))
 def heatmap(self,group):
  arr=np.full((len(METHODS),4),np.nan)
  for r in self.aggs:
   if (r['family'],r['stage'])==group and r['seeds']==4:arr[METHODS.index(r['method']),DIMS.index(r['dim'])]=r['tv_auc1000']
  fig,ax=plt.subplots(figsize=(7.5,10));im=ax.imshow(arr,aspect='auto',cmap='viridis_r',vmin=0,vmax=.7);ax.grid(False)
  ax.set_xticks(range(4),['1D','2D','4D','8D']);ax.set_yticks(range(len(METHODS)),[LABEL[m] for m in METHODS],fontsize=9)
  for i,j in np.ndindex(arr.shape):
   if np.isfinite(arr[i,j]):ax.text(j,i,f'{arr[i,j]:.3f}',ha='center',va='center',color='white' if arr[i,j]>.35 else 'black',fontsize=9)
  fig.colorbar(im,ax=ax,label='Mean histogram TV over response window (lower is better)');ax.set_title(GLABEL[group]+'\n'+('First 1,000 updates after each of three shifts' if group[1]=='mass' else 'Updates 20,001–35,000: split and merge'));fig.tight_layout();self.save(fig,'auc_'+'_'.join(group))
 def eps_plot(self):
  fig,axes=plt.subplots(1,2,figsize=(13,4.7))
  for d,color in zip(DIMS,['#29445b','#07899e','#df5f35','#7952ac']):
   rows=[next(r for r in self.aggs if (r['family'],r['stage'],r['dim'],r['method'])==('double','mass',d,'sinkhorn_e'+format(e,'g'))) for e in EPS]
   axes[0].plot(EPS,[r['tv_auc1000'] for r in rows],'-o',label=f'{d}D',c=color,ms=4)
   axes[1].plot(EPS,[max(1e-8,r['effective_col_l1']) for r in rows],'-o',label=f'{d}D',c=color,ms=4)
  for ax in axes:ax.set_xscale('log');ax.set_xlabel('Raw-cost Sinkhorn epsilon');ax.legend()
  axes[0].set(title='Response accuracy: two-mode mass shifts',ylabel='Mean TV in first 1,000 updates')
  axes[1].set(title='Numerical residual: final 200 updates',ylabel='L1 column residual of row-normalized R/N');axes[1].set_yscale('log')
  fig.tight_layout();self.save(fig,'epsilon_accuracy_residual')
 def atlas(self,group,dim):
  fig,axes=plt.subplots(5,4,figsize=(15,13),sharex=True)
  for ax,method in zip(axes.flat,METHODS):
   t=self.lookup.get((*group,dim,method,0))
   if not t:ax.text(.5,.5,'No completed result',transform=ax.transAxes,ha='center',wrap=True,fontsize=8);ax.set_title(LABEL[method],fontsize=9);continue
   z=self.density(t);dx=np.diff(z['edges']);mid=(z['edges'][:-1]+z['edges'][1:])/2
   ax.plot(mid,z['target_marginals'][0]/dx,'--',c='#263544',lw=1.6,label='Target / coordinate1')
   ax.stairs(z['marginal_mass'][0]/dx,z['edges'],color=COLORS.get(method,'#07899e'),lw=1,label='Actor: 32,768 draws')
   if dim>1:ax.plot(mid,z['marginal_mass'][1:].mean(0)/dx,c='#b3904b',alpha=.8,lw=.8,label='Actor: other coordinates mean');ax.plot(mid,z['target_marginals'][1:].mean(0)/dx,c='#b3904b',ls=':',lw=1,label='Target: other coordinates')
   ax.set_title(LABEL[method]+f"\nTV={float(z['first_axis_tv']):.3f}",fontsize=9);ax.set_xlim(-1,1)
   if not bool(z['reference_reliable']):ax.text(.05,.90,'Reference unreliable',transform=ax.transAxes,color='#ae2436',fontsize=8)
  axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='center',fontsize=9)
  fig.suptitle(GLABEL[group]+f' | {dim}D, 16×64, seed0 | independent final histogram');fig.supxlabel('Action coordinate');fig.supylabel('Density');fig.tight_layout();self.save(fig,'atlas_'+'_'.join(group)+f'_D{dim}')
 def surface(self,step):
  methods=['gmm_learned','exact_learned','sinkhorn_e0.01','sinkhorn_e0.1','argmax_truncated'];fig=plt.figure(figsize=(16,10));examples=[]
  for m in methods:
   t=self.lookup[('double','mass',2,m,0)];z=self.density(t,step);examples.append((m,z))
  z=examples[0][1];edges=z['joint_edges'];mid=(edges[:-1]+edges[1:])/2;X,Y=np.meshgrid(mid,mid,indexing='ij');area=np.diff(edges)[0]**2
  items=[('Exact target',z['target_joint_mass'])]+[(LABEL[m],z['joint_mass']) for m,z in examples]
  for i,(label,mass) in enumerate(items):
   ax=fig.add_subplot(2,3,i+1,projection='3d');ax.plot_surface(X,Y,mass/area,cmap='viridis',linewidth=0,antialiased=False,rcount=64,ccount=64);ax.set(title=label,xlabel='a1',ylabel='a2',zlabel='Density');ax.view_init(elev=28,azim=-60)
  shown='35,000 (independent final sample)' if step=='final_independent' else f'{int(step):,}'
  fig.suptitle(f'2D two-mode target | update {shown} | seed0 | 64×64 histogram, 32,768 samples\nSame action axes; density axes autoscale to expose concentration');fig.tight_layout();self.save(fig,'surface_D2_'+step)
 def assignment(self):
  methods=['gmm_learned','exact_learned','sinkhorn_e0.001','sinkhorn_e0.01','sinkhorn_e0.1'];fig,axes=plt.subplots(2,5,figsize=(18,9));cmap=plt.get_cmap('magma').copy();cmap.set_bad('#d7e0e8')
  for col,m in enumerate(methods):
   t=self.lookup[('double','mass',2,m,0)];p=self.folder(t)/'assignments/025001.npz'
   with np.load(p) as z:
    b=z['b'][0];w=z['w'][0];pos=z['positions'][0]
    if m=='gmm_learned':
     mu=z['mu'][0];ls=z['log_sigma'][0];u=z['u'][0];ell=-.5*(((u[None]-mu[:,None])/np.exp(ls[:,None]))**2+2*ls[:,None]+np.log(2*np.pi)).sum(-1);A=np.exp(ell-logsumexp(ell,axis=0,keepdims=True))*w[None];title='GMM: w × responsibility'
    else:A=z['P'][0];title=LABEL[m]+'\nraw P'
   for row,aa in enumerate([A,A[np.argsort(pos[:,0],kind='stable')][:,np.argsort(b[:,0],kind='stable')]]):
    ax=axes[row,col];im=ax.imshow(np.ma.masked_less_equal(aa,0),aspect='auto',norm=LogNorm(1e-7,.1),cmap=cmap);ax.grid(False);ax.set_title((title+f"\nRow-mass L1={np.abs(A.sum(1)-1/16).sum():.3f}") if row==0 else 'Same assignment, a1 sorted',fontsize=10);ax.set_xlabel('Candidate index'+(' (a1 sorted)' if row else ' (original)'))
  fig.subplots_adjust(top=.82,bottom=.08,left=.055,right=.92,wspace=.24,hspace=.38)
  axes[0,0].set_ylabel('Student row: original');axes[1,0].set_ylabel('Student row: a1 sorted');fig.colorbar(im,ax=axes.ravel().tolist(),label='Assignment mass (log scale); grey = zero',fraction=.015,pad=.02);fig.suptitle('2D / N16 M64 / seed0 / update25,001: original vs a1-sorted assignment\nSorting changes display order only. In GMM, nonuniform row masses are permitted.',y=.96);self.save(fig,'assignment_original_sorted')
 def teacher(self):
  methods=['gmm_learned','exact_learned','sinkhorn_e0.01','sinkhorn_e0.1'];fig,axes=plt.subplots(3,4,figsize=(16,10),sharex=True)
  for col,m in enumerate(methods):
   z=self.density(self.lookup[('double','mass',1,m,0)],'025200');edges=z['edges'][::8];dx=np.diff(edges)
   target=z['target_marginals'][0].reshape(-1,8).sum(1)/dx
   for row,key in enumerate(['proposal_marginals','teacher_marginals','marginal_mass']):
    mass=z[key][0].reshape(-1,8).sum(1)/dx;ax=axes[row,col];ax.stairs(target,edges,color='#263544',ls='--',label='Exact target');ax.stairs(mass,edges,color=COLORS.get(m,'#07899e'),lw=1.4);ax.set_xlim(-1,1)
    if row==0:ax.set_title(LABEL[m])
   axes[0,col].text(.04,.88,'64 proposals',transform=axes[0,col].transAxes,fontsize=9)
  for ax,label in zip(axes[:,0],['Proposal histogram','Weighted teacher histogram','Actor histogram (32,768)']):ax.set_ylabel(label)
  fig.supxlabel('Action');fig.suptitle('After reversal: update25,200, 1D, seed0 | rebinned to64 bins; no KDE\nA spiky 64-sample teacher is expected: compare basin mass as well as fine-bin TV');fig.tight_layout();self.save(fig,'teacher_actor_histograms')
 def design(self):
  x=np.linspace(-1,1,2001);fig,axes=plt.subplots(2,3,figsize=(15,7))
  for col,(title,centers,h,weights,ds) in enumerate([
   ('Separated two modes',[-.65,.65],.12,[[.5,.5],[.8,.2],[.2,.8]],[0,0,0]),
   ('Original three modes',[-.6,0,.6],.1,[[1/3]*3,[.6,.2,.2],[.2,.2,.6]],[0,0,0]),
   ('Three to six modes',[-.6,0,.6],.1,[[1/6]*6]*3,[0,.075,.15])]):
   for i,(w,d) in enumerate(zip(weights,ds)):
    c=np.array(centers)
    if col==2:c=np.ravel(np.array([c-d,c+d]).T)
    f=np.sum(np.array(w)*np.exp(-.5*((x[:,None]-c)/h)**2)/(h*np.sqrt(2*np.pi)),axis=1);norm=np.sum(np.array(w)*(ndtr((1-c)/h)-ndtr((-1-c)/h)))
    label=(f'd={d:g}' if col==2 else str(w));axes[0,col].plot(x,f/norm,label=label);axes[1,col].plot(x,.25*np.log(f))
   axes[0,col].set_title(title);axes[0,col].legend(fontsize=8);axes[1,col].set_xlabel('a1')
  axes[0,0].set_ylabel('Normalized target density');axes[1,0].set_ylabel('Q = .25 log f');fig.tight_layout();self.save(fig,'design_targets')
 def before_after(self):
  fig,axes=plt.subplots(2,4,figsize=(18,8),sharex=True)
  for row,method in enumerate(['gmm_learned','exact_learned']):
   for col,dim in enumerate(DIMS):
    t=self.lookup[('double','mass',dim,method,0)];a=self.density(t,'020000');b=self.density(t);ax=axes[row,col];edges=a['edges'];dx=np.diff(edges);mid=(edges[1:]+edges[:-1])/2
    ax.plot(mid,a['target_marginals'][0]/dx,'k--',label='Same 50:50 target',lw=1.7)
    ax.stairs(a['marginal_mass'][0]/dx,edges,color='#07899e',label='Before changes:20K',lw=1)
    ax.stairs(b['marginal_mass'][0]/dx,edges,color='#dd5c37',label='After cycle:35K',lw=1)
    ax.set_title(f'{LABEL[method]} | {dim}D',fontsize=10);ax.set_xlim(-1,1)
  axes[0,0].set_ylabel('First-coordinate density');axes[1,0].set_ylabel('First-coordinate density');fig.supxlabel('Action coordinate a1');fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.03),ncol=3);fig.suptitle('Same final target, different learned shapes | seed0,32,768 samples\nNo stationary35K control: do not attribute the improvement uniquely to target changes.');fig.tight_layout();self.save(fig,'double_before_after')

def main(src,out):
 r=Report(src,out)
 if (r.out/'findings.json').exists():r.aggs=json.loads((r.out/'findings.json').read_text())
 else:r.summarize()
 r.design()
 for group in GROUPS:
  r.tracking(group)
  if group[1] in ['mass','split']:r.heatmap(group)
  if group[1]=='mass':r.mass_flow(group)
  for dim in DIMS:r.atlas(group,dim)
 r.eps_plot();r.before_after();r.surface('025200');r.surface('final_independent');r.assignment();r.teacher()
 (r.out/'FIGURE_MANIFEST.json').write_text(json.dumps(r.saved,indent=2));print('ALL_FIGURES_DONE',len(r.saved),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--src',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.src,a.out)
