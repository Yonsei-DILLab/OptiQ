"""Read-only posterior-pattern grouping, one algorithm for 2D and 17D."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,ndtr
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage,cut_tree,leaves_list,optimal_leaf_ordering
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,ListedColormap

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
a.out.mkdir(exist_ok=True,parents=True);(a.out/'figures').mkdir(exist_ok=True);(a.out/'arrays').mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
heat=plt.get_cmap('magma').copy();heat.set_bad('black');palette=np.array(plt.get_cmap('tab10').colors);norm=LogNorm(1e-5,1)
manifest=json.loads(a.manifest.read_text())
for name,v in manifest.items():assert hashlib.sha256((a.data/name).read_bytes()).hexdigest()==v['sha256'],name

def read(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def choose2(v):return v*(v-1)/2
def ari(x,y):
 _,x=np.unique(x,return_inverse=True);_,y=np.unique(y,return_inverse=True);table=np.zeros((x.max()+1,y.max()+1),int);np.add.at(table,(x,y),1)
 n=len(x);all_=choose2(n);pairs=choose2(table).sum();ax=choose2(table.sum(1)).sum();by=choose2(table.sum(0)).sum();expect=ax*by/all_;den=.5*(ax+by)-expect
 return float((pairs-expect)/den) if den else 1.
def sil(D,labels):
 K=labels.max()+1;means=np.stack([D[:,labels==k].mean(1) for k in range(K)],1);counts=np.bincount(labels);own=means[np.arange(len(D)),labels]*counts[labels]/np.maximum(counts[labels]-1,1)
 means[np.arange(len(D)),labels]=np.inf;other=means.min(1);score=(other-own)/np.maximum(np.maximum(other,own),1e-15);score[counts[labels]<=1]=0
 return float(score.mean())
def group(logg,ordered=False):
 logg=logg-logsumexp(logg,axis=0,keepdims=True);g=np.exp(logg);np.testing.assert_allclose(g.sum(0),1,atol=1e-12)
 dd=pdist(np.sqrt(g.T)/np.sqrt(2));D=squareform(dd);M=g.shape[1];min_size=max(2,int(np.ceil(.01*M)));scores=[]
 if dd.max()<1e-10:return dict(labels=np.zeros(M,int),K=1,score=0.,scores=[],order=np.arange(M),D=D,g=g,logg=logg)
 Z=linkage(dd,method='average');best=None
 for K in range(2,min(8,M-1)+1):
  labels=cut_tree(Z,n_clusters=K).ravel();counts=np.bincount(labels);score=sil(D,labels);ok=bool(counts.min()>=min_size)
  scores.append(dict(K=K,silhouette=score,min_count=int(counts.min()),eligible=ok))
  if ok and (best is None or score>best[0]+1e-12):best=(score,labels.copy(),K)
 if best is None or best[0]<.25:labels=np.zeros(M,int);score=0.;K=1
 else:score,labels,K=best
 order=leaves_list(optimal_leaf_ordering(Z,dd)) if ordered else leaves_list(Z)
 return dict(labels=labels,K=K,score=score,scores=scores,order=order,D=D,g=g,logg=logg)
def stability(logg,labels):
 out=[];N=logg.shape[0]
 for seed in range(919,924):
  rows=np.random.default_rng(seed).choice(N,max(2,int(.8*N)),replace=False);z=group(logg[rows]);out.append(dict(seed=seed,K=z['K'],ARI=ari(labels,z['labels'])))
 return out
def layout(g,R,coords):
 # Color ID alone is canonicalized AFTER inference, never fed back into fit.
 labels=g['labels'];ks=sorted(range(g['K']),key=lambda k:float(np.median(coords[labels==k,0])));mapping={v:i for i,v in enumerate(ks)};labels=np.array([mapping[v] for v in labels]);g['labels']=labels
 H=R@np.eye(g['K'])[labels];rowgroup=H.argmax(1);leaf=g['order'];co=np.array(sorted(leaf,key=lambda j:labels[j]))
 dr=pdist(np.sqrt(np.maximum(R,0))/np.sqrt(2));ro=leaves_list(linkage(dr,method='average',optimal_ordering=True)) if dr.max()>1e-12 else np.arange(len(R))
 rowleaf=ro.copy();ro=np.array(sorted(ro,key=lambda i:rowgroup[i]));return H,rowgroup,ro,co,rowleaf,leaf
def prepare(tag,logg,logw,coords,centers,extra,ordered=True,stabilize=True):
 start=time.monotonic();g=group(logg,ordered);logw=logw-logsumexp(logw);logJ=g['logg']+logw;loga=logsumexp(logJ,axis=1);R=np.exp(logJ-loga[:,None]);w=np.exp(logw);alpha=np.exp(loga)
 np.testing.assert_allclose(R.sum(1),1,atol=1e-10);np.testing.assert_allclose(np.exp(logJ).sum(0),w,atol=1e-10)
 H,rg,ro,co,rowleaf,colleaf=layout(g,R,coords);st=stability(logg,g['labels']) if stabilize else []
 rec=dict(tag=tag,N=logg.shape[0],M=logg.shape[1],K=g['K'],silhouette=g['score'],scores=g['scores'],cluster_counts=np.bincount(g['labels']).tolist(),teacher_group_mass=np.bincount(g['labels'],weights=w,minlength=g['K']).tolist(),student_group_counts=np.bincount(rg,minlength=g['K']).tolist(),teacher_ESS=float(1/(w*w).sum()),usage_ESS=float(1/(alpha*alpha).sum()),underused_fraction=float((alpha<.1/len(alpha)).mean()),stability=st,stability_mean_ARI=float(np.mean([s['ARI'] for s in st])) if st else None,seconds=time.monotonic()-start,**extra)
 arr=dict(gamma=g['g'],R=R,w=w,alpha=alpha,log_alpha=loga,coordinates=coords,centers=centers,labels=g['labels'],row_groups=rg,row_order=ro,col_order=co,row_seriation=rowleaf,col_seriation=colleaf,H=H,distance=g['D'])
 np.savez_compressed(a.out/'arrays'/f'{tag}.npz',**arr)
 print(json.dumps({k:rec[k] for k in ['tag','K','silhouette','stability_mean_ARI','seconds']}),flush=True)
 return rec,arr

# Independent algebra/behavior checks, before any real-input clustering.
null=np.zeros((10,30));assert group(null)['K']==1
fake=np.full((4,40),-50.);fake[:2,:20]=0;fake[2:,20:]=0;g=group(fake);assert g['K']==2 and ari(g['labels'],np.repeat([0,1],20))==1
assert ari([0,0,1,1],[4,4,8,8])==1 and abs(ari([0,0,1,1],[0,1,0,1])+.5)<1e-12
perm=np.array([3,1,0,2]);np.testing.assert_allclose(group(fake[perm])['D'],g['D'])
np.testing.assert_array_equal(group(fake)['labels'],g['labels'])
validation={'identical_student_K1':True,'two_block_K2':True,'ARI_identity_and_crossing':True,'row_permutation_distance':True,'input_sha256_count':len(manifest)}
toy=[];toyarr={};density={};rl=[];rlarr={};maxgerr=0.;maxrerr=0.
for family in ['double','tri']:
 for seed in range(4):
  run=a.data/'toy'/f'{family}_mass_D2_gmm_learned_N256_M1024_s{seed}';cfg=json.loads((run/'config.json').read_text());z=read(run/'assignments/final_independent.npz');den=read(run/'density/final_independent.npz')
  assert int(z['step'])==35000 and len(den['actions'])==32768
  ed=den['joint_edges'];hist=np.histogram2d(*den['actions'].T,bins=(ed,ed))[0]/32768;np.testing.assert_allclose(hist,den['joint_mass'],atol=1e-12)
  mu=z['mu'][0].astype(float);ls=z['log_sigma'][0].astype(float);u=z['u'][0].astype(float);ell=-.5*(((u[None]-mu[:,None])*np.exp(-ls[:,None]))**2+2*ls[:,None]+np.log(2*np.pi)).sum(-1)
  logw=z['logits'][0].astype(float);ww=np.exp(logw-logsumexp(logw));np.testing.assert_allclose(ww,z['w'][0],atol=2e-6,rtol=2e-5)
  tag=f'toy_{family}_s{seed}';rec,arr=prepare(tag,ell,logw,z['b'][0],z['positions'][0],dict(kind='2D',family=family,seed=seed,step=35000,training_commit=cfg['commit'],joint_histogram_TV=float(den['joint_2d_tv'])),ordered=(seed==0))
  # Ground-truth basin evaluation is AFTER prepare, not an input to grouping.
  truth=np.digitize(z['b'][0,:,0],den['boundaries'][1:-1]);rec['true_basin_ARI']=ari(arr['labels'],truth);cont=np.zeros((rec['K'],len(den['boundaries'])-1),int);np.add.at(cont,(arr['labels'],truth),1);rec['group_basin_counts']=cont.tolist()
  toy.append(rec);toyarr[tag]=arr;density[tag]=den
for step in [50000,650000]:
 z=read(a.data/'rl'/f'actual_64x64_{step}.npz')
 for state in range(len(z['R'])):
  mu=z['mu'][state].astype(float);ls=z['log_sigma'][state].astype(float);c=z['candidates'][state].astype(float);sig=np.exp(ls);Z=ndtr((1-mu)/sig)-ndtr((-1-mu)/sig);assert (Z>0).all()
  ell=(-.5*((c[None]-mu[:,None])/sig[:,None])**2-ls[:,None]-.5*np.log(2*np.pi)-np.log(Z)[:,None]).sum(-1)
  gg=np.exp(ell-logsumexp(ell,axis=0));err=float(np.max(np.abs(gg-z['gamma'][state])));maxgerr=max(maxgerr,err);assert err<.001,err
  logw=z['Q'][state].astype(float)/.25-z['log_q'][state].astype(float)
  tag=f'rl_{step}_s{state}';rec,arr=prepare(tag,ell,logw,c,mu,dict(kind='17D',step=step,state=state,posterior_reconstruction_max_error=err),ordered=(state<8),stabilize=(state<8))
  re=float(np.max(np.abs(arr['R']-z['R'][state])));maxrerr=max(maxrerr,re);assert re<.002,re
  rl.append(rec)
  if state<8:rlarr[tag]=arr
validation.update(max_RL_posterior_reconstruction_error=maxgerr,max_RL_conditional_reconstruction_error=maxrerr,toy_sample_histograms_verified=8)
(a.out/'VALIDATION.json').write_text(json.dumps(validation,indent=2))
(a.out/'summary.json').write_text(json.dumps(dict(analysis_commit=a.commit,training_updates=0,settings=dict(K_range=[2,8],minimum_silhouette=.25,minimum_count='max(2,ceil(.01M))',distance='Hellinger on candidate posterior',linkage='average',stability_keep_fraction=.8,stability_repetitions=5),toy=toy,rl=rl),indent=2))

def save(fig,name):fig.savefig(a.out/'figures'/f'{name}.png',dpi=170,bbox_inches='tight');fig.savefig(a.out/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
def heatmap(ax,x,order='original',bars=False):
 if order=='original':ro=np.arange(len(x['R']));co=np.arange(x['R'].shape[1])
 elif order=='pca':
  X=np.r_[x['centers'],x['coordinates']];_,_,v=np.linalg.svd(X-X.mean(0),full_matrices=False);direction=v[0];ro=np.argsort(x['centers']@direction);co=np.argsort(x['coordinates']@direction)
 elif order=='seriation':ro=x['row_seriation'];co=x['col_seriation']
 else:ro=x['row_order'];co=x['col_order']
 im=ax.imshow(np.ma.masked_less_equal(x['R'][ro][:,co],0),origin='lower',aspect='auto',interpolation='nearest',norm=norm,cmap=heat)
 ax.set(xlabel='Candidate index',ylabel='Student row')
 if bars:
  cb=ax.inset_axes([0,1.005,1,.025]);cb.imshow(palette[x['labels'][co]][None],aspect='auto');cb.set_axis_off()
  rb=ax.inset_axes([-.035,0,.025,1]);rb.imshow(palette[x['row_groups'][ro]][:,None],origin='lower',aspect='auto');rb.set_axis_off()
 return im
def contours(ax,d):
 mid=(d['joint_edges'][1:]+d['joint_edges'][:-1])/2;pdf=d['target_joint_mass'].T/np.diff(d['joint_edges'])[0]**2
 ax.contour(mid,mid,pdf,levels=pdf.max()*np.array([.2,.5,.8]),colors='#565e6a',linewidths=.8,alpha=.6)
 ax.set(xlabel='Action 1',ylabel='Action 2',xlim=(-1,1),ylim=(-1,1));ax.set_aspect('equal')

fig,axs=plt.subplots(2,4,figsize=(18,8),layout='constrained')
for row,family in enumerate(['double','tri']):
 tag=f'toy_{family}_s0';x=toyarr[tag];r=next(q for q in toy if q['tag']==tag);d=density[tag];ax=axs[row,0]
 ax.imshow(d['joint_mass'].T/(2/64)**2,origin='lower',extent=(-1,1,-1,1),aspect='equal',cmap='Blues');contours(ax,d);ax.set_title(f'{family}: actor histogram\n32,768 actions; target contours')
 ax=axs[row,1];contours(ax,d)
 for k in range(r['K']):
  ix=x['labels']==k;ax.scatter(*x['coordinates'][ix].T,s=11,color=palette[k],alpha=.65,label=f'Group {k+1}')
 ax.set_title(f'Candidates: inferred K={r["K"]}\nSilhouette={r["silhouette"]:.2f}; basin ARI={r["true_basin_ARI"]:.2f}');ax.legend(fontsize=7,loc='lower left',ncol=2,framealpha=.8)
 ax=axs[row,2];contours(ax,d);ax.scatter(*x['centers'].T,c=palette[x['row_groups']],s=24,marker='x');ax.set_title('Student centers tanh(mu)\nColor = largest grouped teacher mass')
 im=heatmap(axs[row,3],x,'grouped',True);axs[row,3].set_title('Same R, grouped order\nTop/left strips = inferred groups',pad=20)
fig.colorbar(im,ax=axs[:,3].tolist(),label='Conditional row mass R',shrink=.65);fig.suptitle('2D existing Direct GMM | N256 M1024 | seed 0 | 35K\nGroups use posterior patterns only; target contours are evaluation context',fontsize=14);save(fig,'toy_2d_main')

fig,axs=plt.subplots(2,4,figsize=(16,8),layout='constrained')
for row,family in enumerate(['double','tri']):
 for seed in range(4):
  tag=f'toy_{family}_s{seed}';x=toyarr[tag];r=next(q for q in toy if q['tag']==tag);ax=axs[row,seed];contours(ax,density[tag]);ax.scatter(*x['coordinates'].T,c=palette[x['labels']],s=8,alpha=.7);ax.set_title(f'{family} seed{seed}: K={r["K"]}\nARI basin={r["true_basin_ARI"]:.2f}, subset={r["stability_mean_ARI"]:.2f}')
fig.suptitle('All four trained seeds | identical clustering rule; colors are local group IDs');save(fig,'toy_all_seeds')

fig,axs=plt.subplots(2,4,figsize=(19,8),layout='constrained')
for row,step in enumerate([50000,650000]):
 x=rlarr[f'rl_{step}_s0'];r=next(q for q in rl if q['tag']==f'rl_{step}_s0')
 for col,order in enumerate(['original','pca','seriation','grouped']):
  im=heatmap(axs[row,col],x,order,order=='grouped');title={'original':'Original order','pca':'Action PC1 order','seriation':'Responsibility order (no K)','grouped':f'Responsibility groups: K={r["K"]}'}[order];axs[row,col].set_title(f'{step:,} steps | {title}',pad=20 if order=='grouped' else 8)
fig.colorbar(im,ax=axs.ravel().tolist(),label='Conditional row mass R',shrink=.7);fig.suptitle('17D TRG Humanoid seed0 | N=M64 | same fixed state #0\nIdentical matrix values; reordering only. Group counts are NOT certified Q mode counts.',fontsize=14);save(fig,'rl_main')
for step in [50000,650000]:
 fig,axs=plt.subplots(2,4,figsize=(16,8),layout='constrained')
 for state,ax in enumerate(axs.flat):
  x=rlarr[f'rl_{step}_s{state}'];r=next(q for q in rl if q['tag']==f'rl_{step}_s{state}');im=heatmap(ax,x,'grouped',True);ax.set_title(f'State{state}: K={r["K"]}, sil={r["silhouette"]:.2f}\nsubset ARI={r["stability_mean_ARI"]:.2f}',pad=20)
 fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.6,label='R (log scale)');fig.suptitle(f'{step:,} steps | first eight fixed states, no visual selection');save(fig,f'rl_first8_{step}')
fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
for col,family in enumerate(['double','tri']):
 for r in [q for q in toy if q['family']==family]:
  ss=r['scores'];axs[col].plot([v['K'] for v in ss],[v['silhouette'] for v in ss],'o-',label=f'seed {r["seed"]}')
 axs[col].set(title=f'2D {family}: all tested K',xlabel='Requested groups K',ylabel='Silhouette');axs[col].legend(fontsize=8)
for step in [50000,650000]:
 r=next(q for q in rl if q['tag']==f'rl_{step}_s0');ss=r['scores'];axs[2].plot([v['K'] for v in ss],[v['silhouette'] for v in ss],'o-',label=f'{step:,} steps')
axs[2].set(title='17D state0: all tested K',xlabel='Requested groups K',ylabel='Silhouette');axs[2].legend(fontsize=8)
for ax in axs:ax.axhline(.25,ls='--',c='gray',alpha=.5);ax.grid(alpha=.2)
fig.suptitle('Same K=2..8 search and minimum group-size rule; .25 acceptance threshold');save(fig,'selection_scores')
print('ANALYSIS_COMPLETE',a.out,flush=True)
