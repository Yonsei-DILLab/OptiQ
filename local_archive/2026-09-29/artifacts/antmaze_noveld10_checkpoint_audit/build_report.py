"""Verified inference-only checkpoint comparisons; one training seed per panel."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from antmaze.multimodal.dense_noveld_report import maze
from antmaze.multimodal.analysis import summarize

ROOT=Path(__file__).resolve().parent
SHA='a2b8c7900c67e580e7b1f110775cff4fe9a6963e'
PAIRS=[('v1','optiq',300000),('v2','optiq',100000),('v3','optiq',100000),('v1','sac',200000),('v1','mfpo',300000),('v2','sac',100000)]
def read(p):return json.loads(p.read_text())
def regions(xy,task):
    x,y=xy[...,0],xy[...,1]
    if task=='v1':return (x<-2)&(y>2),(x<-2)&(y<-2)
    if task=='v2':return (x<-2)&(y>2),x>2
    return (x<-2)&(y>2),(x>2)&(y<-2)
def counts(xy,low,shape):
    ix=np.floor((xy-low)/.5).astype(int);ok=(ix>=0).all(-1)&(ix<shape).all(-1)
    out=np.zeros(shape,np.int64);np.add.at(out,tuple(ix[ok].T),1);return out

def load(task,method,step,c):
    d=ROOT/'runs'/f'{task}-{method}-c{c:g}-{step}'
    proof=read(d/'verification.json');assert proof['passed'] and proof['model_unchanged'] and proof['checkpoint_step']==step
    assert proof['diagnostic_source']==SHA
    for name,sha in proof['npz_sha256'].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==sha
    p=read(d/'provenance.json');config=p['config'];assert config['intrinsic']['coefficient']==c and config['seed']==0
    train=dict(np.load(d/'training.npz'));xy=train['xy'];co=train['counts'];mass=co[co>0]/co.sum()
    assert int(co.sum())==step and len(xy)==step
    a,b=regions(xy,task);dist=np.linalg.norm(xy[:,None,:]-train['goals'][None,:,:],axis=2)
    tr=read(d/'training.json');history=tr['episodes'];term=train['dones'].astype(bool)
    row=dict(task=task,method=method,step=step,coefficient=c,visited_bins=int((co>0).sum()),
        effective_bins=float(np.exp(-(mass*np.log(mass)).sum())),top10_fraction=float(np.sort(co.ravel())[-10:].sum()/co.sum()),
        directional_step_fractions=[float(a.mean()),float(b.mean())],
        directional_visited_bins=[int(np.count_nonzero(counts(xy[v],train['low'],co.shape))) for v in (a,b)],
        min_goal_distances=dist.min(0).tolist(),training_successes=int(term.sum()),first_success_step=tr['first_success_step'],
        training_successes_by_goal=[int(np.sum(term&(dist.argmin(1)==g))) for g in range(len(train['goals']))],
        source_commit=config['source_commit'],checkpoint_sha256=p['checkpoint_files_sha256'],rollout={})
    paths={}
    for mode in ('native','policy'):
        z=dict(np.load(d/f'{mode}.npz'));assert len(z['returns'])==100
        np.testing.assert_array_equal(z['initial_simulator_state'],np.broadcast_to(z['initial_simulator_state'][0],z['initial_simulator_state'].shape))
        summary=summarize(task,z['xy'],z['lengths'],z['goal_ids'],z['returns'])
        stored=read(d/f'{mode}.json');assert summary['routes']==stored['routes'] and summary['success_rate']==stored['success_rate']
        mins=[];ar=[];br=[]
        for coords,n,ret,g in zip(z['xy'],z['lengths'],z['returns'],z['goal_ids']):
            path=coords[:n+1];distance=np.linalg.norm(path[:,None,:]-train['goals'][None,:,:],axis=2).min(1)
            np.testing.assert_allclose(ret,-distance[1:].sum(),rtol=2e-6,atol=.02)
            assert bool(g)==bool(distance[-1]<=.50001)
            mins.append(float(distance.min()));aa,bb=regions(path,task);ar.append(bool(aa.any()));br.append(bool(bb.any()))
        row['rollout'][mode]=dict(success_rate=summary['success_rate'],mean_return=summary['mean_return'],
            goals=summary['goal_fractions'],successful_routes=summary['successful_routes'],mean_min_distance=float(np.mean(mins)),
            directional_episode_fractions=[float(np.mean(ar)),float(np.mean(br))],mean_length=float(z['lengths'].mean()))
        paths[mode]=z
    return dict(row=row,config=config,training=train,paths=paths)

def save(fig,name):
    fig.savefig(ROOT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(ROOT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def figures(records,selection,label):
    fig,axes=plt.subplots(len(selection),4,figsize=(14,4.6*len(selection)),squeeze=False,layout='constrained')
    for ri,key in enumerate(selection):
        task,method,step=key
        for ci,(mode,c) in enumerate((('native',.01),('native',10),('policy',.01),('policy',10))):
            r=records[(*key,c)];ax=axes[ri,ci];maze(ax,r['config']);z=r['paths'][mode];s=r['row']['rollout'][mode]
            for xy,n,g in zip(z['xy'],z['lengths'],z['goal_ids']):
                path=xy[:n+1];ax.plot(path[:,0],path[:,1],color={0:'#6e7780',1:'#167ab4',2:'#e47f28'}[int(g)],lw=.7,alpha=.22,zorder=3)
            ax.set_title(f'{task} {method.upper()} | {step//1000}k | c={c:g}\n{mode}: success {s["success_rate"]:.0%}; routes {s["successful_routes"]["observed_modes"]}',fontsize=9)
            ax.set_xlabel(f'Region A/B visited: {s["directional_episode_fractions"][0]:.0%}/{s["directional_episode_fractions"][1]:.0%}',fontsize=8)
    fig.suptitle('Saved-policy trajectories | 100 repeated fixed-full-state rollouts per panel\nGray = failed; blue/orange = reached G1/G2. One training seed; matched steps within each row.',fontsize=12)
    fig.supxlabel('OptiQ native = fresh z, mu-only; policy = fresh z + conditional sigma. SAC native = mean; MFPO native = Q-best-of10.\nRegion visitation includes failed episodes and is not a successful-route count. No external DACER noise or NovelD evaluation reward.',fontsize=8)
    save(fig,label+'-trajectories')
    fig,axes=plt.subplots(len(selection),2,figsize=(9,4*len(selection)),squeeze=False,layout='constrained')
    for ri,key in enumerate(selection):
        maximum=max(records[(*key,c)]['training']['counts'].max() for c in (.01,10))
        for ci,c in enumerate((.01,10)):
            r=records[(*key,c)];d=r['row'];t=r['training'];ax=axes[ri,ci];maze(ax,r['config']);values=t['counts'].astype(float);values[values==0]=np.nan
            im=ax.imshow(values.T,origin='lower',extent=[t['low'][0],t['high'][0],t['low'][1],t['high'][1]],norm=LogNorm(1,maximum),cmap='YlOrRd',interpolation='nearest',zorder=1)
            ax.set_title(f'{key[0]} {key[1].upper()} | {key[2]//1000}k | c={c:g}\n{d["visited_bins"]} bins; effective {d["effective_bins"]:.0f}; top10 {d["top10_fraction"]:.1%}',fontsize=9)
            ax.set_xlabel(f'A/B time: {d["directional_step_fractions"][0]:.1%}/{d["directional_step_fractions"][1]:.1%}',fontsize=8)
        fig.colorbar(im,ax=axes[ri].tolist(),fraction=.03,pad=.02,label='Training visits')
    fig.suptitle('TRAINING exploration | matched budgets | 0.5 m grid\nSame color scale within each row; white = unvisited',fontsize=12)
    save(fig,label+'-training-coverage')

def main():
    records={(*k,c):load(*k,c) for k in PAIRS for c in (.01,10)}
    for k in PAIRS:
        x,y=records[(*k,.01)],records[(*k,10)]
        for mode in ('native','policy'):
            for name in ('initial_simulator_state','env_seeds','policy_batch_seeds'):
                np.testing.assert_array_equal(x['paths'][mode][name],y['paths'][mode][name])
        ca,cb=json.loads(json.dumps(x['config'])),json.loads(json.dumps(y['config']))
        for c in (ca,cb):
            for key in ('profile','source_commit','source_root','steps','expected_updates'):c.pop(key,None)
            c['native'].pop('output_root',None);c['intrinsic'].pop('coefficient')
        assert ca==cb,('Non-coefficient config mismatch',k)
    figures(records,PAIRS[:3],'optiq');figures(records,PAIRS[3:],'baselines')
    for mode in ('native','policy'):
        fig,axes=plt.subplots(3,2,figsize=(8,11),layout='constrained')
        for ri,key in enumerate(PAIRS[:3]):
            for ci,c in enumerate((.01,10)):
                r=records[(*key,c)];ax=axes[ri,ci];maze(ax,r['config']);z=r['paths'][mode];st=r['row']['rollout'][mode]
                for xy,n,g in zip(z['xy'],z['lengths'],z['goal_ids']):
                    path=xy[:n+1];ax.plot(path[:,0],path[:,1],color={0:'#6e7780',1:'#167ab4',2:'#e47f28'}[int(g)],lw=.7,alpha=.25,zorder=3)
                ax.set_title(f"{key[0]} | {key[2]//1000}k | NovelD {c:g}\nSuccess {st['success_rate']:.0%} | successful routes {st['successful_routes']['observed_modes']}",fontsize=10)
        meaning='Random z, mu-only (no conditional sigma)' if mode=='native' else 'Direct policy: random z + conditional sigma'
        fig.suptitle('OptiQ saved-policy trajectories | '+meaning+'\n100 rollouts from identical full state per panel; seed0',fontsize=11)
        fig.supxlabel('Gray = failed rollout; blue/orange = reached G1/G2. No external DACER noise.\nTraining exploration coverage is reported separately.',fontsize=9)
        save(fig,'optiq-'+mode+'-trajectories')
    rows=[r['row'] for r in records.values()]
    (ROOT/'results.json').write_text(json.dumps(dict(inference_only=True,training_seed_count=1,rollouts_per_mode=100,report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=rows),indent=2)+'\n')
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
