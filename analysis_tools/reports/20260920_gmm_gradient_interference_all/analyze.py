"""Read-only aggregation of the three immutable campaigns, without JAX/training."""
import argparse,hashlib,json,math,re
from pathlib import Path
import numpy as np

SOURCES=['1b1c6947d06b','f3aebce92a3e','32bc71599d06']
SIZES=[(16,16),(64,64),(128,128),(256,256),(1024,1024),(2048,2048),(64,4096),(2048,4096)]
STEPS=[0,1,10,100,500,1000,2000,5000,10000,15000,20000]
BRANCHES=['full_adam','momentum_only','left_adam','center_adam','right_adam','left_sgd_matched_norm','center_sgd_matched_norm','right_sgd_matched_norm']

def cdf(x):
    x=np.asarray(x);return sum(np.vectorize(math.erf)((x-c)/(.1*np.sqrt(2)))+1 for c in [-.6,0,.6])/6

def target_mass(edges):return np.diff(cdf(edges))/(cdf(1.)-cdf(-1.))

def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x

def write(path,x):path.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')

def collect(data):
    runs=[];inputs={};validation=[]
    for short in SOURCES:
        source=data/short;manifest=json.loads((source/'repo/SOURCE_MANIFEST.json').read_text())
        val=json.loads((source/'validation/VALIDATION_PASSED.json').read_text());assert val['passed'] and val['commit']==manifest['commit'];validation.append(val)
        assert (source/'runtime/QUEUE_FINISHED.json').exists()
        for p in sorted((source/'runs').glob('N*_s*')):
            n,m,seed=map(int,re.fullmatch(r'N(\d+)_M(\d+)_s(\d+)',p.name).groups())
            cfg=json.loads((p/'config.json').read_text());complete=json.loads((p/'COMPLETE.json').read_text())
            assert complete['step']==20000 and cfg['source_commit']==complete['commit']==manifest['commit']
            assert cfg['n']==n and cfg['seed']==seed and cfg.get('m',m)==m
            assert cfg['temperature']==.25 and cfg['updates']==20000 and cfg['learning_rate']==.0003
            hist=[json.loads(x) for x in (p/'history.jsonl').read_text().splitlines()]
            assert len(hist)==len({h['step'] for h in hist}) and hist[-1]['step']==20000
            evals={int(f.stem.split('_')[-1]):dict(np.load(f,allow_pickle=False)) for f in p.glob('eval_*.npz')}
            diags={int(f.stem.split('_')[-1]):dict(np.load(f,allow_pickle=False)) for f in p.glob('diagnostic_*.npz')}
            assert sorted(diags)==STEPS
            for st,d in diags.items():
                assert d['branch_names'].tolist()==BRANCHES
                assert d['teacher_w'].shape==(m,) and d['training_z'].shape==(n,1)
                assert d['z'].shape==(2048,1) and d['samples'].shape==(32768,)
                assert np.isfinite(d['samples']).all() and np.isfinite(d['gradient_gram']).all()
                np.testing.assert_allclose(d['teacher_w'].sum(),1,atol=2e-6)
                np.testing.assert_allclose(d['assignment_mode_mass'].sum(),1,atol=2e-6)
                labels=np.digitize(d['teacher_b'],[-.3,.3])
                d['teacher_mode_count']=np.bincount(labels,minlength=3)
                d['teacher_mode_mass']=np.bincount(labels,weights=d['teacher_w'],minlength=3)
                np.testing.assert_allclose(d['assignment_mode_mass'].sum(0,dtype=np.float64),d['teacher_mode_mass'],atol=2e-6)
                np.testing.assert_array_equal(d['z'],diags[0]['z'])
                np.testing.assert_allclose(d['gradient_cosine'],d['gradient_cosine'].T,atol=1e-6)
            for st,e in evals.items():
                np.testing.assert_array_equal(np.histogram(e['samples'],e['edges'])[0]/32768,e['histogram'])
                np.testing.assert_allclose(e['target_bin_mass'],target_mass(e['edges']),atol=2e-6)
            final=evals[20000];d=diags[20000];last=hist[-1]
            tv=.5*np.abs(final['histogram']-target_mass(final['edges'])).sum();assert abs(tv-last['histogram_tv'])<2e-6
            np.testing.assert_array_equal(final['samples'],d['samples'])
            for f in p.iterdir():
                if f.suffix in ['.npz','.json','.jsonl'] or f.name=='checkpoint.msgpack':inputs[str(f.relative_to(data))]=hashlib.sha256(f.read_bytes()).hexdigest()
            runs.append(dict(path=p,name=p.name,n=n,m=m,seed=seed,commit=manifest['commit'],history=hist,evals=evals,diags=diags))
    assert len(runs)==32 and {(r['n'],r['m'],r['seed']) for r in runs}=={(n,m,s) for n,m in SIZES for s in range(4)}
    return sorted(runs,key=lambda r:(SIZES.index((r['n'],r['m'])),r['seed'])),inputs,validation

def metrics(r,step=20000):
    d=r['diags'][step];e=r['evals'][step];n,m=r['n'],r['m'];off=~np.eye(3,dtype=bool)
    cosine=d['gradient_cosine'];pair=cosine[np.triu_indices(3,1)]
    teacher=d['teacher_mode_mass'];mass=np.bincount(np.digitize(e['samples'],[-.3,.3]),minlength=3)/32768
    counts=d['specialist_counts'];pos=d['cross_mode_position_rms'][2:5,:3];on=np.diag(pos);other=pos[off]
    tv=.5*np.abs(e['histogram']-e['target_bin_mass']).sum()
    return dict(run=r['name'],n=n,m=m,seed=r['seed'],commit=r['commit'],step=step,
        histogram_tv=tv,actor_basin_tv=.5*np.abs(mass-target_mass([-1,-.3,.3,1])).sum(),actor_basin_mass=mass,
        good_fit=bool(tv<.1),specialist_fraction=counts[:3].sum()/2048,specialist_counts=counts,
        sigma_mean=np.exp(d['log_sigma']).mean(),between_mu_variance=d['mu'].var(),within_variance=np.exp(2*d['log_sigma']).mean(),
        teacher_basin_mass=teacher,teacher_basin_tv=.5*np.abs(teacher-target_mass([-1,-.3,.3,1])).sum(),
        teacher_mode_count=d['teacher_mode_count'],teacher_ess_ratio=1/np.square(d['teacher_w']).sum()/m,teacher_wmax=d['teacher_w'].max(),
        component_ess_ratio=1/np.square(d['alpha']).sum()/n,underused_fraction=(d['alpha']<.1/n).mean(),
        pair_cosine=pair,pair_negative_fraction=(pair<0).mean(),gradient_norm=d['gradient_norm'],
        full_ref_delta=d['delta_reference_nll'][0],momentum_ref_delta=d['delta_reference_nll'][1],
        mode_adam_ref_delta=d['delta_reference_nll'][2:5],mode_sgd_ref_delta=d['delta_reference_nll'][5:8],
        full_teacher_delta=d['delta_mode_nll'][0],mode_adam_teacher_delta=d['delta_mode_nll'][2:5],
        full_total_ref_delta=d['delta_reference_nll'][0].sum(),momentum_total_ref_delta=d['delta_reference_nll'][1].sum(),
        full_total_teacher_delta=d['delta_mode_nll'][0].sum(),
        mode_adam_offdiag_harm_fraction=(d['delta_reference_nll'][2:5][off]>1e-6).mean(),
        mode_sgd_offdiag_harm_fraction=(d['delta_reference_nll'][5:8][off]>1e-6).mean(),
        cross_mode_position_rms=pos,cross_mode_own_mass_change=d['cross_mode_own_mass_change'][2:5,:3],
        offdiag_position_rms=np.nanmean(other) if np.isfinite(other).any() else None,
        diagonal_position_rms=np.nanmean(on) if np.isfinite(on).any() else None,
        movement_ratio=np.nanmean(other)/np.nanmean(on) if np.isfinite(other).any() and np.nanmean(on)>0 else None,
        first_histogram_tv_below_point1=next((h['step'] for h in r['history'] if h['histogram_tv']<.1),None))

def analyze(runs):
    rows=[metrics(r) for r in runs];summaries=[]
    for n,m in SIZES:
        rr=[x for x in rows if (x['n'],x['m'])==(n,m)];out=dict(n=n,m=m,seeds=4,good_fit=sum(x['good_fit'] for x in rr))
        for key in ['histogram_tv','actor_basin_tv','specialist_fraction','sigma_mean','teacher_basin_tv','teacher_ess_ratio','component_ess_ratio','underused_fraction','pair_negative_fraction','full_total_ref_delta','momentum_total_ref_delta','mode_adam_offdiag_harm_fraction','mode_sgd_offdiag_harm_fraction','movement_ratio']:
            vals=np.array([x[key] for x in rr if x[key] is not None]);out[key]=dict(mean=vals.mean() if len(vals) else None,sd=vals.std(ddof=1) if len(vals)>1 else None,n=len(vals))
        out['teacher_missing_mode_snapshots']=sum(np.any(d['teacher_mode_count']==0) for r in runs if (r['n'],r['m'])==(n,m) for d in r['diags'].values())
        out['diagnostic_snapshots']=44
        summaries.append(out)
    return rows,summaries

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    runs,inputs,validation=collect(a.data);rows,summaries=analyze(runs)
    write(a.out/'per_run.json',rows);write(a.out/'summary.json',summaries);write(a.out/'INPUT_MANIFEST.json',dict(files=inputs,validation=validation,run_count=len(runs),diagnostic_count=352))
    print(json.dumps(clean(summaries),indent=2));print('Verified32runs/352diagnostics; no training modified')
