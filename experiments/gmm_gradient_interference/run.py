import argparse,hashlib,json,os,signal,time
from pathlib import Path
from . import bootstrap
from .core import *
from .diagnostics import probe
from flax import serialization


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def save_npz(path,arrays):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)

def verify():
    m=json.loads((bootstrap.ROOT/'SOURCE_MANIFEST.json').read_text())
    assert all(hashlib.sha256((bootstrap.ROOT/f).read_bytes()).hexdigest()==h for f,h in m['files'].items())
    return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--n',type=int,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--until',type=int,required=True);a=p.parse_args()
    manifest=verify();plan=json.loads((Path(__file__).parent/'plan.json').read_text());m=plan['m']
    assert a.n in plan['n_values'] and a.seed in plan['seeds'] and a.until<=plan['updates']
    assert jax.default_backend()=='gpu' and len(jax.devices())==1 and os.environ['CUDA_VISIBLE_DEVICES']=='3'
    assert jax.config.jax_default_matmul_precision is None
    out=a.root/'runs'/f'N{a.n}_M{m}_s{a.seed}';out.mkdir(parents=True,exist_ok=True)
    state,key=initialize(a.seed);train_seconds=0.;diag_seconds=0.;stop=[False]
    ck=out/'checkpoint.msgpack'
    if ck.exists():
        old=serialization.msgpack_restore(ck.read_bytes());assert old['commit']==manifest['commit']
        state=serialization.from_state_dict(state,old['actor']);key=jnp.asarray(old['key'])
        train_seconds=old['train_seconds'];diag_seconds=old['diagnostic_seconds']
    if int(state.step)>=a.until:return
    write(out/'config.json',dict(**plan,n=a.n,seed=a.seed,source_commit=manifest['commit'],source_code_id=manifest['source_code_id']))
    write(out/'RUNNING.json',dict(pid=os.getpid(),start=time.time(),step=int(state.step),commit=manifest['commit'],until=a.until))
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def checkpoint():
        tmp=out/'checkpoint.tmp';tmp.write_bytes(serialization.msgpack_serialize(dict(actor=serialization.to_state_dict(state),key=np.asarray(key),train_seconds=train_seconds,diagnostic_seconds=diag_seconds,commit=manifest['commit'])))
        tmp.replace(ck)
        write(out/'progress.json',dict(step=int(state.step),train_seconds=train_seconds,diagnostic_seconds=diag_seconds,updated=time.time(),commit=manifest['commit']))
    def evaluate():
        nonlocal diag_seconds
        st=int(state.step);t0=time.monotonic();ep=out/f'eval_{st:06d}.npz'
        if not ep.exists():
            z=jax.random.normal(jax.random.PRNGKey(77000+a.seed),(2048,1));mu,ls=heads(state.params,z)
            samples=np.asarray(draw(state.params,jax.random.PRNGKey(88000+a.seed)))
            edges=np.linspace(-1,1,257);mass=np.diff(problems.analytic_cdf(edges,np.asarray(QARG)))
            hist=np.histogram(samples,edges)[0]/len(samples)
            prob=np.asarray(basin_prob(mu,ls));counts=np.bincount(np.where(prob.max(1)>=.8,prob.argmax(1),3),minlength=4)
            data=dict(samples=samples,histogram=hist,edges=edges,target_bin_mass=mass,z=np.asarray(z),mu=np.asarray(mu),log_sigma=np.asarray(ls),basin_prob=prob,specialist_counts=counts)
            save_npz(ep,data)
            record=dict(step=st,histogram_tv=float(.5*np.abs(hist-mass).sum()),specialist_fraction=float((prob.max(1)>=.8).mean()),specialist_counts=counts.tolist(),sigma_mean=float(np.exp(np.asarray(ls)).mean()),train_seconds=train_seconds)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        dp=out/f'diagnostic_{st:06d}.npz'
        if st in plan['diagnostic_steps'] and not dp.exists():
            arrays,summary=probe(state,key,a.n,m,a.seed)
            assert np.isfinite(arrays['samples']).all() and np.isfinite(arrays['gradient_gram']).all()
            save_npz(dp,arrays);write(out/f'diagnostic_{st:06d}.json',dict(step=st,**summary))
        diag_seconds+=time.monotonic()-t0
    started=time.monotonic()
    try:
        evaluate();checkpoint()
        milestones=sorted(set([a.until]+[s for s in plan['diagnostic_steps'] if int(state.step)<s<=a.until]+list(range(1000,a.until+1,1000))))
        for end in milestones:
            if end<=int(state.step):continue
            while int(state.step)<end:
                t0=time.monotonic();state,key,loss=block(a.n,m)(state,key,min(500,end-int(state.step)))
                jax.block_until_ready(state.params);train_seconds+=time.monotonic()-t0
                assert np.isfinite(float(loss)),f'Non-finite training loss at {int(state.step)}'
                checkpoint()
                if stop[0]:return
            evaluate();checkpoint()
            print(json.dumps(dict(step=int(state.step),n=a.n,seed=a.seed,train_seconds=train_seconds,diagnostic_seconds=diag_seconds)),flush=True)
        if int(state.step)==plan['updates']:
            write(out/'COMPLETE.json',dict(step=int(state.step),commit=manifest['commit'],train_seconds=train_seconds,diagnostic_seconds=diag_seconds,finished=time.time()))
    except BaseException as exc:
        checkpoint();write(out/'FAILED.json',dict(error=repr(exc),step=int(state.step),time=time.time()));raise
    finally:
        with (out/'segments.jsonl').open('a') as f:f.write(json.dumps(dict(until=a.until,step=int(state.step),elapsed=time.monotonic()-started,ended=time.time()))+'\n')

if __name__=='__main__':main()
