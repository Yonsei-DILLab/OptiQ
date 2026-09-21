"""One checkpointed segment, with independent evaluation and immutable provenance."""
import argparse,json,os,signal,time
from pathlib import Path
import flax.serialization
import jax
import numpy as np
import wandb
from .engine import cfg_for,initialize,engine,draw
from .metrics import reference,evaluate,component_stats,plot_snapshot


def write_json(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--until',type=int,required=True);a=p.parse_args()
    segment_started=time.monotonic()
    package=Path(__file__).parent;repo=package.parents[1];plan=json.loads((package/'plan.json').read_text())
    c=next(x for x in plan['conditions'] if x['name']==a.condition);cfg=cfg_for(c,a.seed)
    manifest=json.loads((repo/'SOURCE_MANIFEST.json').read_text())
    assert jax.config.jax_default_matmul_precision=='highest'
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3' and len(jax.devices())==1 and jax.default_backend()=='gpu'
    out=a.root/'runs'/f'{a.condition}_s{a.seed}';out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'latest.msgpack';actor,key,target=initialize(cfg);start_step=0;train_seconds=0.
    if checkpoint.exists():
        saved=flax.serialization.msgpack_restore(checkpoint.read_bytes());assert saved['commit']==manifest['commit']
        actor=flax.serialization.from_state_dict(actor,saved['actor']);key=saved['key'];start_step=int(actor.step);train_seconds=float(saved['train_seconds'])
    if start_step>=a.until:return
    cfg.update(commit=manifest['commit'],source_code_id=manifest['source_code_id'],interpretation='recipe comparison; only v5 OT/GMM matched pairs isolate loss')
    write_json(out/'config.json',cfg)
    import hashlib
    rid=hashlib.sha256((manifest['commit']+out.name).encode()).hexdigest()[:12]
    run=wandb.init(entity='OptiQ',project='GMM40_heejoon',id=rid,resume='allow',name=out.name,
        group='legacy-v5-monge-gmm',config=cfg,dir=str(out),save_code=False,mode='online')
    write_json(out/'WANDB.json',dict(id=rid,url=run.url,commit=manifest['commit']))
    algo=engine(c['method'],c['N'],c['M']);locs=np.asarray(target['locs']);scales=np.asarray(target['scales'])
    ref=reference(locs,scales);stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def save():
        saved=dict(actor=flax.serialization.to_state_dict(actor),key=np.asarray(key),train_seconds=train_seconds,commit=manifest['commit'])
        tmp=out/'checkpoint.tmp';tmp.write_bytes(flax.serialization.msgpack_serialize(saved));tmp.replace(checkpoint)
        write_json(out/'state.json',dict(step=int(actor.step),train_seconds=train_seconds,updated=time.time(),commit=manifest['commit'],wandb_id=rid))
    def evaluation():
        step=int(actor.step);ek=jax.random.PRNGKey(1000000+a.seed)
        samples=np.asarray(draw(actor,ek,32768,c['method']))
        assert np.isfinite(samples).all()
        metrics,mode_stats=evaluate(samples,ref,locs,scales)
        t,diag=algo['diagnostics'](actor,jax.random.PRNGKey(2000000+a.seed+step),target)
        t=jax.device_get(t);diag={k:float(v) for k,v in diag.items()}
        ts=component_stats(t['b'],locs,scales,t['w'])
        assigned=component_stats(t['b'],locs,scales,t['A'].sum(0))
        metrics.update(assignment_mode_mass_tv=float(.5*np.abs(assigned['mass']-1/40).sum()),assignment_teacher_mode_tv=float(.5*np.abs(assigned['mass']-ts['mass']).sum()),teacher_mode_mass_tv=float(.5*np.abs(ts['mass']-1/40).sum()),actor_teacher_mode_tv=float(.5*np.abs(mode_stats['mass']-ts['mass']).sum()))
        record=dict(step=step,train_seconds=train_seconds,target_queries=step*c['M'],metrics=metrics,diagnostics=diag)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        write_json(out/'latest_evaluation.json',record)
        np.savez_compressed(out/f'samples_{step:06d}.npz',samples=samples,reference=ref,locs=locs,scales=scales,**mode_stats)
        # Full assignments only at fixed milestones; every evaluation still stores teacher data.
        if step in (0,15000,30000,50000,75000):
            np.savez_compressed(out/f'assignment_{step:06d}.npz',**t)
            plot_snapshot(out,step,samples,ref,locs,mode_stats,t,c['method'])
        else:
            np.savez_compressed(out/f'teacher_{step:06d}.npz',b=t['b'],w=t['w'],q=t['q'],log_q=t['log_q'])
        numeric={f'eval/{k}':v for k,v in metrics.items() if isinstance(v,(int,float))}
        run.log(dict(update=step,train_seconds=train_seconds,**numeric,**{f'train/{k}':v for k,v in diag.items()}))
        print(json.dumps(record),flush=True)
    try:
        if start_step==0:evaluation()
        while int(actor.step)<a.until and not stop[0]:
            before=int(actor.step);count=min(100,a.until-before)
            t0=time.monotonic();actor,key,loss=algo['block'](actor,key,target,count);jax.block_until_ready(actor.params)
            train_seconds+=time.monotonic()-t0
            if not np.isfinite(float(loss)):raise FloatingPointError('Nonfinite training loss')
            if int(actor.step)%1000==0:
                write_json(out/'progress.json',dict(step=int(actor.step),loss=float(loss),train_seconds=train_seconds,updated=time.time()))
                run.log(dict(update=int(actor.step),actor_loss=float(loss),train_seconds=train_seconds))
            if int(actor.step)%5000==0:save();evaluation()
        save()
        if int(actor.step)>=cfg['updates']:write_json(out/'COMPLETE.json',dict(step=int(actor.step),commit=manifest['commit'],time=time.time()))
        with (out/'segments.jsonl').open('a') as f:f.write(json.dumps(dict(start=start_step,end=int(actor.step),commit=manifest['commit'],time=time.time(),segment_wall_seconds=time.monotonic()-segment_started))+'\n')
    except Exception as exc:
        save();write_json(out/'FAILED.json',dict(error=repr(exc),step=int(actor.step),time=time.time()));raise
    finally:run.finish()


if __name__=='__main__':main()
