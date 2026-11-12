"""Exercise completion gates, matched evaluations, tie-breaking, and reuse IDs."""
import json,copy,tempfile
from pathlib import Path
from entropy_selection import choose,TARGETS
from entropy_dependencies import resolve_dependency
P=Path(__file__).resolve().parent
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
def read(p):return json.loads(p.read_text()) if p.exists() else None
def main():
    protocol=read(P/'protocol.json');old=read(P.parent/'fetch-adroit-f1-a1-temp-20261112/protocol.json')
    original_ids={j['id'] for j in old['rows']}
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        for family,task in [('fetch','slide'),('fetch','pick-and-place'),('adroit','relocate')]:
            key=family+'-'+task
            for temp in [.05,.1,.25,.5,1.]:
                (root/'selections'/(key+'.json')).unlink(missing_ok=True)
                (root/'entropy-selections'/(key+'.json')).unlink(missing_ok=True)
                t=dict(family=family,task=task,temperature=temp,pilot_seeds=[5,6],expected_steps=5000192,common_steps=[4950016,4960000,4970240,4980224,4995072])
                j=dict(family=family,task=task,seed=2,status='waiting_temperature',temperature=None)
                resolve_dependency(j,root,read)
                if not (root/'selections'/(key+'.json')).exists():assert j['status']=='waiting_temperature'
                save(root/'selections'/(key+'.json'),t);resolve_dependency(j,root,read);assert j['status']=='queued' and j['temperature']==temp
                for h in [-.9,.5]:
                    j.update(status='waiting_entropy',seed=2)
                    save(root/'entropy-selections'/(key+'.json'),dict(t,target_entropy_per_dim=h))
                    resolve_dependency(j,root,read)
                    assert j['status']==('reused_reference' if h==-.9 else 'queued')
                    if h==-.9:assert j['reference_id'] in original_ids,j
        rows=[];t=dict(family='adroit',task='pen',temperature=.1,pilot_seeds=[5,6],expected_steps=5000192)
        steps=[4950016,4960000,4970240,4980224,4995072]
        for h in TARGETS:
            for seed in [5,6]:
                o=root/f'{h}-{seed}'
                j=dict(id=o.name,family='adroit',task='pen',method='ibolt',temperature=.1,seed=seed,phase='pilot' if h==-.9 else 'entropy_pilot',dacer_target_entropy_per_dim=h)
                save(o/'config.json',dict(seed=seed,batch_size=256,num_envs=256,utd=.25,demonstration_dataset_sha256='fixed-demo',critic_layer_norm=True,offline_pretraining_updates=0))
                save(o/'ibolt-config.json',dict(temperature=.1,dacer=dict(enabled=True,target_entropy_per_dim=h,noise_scale=.1),alg=dict(actor=dict(hidden_dims=[256,256]),critic=dict(hs=[256,256]))))
                save(o/'demonstration-validation.json',dict(passed=True,replayed=True,successful_trajectories=25))
                evaluations=[dict(steps=s,updates=(s-12800)//4,primary_mode='zero_z',env_seeds=[seed]*10,policy_seeds=[seed+100]*10,zero_z=dict(success_rate=.5,episodes=[{}]*10)) for s in steps]
                (o/'evaluations.jsonl').write_text('\n'.join(map(json.dumps,evaluations))+'\n')
                rows.append(dict(job=j,output=o,done=dict(completed=True,steps=5000192,updates=(5000192-12800)//4)))
        assert choose(rows[:-1],t) is None
        incomplete=copy.deepcopy(rows);incomplete[0]['done']=None;assert choose(incomplete,t) is None
        assert choose(rows,t)['target_entropy_per_dim']==-.9
        for r in rows:
            if r['job']['dacer_target_entropy_per_dim']==.5:
                f=r['output']/'evaluations.jsonl';es=[json.loads(l) for l in f.read_text().splitlines()]
                for e in es:e['zero_z']['success_rate']=.8
                f.write_text('\n'.join(map(json.dumps,es))+'\n')
        assert choose(rows,t)['target_entropy_per_dim']==.5
        f=rows[-1]['output']/'evaluations.jsonl';es=[json.loads(l) for l in f.read_text().splitlines()];es[-1]['env_seeds'][0]+=1;f.write_text('\n'.join(map(json.dumps,es))+'\n')
        try:choose(rows,t)
        except AssertionError:pass
        else:raise AssertionError('Mismatched evaluation seeds accepted')
    assert len(protocol['rows'])==104
    result=dict(passed=True,checks=['temperature completion dependency','entropy completion dependency','all original reference IDs','incomplete grid blocked','tie chooses lower entropy','winner selected on pilot only','mismatched evaluation seed rejected','104 unique queued candidates'])
    save(P/'queue-check.json',result);print(json.dumps(result))
if __name__=='__main__':main()
