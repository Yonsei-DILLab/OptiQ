"""Select only after all five entropy targets finish on both pilot seeds."""
import json,hashlib,statistics,math
TARGETS=[-.9,0.,.25,.5,.75]
STEPS=5000192
def load(p):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return None
def choose(rows,temperature_selection):
    family=temperature_selection['family'];task=temperature_selection['task'];temp=temperature_selection['temperature']
    candidates={}
    for r in rows:
        j=r['job']
        if (j['family'],j['task'],j['method'])!=(family,task,'ibolt'):continue
        if j['phase']=='pilot' and j['temperature']==temp:h=-.9
        elif j['phase']=='entropy_pilot' and j['temperature']==temp:h=j['dacer_target_entropy_per_dim']
        else:continue
        key=(h,j['seed']);assert key not in candidates,key;candidates[key]=r
    if set(candidates)!={(h,s) for h in TARGETS for s in [5,6]}:return None
    if any(not r['done'] or not r['done'].get('completed') or r['done']['steps']!=STEPS for r in candidates.values()):return None
    curves={};evidence=[];demo_hashes=set()
    for (h,seed),r in candidates.items():
        j=r['job'];o=r['output'];c=load(o/'config.json');cfg=load(o/('mujoco-config.json' if family=='fetch' else 'ibolt-config.json'))
        assert c['seed']==seed and c['batch_size']==256 and c['num_envs']==256
        assert cfg['temperature']==temp and cfg['dacer']['target_entropy_per_dim']==h and cfg['dacer']['enabled']
        assert cfg['dacer']['noise_scale']==(2. if family=='fetch' else .1)
        assert cfg['alg']['actor']['hidden_dims']==cfg['alg']['critic']['hs']==[256,256]
        assert c['utd']==(.125 if family=='fetch' else .25)
        assert r['done']['updates']==int((STEPS-12800)*c['utd'])
        if family=='adroit':
            proof=load(o/'demonstration-validation.json');assert proof['passed'] and proof['replayed'] and proof['successful_trajectories']==25
            demo_hashes.add(c['demonstration_dataset_sha256']);assert c['critic_layer_norm'] and c['offline_pretraining_updates']==0
        elif task=='slide':assert c['distance_threshold_m']==.20 and c['protocol_variant']=='F1-slide-20cm'
        f=o/'evaluations.jsonl';curve={}
        for line in f.read_text().splitlines():
            e=json.loads(line);v=e[e['primary_mode']];n=v['episodes'];n=len(n) if isinstance(n,list) else n
            if n==10 and e['steps']>0:
                assert math.isfinite(v['success_rate']) and 0<=v['success_rate']<=1
                curve[e['steps']]=e
        curves[h,seed]=curve
        evidence.append(dict(id=j['id'],target_entropy_per_dim=h,seed=seed,
                             evaluations_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                             config_sha256=hashlib.sha256((o/'config.json').read_bytes()).hexdigest()))
    if family=='adroit':assert len(demo_hashes)==1
    common=sorted(set.intersection(*(set(c) for c in curves.values())))
    assert len(common)>=5 and common[-1]>4900000
    last=common[-5:]
    for seed in [5,6]:
        for step in common:
            es=[curves[h,seed][step] for h in TARGETS]
            assert len({e['updates'] for e in es})==1 and len({e['primary_mode'] for e in es})==1
            if family=='adroit':
                assert all(e['env_seeds']==es[0]['env_seeds'] and e['policy_seeds']==es[0]['policy_seeds'] for e in es)
    scores=[]
    for h in TARGETS:
        # Every regular evaluation contains exactly ten episodes. Compare the
        # integer success counts so representation error cannot break a tie.
        counts=[]
        for s in [5,6]:
            total=0
            for t in last:
                e=curves[h,s][t];rate=e[e['primary_mode']]['success_rate']
                successes=round(rate*10)
                assert math.isclose(rate*10,successes,rel_tol=0,abs_tol=1e-8)
                total+=successes
            counts.append(total)
        vals=[n/(10*len(last)) for n in counts]
        scores.append(dict(target_entropy_per_dim=h,per_seed=vals,
                           mean=sum(counts)/(20*len(last)),success_counts=counts,
                           total_successes=sum(counts),total_episodes=20*len(last)))
    winner=max(scores,key=lambda x:(x['total_successes'],-x['target_entropy_per_dim']))
    return dict(family=family,task=task,temperature=temp,target_entropy_per_dim=winner['target_entropy_per_dim'],
                pilot_seeds=[5,6],expected_steps=STEPS,common_steps=last,scores=scores,evidence=evidence,
                rule='Last five common regular primary-success evaluations, mean over seeds5/6; ties lower entropy target',
                temperature_selection_sha256=hashlib.sha256(json.dumps(temperature_selection,sort_keys=True).encode()).hexdigest(),
                evaluation_pairing='Exact update counts and modes; Adroit env/policy seed arrays equal; Fetch unchanged evaluator seed=700000+1000*training_seed')
