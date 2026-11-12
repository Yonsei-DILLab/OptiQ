"""Freeze parameter-only source variants and append-only dependency queue specs."""
from pathlib import Path
import json, hashlib, shutil, copy, ast, difflib

P=Path(__file__).resolve().parent
OLD=P.parent/'fetch-adroit-f1-a1-temp-20261112'
R='/workspace/heechan-unified-f1-a1-20261112'
TARGETS=[-.9,0.,.25,.5,.75]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def replace(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)

provenance={}
for variant,base in [('fetch',OLD/'fetch-source'),('fetch-slide-20cm',OLD/'slide-20cm-20261112/fetch-slide-20cm-source'),('adroit',OLD/'adroit-source')]:
    dest=P/('entropy-'+variant+'-source');dest.mkdir(exist_ok=True)
    original=json.loads((base/'source-sha256.json').read_text())
    for name,h in original.items():
        assert digest(base/name)==h,name
        out=dest/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/name,out)
    name='adroit_rebuttal/run.py' if variant=='adroit' else 'rebuttal_tools/fetch_suite_5m_worker.py'
    s=(dest/name).read_text();before=s
    if variant=='adroit':
        s=replace(s,'            validate_config(cfg)\n',"            validate_config(cfg)\n            target = float(spec['dacer_target_entropy_per_dim'])\n            assert target in [-.9, 0., .25, .5, .75]\n            cfg.dacer.target_entropy_per_dim = target\n            assert cfg.dacer.enabled and cfg.dacer.noise_scale == .1\n")
        s=replace(s,'                backend=jax.default_backend())',"                dacer_target_entropy_per_dim=target,\n                dacer_total_target_entropy=target * int(env.action_space.shape[0]),\n                backend=jax.default_backend())")
    else:
        s=replace(s,"    temperature = spec['temperature']","    temperature = spec['temperature']\n    target = float(spec['dacer_target_entropy_per_dim'])\n    assert target in [-.9, 0., .25, .5, .75]")
        s=replace(s,'dacer_target_entropy_per_dim=-.9, dacer_noise_scale=2.,','dacer_target_entropy_per_dim=target, dacer_total_target_entropy=4*target, dacer_noise_scale=2.,')
        s=replace(s,"parameter_selection='Fixed five-temperature pilot grid; confirm only on independent seeds'","parameter_selection='Frozen selected T; DACER entropy pilot seeds 5/6; independent confirmation seeds 0-4'")
        s=replace(s,'cfg.dacer.noise_scale=2.; cfg.dacer.target_entropy_per_dim=-.9','cfg.dacer.noise_scale=2.; cfg.dacer.target_entropy_per_dim=target')
        s=replace(s,"'alg.actor.temperature','wandb.group'}","'alg.actor.temperature','wandb.group','dacer.target_entropy_per_dim'}")
    ast.parse(s);(dest/name).write_text(s)
    manifest={n:digest(dest/n) for n in original};write(dest/'source-sha256.json',manifest)
    changed=[n for n in original if manifest[n]!=original[n]];assert changed==[name] or set(changed)=={name}
    (P/(variant+'-source.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='original/'+name,tofile='entropy/'+name)))
    provenance[variant]=dict(parent_manifest_sha256=digest(base/'source-sha256.json'),manifest_sha256=digest(dest/'source-sha256.json'),changed_files=changed)

original_protocol=json.loads((OLD/'protocol.json').read_text())
rows=[];assignments={h:[] for h in [1,3,4,5]}
tasks=[('fetch',t) for t in ['reach','push','pick-and-place','slide']]+[('adroit',t) for t in ['door','hammer','pen','relocate']]
lanes={'fetch':[1,4,1,3],'adroit':[4,3,4,5]}
indices={'fetch':0,'adroit':0}
for family,task in tasks:
    template=next(j for j in original_protocol['rows'] if j['family']==family and j['task']==task and j['phase']=='pilot')
    variant='fetch-slide-20cm' if task=='slide' else family
    prefix=f'{family}-{task}'+('-r20cm' if task=='slide' else '')+'-ibolt-entropy'
    for phase,seeds,targets in [('entropy_pilot',[5,6],TARGETS[1:]),('entropy_confirm',list(range(5)),[None])]:
        for target in targets:
            for seed in seeds:
                h=lanes[family][indices[family]%4];indices[family]+=1
                tag='selected' if target is None else 'h'+str(target).replace('.','p')
                jid=f'{prefix}-{tag}-s{seed}-'+('pilot' if phase=='entropy_pilot' else 'confirm')
                j=copy.deepcopy(template)
                j.update(id=jid,host=h,seed=seed,temperature=None,dacer_target_entropy_per_dim=target,
                         phase=phase,status='waiting_temperature' if phase=='entropy_pilot' else 'waiting_entropy',
                         output=R+'/runs/'+jid,source=R+'/entropy-'+variant+'-source',
                         source_manifest=R+'/entropy-'+variant+'-source/source-sha256.json',
                         source_manifest_sha256=provenance[variant]['manifest_sha256'],
                         campaign='fetch-adroit-f1-a1-entropy-20261112',
                         dependency_selection=family+'-'+task,steps=5000192)
                rows.append(j);assignments[h].append(j)
protocol=dict(campaign='fetch-adroit-f1-a1-entropy-20261112',root=R,central=original_protocol['central'],
    actor_temperature='Frozen original temperature-pilot winner, after all ten temperature pilots finish',
    target_entropy_per_dim=TARGETS,total_entropy='target_per_dim * actual environment action dimension',
    pilot_seeds=[5,6],confirmation_seeds=list(range(5)),steps=5000192,
    selection_rule='Mean of last five exact common regular primary-success evaluations across seeds 5/6; ties smaller entropy target',
    reference_reuse='Selected-temperature original -0.9 pilot pair; if -0.9 wins, original matching confirmation runs',
    pilot_runs=80,reused_pilot_runs=16,new_pilot_runs=64,confirmation_runs=40,
    unchanged=dict(f1=original_protocol['f1'],a1=original_protocol['a1'],dacer_alpha_lr=.03,initial_alpha=.27,interval_updates=10000),
    sources=provenance,rows=rows,original_experiments_preserved=True)
assert len(rows)==104 and len({j['id'] for j in rows})==104
write(P/'protocol.json',protocol)
for h,jobs in assignments.items():write(P/f'additions-host{h}.json',jobs)

controller=(OLD/'controller.py').read_text()
controller=replace(controller,"   if j['status']=='waiting_selection':", "   if j['status'] in ('waiting_temperature','waiting_entropy'):\n    from entropy_dependencies import resolve_dependency\n    resolve_dependency(j,ROOT,read)\n   if j['status']=='waiting_selection':")
controller=replace(controller,"   if j['phase']=='smoke':return True", "   if j['phase'].startswith('entropy_'):\n    proof=read(ROOT/'entropy-archive-ready.json',{})\n    if not proof.get('passed') or proof.get('source_manifests',{}).get(Path(j['source']).name)!=j['source_manifest_sha256']:return False\n   if j['phase']=='smoke':return True")
controller=replace(controller,"   assert cfg['dacer']['enabled'] and cfg['temperature']", "   if j['phase'].startswith('entropy_'):\n    assert cfg['dacer']['target_entropy_per_dim']==j['dacer_target_entropy_per_dim']\n    assert cfg['dacer']['noise_scale']==(2. if j['family']=='fetch' else .1)\n    assert cfg['dacer']['alpha_lr']==.03 and cfg['dacer']['initial_alpha']==.27 and cfg['dacer']['interval_updates']==10000\n   assert cfg['dacer']['enabled'] and cfg['temperature']")
controller=replace(controller," source=Path(j.get('source',ROOT/(j['family']+'-source')));env=", " source=Path(j.get('source',ROOT/(j['family']+'-source')))\n if j['phase'].startswith('entropy_'):assert hashlib.sha256(Path(j['source_manifest']).read_bytes()).hexdigest()==j['source_manifest_sha256']\n env=")
controller=replace(controller,"    for g in gpu_state():", "    for g in gpu_state():\n     if g['index'] in p.get('excluded_gpus',[]):continue")
ast.parse(controller);(P/'controller.py').write_text(controller)
gateway=(OLD/'archive_gateway.py').read_text()
addition="""elif args==['entropy-selection']:
 d=json.load(sys.stdin)
 assert d['family'] in ['fetch','adroit'] and d['temperature'] in [.05,.1,.25,.5,1.]
 assert d['target_entropy_per_dim'] in [-.9,0.,.25,.5,.75] and d['pilot_seeds']==[5,6] and d['expected_steps']==5000192
 key=d['family']+'-'+d['task'];assert key in json.loads((BASE/'entropy-protocol.json').read_text())['task_keys']
 ts=json.loads((BASE/'selections'/(key+'.json')).read_text());assert ts['temperature']==d['temperature']
 p=BASE/'entropy-selections';p.mkdir(exist_ok=True);dest=p/(key+'.json')
 if dest.exists():assert json.loads(dest.read_text())==d
 else:
  tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(dest)
 print('entropy selection saved')
elif args==['entropy-archive-ready']:
 d=json.load(sys.stdin);assert d['passed']
 assert d['source_manifests']==json.loads((BASE/'entropy-transfer-proof.json').read_text())['source_manifests']
 assert d['protocol_sha256']==hashfile(BASE/'entropy-protocol.json')
 tmp=BASE/'entropy-archive-ready.tmp';tmp.write_text(json.dumps(d,indent=2));tmp.replace(BASE/'entropy-archive-ready.json')
 print('entropy archive verified')
"""
gateway=replace(gateway,"else:raise SystemExit('Restricted archive command rejected')",addition+"else:raise SystemExit('Restricted archive command rejected')")
ast.parse(gateway);(P/'archive_gateway.py').write_text(gateway)
protocol['task_keys']=[f'{f}-{t}' for f,t in tasks];write(P/'protocol.json',protocol)
write(P/'build-audit.json',dict(passed=True,source_changes=provenance,new_jobs=len(rows),by_host={h:len(v) for h,v in assignments.items()}))
print(json.dumps(json.loads((P/'build-audit.json').read_text()),indent=2))
