"""Create an isolated categorical-target ablation from the original campaign.

Run on login4. Does not edit production OptiQ or the previous experiment snapshot.
"""
import argparse, hashlib, json, shutil, time
from pathlib import Path

CASES=['modes2d_8','separable_8','modes2d_4','separable_4','unimodal','asymmetric_1.3_0.08']
ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--out',required=True);ap.add_argument('--scripts',required=True)
a=ap.parse_args();base=Path(a.base);out=Path(a.out);scripts=Path(a.scripts)
out.mkdir(parents=True,exist_ok=False);(out/'runs').mkdir();(out/'logs').mkdir()
shutil.copytree(base/'code',out/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
code=out/'code';algorithm=code/'optiq_dime/algorithm.py';original=algorithm.read_text()
needle='''            elif transport_target_mode == "barycentric":'''
assert original.count(needle)==1
branch='''            elif transport_target_mode == "categorical":
                # Independent draw per row. Preserve the original RNG split and
                # returned key so actor/proposal streams match the argmax run.
                target_key = jax.random.fold_in(key, 0x434154)
                row_logits = jnp.where(
                    row_distribution > 0, jnp.log(row_distribution), -jnp.inf
                )
                selected_indices = jax.random.categorical(
                    target_key, row_logits, axis=-1
                )
                selected_actions = jax.vmap(lambda actions, indices: actions[indices])(
                    proposals, selected_indices
                )
'''
algorithm.write_text(original.replace(needle,branch+needle))
frozen=code/'analysis_boltzmann/frozen.py';old=frozen.read_text()
old=old.replace('    state=actor_state(p.dim,args.seed)\n','    state=actor_state(p.dim,args.seed)\n    if args.initial_checkpoint:\n        state=serialization.from_bytes(state,Path(args.initial_checkpoint).read_bytes())\n')
old=old.replace('import argparse,csv,json,time\n','import argparse,csv,json,time\nfrom pathlib import Path\n')
old=old.replace("if args.initialization=='coverage':","if args.initialization=='coverage' and not args.initial_checkpoint:")
old=old.replace("'updates':2000 if args.initialization=='coverage' else 0","'updates':2000 if args.initialization=='coverage' and not args.initial_checkpoint else 0, 'initial_checkpoint':args.initial_checkpoint, 'initialization_reused':bool(args.initial_checkpoint)")
old=old.replace("    run(ap.parse_args())", "    ap.add_argument('--initial-checkpoint',default=None)\n    run(ap.parse_args())")
old=old.replace('    def diagnose(step):','    def diagnose(step, distribution_only=False):')
needle="        (out/f'actor_{step}.msgpack').write_bytes(serialization.to_bytes(state))\n"
assert old.count(needle)==1
old=old.replace(needle,needle+'''        if distribution_only:
            print(json.dumps({'step':step,'elapsed':time.monotonic()-start,'case':args.case,'distribution_only':True}),flush=True)
            return
''')
needle='        if step in checkpoints: diagnose(step)'
assert old.count(needle)==1
old=old.replace(needle,'''        if step in checkpoints: diagnose(step)
        elif step in (1,10,50): diagnose(step,distribution_only=True)
        if step%100==0:
            (out/'training.json').write_text(json.dumps(train,indent=2))''')
frozen.write_text(old)
for name in ['categorical_worker.py','categorical_smoke.py','categorical_summary.py','checkpoint_integrity.py']:
    shutil.copy2(scripts/name,code/'analysis_boltzmann'/name)
tasks=[]
# Obtain paired evidence on a case across all seeds before moving to the next.
for case in CASES:
    for seed in range(5):
        for init in ['default','coverage']:
            name=f'frozen_{case}_{init}_seed{seed}';initial=base/'runs'/name/'actor_0.msgpack'
            assert initial.exists() and (initial.parent/'COMPLETE').exists(),initial
            manifest=json.loads((initial.parent/'manifest.json').read_text())
            assert manifest['source_sha256']['optiq_dime/algorithm.py']==hashlib.sha256(original.encode()).hexdigest(), 'Original training algorithm snapshot differs'
            tasks.append(dict(case=case,seed=seed,initialization=init,initial_checkpoint=str(initial),
                              initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),out=str(out/'runs'/name)))
(out/'tasks.json').write_text(json.dumps(tasks,indent=2))
protocol=dict(created=time.time(),baseline=str(base),seeds=list(range(5)),cases=CASES,initializations=['default','coverage'],
    changed_parameter='transport_target_mode: argmax -> categorical',sinkhorn_iterations=30,temperature=.25,
    actor_updates=20000,policy_samples=16,proposals_per_policy_sample=4,repetitions=2000,ks=[1,8,16,50,64,256,1024],
    early_distribution_checkpoints=[1,10,50],full_checkpoints=[0,100,1000,5000,20000],initial_checkpoint_reused=True,
    target_rng='jax.random.fold_in(returned_actor_key, 0x434154); original key splitting unchanged',
    configuration_dispatch='Both no-argument config() and positional config(seed) set explicitly; actual frozen wrapper validated',
    max_concurrent_gpus=2,expansion_enabled=False,production_code_changed=False,
    original_algorithm_sha256=hashlib.sha256(original.encode()).hexdigest(),
    categorical_algorithm_sha256=hashlib.sha256(algorithm.read_bytes()).hexdigest())
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
print(json.dumps(dict(campaign=str(out),tasks=len(tasks),protocol=protocol),indent=2))
