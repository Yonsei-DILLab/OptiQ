"""Collect provenance and original fixed-probe samples, never infer missing states."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

CASES = ['t00_reference', 'n00_spike_ramp', 'n07_spike_flat_ramp',
         't01_two_offset', 't05_unequal_mass']
OLD_STEPS = [0, 1000, 10000, 25000, 50000, 75000, 100000]
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def original_folder(workspace, case, method, seed, step):
    if step == 100000:
        return workspace/'studies/20260926_kl_appendix_six/parent/runtime/confirm'/case/f'{method}_s{seed}'
    if method == 'forward':
        group = 'forward_nongmm' if case.startswith('n') else 'forward_gmm'
        stage = 'screen' if seed < 2 else 'validate_seeds'
        suffix = Path(stage)/case/f'{method}_s{seed}'
    else:
        group = 'reverse_main' if case in CASES[:2] else 'reverse_vast'
        suffix = Path(case)/f'{method}_s{seed}'
    return workspace/'studies/20260926_kl_five_progress/inputs'/group/suffix

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    tasks=[];probes={}
    for method in ['forward','reverse']:
        for case in CASES:
            for seed in range(4):
                folder=original_folder(a.workspace,case,method,seed,100000)
                run=read(folder/'RUN.json');cfg=run['config'].copy()
                for k in ['cases','authorization','validation_seeds','seeds']: cfg.pop(k,None)
                assert cfg['n']==cfg['m']==128 and cfg['batch']==32
                assert run['L']==(1048576 if method=='reverse' else 0)
                task=dict(index=len(tasks),case=case,method=method,seed=seed,L=run['L'],
                    config=cfg,original_commit=run['source_commit'],
                    initial_parameter_sha256=run['initial_parameter_sha256'],
                    original_final_checkpoint_sha256=sha(folder/'checkpoint.msgpack'),
                    original_run_file=str(folder/'RUN.json'),original_run_sha256=sha(folder/'RUN.json'))
                tasks.append(task)
                for step in OLD_STEPS:
                    file=original_folder(a.workspace,case,method,seed,step)/f'samples_{step:06d}.npz'
                    with np.load(file) as z:
                        count=len(z['actions']);mass=z['histogram_mass']
                        assert len(mass)==512 and np.isclose(mass.sum(),1)
                        assert np.allclose(np.histogram(z['actions'],z['edges'])[0]/count,mass)
                        probes[f'{task["index"]}:{step}']=dict(file=str(file),sha256=sha(file),
                            count=count,probe_draw_count=16384 if count>32768 else count,
                            actions=z['actions'].ravel()[:128].tolist(),
                            mu=z['mu'].ravel()[:128].tolist(),sigma=z['sigma'].ravel()[:128].tolist(),
                            histogram_mass=mass.tolist())
    (a.out/'TASKS.json').write_text(json.dumps(tasks,indent=2)+'\n')
    (a.out/'REFERENCE.json').write_text(json.dumps(probes,indent=2)+'\n')
    (a.out/'INPUT_MANIFEST.json').write_text(json.dumps({p.name:sha(p) for p in a.out.glob('*.json')
        if p.name!='INPUT_MANIFEST.json'},indent=2)+'\n')
    print(f'{len(tasks)} tasks; {len(probes)} original time points verified')
if __name__=='__main__': main()
