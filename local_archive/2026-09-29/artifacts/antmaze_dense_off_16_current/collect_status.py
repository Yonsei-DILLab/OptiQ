"""Read-only snapshots for the sixteen-policy dense/NovelD-off campaign."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
CAMPAIGN = 'antmaze-dense-off-16-current-s0-20260924'
SOURCE = 'a70e6bb1c59407200f7ff2a65cd09b21fff0c1f3'
HOSTS = ['vast-heechan-180', 'vast-heechan-199']
REMOTE = r'''
import json, os, pathlib, subprocess, datetime
root=pathlib.Path('/home/heechan/optiq-experiments/antmaze-dense-off-16-current-s0-20260924')
def read(path):
 return json.loads(path.read_text()) if path.exists() else None
out={'collected_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for name in ['manifest.json','status.json','failure.json','registration.json','result.json']:
 out[name]=read(root/name)
out['jobs']={}
for entry in out['manifest.json']['jobs']:
 jid=entry['id'];row={'job':read(root/'jobs'/f'{jid}.json')}
 for phase in ['preflight','runs']:
  folder=root/phase/jid
  if not folder.exists():continue
  row[phase]={}
  for name in ['config.json','progress.json','result.json','failure.json','wandb.json',
               'checkpoint-verification.json','parameter-audit.json','optiq-profile-verification.json']:
   value=read(folder/name)
   if value is not None:row[phase][name]=value
  if (folder/'policy-checkpoints').exists():
   row[phase]['policy_proofs']=[read(p) for p in sorted((folder/'policy-checkpoints').glob('*/verification.json'))]
 out['jobs'][jid]=row
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
out['gpu_processes']=[]
for line in gpu.splitlines():
 parts=[x.strip() for x in line.split(',')]
 try:command=pathlib.Path('/proc/'+parts[0]+'/cmdline').read_bytes().replace(b'\x00',b' ').decode()
 except FileNotFoundError:command='exited during snapshot'
 out['gpu_processes'].append(dict(pid=int(parts[0]),gpu=parts[1],memory=parts[2],command=command))
for name in ['controller.log','controller.err']:
 p=root/name;out[name]=p.read_text()[-5000:] if p.exists() else None
print(json.dumps(out))
'''


def collect(host):
    proc = subprocess.run(['ssh', host, 'python3 - <<\'PY\'\n' + REMOTE + '\nPY'],
                          capture_output=True, text=True, check=True)
    # The SSH greeting is on stderr; stdout contains only this JSON record.
    return host, json.loads(proc.stdout)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--require-initial-learning', action='store_true')
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = ROOT / 'snapshots' / timestamp
    destination.mkdir(parents=True, exist_ok=False)
    (ROOT / 'latest').mkdir(exist_ok=True)
    snapshots = dict(ThreadPoolExecutor(max_workers=2).map(collect, HOSTS))
    rows = []
    for host, data in snapshots.items():
        serialized = json.dumps(data, indent=2) + '\n'
        (destination / (host + '.json')).write_text(serialized)
        (ROOT / 'latest' / (host + '.json')).write_text(serialized)
        manifest, status = data['manifest.json'], data['status.json']
        assert manifest['source_commit'] == status['source_commit'] == SOURCE
        assert manifest['wandb_project'] == 'antmaze'
        assert len(manifest['jobs']) == 8
        assert not status['failed'] and data['failure.json'] is None
        for entry in manifest['jobs']:
            jid = entry['id']; row = data['jobs'][jid]
            for phase in ['preflight', 'runs']:
                phase_data = row.get(phase, {})
                cfg = phase_data.get('config.json')
                if not cfg:
                    continue
                assert cfg['source_commit'] == SOURCE and cfg['reward_profile'] == 'dense'
                assert cfg['noveld_enabled'] is False and cfg['noveld_coefficient'] == 0
                assert cfg['num_envs'] == 256 and cfg['batch_size'] == 4096
                assert cfg['updates_per_vector_step'] == 8 and cfg['interim_eval_episodes'] == 40
                assert cfg['final_eval_episodes'] == 100 and cfg['eval_starts'] == 'random'
                assert cfg['wandb_project'] == 'antmaze'
                progress = phase_data.get('progress.json')
                if progress:
                    assert progress['rnd_updates'] == 0
                proof = phase_data.get('checkpoint-verification.json')
                if proof:
                    assert proof['readback_verified'] and proof['dense_replay_verified']
                    assert proof['intrinsic_enabled'] is False
                audit = phase_data.get('parameter-audit.json')
                if audit:
                    assert audit['passed'] and set(audit['final']) == {'actor', 'critic'}
                if entry['method'] == 'optiq':
                    profile = phase_data['optiq-profile-verification.json']
                    assert profile['verified'] and profile['rnd_lrs'] == []
                    alg = cfg['native']['alg']
                    assert alg['actor']['hidden_dims'] == alg['critic']['hs'] == [256]*3
                    assert alg['actor']['temperature'] == .01
                    assert alg['optimizer']['lr_actor'] == 3e-4 and alg['optimizer']['lr_critic'] == 5e-4
                elif entry['method'] == 'dipo':
                    assert cfg['native']['algo']['v_min'] == -6000.
                    assert cfg['native']['intrinsic']['enabled'] is False
            if args.require_initial_learning and entry['method'] in ('optiq', 'dipo'):
                pre = row['preflight']['result.json']
                assert pre['completed'] and pre['updates'] == 8 and pre['rnd_updates'] == 0
                assert row['runs']['progress.json']['updates'] > 0
                assert row['runs']['wandb.json']['project'] == 'antmaze'
            rows.append(dict(host=host, id=jid, phase=(row['job'] or {}).get('phase', 'pending'),
                             progress=row.get('runs', {}).get('progress.json'),
                             wandb=row.get('runs', {}).get('wandb.json')))
    output = dict(source_commit=SOURCE, campaign=CAMPAIGN, timestamp_utc=timestamp,
                  verified=True, initial_eight_learning_verified=args.require_initial_learning,
                  hosts={host: data['status.json'] for host, data in snapshots.items()}, runs=rows)
    (ROOT / 'launch-verification.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
