"""Validate actual new configurations against preserved DACER-off T1 controls."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = '5baa5b3463416cdfdcad465e8f862f3729802568'
CONTROL = '484f92e7d6d34c964d85b4493ff17c5a9ebcf32e'
reference_root = ROOT.parent / 'antmaze_dacer_off_t1/reports/20260924T085941Z'
controls = {}
for task in ('v1', 'v3', 'v4'):
    folder = next(reference_root.glob(f'*/runs/{task}-optiq-dacer-off-T1-s0'))
    parameter_path = folder / 'parameter-audit.json'
    if parameter_path.exists():
        parameters = json.loads(parameter_path.read_text())
    else:
        old_snapshot = json.loads((ROOT.parent / 'antmaze_dacer_off_t1' /
                                   (folder.parents[1].name + '.json')).read_text())
        parameters = old_snapshot[f'preflight/{task}-optiq-dacer-off-T1-s0/parameter-audit.json']
    controls[task] = (json.loads((folder / 'config.json').read_text()),
                      parameters)
jobs, running, pending, preflight = [], [], [], []
for host in ('vast-heechan-180', 'vast-heechan-199'):
    data = json.loads((ROOT / (host + '.json')).read_text())
    manifest, status = data['manifest.json'], data['status.json']
    assert manifest['source_commit'] == data['frozen_head'] == data['branch_head'] == SOURCE
    assert not data['frozen_status']
    assert not status['failed'] and not status['pending_held'] and 'failure.json' not in data
    assert manifest['replay_capacity'] == 1000000
    gpu_pids = {int(line.split(',')[0]) for line in data['gpu_processes'].splitlines() if line.strip()}
    for entry in manifest['jobs']:
        jobs.append((entry['task'], entry['temperature']))
        assert entry['method'] == 'optiq' and entry['dacer'] == 'off'
        assert entry['reward_profile'] == 'dense' and entry['noveld'] == 'off'
        if entry['id'] in status['pending']:
            pending.append(dict(host=host, id=entry['id']))
            continue
        ref, initial = controls[entry['task']]
        assert ref['source_commit'] == CONTROL
        for phase in ('preflight', 'runs'):
            prefix = f'{phase}/{entry["id"]}/'
            config = data[prefix + 'config.json']
            assert config['source_commit'] == SOURCE and config['seed'] == 0
            assert config['temperature'] == entry['temperature']
            assert config['temperature_schedule'] is None
            assert config['dacer_enabled'] is False and config['noveld_enabled'] is False
            alg = copy.deepcopy(config['native']['alg'])
            assert alg['actor']['temperature'] == entry['temperature']
            alg['actor']['temperature'] = 1.
            assert alg == ref['native']['alg'], entry['id']
            assert config['native']['dacer'] == ref['native']['dacer']
            for key in ('num_envs', 'batch_size', 'updates_per_vector_step', 'warmup_transitions',
                        'reward_profile', 'noveld_enabled', 'eval_starts', 'random_init',
                        'eval_interval', 'interim_eval_episodes', 'final_eval_episodes',
                        'save_intermediate_policy', 'policy_checkpoint_interval'):
                assert config[key] == ref[key], (entry['id'], key)
            proof = data[prefix + 'dacer-disabled-verification.json']
            assert proof['verified'] and proof['train_equals_direct_policy_at_same_rng']
            assert proof['extra_noise_std'] == proof['regulator_updates'] == 0
            profile = data[prefix + 'optiq-profile-verification.json']
            assert profile['verified']
            if phase == 'preflight':
                result = data[prefix + 'result.json']
                assert result['completed'] and result['steps'] == 8448 and result['updates'] == 8
                assert result['rnd_updates'] == result['dacer_updates'] == result['dacer_noise_std'] == 0
                assert result['checkpoint']['dense_replay_verified']
                assert data[prefix + 'parameter-audit.json']['initial'] == initial['initial']
                preflight.append(entry['id'])
            else:
                assert config['steps'] == ref['steps'] == entry['steps']
                progress, wb = data[prefix + 'progress.json'], data[prefix + 'wandb.json']
                assert progress['updates'] > 8 and progress['step'] > 8448
                assert progress['rnd_updates'] == progress['dacer_updates'] == progress['dacer_noise_std'] == 0
                assert wb['mode'] == 'online' and wb['project'] == config['wandb_project'] == 'antmaze'
                learners = [p for p in data['processes'] if p['pid'] in gpu_pids
                            and 'antmaze_experiments.run ' in p['cmd']
                            and '/runs/' + entry['id'] + ' ' in p['cmd']]
                assert len(learners) == 1, (entry['id'], [p['pid'] for p in learners])
                assert learners[0]['pid'] in gpu_pids
                assert learners[0]['cwd'].endswith(SOURCE)
                gpu = next(j['gpu'] for j in status['running'] if j['id'] == entry['id'])
                running.append(dict(host=host, id=entry['id'], gpu=gpu, pid=learners[0]['pid'],
                                    step=progress['step'], updates=progress['updates'],
                                    budget=config['steps'], wandb=wb['url']))
assert len(jobs) == 9 and len(set(jobs)) == 9
assert set(jobs) == {(v, t) for v in ('v1', 'v3', 'v4') for t in (3., 5., 10.)}
assert len(running) == len(preflight) == 8
assert pending == [dict(host='vast-heechan-180', id='v1-optiq-dacer-off-T10-s0')]
result = dict(verified=True, time=datetime.now(timezone.utc).isoformat(), source=SOURCE,
              comparison_source=CONTROL, training=running, pending=pending,
              preflight_passed=preflight, verified_change='Only teacher temperature; all other learning and evaluation settings match T1',
              source_change='Registration, protocol, manifest test and AGENTS only; training source unchanged',
              live_proof='Eight actual GPU learner PIDs, online W&B, positive update counts, no DACER or NovelD updates',
              queued_job='Registered once; preflight will execute when the first host180 GPU is free')
(ROOT / 'registration-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
