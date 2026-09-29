"""Verify the live OFF campaign against preserved T1 controls, without mutation."""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = '484f92e7d6d34c964d85b4493ff17c5a9ebcf32e'
controls = Path((ROOT.parent/'antmaze_anneal_baselines/latest-report-path.txt').read_text().strip())
reference = {}
for task in ('v1', 'v2', 'v3', 'v4'):
    run = next(controls.glob(f'*/antmaze-optiq-dense-off-T1-s0-20260924/runs/{task}-optiq-s0'))
    reference[task] = (json.loads((run/'config.json').read_text()),
                       json.loads((run/'parameter-audit.json').read_text()))
records = []
for host in ('vast-heechan-180', 'vast-heechan-199'):
    data = json.loads((ROOT/(host+'.json')).read_text())
    manifest, status = data['manifest.json'], data['status.json']
    assert manifest['source_commit'] == SOURCE == data['branch_head']
    assert not data['branch_status'] and not status['failed'] and not status['pending_held']
    assert 'failure.json' not in data and not status['pending']
    for job in manifest['jobs']:
        assert job['dacer'] == 'off' and job['temperature'] == 1
        assert job['reward_profile'] == 'dense' and job['noveld'] == 'off'
        ref, param = reference[job['task']]
        for phase in ('preflight', 'runs'):
            prefix = f'{phase}/{job["id"]}/'
            cfg = data[prefix+'config.json']
            assert cfg['source_commit'] == SOURCE and not cfg['dacer_enabled']
            assert cfg['temperature_schedule'] is None
            assert cfg['native']['alg'] == ref['native']['alg']
            dc = copy.deepcopy(cfg['native']['dacer'])
            assert not dc['enabled']
            dc['enabled'] = True
            assert dc == ref['native']['dacer']
            for key in ('num_envs', 'batch_size', 'updates_per_vector_step',
                        'warmup_transitions', 'reward_profile', 'noveld_enabled', 'eval_starts'):
                assert cfg[key] == ref[key], key
            proof = data[prefix+'dacer-disabled-verification.json']
            assert proof['verified'] and proof['train_equals_direct_policy_at_same_rng']
            assert proof['state_serializable_without_regulator']
            assert proof['extra_noise_std'] == proof['regulator_updates'] == 0
            assert prefix+'dacer_regulator.json' not in data
            if phase == 'preflight':
                result = data[prefix+'result.json']
                assert result['completed'] and result['steps'] == 8448 and result['updates'] == 8
                assert result['rnd_updates'] == result['dacer_updates'] == result['dacer_noise_std'] == 0
                assert result['checkpoint']['dense_replay_verified']
                for key in ('actor', 'critic'):
                    assert data[prefix+'parameter-audit.json']['initial'][key] == param['initial'][key]
            else:
                for key in ('steps', 'eval_interval', 'interim_eval_episodes',
                            'final_eval_episodes', 'save_intermediate_policy'):
                    assert cfg[key] == ref[key], key
                progress, wb = data[prefix+'progress.json'], data[prefix+'wandb.json']
                assert progress['updates'] > 8 and not progress['dacer_enabled']
                assert progress['dacer_updates'] == progress['dacer_noise_std'] == progress['rnd_updates'] == 0
                assert wb['mode'] == 'online' and wb['project'] == 'antmaze'
                records.append(dict(host=host, id=job['id'], task=job['task'],
                                    step=progress['step'], updates=progress['updates'],
                                    budget=cfg['steps'], wandb=wb['url']))
assert len(records) == 4 and {r['task'] for r in records} == {'v1','v2','v3','v4'}
result = dict(verified=True, time=datetime.now(timezone.utc).isoformat(), source=SOURCE,
              training=records, preflight_passed=4,
              control_comparison='Native model/optimizer/reward/evaluation and initial actor/critic hashes match T1 controls; only dacer.enabled changed.',
              live_proof='Zero regulator/noise/RND updates; actual train actions equal direct-policy actions at identical restored RNG. Conditional sigma and random latent preserved.')
(ROOT/'registration-verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
