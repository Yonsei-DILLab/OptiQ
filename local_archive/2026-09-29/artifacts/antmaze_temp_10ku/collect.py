"""Archive the two frozen AntMaze temperature shards after all eight runs finish."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = 'antmaze-v1-optiq-temp-10ku-s0-20260923'
SHADOW_NAME = 'antmaze-v1-optiq-temp-10ku-samehost180-s0-20260923'
SOURCE = 'a8250c4cde21c8748f99c33329b0540bf54b19a3'

for label, host, name in (
    ('vast-heechan-180', 'vast-heechan-180', NAME),
    ('vast-heechan-199', 'vast-heechan-199', NAME),
    ('vast-heechan-180-shadow', 'vast-heechan-180', SHADOW_NAME),
):
    shard = ROOT / label
    shard.mkdir(exist_ok=True)
    remote = f'/home/heechan/optiq-experiments/{name}/'
    subprocess.run(['rsync', '-az', '--exclude=wandb/', '--exclude=checkpoint*.pt',
                    f'{host}:{remote}', str(shard) + '/'], check=True)
    remote_hashes = subprocess.check_output(['ssh', host,
        f'sha256sum {remote}runs/*/checkpoint-final.pt'], text=True)
    hashes = {Path(line.split(maxsplit=1)[1]).parent.name: line.split()[0]
              for line in remote_hashes.splitlines() if line}
    manifest = json.loads((shard / 'manifest.json').read_text())
    status = json.loads((shard / 'status.json').read_text())
    assert manifest['source_commit'] == status['source_commit'] == SOURCE
    assert len(status['completed']) == 4 and not status['failed'] and not status['running']
    assert not status['pending']
    for job in manifest['jobs']:
        key = job['id']
        path = shard / 'runs' / key
        config = json.loads((path / 'config.json').read_text())
        result = json.loads((path / 'result.json').read_text())
        proof = json.loads((path / 'checkpoint-verification.json').read_text())
        assert config['source_commit'] == result['source_commit'] == SOURCE
        assert config['temperature'] == config['native']['alg']['actor']['temperature'] == job['temperature']
        assert config['steps'] == result['steps'] == 328192
        assert config['expected_updates'] == result['updates'] == 10000
        assert result['rnd_updates'] == 10000
        assert proof['sha256'] == result['checkpoint']['sha256']
        assert proof['readback_verified'] and proof['sparse_replay_verified']
        digest = hashes[key]
        assert digest == proof['sha256']
        for eval_label in ('native-natural', 'native-fixed', 'policy-natural',
                           'policy-fixed', 'zero_z-natural', 'zero_z-fixed'):
            base = path / 'evaluations' / '0000328192' / eval_label
            assert (base / 'rollouts.npz').exists()
            assert json.loads((base / 'summary.json').read_text())['episodes'] == 100
        print(json.dumps(dict(host=host, shard=label, job=key, checkpoint_sha256=digest,
                              steps=result['steps'], updates=result['updates'])))

initial = []
for label in ('vast-heechan-180', 'vast-heechan-180-shadow'):
    for audit in (ROOT / label / 'preflight').glob('*/parameter-audit.json'):
        record = json.loads(audit.read_text())['initial']
        initial.append(tuple(record[x]['sha256'] for x in
                             ('actor', 'critic', 'rnd_predictor', 'rnd_target')))
assert len(initial) == 8 and len(set(initial)) == 1, 'Same-host initial weights differ'
print('Eight same-host initial actor/critic/RND weights match exactly.')
