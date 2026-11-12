"""Relocate only disk-blocked, never-started jobs to validated free workers."""
from pathlib import Path
import sys, json, subprocess, hashlib, copy, collections, time

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parent))
from deploy import freeze, resume
R = '/workspace/heechan-unified-f1-a1-20261112'
TAG = P.name
HOSTS = [3, 4, 5]

def save(n, d):
    (P / n).write_text(json.dumps(d, indent=2))

def remote(h, code, **kw):
    data = dict(ROOT=R, TAG=TAG, HOST=h, **kw)
    script = '\n'.join(f'{k}={v!r}' for k, v in data.items()) + '\n' + code
    r = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                        f'ibolt-server-{h}', 'python3 -'], input=script, text=True,
                       capture_output=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout)

mutation = r'''
from pathlib import Path
import json,fcntl,time,collections
p=Path(ROOT);assert (p/'PAUSE').read_text().startswith(TAG)
lock=(p/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old=json.loads((p/'plan.json').read_text());assert old==BEFORE
old_ids={j['id'] for j in old['jobs']};new_ids={j['id'] for j in AFTER['jobs']}
for j in old['jobs']+AFTER['jobs']:
 if j['id'] in old_ids^new_ids:
  assert j['status']=='queued' and not j.get('pid')
  assert not Path(j['output']).exists() and not (p/'specs'/(j['id']+'.json')).exists()
AFTER['counts']=dict(collections.Counter(j['status'] for j in AFTER['jobs']))
AFTER['updated_unix']=time.time()
(p/TAG/'movement.json').write_text(json.dumps(MOVEMENT,indent=2))
tmp=p/'plan.rebalance.tmp';tmp.write_text(json.dumps(AFTER,indent=2));tmp.replace(p/'plan.json')
print(json.dumps(dict(host=HOST,counts=AFTER['counts'])))
'''

if __name__ == '__main__':
    before = {}
    committed = False
    try:
        for h in HOSTS:
            pre = json.loads((P / f'preflight-host{h}.json').read_text())
            remote(h, "from pathlib import Path\n(Path(ROOT)/TAG).mkdir(exist_ok=True)\nprint('{}')")
            before[h] = remote(h, freeze, OLD_CONTROLLER=pre['hashes']['controller.py'],
                               OLD_GATEWAY=pre['hashes']['archive_gateway.py'])
            save(f'before-host{h}.json', before[h])
            print('HELD', h, flush=True)
        after = copy.deepcopy(before)
        jobs = [j for j in before[3]['jobs'] if j['status'] == 'queued'
                and j['id'] != 'adroit-pen-ibolt-entropy-selected-s3-confirm']
        assert len(jobs) == 15
        moved = []
        for j in jobs:
            assert not j.get('pid')
            h = 4 if j['family'] == 'fetch' else 5
            assert not any(x['id'] == j['id'] for x in after[h]['jobs'])
            variant = Path(j['source']).name.removesuffix('-source')
            assert before[3]['source_manifests'][variant] == before[h]['source_manifests'][variant]
            if j['family'] == 'adroit':
                assert before[3]['datasets'][j['task']] == before[h]['datasets'][j['task']]
            q = copy.deepcopy(j)
            q['host'] = h
            q.setdefault('placement_history', []).append(dict(from_host=3, to_host=h, at=time.time(), reason=TAG))
            after[h]['jobs'].append(q)
            after[3]['jobs'] = [x for x in after[3]['jobs'] if x['id'] != j['id']]
            moved.append(dict(id=j['id'], from_host=3, to_host=h, family=j['family']))
        ids_before = [j['id'] for p in before.values() for j in p['jobs']]
        ids_after = [j['id'] for p in after.values() for j in p['jobs']]
        assert len(ids_after) == len(set(ids_after)) and set(ids_before) == set(ids_after)
        for h in HOSTS:
            assert [j for j in before[h]['jobs'] if j['status']=='running'] == [j for j in after[h]['jobs'] if j['status']=='running']
            assert sum(j['status'] in ['running','queued'] for j in after[h]['jobs']) <= (15 if h==5 else 20)
        movement = dict(at=time.time(), reason='Source disk admission gate; preserve learners, source and dataset hashes', moves=moved)
        save('movement.json', movement)
        for h in HOSTS:
            save(f'prepared-host{h}.json', after[h])
            print(remote(h, mutation, BEFORE=before[h], AFTER=after[h], MOVEMENT=movement), flush=True)
        # Verify all three committed plans before allowing any admission.
        for h in HOSTS:
            live = remote(h, "from pathlib import Path\nprint((Path(ROOT)/'plan.json').read_text())")
            assert live['jobs'] == after[h]['jobs']
        committed = True
        save('committed.json', dict(passed=True, at=time.time(), moved=15, learners_preserved=True))
    finally:
        if not committed:
            # All controllers are still paused: restore every held plan, including
            # an ambiguous transport failure after a possible target write.
            for h, old in before.items():
                remote(h, "from pathlib import Path\nimport json\np=Path(ROOT);t=p/'plan.restore.tmp';t.write_text(json.dumps(OLD,indent=2));t.replace(p/'plan.json');print('{}')", OLD=old)
        for h in before:
            print(remote(h, resume), flush=True)
