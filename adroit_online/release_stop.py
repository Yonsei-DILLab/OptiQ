"""Stop exactly the two completed Adroit instances; destruction is unsupported."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
import shlex

from .release_monitor import SERVERS, ready, snapshot

FIELDS = ('id', 'actual_status', 'intended_status', 'public_ipaddr', 'num_gpus',
          'gpu_name', 'machine_id', 'disk_space', 'disk_usage')


def validate_gate(plan, audit, now):
    assert plan['final_action'] == 'stop_only' and plan['destroy_authorized'] is False
    assert plan['preserve_instance_and_disk'] is True
    assert audit['passed'] and audit['final_action'] == 'stop_only'
    assert audit['destroy_authorized'] is False and ready(audit['completion'])
    assert 0 <= now - audit['verified_unix'] <= 1800, 'Archive audit is stale'
    receipts = audit['receipts']
    assert len(receipts) == 2 and {r['instance_id'] for r in receipts} == {s[0] for s in SERVERS}
    assert all(r['passed'] and r['files'] > 0 for r in receipts)
    if audit['scope'] == 'essential':
        assert plan['backup_gate'] == 'essential_with_full_source_hash_manifest'
    else:
        assert audit['scope'] == 'full'


def remote_api(server, stop=False):
    instance, host, port = server
    assert instance in (60326900, 60327673)
    # Per-instance keys are source-IP restricted. Keep authentication on that host.
    # The sole supported mutation is PUT state=stopped; no deletion endpoint exists.
    code = '''import json,os,requests
instance=EXPECTED_INSTANCE;stop=STOP_REQUEST
assert int(os.environ['CONTAINER_ID'])==instance
url='https://console.vast.ai/api/v0/instances/'+str(instance)+'/'
headers={'Authorization':'Bearer '+os.environ['CONTAINER_API_KEY']}
if stop:r=requests.put(url,headers=headers,json={'state':'stopped'},timeout=30)
else:r=requests.get(url+'?owner=me',headers=headers,timeout=30)
d=r.json()
if stop:
 out=dict(instance_id=instance,http_status=r.status_code,accepted=d.get('success') is True,requested_state='stopped')
else:
 v=d.get('instances')
 assert r.status_code==200 and isinstance(v,dict) and int(v['id'])==instance
 out={k:v.get(k) for k in ('id','actual_status','intended_status','public_ipaddr','num_gpus','gpu_name','machine_id','disk_space','disk_usage')}
print(json.dumps(out),flush=True)
'''.replace('EXPECTED_INSTANCE', str(instance)).replace('STOP_REQUEST', str(stop))
    result = subprocess.run(['ssh', '-S', 'none', '-o', 'BatchMode=yes', '-o',
        'ConnectTimeout=15', '-p', str(port), 'heejoonorm@' + host,
        'python3 -c ' + shlex.quote(code)], capture_output=True, text=True, timeout=45)
    assert result.stdout.strip(), f'No provider response for instance {instance}'
    return json.loads(result.stdout)


def stopped(value):
    return (value.get('actual_status') == 'stopped' or
            (value.get('actual_status') == 'exited' and value.get('intended_status') == 'stopped'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    plan = json.loads(a.plan.read_text()); audit = json.loads(a.audit.read_text())
    validate_gate(plan, audit, time.time()); a.output.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).with_name('release_probe.py').read_text()
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(lambda s: snapshot(s, code), SERVERS))
    assert ready(reports), 'A fresh completion/idle-GPU check failed'
    before = {}; events = []
    source_commit = json.loads((Path(__file__).resolve().parents[1] / 'source-provenance.json').read_text())['commit']
    for server in SERVERS:
        instance, host, _ = server; value = remote_api(server)
        assert value['public_ipaddr'] == host and value['num_gpus'] == 4
        assert 'RTX5090' in value['gpu_name'].replace(' ', '').replace('_', '')
        assert value['actual_status'] == 'running', (instance, value['actual_status'])
        before[str(instance)] = value
    (a.output / 'before-stop.json').write_text(json.dumps(dict(
        observed_unix=time.time(), final_action='stop_only', destroy_authorized=False,
        source_commit=source_commit, audit_sha256=hashlib.sha256(a.audit.read_bytes()).hexdigest(),
        provider=before, completion=reports), indent=2) + '\n')
    for server in SERVERS:
        event = dict(instance_id=server[0], request_unix=time.time(), requested_state='stopped')
        try: event.update(remote_api(server, stop=True))
        except Exception as error:
            event['accepted'] = None; event['error'] = type(error).__name__
        events.append(event)
        (a.output / 'requests.json').write_text(json.dumps(events, indent=2) + '\n')
        print(json.dumps(event), flush=True)
    result = dict(observed_unix=time.time(), final_action='stop_only', destroy_authorized=False,
                  source_commit=source_commit, requests=events,
                  all_requests_accepted=all(e.get('accepted') is True for e in events),
                  stopped_state_verified=False,
                  verification_required='Vast account console; instance API keys work only from their own host')
    (a.output / 'latest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    assert result['all_requests_accepted'], 'At least one stop request lacks a provider acknowledgement'


if __name__ == '__main__':
    main()
