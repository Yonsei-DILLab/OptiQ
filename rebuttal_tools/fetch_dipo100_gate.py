"""Read only prespecified completion records; gate DiPo on all 40 SAC/iBOLT runs."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def inspect_dependency(spec):
    plan_path = Path(spec['plan'])
    plan = json.loads(plan_path.read_text())
    expected = {(m,t,s) for m in spec['methods'] for t in spec['tasks'] for s in range(5)}
    selected = {}
    for job in plan['jobs']:
        key = (job['method'],job.get('task','pick-and-place'),job['seed'])
        if key in expected:
            if key in selected:
                raise ValueError('Duplicate dependency: '+str(key))
            selected[key] = job
    if set(selected) != expected:
        raise ValueError('Dependency population differs from its fixed manifest')
    rows = []
    for key, job in sorted(selected.items()):
        method, task, seed = key
        out = Path(job['output'])
        marker = out/'verified-completion.json'
        row = dict(method=method,task=task,seed=seed,status=job['status'],completed=False)
        if marker.exists():
            done = json.loads(marker.read_text())
            cfg = json.loads((out/'config.json').read_text())
            assert done['completed'] and not done['smoke']
            assert (done['method'],done['seed'],done['steps'],done['updates']) == (method,seed,5000192,623424)
            assert done['observed_utd'] == .125
            assert (cfg['task'],cfg['num_envs'],cfg['batch_size'],cfg['utd']) == (task,256,256,.125)
            assert cfg['HER'] and cfg['normalize_observations']
            final = json.loads((out/'evaluations'/'005000192.json').read_text())
            assert final[final['primary_mode']]['episodes'] == 100
            row.update(completed=True,finished_unix=done['finished_unix'],verified_marker=str(marker))
        rows.append(row)
    count = sum(row['completed'] for row in rows)
    return dict(ready=count==len(expected),completed=count,expected=len(expected),
        rows=rows,plan=str(plan_path),checked_unix=time.time())


def collect_gate(plan):
    results, errors = {}, {}
    for dependency in plan['dependencies']:
        label = dependency['id']
        try:
            if dependency.get('local'):
                result = inspect_dependency(dependency)
            else:
                command = ['ssh','-T','-n','-p',str(dependency['port']),'-i',plan['reader_private_key'],
                    '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=10',
                    '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+plan['reader_known_hosts'],
                    '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2',
                    dependency['user']+'@'+dependency['host'],'status']
                response = subprocess.run(command,capture_output=True,text=True,timeout=25)
                if response.returncode:
                    raise RuntimeError(response.stderr[-500:])
                result = json.loads(response.stdout)
            assert result['expected'] == dependency['expected']
            results[label] = result
        except Exception as error:
            errors[label] = repr(error)
    expected = sum(d['expected'] for d in plan['dependencies'])
    completed = sum(r['completed'] for r in results.values())
    assert expected == 40
    return dict(ready=not errors and completed==40 and len(results)==len(plan['dependencies']),
        completed=completed,expected=40,dependencies=results,errors=errors,checked_unix=time.time())


def revoke_readers(plan):
    results = {}
    for dependency in plan['dependencies']:
        if dependency.get('local'):
            continue
        try:
            command = ['ssh','-T','-n','-p',str(dependency['port']),'-i',plan['reader_private_key'],
                '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=10',
                '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+plan['reader_known_hosts'],
                dependency['user']+'@'+dependency['host'],'revoke']
            response = subprocess.run(command,capture_output=True,text=True,timeout=20)
            assert response.returncode == 0, response.stderr[-500:]
            results[dependency['id']] = json.loads(response.stdout)
        except Exception as error:
            results[dependency['id']] = dict(error=repr(error))
    if all(value.get('revoked') for value in results.values()):
        Path(plan['reader_private_key']).unlink(missing_ok=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reader-config',type=Path,required=True)
    args = parser.parse_args()
    config = json.loads(args.reader_config.read_text())
    if os.environ.get('SSH_ORIGINAL_COMMAND') == 'revoke':
        authorized = Path.home()/'.ssh/authorized_keys'
        original = authorized.read_text()
        lines = [line for line in original.splitlines(keepends=True) if line.rstrip('\r\n') != config['authorization_line']]
        temp = authorized.with_name('authorized_keys.'+config['campaign']+'.tmp')
        temp.write_text(''.join(lines))
        temp.chmod(0o600)
        temp.replace(authorized)
        print(json.dumps(dict(revoked=True,campaign=config['campaign'])))
    else:
        print(json.dumps(inspect_dependency(config)),flush=True)


if __name__ == '__main__':
    main()
