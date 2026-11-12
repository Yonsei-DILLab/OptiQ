"""Build a private scheduler sidecar; frozen learner sources stay unchanged.

The replacement controller adopts existing PID/start-time identities from the
original plan. Deployment must retire only the original controller PID, never
its process group. No failed, cancelled, or superseded job is retried.
"""
import ast


def build(original):
    needle = "plan=json.loads(path.read_text());children={};stopping=False"
    assert original.count(needle) == 1
    original = original.replace(needle, needle + "\n    policy_path=Path(os.environ['IBOLT_CAPACITY_POLICY'])\n    admission_lock=Path('/workspace/.ibolt-capacity-admission.lock').open('a')")
    needle = "    def resources():\n"
    assert original.count(needle) == 1
    original = original.replace(needle, '''    def resources():
        policy=json.loads(policy_path.read_text())
        plan.update(policy['limits'])
        plan['capacity_policy']=str(policy_path)
        peer_jobs=[]
        for filename in policy['peer_plans']:
            other=plan if Path(filename)==path else json.loads(Path(filename).read_text())
            for key in ('jobs','preflights'):
                rows=other.get(key,[])
                if isinstance(rows,list):peer_jobs.extend(rows)
        peers=[j for j in peer_jobs if j.get('status') in ('initializing','running')
               and j.get('pid') and identity(j['pid'])==j.get('start_ticks')]
''')
    needle = "        for j in plan['jobs']+plan['preflights']:\n"
    assert original.count(needle) >= 1
    # Only resources()'s census; keep update/error handling scoped to this plan.
    original = original.replace(needle, "        for j in peers:\n", 1)
    needle = "        return gpus,[cs for k,cs in topology.items() if k not in reserved]"
    assert original.count(needle) == 1
    original = original.replace(needle, '''        for i,g in gpus.items():
            g['peer_active']=sum(j.get('gpu')==i for j in peers)
            g['peer_initializing']=sum(j.get('gpu')==i and j['status']=='initializing' for j in peers)
        return gpus,[cs for k,cs in topology.items() if k not in reserved]''')
    needle = "            try:\n                gpus,available=resources()"
    assert original.count(needle) == 1
    original = original.replace(needle, "            fcntl.flock(admission_lock,fcntl.LOCK_EX)\n" + needle)
    needle = "                time.sleep(10);continue"
    assert original.count(needle) == 1
    original = original.replace(needle, "                fcntl.flock(admission_lock,fcntl.LOCK_UN)\n" + needle)
    needle = "                if passed:\n                    order=plan['task_order'][j['method']]"
    assert original.count(needle) == 1
    original = original.replace(needle, "                if j['method'] not in policy['allowed_methods']:continue\n                if passed and not policy.get('parallel_tasks',False):\n                    order=plan['task_order'][j['method']]")
    # Failed task cohorts stay blocked; superseded methods are never released.
    needle = "                if len(active)>=plan['max_active']"
    assert original.count(needle) == 1
    original = original.replace(needle, "                if any(x['method']==j['method'] and x['task']==j['task'] and x['status']=='failed' for x in plan['jobs']):continue\n" + needle)
    needle = "and counts[i]<plan['max_per_gpu'] and initializing[i]<2]"
    assert original.count(needle) == 1
    original = original.replace(needle, "and counts[i]<plan['max_per_gpu'] and g['peer_active']<policy['global_max_per_gpu'] and g['peer_initializing']<2]")
    needle = "                spawn(j,gpu,available[:3]);available=available[3:];active.append(j);gpus[gpu]['free_mib']-=3072"
    assert original.count(needle) == 1
    original = original.replace(needle, needle + ";gpus[gpu]['peer_active']+=1;gpus[gpu]['peer_initializing']+=1")
    needle = "            atomic(path,plan)\n            if passed and all"
    assert original.count(needle) == 1
    original = original.replace(needle, "            atomic(path,plan)\n            fcntl.flock(admission_lock,fcntl.LOCK_UN)\n            if passed and all")
    needle = "        for j in plan['jobs']+plan['preflights']:\n            if j.get('pid')"
    assert original.count(needle) == 1
    original = original.replace(needle, "        for j in plan['jobs']+plan['preflights']:\n            if stopping and j.get('pid')")
    # The policy is reloaded inside resources(), before every admission cycle.
    original = original.replace("        policy=json.loads(policy_path.read_text())", "        nonlocal policy\n        policy=json.loads(policy_path.read_text())")
    original = original.replace("    topology={}", "    policy=json.loads(policy_path.read_text())\n    topology={}", 1)
    ast.parse(original)
    return original
