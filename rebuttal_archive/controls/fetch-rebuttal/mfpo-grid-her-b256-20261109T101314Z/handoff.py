"""Replace only the host2 scheduler and MFPO jobs; adopt every other training PID."""
import copy,json,os,pathlib,signal,subprocess,sys,time

root=pathlib.Path('/workspace/iBOLT')
oldfolder=root/'outputs/fetch-rebuttal/her-b256-utd1-20261109T082908Z/host2'
folder=root/'outputs/fetch-rebuttal/mfpo-grid-her-b256-20261109T101314Z/host2'
source=folder/'source';sys.path.insert(0,str(source))
from fetch_rebuttal.campaign import process_live
from fetch_rebuttal.run import save_json,guard
guard()
plan=json.loads((folder/'plan.json').read_text())
assert (folder/'grid-ready.json').exists() and len(plan['mfpo_preflights'])==9
assert all(x['status']=='completed' for x in plan['mfpo_preflights'].values())
assert plan['status']=='prepared'
old=json.loads((oldfolder/'plan.json').read_text())
program='fetch-her-b256-utd1-20261109T082908Z-host2'
pid=int(subprocess.check_output(['sudo','-n','supervisorctl','pid',program],text=True).strip())
assert pid==old['controller_pid'] and pid>1
assert b'fetch_rebuttal.her_campaign' in pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
others=[copy.deepcopy(j) for j in old['jobs'] if j['method']!='mfpo' and j['status'] in ('running','initializing')]
assert len(others)==15 and all(process_live(j) for j in others)
save_json(folder/'previous-plan-before-handoff.json',old)
evidence=dict(started_unix=time.time(),previous_controller_pid=pid,
    adopted=[{k:j[k] for k in ('id','pid','start_ticks','output','cpus')} for j in others],superseded=[])
# A signal to the controller PID only, never to its process group.
os.kill(pid,signal.SIGTERM)
for _ in range(50):
    try:
        state=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
        if state=='Z':break
    except FileNotFoundError:break
    time.sleep(.1)
else:raise RuntimeError('Previous scheduler did not stop')
assert all(process_live(j) for j in others),'Unrelated training PID changed during scheduler handoff'
for job in old['jobs']:
    if job['method']!='mfpo':continue
    row=copy.deepcopy(job)
    if process_live(job):os.kill(job['pid'],signal.SIGTERM)
    output=pathlib.Path(job['output'])
    progress=output/'progress.json'
    job.update(status='superseded',superseded_unix=time.time(),
        superseded_reason='User requested MFPO entropy/initial-temperature grid and selected reruns',
        superseded_by=str(folder/'plan.json'))
    if progress.exists():job['progress']=json.loads(progress.read_text())
    if output.exists():save_json(output/'superseded.json',{k:job[k] for k in ('status','superseded_unix','superseded_reason','superseded_by')})
    evidence['superseded'].append(dict(id=job['id'],pid=job.get('pid'),previous_status=row['status'],
        steps=job.get('progress',{}).get('steps',0),output=job['output']))
for _ in range(100):
    live=[j for j in old['jobs'] if j['method']=='mfpo' and process_live(j)]
    if not live:break
    time.sleep(.1)
else:
    for job in live:os.kill(job['pid'],signal.SIGKILL)
    time.sleep(1)
assert not any(process_live(j) for j in old['jobs'] if j['method']=='mfpo')
assert all(process_live(j) for j in others)
plan['jobs']=old['jobs']+plan['jobs']
plan['preflights']=old['preflights']
plan['selections']=old.get('selections',{})
plan['controller_handoff']=dict(previous_plan=str(oldfolder/'plan.json'),previous_program=program,
    adopted_pids=len(others),superseded_mfpo_jobs=20,completed_unix=time.time())
plan['status']='prepared';plan.pop('controller_pid',None)
plan['max_initializing_per_gpu']=5
save_json(folder/'plan.json',plan)
newprogram='fetch-mfpo-grid-her-b256-20261109T101314Z-host2'
subprocess.run(['sudo','-n','supervisorctl','start',newprogram],check=True)
save_json(oldfolder/'controller-handoff.json',dict(plan=str(folder/'plan.json'),program=newprogram,
    reason='MFPO grid; other host2 learner processes adopted without restart',created_unix=time.time()))
assert all(process_live(j) for j in others)
evidence.update(completed_unix=time.time(),other_training_pids_preserved=True,new_program=newprogram)
save_json(folder/'handoff-verification.json',evidence)
print(json.dumps(dict(program=newprogram,adopted_training_pids=len(others),
    mfpo_stopped=sum(j['previous_status'] in ('running','initializing') for j in evidence['superseded']),
    superseded_followups=sum(j['previous_status']=='waiting_for_previous_task' for j in evidence['superseded']),
    new_grid_runs=18,selected_fresh_reruns=20)),flush=True)
