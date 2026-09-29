from pathlib import Path
import configparser,fcntl,json,os,psutil,shutil,subprocess,time
ops=Path('/home/heechan/OptiQ-ops');jobs=ops/'supervisor/jobs';archive=ops/'supervisor/disabled-antmaze-20260925'
ctl=['/usr/local/bin/supervisorctl','-c',str(ops/'supervisor/supervisord.conf')]
receipt=ops/'host-reservation-gmm40-20260925.json'
assert not receipt.exists(),'Reservation already recorded; inspect instead of repeating'
def processes():
 rows=[]
 for p in psutil.process_iter(['pid','cmdline','status']):
  try:
   if p.pid!=os.getpid() and 'antmaze' in ' '.join(p.info['cmdline'] or []).lower():rows.append(p.info)
  except(psutil.NoSuchProcess,psutil.AccessDenied):pass
 return rows
before=processes()
assert not before,('Unexpected AntMaze process needs scoped termination',before)
configs=[]
for path in sorted(jobs.glob('*antmaze*.conf')):
 cfg=configparser.ConfigParser(interpolation=None);cfg.read(path)
 assert len(cfg.sections())==1
 section=cfg.sections()[0];service=section.removeprefix('program:')
 assert section.startswith('program:antmaze-') and service==path.stem
 configs.append((path,service))
assert configs
archive.mkdir(exist_ok=True)
status_before=subprocess.run(ctl+['status'],text=True,capture_output=True).stdout
actions=[]
for path,service in configs:
 assert not (archive/path.name).exists()
 stopped=subprocess.run(ctl+['stop',service],text=True,capture_output=True)
 assert stopped.returncode==0 or 'not running' in stopped.stdout,(service,stopped.stdout,stopped.stderr)
 shutil.move(str(path),str(archive/path.name))
 actions.append({'service':service,'archive':str(archive/path.name),'stop':stopped.stdout.strip()})
subprocess.run(ctl+['reread'],check=True,capture_output=True,text=True)
for _,service in configs:
 result=subprocess.run(ctl+['update',service],check=True,capture_output=True,text=True)
 actions.append({'service':service,'unload':result.stdout.strip()})
remaining=processes();assert not remaining
status_after=subprocess.run(ctl+['status'],text=True,capture_output=True).stdout
assert not any(line.split() and line.split()[0].startswith('antmaze-') for line in status_after.splitlines())
queues=[]
for root in Path('/home/heechan/optiq-experiments').glob('*antmaze*'):
 status=root/'status.json'
 if status.exists():
  d=json.loads(status.read_text());assert not d.get('running') and not d.get('pending'),root
  queues.append({'root':str(root),'running':0,'pending':0})
locks=[]
for gpu in range(4):
 with (ops/f'locks/gpu-{gpu}.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
  except BlockingIOError:free=False
  locks.append({'gpu':gpu,'lock_free':free})
d={'time':time.time(),'host':'vast1','reserved_for':'GMM40','antmaze_allowed':False,'antmaze_training_or_evaluation_allowed':False,'policy_source_commit':'ac314a63658d4458308f420aee089ceaeded10b0','antmaze_processes':remaining,'queues':queues,'gpu_locks':locks,'archived_services':actions,'supervisor_before':status_before,'supervisor_after':status_after,'preserved':'All source snapshots, experiment logs/results/checkpoints, NM/ablation data; instance remains running; no GMM40 job launched'}
receipt.write_text(json.dumps(d,indent=2)+'\n')
(ops/'ANTMAZE_HOST_POLICY.md').write_text('# Host reserved for GMM40\n\nUser instruction 2026-09-25: vast1 (RTX4090x4) must not run AntMaze training, preflights, evaluations or background watchers. Use vast-heechan-180/199 for approved AntMaze work. Preserve all old results, source snapshots and NM/ablation data. The instance remains available for GMM40; no new GMM40 experiment is authorized by this reservation alone. See host-reservation-gmm40-20260925.json.\n')
print(json.dumps(d))
