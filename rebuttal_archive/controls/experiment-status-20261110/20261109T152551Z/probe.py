"""Read-only instance, GPU, and experiment inspection. No process control."""
import collections,csv,io,json,os,pathlib,shutil,subprocess,time
P=pathlib.Path
def read_json(path):
    try:return json.loads(P(path).read_text())
    except (OSError,ValueError):return None
def kv(path):
    try:return dict(line.split() for line in P(path).read_text().splitlines())
    except (OSError,ValueError):return {}
def proc(pid):
    try:
        fields=P(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),state=fields[0],start_ticks=int(fields[19]),
                    cpu_seconds=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),
                    rss_mib=int(fields[21])*os.sysconf('SC_PAGE_SIZE')/2**20)
    except (OSError,ValueError):return None
def cpu():
    a=list(map(int,P('/proc/stat').read_text().splitlines()[0].split()[1:]));return sum(a),a[3]+a[4]
def gpu_sample():
    query='index,name,uuid,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit'
    r=subprocess.run(['nvidia-smi','--query-gpu='+query,'--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    rows=[]
    for a in csv.reader(io.StringIO(r.stdout),skipinitialspace=True):
        rows.append(dict(index=int(a[0]),name=a[1],uuid=a[2],total_mib=float(a[3]),used_mib=float(a[4]),
            gpu_util=float(a[5]),memory_util=float(a[6]),temperature_c=float(a[7]),power_w=float(a[8]),power_limit_w=float(a[9])))
    return dict(time=time.time(),gpus=rows)
start=time.time();cpu0=cpu();events0=kv('/sys/fs/cgroup/memory.events');samples=[]
for i in range(6):
    samples.append(gpu_sample())
    if i<5:time.sleep(5)
cpu1=cpu();gpus=[]
for g in samples[-1]['gpus']:
    measurements=[next(x for x in s['gpus'] if x['index']==g['index']) for s in samples]
    g.update(util_mean=sum(x['gpu_util'] for x in measurements)/len(measurements),
        util_min=min(x['gpu_util'] for x in measurements),util_max=max(x['gpu_util'] for x in measurements),
        free_mib=g['total_mib']-g['used_mib'])
    gpus.append(g)
apps=[]
r=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
for a in csv.reader(io.StringIO(r.stdout),skipinitialspace=True):
    if len(a)>=3:apps.append(dict(gpu_uuid=a[0],pid=int(a[1]),used_mib=a[2]))
processes=[]
for p in P('/proc').glob('[0-9]*'):
    try:
        args=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\0') if s]
        if not args or 'python' not in P(args[0]).name:continue
        module=args[args.index('-m')+1] if '-m' in args else ''
        if not module and len(args)>1 and args[1].endswith('.py'):module=P(args[1]).name
        if not any(s in ' '.join(args) for s in ('fetch_rebuttal','pointmaze','gmm40')):continue
        row=proc(p.name)
        if not row:continue
        row['module']=module
        row['settings']={flag:args[args.index(flag)+1] for flag in ('--method','--maze','--task','--seed','--temperature','--plan','--output') if flag in args and args.index(flag)+1<len(args)}
        row['affinity']=sorted(os.sched_getaffinity(int(p.name)))
        env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
        row['visible_gpu']=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode(errors='replace')
        processes.append(row)
    except (OSError,ProcessLookupError):continue
base=P('/workspace/iBOLT/outputs/fetch-rebuttal/her-b256-utd1-20261109T082908Z')/f'host{HOST}'
handoff=read_json(base/'controller-handoff.json');planpath=P(handoff['plan']) if handoff else base/'plan.json'
plan=read_json(planpath);fetch=None
if plan:
    jobs=[]
    for j in plan['jobs']:
        o=P(j['output']);progress=read_json(o/'progress.json') or {};failure=read_json(o/'failure.json')
        row={k:j.get(k) for k in ('id','method','task','seed','phase','status','gpu','pid','start_ticks')}
        row.update(steps=progress.get('steps',0),updates=progress.get('updates',0),
            progress_time=progress.get('observed_unix'),progress_mtime=(o/'progress.json').stat().st_mtime if (o/'progress.json').exists() else None,
            error=str(failure.get('error','failure recorded'))[:700] if failure else j.get('error'))
        if row['status'] in ('running','initializing'):
            info=proc(j.get('pid',0));row['live']=bool(info) and info['state']!='Z' and info['start_ticks']==j.get('start_ticks')
        jobs.append(row)
    fetch=dict(plan=str(planpath),status=plan['status'],checked_unix=plan.get('checked_unix'),
               counts=dict(collections.Counter(j['status'] for j in jobs)),jobs=jobs,
               selections=plan.get('selections',{}),mfpo_selection=plan.get('mfpo_selection'))
maze='medium' if HOST==1 else 'hard'
pointmaze=[]
for o in sorted(P('/workspace/iBOLT/outputs/pointmaze-rebuttal/baseline-queue-20261107T162000Z/runs').glob(maze+'-*')):
    progress=read_json(o/'progress.json') or {};failure=read_json(o/'failure.json');completed=read_json(o/'completed.json')
    pointmaze.append(dict(name=o.name,status='completed' if completed else 'failed' if failure else progress.get('status','unknown'),
        steps=progress.get('steps',0),updates=progress.get('updates',0),progress_mtime=(o/'progress.json').stat().st_mtime if (o/'progress.json').exists() else None,
        error=str(failure.get('error','failure recorded'))[:700] if failure else None))
mem={line.split(':')[0]:int(line.split()[1])*1024 for line in P('/proc/meminfo').read_text().splitlines()}
current=int(P('/sys/fs/cgroup/memory.current').read_text());maximum=P('/sys/fs/cgroup/memory.max').read_text().strip()
effective_available=min(mem['MemAvailable'],int(maximum)-current) if maximum!='max' else mem['MemAvailable']
disk=shutil.disk_usage('/workspace');events1=kv('/sys/fs/cgroup/memory.events')
supervisor=subprocess.run(['sudo','-n','supervisorctl','status'],capture_output=True,text=True,timeout=15)
services=[line for line in supervisor.stdout.splitlines() if any(s in line for s in ('fetch-','pointmaze-','gmm40-'))]
result=dict(host=HOST,started_unix=start,finished_unix=time.time(),sampling_seconds=samples[-1]['time']-samples[0]['time'],
    gpus=gpus,gpu_samples=samples,gpu_apps=apps,experiment_processes=processes,
    cpu=dict(allowed_logical=len(os.sched_getaffinity(0)),loadavg=os.getloadavg(),host_busy_percent=100*(1-(cpu1[1]-cpu0[1])/(cpu1[0]-cpu0[0])),cpu_max=P('/sys/fs/cgroup/cpu.max').read_text().strip()),
    ram=dict(current_gib=current/2**30,limit=maximum,effective_available_gib=effective_available/2**30),
    disk=dict(total_gib=disk.total/2**30,free_gib=disk.free/2**30,used_percent=disk.used/disk.total*100),
    memory_events=events1,memory_events_delta={k:int(v)-int(events0.get(k,0)) for k,v in events1.items()},
    services=services,fetch=fetch,pointmaze=pointmaze)
print(json.dumps(result,allow_nan=False))
