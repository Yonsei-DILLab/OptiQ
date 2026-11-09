
import pathlib,subprocess,concurrent.futures,json,shlex,time
folder=pathlib.Path(pathlib.Path('/tmp/fetch-normalized-campaign-path').read_text());remote='/workspace/iBOLT/'+str(folder)
opts=['-o','BatchMode=yes','-o','ConnectTimeout=8','-o','KexAlgorithms=curve25519-sha256','-o','Ciphers=chacha20-poly1305@openssh.com','-o','ControlMaster=no','-o','ControlPath=none']
def probe(h):
 code="import pathlib,json,time;p=pathlib.Path("+repr(remote+f'/host{h}')+");d=json.loads((p/'plan.json').read_text());r=dict(host="+str(h)+",checked=time.time(),status=d['status'],counts=d.get('counts'),error=d.get('error'),preflights={},jobs=[{k:j.get(k) for k in ('id','status','pid','gpu','progress','error','waiting_reason')} for j in d['jobs'] if j['status'] not in ('waiting_for_previous_task',)]);r['preflights']={str(f.parent.name):json.loads(f.read_text()) for f in (p/'preflights').glob('*/completed.json')};r['preflight_errors']={str(f.parent.name):json.loads(f.read_text()) for f in (p/'preflights').glob('*/failure.json')};print(json.dumps(r))"
 p=subprocess.run(['ssh',*opts,f'ibolt-server-{h}','python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=20)
 if p.returncode:return {'host':h,'error':p.stderr[-500:]}
 d=json.loads(p.stdout);(folder/f'live-host{h}.json').write_text(json.dumps(d,indent=2)+'\n')
 print('HOST',h,d['status'],d['counts'],'PREFLIGHTS',list(d['preflights']),'ERROR',d['error'],'PREFLIGHT_ERRORS',d['preflight_errors'],flush=True)
 groups={}
 for j in d['jobs']:
  key=j['status'];groups.setdefault(key,[]).append(j)
 for state,js in groups.items():
  steps=[(j['progress'] or {}).get('steps',0) for j in js];updates=[(j['progress'] or {}).get('updates',0) for j in js]
  print(state,len(js),'steps',min(steps),max(steps),'updates',min(updates),max(updates),flush=True)
  if state=='failed':print(js,flush=True)
 return {'host':h,'saved':True}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 for r in ex.map(probe,[1,2]):print(json.dumps(r),flush=True)
