
import pathlib,subprocess,shlex,base64,zlib,concurrent.futures,json,datetime,collections
folder=pathlib.Path(__file__).resolve().parent;remote='/workspace/iBOLT/'+str(folder.relative_to(pathlib.Path.cwd()));code=(folder/'status_probe.py').read_text()
opts=['-o','BatchMode=yes','-o','ConnectTimeout=8','-o','KexAlgorithms=curve25519-sha256','-o','Ciphers=chacha20-poly1305@openssh.com','-o','ControlMaster=no','-o','ControlPath=none']
def probe(h):
 enc=base64.b64encode(zlib.compress(('BASE='+repr(remote+f'/host{h}')+'\n'+code).encode())).decode()
 p=subprocess.run(['ssh',*opts,f'ibolt-server-{h}','python3 -c '+shlex.quote('import zlib,base64;exec(zlib.decompress(base64.b64decode('+repr(enc)+')))')],capture_output=True,text=True,timeout=25)
 if p.returncode:raise RuntimeError(p.stderr[-1200:])
 d=json.loads(p.stdout);d['host']=h;(folder/f'latest-status-host{h}.json').write_text(json.dumps(d,indent=2)+'\n')
 print('HOST',h,datetime.datetime.fromtimestamp(d['checked_unix'],datetime.timezone(datetime.timedelta(hours=9))).isoformat(),d['status'],d['counts'],'GPU',d['gpu'],flush=True)
 groups=collections.defaultdict(list)
 for r in d['jobs']:groups[(r['method'],r['status'])].append(r)
 for group,rs in groups.items():
  steps=[r.get('progress',{}).get('steps',0) for r in rs];updates=[r.get('progress',{}).get('updates',0) for r in rs]
  print(json.dumps({'group':group,'count':len(rs),'steps':[min(steps),max(steps)],'updates':[min(updates),max(updates)],'errors':[r['error'] for r in rs if r.get('error')]}),flush=True)
 return d
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:data=list(ex.map(probe,[1,2]))
(folder/'latest-status-input.json').write_text(json.dumps(data,indent=2)+'\n')
