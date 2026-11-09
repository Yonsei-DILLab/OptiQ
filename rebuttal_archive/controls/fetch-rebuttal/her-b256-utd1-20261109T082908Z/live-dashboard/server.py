"""Read-only local dashboard for the two Fetch experiment servers."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import base64
import gzip
import hashlib
import json
import mimetypes
import os
import shlex
import subprocess
import threading
import time
import urllib.parse
import zlib

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent.name
REMOTE = '/workspace/iBOLT/outputs/fetch-rebuttal/' + CAMPAIGN
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
    '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2',
    '-o', 'KexAlgorithms=curve25519-sha256', '-o', 'Ciphers=chacha20-poly1305@openssh.com',
    '-o', 'ControlMaster=no', '-o', 'ControlPath=none']
REMOTE_QUERY = r'''
import pathlib,json,time
p=pathlib.Path(BASE);handoff=p/'controller-handoff.json'
planpath=pathlib.Path(json.loads(handoff.read_text())['plan']) if handoff.exists() else p/'plan.json'
plan=json.loads(planpath.read_text())
r=dict(source_time=time.time(),status=plan['status'],counts=plan.get('counts',{}),selections=plan.get('selections',{}),mfpo_selection=plan.get('mfpo_selection'),jobs=[])
for j in plan['jobs']:
 o=pathlib.Path(j['output']);row={k:j.get(k) for k in ('id','task','method','seed','temperature','phase','status','gpu','target_entropy_coeff','initial_temperature')}
 row['steps']=0;row['updates']=0
 f=o/'progress.json'
 if f.exists():
  try:
   progress=json.loads(f.read_text());row['steps']=progress.get('steps',0);row['updates']=progress.get('updates',0);row['progress_time']=progress.get('observed_unix')
  except (ValueError,OSError):pass
 row['evaluations']=[];f=o/'evaluations.jsonl'
 if f.exists():
  records={}
  for line in f.read_text().splitlines():
   if not line.strip():continue
   try:
    e=json.loads(line);v=e[e['primary_mode']];n=v['episodes']
    records[e['steps']]=dict(step=e['steps'],episodes=n,final=v['success_rate'],any=v['any_success_rate'],new_final=v.get('new_final_success_count',0)/n,new_any=v.get('new_any_success_count',0)/n,return_=v['mean_return'],initial=sum(v.get('initial_success',[])))
   except (ValueError,KeyError,ZeroDivisionError):continue
  row['evaluations']=[records[k] for k in sorted(records)]
 if j.get('error'):row['error']=str(j['error'])[-800:]
 r['jobs'].append(row)
print(json.dumps(r,separators=(',',':'),allow_nan=False))
'''


def query_host(host):
    code = 'BASE=' + repr(REMOTE + f'/host{host}') + '\n' + REMOTE_QUERY
    blob = base64.b64encode(zlib.compress(code.encode())).decode()
    command = 'python3 -c ' + shlex.quote('import zlib,base64;exec(zlib.decompress(base64.b64decode(' + repr(blob) + ')))')
    result = subprocess.run(['/usr/bin/ssh', *SSH_OPTIONS, f'ibolt-server-{host}', command],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=22)
    if result.returncode:
        raise RuntimeError('SSH 연결 실패 또는 응답 시간 초과')
    value = json.loads(result.stdout)
    if not isinstance(value.get('jobs'), list):
        raise ValueError('서버 데이터 형식 오류')
    value.update(id=host, ok=True, last_success=time.time(), error=None)
    return value


class Collector:
    def __init__(self, interval=30):
        self.interval = interval
        self.lock = threading.Lock()
        self.refresh = threading.Event()
        self.stop = threading.Event()
        self.collecting = False
        self.hosts = {}
        self.last_attempt = None
        self.revision = ''
        cache = ROOT/'data-cache.json'
        if cache.exists():
            try:
                saved = json.loads(cache.read_text())
                self.hosts = {int(h['id']): dict(h, ok=False, error='재연결 중') for h in saved.get('hosts', [])}
            except (OSError, ValueError, KeyError):
                pass

    def snapshot(self):
        with self.lock:
            hosts = [self.hosts.get(i, dict(id=i, ok=False, jobs=[], error='연결 중', last_success=None)) for i in (1,2)]
            return dict(campaign=CAMPAIGN, generated_at=time.time(), last_attempt=self.last_attempt,
                collecting=self.collecting, interval=self.interval, revision=self.revision, hosts=hosts,
                settings=dict(batch_size=256, utd=1, her=True, normalize_observations=True,
                    budget=1000000, temperatures=[.05,.1,.25,.5,1,5,10], baseline_seeds=[0,1,2,3,4]))

    def collect_once(self):
        with self.lock:
            self.collecting = True
            self.last_attempt = time.time()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {h:pool.submit(query_host,h) for h in (1,2)}
            for host,future in futures.items():
                try:
                    value = future.result()
                except Exception as error:
                    with self.lock:
                        previous = self.hosts.get(host, dict(id=host, jobs=[], last_success=None))
                        self.hosts[host] = dict(previous, ok=False, error=str(error))
                else:
                    with self.lock:
                        self.hosts[host] = value
        with self.lock:
            self.collecting = False
            self.revision = hashlib.sha256(json.dumps(self.hosts,sort_keys=True).encode()).hexdigest()[:12]
        state = self.snapshot()
        temporary = ROOT/'data-cache.tmp'
        temporary.write_text(json.dumps(state,separators=(',',':'),allow_nan=False)+'\n')
        temporary.replace(ROOT/'data-cache.json')

    def run(self):
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                self.collect_once()
            except Exception as error:
                print('Collector error:', type(error).__name__, flush=True)
                with self.lock:
                    self.collecting = False
            self.refresh.wait(max(3, self.interval-(time.monotonic()-started)))
            self.refresh.clear()


def handler_for(collector):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, pattern, *args):
            if args and str(args[1] if len(args)>1 else '') not in ('200','304'):
                super().log_message(pattern,*args)

        def local_host(self):
            return self.headers.get('Host','').split(':')[0] in ('127.0.0.1','localhost')

        def reply(self, body, content_type, status=200):
            if isinstance(body,str):
                body = body.encode()
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'")
            if 'gzip' in self.headers.get('Accept-Encoding','') and len(body)>2000:
                body=gzip.compress(body)
                self.send_header('Content-Encoding','gzip')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.local_host():
                return self.reply('Invalid host','text/plain',403)
            path=urllib.parse.urlparse(self.path).path
            if path=='/api/state':
                return self.reply(json.dumps(collector.snapshot(),separators=(',',':'),allow_nan=False),'application/json')
            if path=='/health':
                return self.reply(json.dumps(dict(ok=True,pid=os.getpid())),'application/json')
            allowed={'/':'index.html','/index.html':'index.html','/app.js':'app.js','/analytics.js':'analytics.js','/styles.css':'styles.css','/favicon.svg':'favicon.svg'}
            name=allowed.get(path)
            if not name or not (ROOT/name).is_file():
                return self.reply('Not found','text/plain',404)
            content_type={'js':'application/javascript','css':'text/css','html':'text/html','svg':'image/svg+xml'}[name.rsplit('.',1)[1]]
            return self.reply((ROOT/name).read_bytes(),content_type+'; charset=utf-8')

        def do_POST(self):
            origin=self.headers.get('Origin')
            if not self.local_host() or (origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}')):
                return self.reply('Invalid origin','text/plain',403)
            if urllib.parse.urlparse(self.path).path!='/api/refresh':
                return self.reply('Not found','text/plain',404)
            collector.refresh.set()
            return self.reply('{"requested":true}','application/json',202)
    return Handler


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--interval',type=int,default=30)
    parser.add_argument('--collect-once',action='store_true')
    args=parser.parse_args()
    collector=Collector(args.interval)
    if args.collect_once:
        collector.collect_once()
        print(json.dumps(dict(hosts=[dict(id=h['id'],ok=h['ok'],jobs=len(h['jobs'])) for h in collector.snapshot()['hosts']])))
        return
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(collector))
    threading.Thread(target=collector.run,name='fetch-log-collector',daemon=True).start()
    print(f'Fetch live dashboard: http://127.0.0.1:{args.port}',flush=True)
    try:
        server.serve_forever()
    finally:
        collector.stop.set();collector.refresh.set();server.server_close()


if __name__=='__main__':
    main()
