"""Correct only the integer-count Pen selection tie; preserve all trial data."""
from pathlib import Path
import sys,json,subprocess,hashlib,time,zlib,base64
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent))
from deploy import freeze,resume
R='/workspace/heechan-unified-f1-a1-20261112'
TAG='entropy-tie-correction-20261112T1220'
D=json.loads((P/'correction.json').read_text())
def remote(h,code,**extra):
    values=dict(ROOT=R,TAG=TAG,HOST=h,OLD=D['old'],NEW=D['new'],**extra)
    script='\n'.join(f'{k}={v!r}' for k,v in values.items())+'\n'+code
    packed=base64.b64encode(zlib.compress(script.encode())).decode()
    script=f'import zlib,base64\nexec(zlib.decompress(base64.b64decode({packed!r})))\n'
    for attempt in range(1 if code in [freeze,resume] else 4):
        r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','KexAlgorithms=curve25519-sha256','-o','ConnectTimeout=15',f'ibolt-server-{h}','python3 -'],input=script,text=True,capture_output=True,timeout=60)
        if r.returncode==0:break
        print('TRANSPORT_RETRY',h,attempt+1,flush=True)
    assert r.returncode==0,r.stdout+r.stderr
    return json.loads(r.stdout)
if __name__=='__main__':
    held=False
    try:
        for h in range(1,6):
            print(remote(h,r'''
from pathlib import Path
import json,hashlib
p=Path(ROOT);a=p/TAG;a.mkdir(exist_ok=True);f=p/'entropy-selections/adroit-pen.json'
assert json.loads(f.read_text())==OLD
(a/'adroit-pen.before.json').write_bytes(f.read_bytes())
(a/'adroit-pen.corrected.json').write_text(json.dumps(NEW,indent=2))
print(json.dumps(dict(host=HOST,staged=True)))
'''),flush=True)
        pre=json.loads((P.parent/'status-rebalance-20261112T1210/preflight-host3.json').read_text())
        before=remote(3,freeze,OLD_CONTROLLER=pre['hashes']['controller.py'],OLD_GATEWAY=pre['hashes']['archive_gateway.py']);held=True
        (P/'host3-plan.before.json').write_text(json.dumps(before,indent=2))
        for h in range(1,6):
            print(remote(h,r'''
from pathlib import Path
import json
p=Path(ROOT);f=p/'entropy-selections/adroit-pen.json';assert json.loads(f.read_text()) in [OLD,NEW]
t=f.with_suffix('.corrected.tmp');t.write_text(json.dumps(NEW,indent=2));t.replace(f)
print(json.dumps(dict(host=HOST,corrected=True)))
'''),flush=True)
        print(remote(3,r'''
from pathlib import Path
import json,collections,time,fcntl
p=Path(ROOT);assert (p/'PAUSE').read_text().startswith(TAG)
lock=(p/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
d=json.loads((p/'plan.json').read_text());assert d==BEFORE
jj=[j for j in d['jobs'] if j['id']=='adroit-pen-ibolt-entropy-selected-s3-confirm'];assert len(jj)==1
j=jj[0];assert j['status']=='queued' and not j.get('pid') and not Path(j['output']).exists()
j['status']='not_selected';j['not_selected_reason']='Exact-count tie correction selects H=-0.9; existing T=0.1 confirmation is reused'
j['correct_reference_id']='adroit-pen-ibolt-s3-t0p1-confirm'
d['counts']=dict(collections.Counter(x['status'] for x in d['jobs']));d['updated_unix']=time.time()
t=p/'plan.tie.tmp';t.write_text(json.dumps(d,indent=2));t.replace(p/'plan.json')
print(json.dumps(dict(host=HOST,unstarted_incorrect_confirmation_cancelled=True)))
''',BEFORE=before),flush=True)
        (P/'workers-applied.json').write_text(json.dumps(dict(at=time.time(),hosts=list(range(1,6)),corrected_H=-.9,completed_auxiliary_trials_preserved=True),indent=2))
    finally:
        if held:print(remote(3,resume),flush=True)
