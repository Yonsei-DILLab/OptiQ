"""Central dependency coordinator; no learner access or result-based retries."""
from central_collector import ROOT,NEW,HOSTS,remote,digest
from entropy_selection import choose,load
from pathlib import Path
import json,time,traceback,fcntl

def write(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(p)
def cycle():
    rows=[];ready=[];selections=[]
    protocol=load(ROOT/'entropy-protocol.json');assert protocol
    for h in HOSTS:
        base=ROOT/f'hosts/host{h}/workspace'/NEW
        local=load(base/'entropy-protocol.json');proof=load(base/'entropy-transfer-proof.json')
        if local and proof:
            assert local==protocol
            ack=load(base/'entropy-archive-ready.json')
            if not ack or ack.get('source_manifests')!=proof['source_manifests']:
                for folder,sha in proof['source_manifests'].items():
                    source=base/folder;m=source/'source-sha256.json'
                    assert digest(m)==sha
                    for name,value in load(m).items():assert digest(source/name)==value,(h,folder,name)
                remote(h,'entropy-archive-ready',dict(passed=True,source_manifests=proof['source_manifests'],protocol_sha256=digest(base/'entropy-protocol.json'),verified_at=time.time()))
            ready.append(h)
        plan=load(base/'plan.json')
        if not plan:continue
        for j in plan['jobs']:
            if j['phase'] in ['pilot','entropy_pilot']:
                o=base/'runs'/j['id'];rows.append(dict(job=j,output=o,done=load(o/('completed.json' if j['family']=='adroit' else 'verified-completion.json'))))
    dest=ROOT/'entropy-selections';dest.mkdir(exist_ok=True)
    for key in protocol['task_keys']:
        t=load(ROOT/'entropy-temperature-selections'/(key+'.json')) or load(ROOT/'selections'/(key+'.json'))
        if not t:continue
        chosen=load(dest/(key+'.json'))
        if not chosen:
            chosen=choose(rows,t)
            if chosen:
                chosen['selected_at']=time.time();write(dest/(key+'.json'),chosen)
                print('ENTROPY_SELECTED',json.dumps(chosen),flush=True)
        if chosen:
            for h in ready:remote(h,'entropy-selection',chosen)
            selections.append(dict(task=key,temperature=chosen['temperature'],target_entropy_per_dim=chosen['target_entropy_per_dim']))
    out=dict(at=time.time(),passed=True,archive_ready_hosts=ready,selections=selections)
    write(ROOT/'entropy-coordinator-status.json',out);print(json.dumps(out),flush=True)

if __name__=='__main__':
    lock=(ROOT/'entropy-coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:cycle()
        except Exception as e:
            write(ROOT/'entropy-coordinator-status.json',dict(at=time.time(),passed=False,error=repr(e),traceback=traceback.format_exc()))
        time.sleep(45)
