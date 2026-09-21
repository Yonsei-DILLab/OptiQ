"""Reproduce a buffered-summary startup gate and verify bounded failures."""
from types import SimpleNamespace
from online_runner import verify_upload

class Clock:
    def __init__(self): self.t=0
    def now(self):return self.t
    def sleep(self,s):self.t+=s

class Run:
    entity='entity';project='project';id='run'
    def __init__(self):self.calls=[];self.summary={}
    def log(self,row,commit):
        assert commit is True
        assert 'env_steps' not in row  # Do not manufacture learning steps.
        self.calls.append(row)

class API:
    def __init__(self,run,clock,delay):self.run_=run;self.clock=clock;self.delay=delay
    def flush(self):pass
    def run(self,path):
        assert path=='entity/project/run'
        visible=self.run_.calls[-1] if self.run_.calls and self.clock.t>=self.delay else {}
        return SimpleNamespace(summary=visible)

c=Clock();r=Run();api=API(r,c,65)
result=verify_upload(r,0,lambda:api,timeout=300,clock=c.now,sleep=c.sleep,token='new')
assert result['verified'] and result['elapsed_seconds']==65 and len(r.calls)==3
assert all(v['ops/upload_probe_env_step']==0 for v in r.calls)
c=Clock();r=Run();api=API(r,c,999)
try:verify_upload(r,17,lambda:api,timeout=20,clock=c.now,sleep=c.sleep,token='new')
except RuntimeError:pass
else:raise AssertionError('Must fail if no remote receipt ever appears')
assert c.t==20
c=Clock();r=Run()
class Flaky(API):
    def run(self,path):
        if self.clock.t<10:raise ConnectionError('redacted')
        return super().run(path)
assert verify_upload(r,123,lambda:Flaky(r,c,10),clock=c.now,sleep=c.sleep)['verified']
print('PASS: committed history, delayed visibility >50s, no false success, transient retry, unchanged env-step axis')
