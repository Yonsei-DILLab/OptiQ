
import pathlib,json,pickle,numpy as np,sys
p=pathlib.Path(sys.argv[1]);rows=[]
for done in p.glob('*/completed.json'):
 o=done.parent;c=json.loads((o/'config.json').read_text());f=o/'checkpoints/final-replay.npz'
 if f.exists():
  b=np.load(f);x=b['observations'] if 'observations' in b else b['state'];y=b['next_observations'] if 'next_observations' in b else b['next_state'];r=b['rewards'] if 'rewards' in b else b['reward'];t=b['terminals'] if 'terminals' in b else 1-b['not_done']
 else:
  b=pickle.load((o/'checkpoints/final-replay.pkl').open('rb'));x=b.observations[:b.size(),0];y=b.next_observations[:b.size(),0];r=b.rewards[:b.size(),0];t=b.dones[:b.size(),0]*(1-b.timeouts[:b.size(),0])
 assert np.isin(r,[-1,0]).all() and not t.any()
 assert x.shape[1]==28 and np.mean(x[:,0])>.5
 mask=np.arange(len(x)-1)%50!=49;err=float(np.max(np.abs(y[:-1][mask]-x[1:][mask])));assert err<1e-5,err
 n=json.loads((o/'checkpoints'/('normalizer_%09d.json'%c['steps'])).read_text())
 rows.append(dict(method=c['method'],updates=json.loads(done.read_text())['updates'],batch=c['batch_size'],raw_replay=True,temporal_link_max_error=err,timeout_bootstrap=True,norm_count=n['count']))
print(json.dumps(rows))
