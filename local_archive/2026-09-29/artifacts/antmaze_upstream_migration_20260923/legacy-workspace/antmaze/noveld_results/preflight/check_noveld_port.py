import ast, json, hashlib
from pathlib import Path
from collections.abc import Sequence
import torch, numpy as np
from antmaze.multimodal.noveld import NovelD, Replay
namespace=dict(torch=torch,nn=torch.nn,np=np,Sequence=Sequence,F=torch.nn.functional,clip_grad_norm_=torch.nn.utils.clip_grad_norm_)
for filename,names in [('ddiffpg-mlp.py',{'RNDModel'}),('ddiffpg-intrinsic.py',{'IntrinsicM','Embedder','get_embedder'})]:
 tree=ast.parse(Path('/tmp',filename).read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,'exec'),namespace)
port=NovelD(29,0)
reference=namespace['IntrinsicM'].__new__(namespace['IntrinsicM'])
reference.pos_enc=True;reference.env_name='antmaze-v1';reference.type='noveld';reference.normalize=False
reference.embedder,_=namespace['get_embedder'](10,input_dims=2)
reference.rnd_model=namespace['RNDModel'](69).cuda()
reference.rnd_model.predictor.load_state_dict(port.predictor.state_dict());reference.rnd_model.target.load_state_dict(port.target.state_dict())
rng=np.random.default_rng(88);obs=rng.normal(size=(4096,29)).astype('float32');nxt=obs+rng.normal(0,.3,obs.shape).astype('float32')
s=torch.as_tensor(obs,device='cuda');ns=torch.as_tensor(nxt,device='cuda')
with torch.no_grad():
 np.testing.assert_allclose(reference.encode_obs(s).cpu(),port.encode(s).cpu(),rtol=0,atol=0)
 expected=reference.compute_reward(s,ns).cpu().numpy().ravel()
actual=port.reward_and_update(obs,nxt,np.full(len(obs),-8.))
np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
replay=Replay(5000,0,port);replay.add_batch(obs,np.zeros((4096,8)),np.full(4096,-8.),nxt,np.zeros(4096))
before=port.updates;replay.diagnostic=True;data=replay.sample(4096)
assert len(data.observations)==256 and port.updates==before
assert np.all(data.rewards.numpy()==-8.)
replay.sample(4096);assert port.updates==before+1
print(json.dumps(dict(passed=True,encoding_exact=True,upstream_reward_max_error=float(abs(actual-expected).max()),rnd_updates=port.updates,diagnostic_does_not_train_rnd=True)))
