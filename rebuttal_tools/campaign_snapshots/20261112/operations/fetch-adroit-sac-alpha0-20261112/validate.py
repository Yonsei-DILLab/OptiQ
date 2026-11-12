from pathlib import Path
import sys,json,hashlib,inspect,traceback
import torch,jax
from rebuttal_tools.check_fetch_jax_sac import run
from baselines.jax_sac import JaxSAC
family=sys.argv[1];out=Path(sys.argv[2]);out.mkdir(exist_ok=True,parents=True)
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
results=[]
try:
 assert jax.default_backend()=='gpu'
 for task in ([None] if family=='fetch' else ['door','hammer','pen','relocate']):
  results.append(run(2,'cpu',out,task,0.))
 # The existing automatic-alpha path is unchanged, and also numerically checked.
 if family=='fetch':results.append(run(2,'cpu',out,None,'auto'))
 report=dict(passed=True,results=results,module_sha256=hashlib.sha256(Path(inspect.getfile(JaxSAC)).read_bytes()).hexdigest())
except BaseException as e:
 report=dict(passed=False,results=results,error=repr(e),traceback=traceback.format_exc())
(out.parent/f'validation-{family}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'passed':report['passed'],'family':family,'error':report.get('error')}),flush=True)
