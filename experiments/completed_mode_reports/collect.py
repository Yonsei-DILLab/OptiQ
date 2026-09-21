"""Export only completed-run reporting artifacts; never touch checkpoints."""
import hashlib,io,json,tarfile,zipfile,time
from pathlib import Path
root=Path('/data1/heejoonorm/OptiQ/studies')
out=root/'20260921_completed_mode_reports';out.mkdir(exist_ok=True)
names=['20260921_gmm_mode_gradient','20260921_gmm_mode_gradient_batch32','20260921_gmm_mode_selection_frozen']
kept={'H','training_z','training_mu','training_log_sigma','gradient_gram','gradient_cosine','gradient_norm','routed_gradient_norm','routed_batch_metrics','single_group_gradient_cosine','teacher_mode_mass','batch_teacher_mode_mass','routed_method_names','routed_fixed_latent_delta','routed_branch_histograms','alpha','candidates','Q','log_q','logits','weights','candidate_labels','output_mode_gradients'}
manifest={'time':time.time(),'inputs':{},'exported':{}}
with tarfile.open(out/'report_inputs.tar.gz','w:gz') as tar:
 def add(p,data=None):
  rel=str(p.relative_to(root));original=p.read_bytes();manifest['inputs'][rel]=hashlib.sha256(original).hexdigest()
  data=original if data is None else data
  manifest['exported'][rel]=hashlib.sha256(data).hexdigest()
  ti=tarfile.TarInfo(rel);ti.size=len(data);tar.addfile(ti,io.BytesIO(data))
 for name in names:
  study=root/name;add(study/'campaign/SOURCE_MANIFEST.json')
  for run in sorted((study/'campaign/runtime/runs').iterdir()):
   if not (run/'COMPLETE.json').exists():continue
   cfg=json.loads((run/'config.json').read_text())
   for fname in ['config.json','COMPLETE.json','LAUNCH.json','REFERENCE.json','history.jsonl','training.jsonl','diagnostic_020000.json','eval_020000.npz']:
    p=run/fname
    if p.exists():add(p)
   p=run/'diagnostic_020000.npz'
   extra={'joint'} if name.endswith('frozen') and cfg['seed']==0 else set()
   buf=io.BytesIO()
   with zipfile.ZipFile(p) as src,zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as dst:
    for z in src.namelist():
     if Path(z).stem in kept|extra:dst.writestr(z,src.read(z))
   add(p,buf.getvalue())
   if name.endswith('frozen') and cfg['seed']==0:
    for st in [0,1000,5000]:add(run/f'eval_{st:06d}.npz')
(out/'INPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'archive':str(out/'report_inputs.tar.gz'),'bytes':(out/'report_inputs.tar.gz').stat().st_size,'files':len(manifest['inputs'])}))
