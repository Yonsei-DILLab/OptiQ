from pathlib import Path
import sys,json,shutil,collections
base=Path('/lustre/hobbit9882/OptiQ-nonstationary-q-20260917/extensions/20260918_nd')
sys.path.insert(0,str(base/'operations_a08517e_reduced/reporting'))
from build_report import Report,GROUPS,DIMS
# Reporting-only label correction for variable matrix sizes.
import build_report
original_atlas=Report.atlas
def labeled_atlas(self,group,dim):
 self.matrix_label=f"{self.ts[0]['n']}×{self.ts[0]['m']}" if self.ts else 'unknown'
 return original_atlas(self,group,dim)
# Patch the plotting function text in a separate reporting module, not training source.
report_source=Path(build_report.__file__).read_text().replace("16×64, seed0 | independent final histogram", "{getattr(self,'matrix_label','16×64')}, seed0 | independent final histogram")
# Avoid nested single quote in the original f-string.
report_source=report_source.replace("getattr(self,'matrix_label','16×64')", 'self.matrix_label')
ns=dict(build_report.__dict__);exec(compile(report_source,'reporting_size_label','exec'),ns)
Report=ns['Report'];orig_init=Report.__init__
def sized_init(self,src,out,n=16,m=64):
 orig_init(self,src,out,n,m);self.matrix_label=f'{n}×{m}'
Report.__init__=sized_init
src=base/'report_input_completed_20260919';out=base/'report_output_completed_20260919';out.mkdir(exist_ok=True)
old=base/'cc11f537af33/report_output_20260919/figures'
small=out/'N16_M64';small.mkdir(exist_ok=True)
if old.exists():shutil.copytree(old,small/'figures',dirs_exist_ok=True)
r=Report(src,small);r.summarize()
if not old.exists():
 r.design()
 for group in GROUPS:
  r.tracking(group)
  if group[1] in ['mass','split']:r.heatmap(group)
  if group[1]=='mass':r.mass_flow(group)
  for dim in DIMS:r.atlas(group,dim)
 r.eps_plot();r.before_after();r.surface('025200');r.surface('final_independent');r.assignment();r.teacher()
else:
 r.tracking(('tri','closed'))
 for dim in DIMS:r.atlas(('tri','closed'),dim)
print('SMALL_DONE',len(r.rows),flush=True)
for n,m in [(256,16384),(1024,4096)]:
 r=Report(src,out/f'N{n}_M{m}',n,m);r.summarize()
 for group in GROUPS:
  ts=[t for t in r.ts if (t['family'],t['stage'])==group]
  if not ts:continue
  r.tracking(group)
  for dim in sorted({t['dim'] for t in ts}):r.atlas(group,dim)
 print('SIZE_DONE',n,m,len(r.rows),flush=True)
shutil.copy2(src/'inventory.json',out/'inventory.json')
print('COMPLETE_ANALYSIS',out,flush=True)
