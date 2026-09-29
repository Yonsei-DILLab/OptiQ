"""Re-layout the exact user-provided GMM40 figure; preserve all samples/style."""
from pathlib import Path
import importlib.util, hashlib, json
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCE=ROOT/'artifacts/gmm40_paper_ibolt256_sql256_4seed/build_figure.py'
ORIGINAL=ROOT/'artifacts/gmm40_paper_ibolt256_sql256_4seed/gmm40_seed2_ibolt256_sql256_main.png'
ATTACHMENT=Path('/Users/yunheechan/Downloads/gmm40.png')
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ORIGINAL)==sha(ATTACHMENT)
spec=importlib.util.spec_from_file_location('gmm40_paper_original',SOURCE)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
# Original validator checks every raw sample, target, optimizer audit and metric.
target,samples,rows,provenance=original.check()
assert original.ALPHA==0.2 and original.AREA==2.1
original.common.style()
plt=original.plt
plt.rcParams.update({
    'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans',
    'font.size':18,'axes.titlesize':24,'axes.labelsize':22,
    'xtick.labelsize':18,'ytick.labelsize':18,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
})
fig,axes=plt.subplots(1,6,figsize=(19.8,4.05))
grid=original.common.contour_grid(target)
methods=('Ground truth',)+original.METHODS
for i,(ax,name) in enumerate(zip(axes,methods)):
    title=f'({chr(97+i)}) {name}'+(' (ours)' if name=='iBOLT' else '')
    original.panel(ax,samples[name],grid,title,ylabel=(i==0))
    ax.set_title(title,fontsize=24,pad=13,fontweight='bold' if name=='iBOLT' else 'normal')
    ax.set_xlabel(r'$a_1$',fontsize=22,labelpad=4)
    if i==0:ax.set_ylabel(r'$a_2$',fontsize=22,labelpad=4)
    ax.tick_params(axis='both',labelsize=18,pad=3)
    np.testing.assert_array_equal(ax.collections[-1].get_offsets(),samples[name])
fig.subplots_adjust(left=.043,right=.994,bottom=.22,top=.80,wspace=.30)
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
# Check neighboring panel titles do not collide at publication font sizes.
for left,right in zip(axes[:-1],axes[1:]):
    assert left.title.get_window_extent(renderer).x1 < right.title.get_window_extent(renderer).x0
stem='gmm40_1x6_alpha0p2_s2p1_largefont'
for ext in ('png','pdf','svg'):
    fig.savefig(OUT/f'{stem}.{ext}',dpi=360,bbox_inches='tight',pad_inches=.08)
plt.close(fig)
provenance.update({
    'input_figure':str(ATTACHMENT),'input_figure_sha256':sha(ATTACHMENT),
    'matched_original_figure':str(ORIGINAL),'original_plotter':str(SOURCE),
    'original_plotter_sha256':sha(SOURCE),'layout':'1x6','all_samples_unchanged':True,
    'contours_unchanged':True,'point_alpha':.2,'point_area_pt2':2.1,
    'font_family':'DejaVu Sans','title_pt':24,'axis_label_pt':22,'tick_label_pt':18,
    'ibolt_title_bold_only':True,'figure_inches':[19.8,4.05],
    'render_script_sha256':sha(Path(__file__)),
    'outputs':{ext:{'path':str(OUT/f'{stem}.{ext}'),'sha256':sha(OUT/f'{stem}.{ext}')} for ext in ('png','pdf','svg')},
})
(OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
# Display-sized QA preview; full-resolution paper image remains untouched.
im=Image.open(OUT/f'{stem}.png');size=im.size
im.thumbnail((2200,800));im.save(OUT/'preview.png')
print(json.dumps({'image':str(OUT/f'{stem}.png'),'size':size,'title_font':24,'tick_font':18,'sample_checks':'passed','exact_source_image_match':True},ensure_ascii=False))
