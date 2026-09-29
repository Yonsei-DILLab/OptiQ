from pathlib import Path
import runpy,contextlib,io,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    d=runpy.run_path(str(root/'plot_results.py'))
rows=sorted([r for r in d['records'] if 'progress100_geodesic_no_bonus' in r['job']],key=lambda r:r['task'])
fig,axs=plt.subplots(1,len(rows),figsize=(13,4.8),squeeze=False)
for ax,r in zip(axs[0],rows):
    d['mod'].audit.decorate(ax,r['task'])
    xy,goals,families,metric=d['panels'][r['job'],'fixed']
    for p,g,f in zip(xy,goals,families):
        ax.plot(p[:,0],p[:,1],color=d['mod'].COLORS[f],alpha=.5,lw=1)
        if not g:ax.scatter(*p[-1],marker='x',color='#c43150',s=16)
    ax.scatter(0,0,marker='*',color='black',s=70,zorder=10)
    ax.set_title(f"{r['task'].upper()} | {r['step']/1e6:.2f}M\n"+', '.join(f'{k}: {v}/40' for k,v in metric['corridors'].items())+f" | success {metric['successes']}/40",fontsize=11)
fig.suptitle('Step penalty ON: 100 x geodesic progress - 1; B=0\nFixed original full-state reset | Direct policy (random z + conditional sigma)',fontsize=13)
fig.tight_layout(rect=(0,.14,1,.86))
fig.text(.5,.025,'Seed 0; 40 episodes per checkpoint; different training steps across mazes. V1 retains random resets.',ha='center',fontsize=10)
fig.savefig(d['out']/'step_penalty_fixed.png',dpi=150);plt.close(fig)
for r in rows:
    files=list(d['out'].glob(f"*/runs/{r['job']}/step_*/provenance.json"))
    p=json.loads(files[0].read_text())
    spec=p['training_config']['reward_specification']
    assert spec['step_cost']==1 and spec['progress_scale']==100 and not spec['success_bonus_enabled']
    print(r['task'],r['step'],r['modes']['fixed'],spec['formula'])
text='''# Step penalty 고정 시작점 확인

보상 100*(d_current-d_next)-1, geodesic, B=0. T=1, DACER/NovelD OFF.
각 체크포인트 seed0의 direct policy(random z+conditional sigma)를 원래 학습과 같은 고정 전체 상태에서 40회 평가.

'''
for r in rows:
    m=r['modes']['fixed']
    text+=f"- {r['task']} {r['step']:,} steps: 통로 {m['corridors']}, 성공 {m['successes']}/40.\n"
text+='\n해당 체크포인트에서는 모두 단일 통로이며, step penalty를 줬다고 양쪽 경로가 유지되지는 않았다. penalty 없는 비교군과 학습량이 달라 penalty의 인과효과를 분리한 결과는 아니다.\n\n원자료 SHA256·전체 초기 상태 일치·모델 불변 검증 통과. 일부 서버 재평가의 W&B 업로드는 API 인증 누락으로 실패했으나 검증된 로컬 궤적/지표는 정상 보관됐다. 원래 학습 로그와 구분한다.\n\n![고정 시작 평가](step_penalty_fixed.png)\n'
(d['out']/'STEP_PENALTY_KO.md').write_text(text)
print(d['out']/'step_penalty_fixed.png')
