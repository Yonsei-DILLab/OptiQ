"""Primary routefast figures, verified full-start scores and explicit assistance."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from antmaze.evaluation import atomic_json
from .report import draw_paths
from .verify import verify_campaign


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    a=p.parse_args();root=a.root;out=root/"report";out.mkdir(exist_ok=True)
    validation=verify_campaign(root)
    atomic_json(out/"validation.json",validation)
    methods=["optiq","sac","mfpo","meow"]
    records={m:json.loads((root/"runs"/f"v1-{m}-s0"/"result.json").read_text()) for m in methods}
    fig,axes=plt.subplots(2,4,figsize=(14,8.5))
    for col,m in enumerate(methods):
        folder=root/"runs"/f"v1-{m}-s0"
        for row,label in enumerate(("policy-fixed","policy-natural")):
            ax=axes[row,col];draw_paths(ax,folder,"v1",label)
            ax.set_title(m.upper()+"\n"+ax.get_title(),fontsize=10)
            ax.set_xlabel("Identical full starting state" if row==0 else "Original random starting distribution")
    fig.suptitle("AntMaze v1 · seed 0 · 100k interactions / 91,808 updates\n"
        "Training: route shaping + reset curriculum + NovelD; evaluation: direct policy, original starts",fontsize=13)
    fig.text(.5,.01,"100 rollouts per panel · Gray: failure · Blue: success · No curriculum resets or planner at evaluation",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.03,1,.93));fig.savefig(out/"routefast-trajectories.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    summary={}
    for m,r in records.items():
        folder=root/"runs"/f"v1-{m}-s0"
        h=json.loads((folder/"history-policy-natural.json").read_text())
        axes[0].plot([x["step"] for x in h],[x["success_rate"] for x in h],label=m.upper())
        axes[1].plot([x["step"] for x in h],[x["mean_min_distance"] for x in h],label=m.upper())
        progress=json.loads((folder/"progress.json").read_text())
        summary[m]=dict(steps=r["steps"],updates=r["updates"],seconds=r["seconds"],
            fixed=r["summaries"]["policy-fixed"],natural=r["summaries"]["policy-natural"],
            training={k:v for k,v in progress.items() if k.startswith("training_")})
    for ax in axes:
        ax.set_xlabel("Environment interactions");ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel("Success fraction (10 full-start rollouts)");axes[0].set_ylim(-.03,1.03)
    axes[1].set_ylabel("Mean closest goal distance (m)")
    fig.suptitle("One training seed per method · Evaluation excludes training assistance")
    fig.tight_layout();fig.savefig(out/"routefast-learning.png",dpi=180);plt.close(fig)
    atomic_json(out/"routefast-summary.json",summary)
    lines=["# DDiffPG 기반 빠른 길찾기: v1 / seed 0 / 100k", "",
        "모든 평가 궤적은 원래 출발점 조건에서 직접 정책을 샘플링했다. 학습에만 우회거리 보상, 시작점 curriculum, NovelD를 적용했다.",
        "100k interaction 중 warmup8192, 실제 learner/RND91808회 업데이트. 4개 모두 독립 학습했고 단일 시드 예비 결과다.","",
        "| 방법 | 고정 출발 성공/100 | 원래 랜덤 출발 성공/100 | 고정 출발 성공 경로 |", "|---|---:|---:|---|"]
    for m,s in summary.items():
        lines.append(f"| {m} | {round(s['fixed']['success_rate']*100)} | {round(s['natural']['success_rate']*100)} | {s['fixed']['successful_routes']['counts']} |")
    lines.extend(["", "Curriculum 학습 중 성공 횟수는 원래 출발점 평가 성공률과 구분한다. 성능이 올라도 보상·curriculum·UTD 등이 함께 바뀌었으므로 단독 요인의 효과로 해석하지 않는다.",
        "정책 입력에는 경로/waypoint를 추가하지 않았지만, 학습 보상이 미로 지도 정보를 이용하므로 무보조 탐색 benchmark로 부르지 않는다.",
        "검증: 원자료에서 평가 보상·목표 도달·경로·초기 full state 재계산, 정확한 update/transition 수와 checkpoint hash, actor/critic 변경 및 RND target 보존 확인.",
        "",f"Source: {validation['training_source_commit']}"])
    (out/"ROUTEFAST_REPORT_KO.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":main()
