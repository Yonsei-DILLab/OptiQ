# Positive DACER target16개 최종 요약

W&B에서16개 모두 finished,508416total transitions/15632learner updates/최종100회 직접 정책 평가를 확인했습니다. 목표 entropy는 차원당 +0.1,+0.5,+0.7,+0.9이며500learner updates마다 갱신합니다. 모든 실험 seed0, teacher T1입니다.

| 환경 | +0.1 | +0.5 | +0.7 | +0.9 |
|---|---:|---:|---:|---:|
|v1|[0%](https://wandb.ai/OptiQ/antmaze/runs/mfaz1k6t)|[0%](https://wandb.ai/OptiQ/antmaze/runs/sxtwgirz)|[0%](https://wandb.ai/OptiQ/antmaze/runs/zy4n8k5a)|[0%](https://wandb.ai/OptiQ/antmaze/runs/kk2y92f3)|
|v2|[100%](https://wandb.ai/OptiQ/antmaze/runs/86pvpd59)|[100%](https://wandb.ai/OptiQ/antmaze/runs/e4dxy0g3)|[100%](https://wandb.ai/OptiQ/antmaze/runs/jqnq5pfa)|[100%](https://wandb.ai/OptiQ/antmaze/runs/zj6e792j)|
|v3|[1%](https://wandb.ai/OptiQ/antmaze/runs/ykojf55f)|[48%](https://wandb.ai/OptiQ/antmaze/runs/fowum0dg)|[78%](https://wandb.ai/OptiQ/antmaze/runs/x01sqa1w)|[19%](https://wandb.ai/OptiQ/antmaze/runs/4lyuiibk)|
|v4|[0%](https://wandb.ai/OptiQ/antmaze/runs/xb2zi45n)|[0%](https://wandb.ai/OptiQ/antmaze/runs/z1uguklk)|[0%](https://wandb.ai/OptiQ/antmaze/runs/x5j8mst0)|[0%](https://wandb.ai/OptiQ/antmaze/runs/pv4gv927)|

![성공률](final_success_wandb.png)

성공률과 다중 경로 유지는 다릅니다. 수집한 원시 궤적에서는 v2는 동일 목표/경로로 집중했고, v3의 성공도 한쪽에 집중했습니다. v1/v4는 최종 성공이 없습니다. 16개 중14개는 최종100회 원시 궤적과 체크포인트 증명까지 로컬 검증했습니다. v3 +.5 및 v4 +.7의 최종 성공률은 W&B 기록이며,199서버 접속 중단으로 이 두 정책의 최종 원시 궤적 수집은 남아 있습니다. 해당 정책의500224step/40회 궤적을100회 최종 궤적으로 바꾸어 표기하지 않았습니다.

직접 정책은 random z+conditional sigma, 외부 DACER 행동잡음은 평가에서 제외합니다. v1은 학습과 동일한 랜덤 시작, v2~v4는 원래 고정 full state입니다. 보상은100*(Euclidean distance decrease), 성공 bonus0,step penalty0,NovelD OFF입니다. 경로 유지 목표는 아직 달성하지 못했습니다.
