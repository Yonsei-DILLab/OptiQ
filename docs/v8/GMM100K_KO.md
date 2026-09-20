# Raw-z OptiQ v8: GMM40 100K 실행 기록

2026-09-20 01:17 UTC 시작. 사용자 지시에 따라 실험부터 시작한 뒤 문서를 정리했다.
현재 진행 중인 실험이며 100K 완료나 분포 복원 성공을 주장하는 기록이 아니다.

실행 코드 commit: `7b0076b9ed5e9ed31e8d6415eb2157f38520ca34` (`v8`).
이후 문서·docstring 정리 commit은 이 실행의 고정 source를 변경하지 않는다.

| GPU | Seed | 요청한 updates | Supervisor |
|---|---:|---:|---|
| 0 | 0 | 100000 | optiq-v8-rawz-s0-100k |
| 1 | 1 | 100000 | optiq-v8-rawz-s1-100k |
| 2 | 2 | 100000 | optiq-v8-rawz-s2-100k |
| 3 | 3 | 100000 | optiq-v8-rawz-s3-100k |

모두 새 seed별 초기화에서 시작했다. 이전 Gaussian-cost prototype의 중단 checkpoint를
재개하지 않았다. 그 prototype의 실패 로그와 source는 별도 디렉터리에 보존했다.

## 설정

- Raw normal latent 적분점 H=4096. OT source는 g의 mean이나 Gaussian이 아니다.
- Teacher Gaussian256개에서 각 행동 하나 → importance correction →16개 사전 재표집.
- Cost=||z−teacher_pre_tanh_u||², 거리 epsilon=.1, Boltzmann alpha=1.
- 상태/lane마다 fresh Sinkhorn, min/max10/2000, 행·열 relative tolerance1e-3.
- 16개 선택 latent의 새 행동으로 조건부 SAC loss. 추가 source importance ratio 없음.
- Batch256, MLP256×2, Adam3e-4, actor clipping 없음, 초기 sigma=.5.
- 평가10000개, reference seed20260917; checkpoint는0·1K·5K 간격 및 최종.
- Critic을 학습하지 않는 fixed-Q GMM40 실험이다. RL 장기 학습은 시작하지 않았다.

## 산출물

이 인스턴스의 campaign root:

```text
/root/optiq-experiments/v8/gmm40-training/rawz-H4096-P256-K16-T1-e01-20260920T011741Z
```

- `manifest.json`: commit, GPU/seed, 명령 및 설정.
- `source/`: 변경하지 않는 detached worktree.
- `preflight.json`: 실행 source의 SHA256과 GPU300 updates·정확한 재개 검증.
- `seedN/config.json`, `training.jsonl`, `status.json`, `runtime.json`.
- `seedN/checkpoints/`, `checkpoint_audits.json`: actor·Adam·RNG 저장 및 검증.
- `seedN/evaluations/step_*/`: GT 중심을 위에 표시한 samples.png와 평가 지표.
- `logs/seedN.log`: 각 worker 출력.

실행 직후 seed0–3 모두 실제500–600 updates와 OT 수렴 통과를 확인했다.
그 구간의 warm median은 약56–66ms/update였으며, 최대 반복은356회 이하였다.
이는 초기 구간 관측치이며 장기 속도·분포 복원 성능의 보장은 아니다.
수렴 오류·비유한 입력이면 해당 업데이트를 거부하고 유효 prefix를 저장한다.

첫 1K 평가에서 seed0–3의 coverage는 모두18/40, 모드 근처 비율은
68.73–70.42%였다. 초기 분포보다 근처 비율은 증가했지만 MMD²는
0.038–0.041에서0.059–0.068로, SW2는약7.2에서10.9–11.6으로 악화했다.
따라서 일부 모드에 집중하는 초기 경향이며 전체 분포 복원 성공으로 해석하지 않는다.
1K의 actor·Adam·RNG 체크포인트4개 저장 및 감사가 통과했다.

외부 baseline은 기존 원본 구현을 `gmm40-baseline/`에 보존했다. 이번 실행과 동시에
baseline 재학습을 시작하지 않았으며, 과거 결과와 비교할 때 설정 차이를 구분해야 한다.
