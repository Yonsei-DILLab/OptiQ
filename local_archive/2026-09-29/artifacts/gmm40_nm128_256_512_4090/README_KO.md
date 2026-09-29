# iBOLT GMM40 N/M ablation

- 요청: N=M 128,256,512 × seed0,1,2,3 =12개, 각100k actor update.
- 서버: `vast1`, RTX4090×4. 기존 GPU 작업 및 GPU 잠금을 존중하고 빈 슬롯만 사용.
- 학습 소스: `c429fbb2b22abd607e87b28d7577aa8c0ae4c536`, `direct-gmm-trg`, 실행 전 push 완료.
- 원격 경로: `/home/heechan/optiq-experiments/gmm40-ibolt-nm128-256-512-100k-4seed-4090-20260925`.
- W&B: `OptiQ/gmm-trg`, group은 원격 경로의 마지막 이름과 동일.
- 모델256×3 GELU, batch256, Adam3e-4, T1, beta1, random latent.
- mean-head variance scale16, log sigma[-5,-3.5], 초기-4, teacher std floor.05.
- 기존 논문용100k·256×3 N=M64 튜닝 설정에서 N/M만 변경.
- 기존 평가/저장 주기 유지, 매번 full-policy/μ-only 각각10,000개를 별도 저장.
- 100k optimizer audit와12개 완료 뒤 N/M별4seed 집계·분포 PDF/PNG 및 최종 archive 생성.

상태 수집:

```
/tmp/optiq-antmaze-report-20260922/bin/python artifacts/gmm40_nm128_256_512_4090/collect_status.py
```

완료 후 `--archive-completed`를 추가하면 최종 자료를 내려받아 SHA256을 검증하고 압축을 해제한다.
이는 기존 학습을 변경하거나 재시작하지 않는 읽기 전용 수집 명령이다.
