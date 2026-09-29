# OptiQ / iBOLT GMM40 원시 샘플

100,000 actor optimizer updates 시점에 실제 저장된 원본 평가 데이터입니다.
재학습·재샘플링·필터링·정규화·순서 변경 없이 원본 파일을 그대로 복사했습니다.

- `nm64_seed3_100k/`: N=M=64, seed 3. 저장된 두 평가 모드 모두 coverage 40/40.
- `nm256_seed2_100k/`: N=M=256, seed 2. 저장된 두 평가 모드 모두 coverage 40/40.
- N=M=64는 요청에 따라 40/40인 seed 3을 선택했습니다. 전체 seed 평균을 나타내는 자료가 아닙니다.

각 폴더:
- `samples.npy`: fresh random latent + conditional sigma를 포함한 full-policy 원본 샘플.
- `samples_mu_only.npy`: fresh random latent, conditional sigma noise를 제거한 mu-only 원본 샘플.
- 두 배열은 각 (10000, 2), float32입니다. 각 행은 시각화 전 물리 좌표 (x1,x2)이며 이미 x=40*a 변환이 적용되어 있습니다. 다시 40을 곱하지 마세요.
- `metrics.json` / `metrics_mu_only.json`: 각각 해당 샘플의 원래 평가 지표.
- `target_definition.json`: 40개 GT 중심, 표준편차, 혼합 가중치, 범위 정의. GT 표준편차는 actor sigma와 다릅니다.
- `config.json`: 원래 실행 설정과 학습 소스 SHA.
- `update_count_audit.json`: 100k optimizer update 완료 증명.
- `metrics.jsonl`: 원래 저장된 학습/평가 지표 이력.
- `wandb_status.json`: 원래 W&B run 식별 정보.

공통 설정: 256x3 GELU, batch256, Adam3e-4, T1, beta1, random Gaussian latent,
mean-head variance scale16, actor log sigma[-5,-3.5], 초기-4, teacher-only sigma floor .05.
고정 Q=log p_GMM40를 사용하는 분포 모사 실험이며 critic/replay 학습은 없습니다.

Coverage는 기존 기준(가장 가까운 표준화 GT 중심의 3-sigma 안 샘플 수가
metrics의 coverage_threshold 이상인 성분 수)입니다. 원시 좌표로 counts/coverage를 재검증했습니다.
SHA256SUMS.txt와 MANIFEST.json에 원본 보관본과의 바이트 일치 검증을 기록했습니다.

읽기 예시 (이 폴더에서 실행):
```python
import numpy as np
x64 = np.load('nm64_seed3_100k/samples.npy', allow_pickle=False)
x256 = np.load('nm256_seed2_100k/samples.npy', allow_pickle=False)
mu64 = np.load('nm64_seed3_100k/samples_mu_only.npy', allow_pickle=False)
mu256 = np.load('nm256_seed2_100k/samples_mu_only.npy', allow_pickle=False)
```
