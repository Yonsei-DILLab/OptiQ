# Final OptiQ v2

사용자가 2026-09-11 최종 버전으로 선택한 **continuous-latent checked K64** 구성이다.
Humanoid-v4 seed 0·1·2·3의 900K–1M 평균은 5362.67, 최종 정책 새 평가 평균은
5397.09다. 해당 학습 코드의 기준 커밋은 `8cb4f237aacb61113f65e693e23236f17d77f978`이다.

- [독립 구현용 상세 의사코드](PSEUDOCODE.md): 수식, shape, 난수, gradient, 실행 순서.
- [알고리즘 설명](ALGORITHM_KO.md): 읽기 쉬운 개요와 원안 대비 변경.
- [실행·이식·재현 안내](REPRODUCIBILITY.md): 설치, 명령, 시드, 평가 및 저장 규칙.
- [검토 결과](REVIEW.md): 설정 동등성, 수치 및 실행 검증과 한계.
- [완료된 실험 결과](RESULTS_KO.md), [이론 조건](THEORY.md).
- [실험 설정 원본의 수치](REFERENCE_CONFIG.json), [출처·해시 manifest](MANIFEST.json).
- [구버전 경로 안내](../archive/v2/README.md).

기본 실행: `scripts/run_v2.sh 0`. 설정만 확인: `scripts/run_v2.sh 0 --check`.
`mujoco_v2`, `mujoco_v2_checked`, `v2/final`은 같은 최종 설정을 읽는다.
실제 설정 정의는 [configs/v2/final.yaml](../../configs/v2/final.yaml)에 한 번만 둔다.
공통 학습 구현은 `optiq_dime/`에 유지한다. 이 경로에 보존된 finite/proximal 옵션은
최종 설정에서 활성화되지 않는다.
