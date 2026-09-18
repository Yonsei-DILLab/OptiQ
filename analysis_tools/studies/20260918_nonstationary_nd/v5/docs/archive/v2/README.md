# Archived v2 variants

최종 버전은 [continuous-latent checked K64](../../v2/README.md)이다.
여기는 원안과 후속 비교 구성을 재현하기 위한 보관 경로다. 학습 기록·체크포인트는
기존 `outputs/` 위치를 유지한다. 원안의 이름을 최종 결과에 붙여 해석하지 않는다.

| 구성 | 명시적인 Hydra 경로 | 특징 |
|---|---|---|
| 첨부 원안의 최초 구현 | `archive/v2/original` | realized-u KDE h=.8, argmax MSE, T=.5 |
| 조건부 proposal 비교 | `archive/v2/conditional` | learned conditional mixture + full NLL, guard 없음 |
| 강한 수락 검사 비교 | `archive/v2/guarded` | min-Q teacher, 표준오차 margin 2 |
| 유한 latent 비교 | `archive/v2/finite` | 실제 16-component 정책, σ 상한 .2, latent residual .5 |
| Proximal 비교 | `archive/v2/proximal` | 실제 유한 혼합, ESS가 Q와 density의 공통 계수 η를 선택 |

`configs/archive/v2/`가 위 설정의 실제 정의 위치다. 예:

```bash
OPTIQ_CONFIG=archive/v2/original scripts/run_v2.sh 0 --check
```

`mujoco_v2_conditional`, `mujoco_v2_guarded`, `mujoco_v2_finite`,
`mujoco_v2_proximal`은 예전 명령·supervisor·검증 도구 호환용 작은 alias 파일이다.
이 네 이름의 resolved configuration은 경로 정리 전과 동일하다.
**기존 `mujoco_v2` 이름의 기본값만 최종 버전으로 승격했다.** 원안을 재현하려면
반드시 `archive/v2/original`을 지정한다. 초기 calibration/screen 도구와 원안 테스트는
이 보관 경로를 명시하도록 수정했다.

## 문서와 도구

- [첨부 원안](V2_IDAC_PSEUDOCODE.txt), [최초 구현 설명](V2_IDAC.md).
- [초기 calibration](V2_CALIBRATION_RESULTS.md), [전체 조사 이력](V2_IMPROVEMENT.md).
- [유한 혼합 구성](V2_FINITE_MIXTURE.md), [유한 혼합 중단 결과](V2_FINITE_SCREEN_RESULTS_KO.md).
- [Projection 진단](V2_PROJECTION_DIAGNOSIS_KO.md), [proximal 구성](V2_PROXIMAL_TARGET.md).

공통 source 및 평가 도구는 `optiq_dime/`, `scripts/`에 남긴다. 이미 저장된 manifest와
실행 중인 supervisor가 이 경로·파일 해시를 참조하므로 임의 이동하지 않는다.
특히 `evaluate_v2_confirmation.py`, `report_v2_finite.py`, `evaluate_v2_finite.py`,
`monitor_v2_proximal.py`, `finish_v2_finite.py`는 기존 기록과 실행을 지원한다.
`scripts/supervisor_v2_{screen,conditional_screen,guarded_screen,finite,proximal}.sh`와
대응하는 `deploy/supervisor/` 파일은 보관 구성의 명시적 실행 도구다.
경로 정리는 새 실험을 시작하거나 기존 프로세스를 중단하지 않는다.
