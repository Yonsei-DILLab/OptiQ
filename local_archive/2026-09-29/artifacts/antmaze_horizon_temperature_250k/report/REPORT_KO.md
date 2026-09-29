# Gamma/temperature 단기 실험

현재 수집된 정책별 결과입니다. 기존 H/d +0.7 대조군을 포함하며, 표의 step이 다르면 최종 성능을 직접 비교하지 않습니다. 정확히 같은 step의 비교만 results.json의 exact_step_matched_comparisons에 보관합니다.

| 환경 | 조건 | 평가 | total step | 횟수 | 첫 통로 | 성공 통로 | 성공률 |
|---|---|---|---:|---:|---|---|---:|
| v1 | control | native | 200192 | 40 | {'uncommitted': 40} | {} | 0.0% |
| v1 | control | policy | 200192 | 40 | {'uncommitted': 40} | {} | 0.0% |
| v1 | gamma999 | native | 258304 | 100 | {'uncommitted': 100} | {} | 0.0% |
| v1 | gamma999 | policy | 258304 | 100 | {'uncommitted': 100} | {} | 0.0% |
| v1 | gamma999_temp3 | native | 258304 | 100 | {'uncommitted': 99, 'upper': 1} | {} | 0.0% |
| v1 | gamma999_temp3 | policy | 258304 | 100 | {'uncommitted': 100} | {} | 0.0% |
| v3 | control | native | 200192 | 40 | {'left': 27, 'uncommitted': 13} | {} | 0.0% |
| v3 | control | policy | 200192 | 40 | {'uncommitted': 18, 'left': 19, 'right': 3} | {} | 0.0% |
| v3 | gamma999 | native | 258304 | 100 | {'left': 54, 'right': 20, 'uncommitted': 26} | {} | 0.0% |
| v3 | gamma999 | policy | 258304 | 100 | {'left': 54, 'right': 33, 'uncommitted': 13} | {} | 0.0% |
| v3 | gamma999_temp3 | native | 258304 | 100 | {'left': 17, 'right': 79, 'uncommitted': 4} | {'right': 49} | 49.0% |
| v3 | gamma999_temp3 | policy | 258304 | 100 | {'left': 20, 'right': 80} | {'right': 33} | 33.0% |
| v3 | temp3 | native | 258304 | 100 | {'left': 47, 'right': 50, 'uncommitted': 3} | {'right': 14} | 14.0% |
| v3 | temp3 | policy | 258304 | 100 | {'left': 43, 'right': 48, 'uncommitted': 9} | {'right': 16} | 16.0% |
| v4 | control | native | 200192 | 40 | {'lower': 19, 'upper': 17, 'uncommitted': 4} | {} | 0.0% |
| v4 | control | policy | 200192 | 40 | {'upper': 16, 'lower': 18, 'uncommitted': 6} | {} | 0.0% |
| v4 | gamma999 | native | 258304 | 100 | {'upper': 38, 'uncommitted': 25, 'lower': 37} | {} | 0.0% |
| v4 | gamma999 | policy | 258304 | 100 | {'lower': 39, 'upper': 30, 'uncommitted': 31} | {} | 0.0% |
| v4 | gamma999_temp3 | native | 258304 | 100 | {'upper': 16, 'uncommitted': 53, 'lower': 31} | {} | 0.0% |
| v4 | gamma999_temp3 | policy | 258304 | 100 | {'upper': 23, 'uncommitted': 48, 'lower': 29} | {} | 0.0% |
| v4 | temp3 | native | 258304 | 100 | {'lower': 39, 'upper': 41, 'uncommitted': 20} | {} | 0.0% |
| v4 | temp3 | policy | 258304 | 100 | {'lower': 34, 'uncommitted': 39, 'upper': 27} | {} | 0.0% |

![동일 step 비교](matched_trajectories_v1.png)
![동일 step 비교](matched_trajectories_v3.png)
![동일 step 비교](matched_trajectories_v4.png)

policy는 random-z와 conditional sigma를 모두 사용하는 직접 정책입니다. native는 random-z μ-only 보조 결과입니다. 두 평가 모두 외부 DACER 행동잡음을 더하지 않습니다. v1은 원래 랜덤 시작이고 v3/v4는 원래 고정 full state입니다.

통로 진입·목표 도달·후기 체크포인트에서의 유지 여부를 따로 판단합니다. 단일 seed의 일시적인 양쪽 방문으로 목표 달성을 선언하지 않습니다.

![직접 정책](latest_trajectories_policy.png)
![직접 정책 곡선](learning_curves_policy.png)
![mu-only 보조](latest_trajectories_native.png)
![mu-only 곡선](learning_curves_native.png)

학습 source eee04de7f2c8a34feffda3d0fc376ff9ea1dfe45, 대조군 2564b59faa0d319eece496b93eff0f19359efc37. 별도 후처리이며 학습·원시 자료를 변경하지 않았습니다.
