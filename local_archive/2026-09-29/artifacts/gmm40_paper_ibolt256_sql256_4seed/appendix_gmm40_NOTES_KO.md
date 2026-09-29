# GMM40 Appendix 작성 근거

대상은 본문 `fig:gmm40_sampling` 및 `tab:gmm40_sampling_quality`의 100k / iBOLT N=M256 / full-policy 비교다. 원고 전체를 수정하지 않았으며 `appendix_gmm40.tex`는 기존 GMM40 TODO subsection을 대체할 삽입용 파일이다.

- 원자료 및 20개 seed별 지표: 같은 디렉터리 `per_seed.csv`, `results.json`, `build_figure.py`.
- 공통 타깃: `artifacts/gmm40_5090_queue/results/results/target/definition.json`.
- iBOLT: `artifacts/gmm40_dacer_off_nm_4090/inputs/gmm40-ibolt-nm128-256-512-100k-4seed-4090-20260925/results/ibolt_nm256_s2_100k/config.json`, source c429fbb2b22abd607e87b28d7577aa8c0ae4c536. 실행 adapter와 policy/proposal source를 대조했다.
- SAC/DIPO/MFPO: source 87d5d8ff210569ace7e59bd8a53ad02141b67f0a의 보존 adapter (`artifacts/gmm40_5090_queue/worktree/gmm40/torch_agents.py`, `mfpo.py`). CLI 공통 필드인 width/depth는 MFPO/DIPO 실제 구조를 나타내지 않으므로 그대로 사용하지 않았다.
- SQL: source 8a73d69112d73fd88867bfe75d42ba9a4348a379, `artifacts/gmm40_sql_particles_100k/shard1/results/sql_k256_s2_100k/config.json`; 포트 코드 `tmp/gmm40-nm-4090-source/gmm40/sql_jax.py`와 `SQL_BASELINE.md`.
- 타깃 upstream: DiKL b7df982bec4dc83fed65f98feeecdde928089272, `DiKL/energy/mog40.py`. 보존된 target 생성기 및 target JSON에 따라 기재했다. 재현 가능한 center 집합은 JSON을 함께 제공한다.

원고와 맞춰야 할 사항:
1. Common Settings의 모든 실험 RTX5090, SAC JAX, SQL TensorFlow라는 문구는 이 GMM40 실험에는 맞지 않는다. 해당 문장을 관련 benchmark로 한정하거나 예외를 명시해야 한다. SQL 실행 GPU는 이 설명에 미확인 수치를 추가하지 않았다.
2. Method의 q=finite actor mixture 설명과 달리, GMM40에서는 proposal-only sigma floor 0.05가 actor 상한 exp(-3.5)보다 크므로 proposal과 actor likelihood mixture가 다르다. 부록에 명확히 구분했으며, main method에도 이 실험용 proposal 변형을 연결하는 문장을 넣을 수 있다.
3. 40/40은 component-center 기반 coverage이며 density local maxima 40개를 별도로 증명한 결과가 아니다.
4. 표와 그림은 conditional sigma를 포함한 full-policy 결과다. MMD는 10k 전체가 아니라 앞 2,048개로 계산한다.
5. 동일한 100k actor updates는 Q-query 수나 계산량이 동일함을 의미하지 않는다.
6. PDF는 일반 article 클래스의 레이아웃 검사용이다. ICLR 본문에는 subsection 파일만 삽입한다.
