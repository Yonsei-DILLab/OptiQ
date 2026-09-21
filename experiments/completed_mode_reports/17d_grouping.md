# 17D에서 assignment 구조를 찾는 방법

현재 질문은 “17D action에 몇 개의 Gaussian cluster가 있는가”보다 **“같은 student들이 담당하는 candidate들의 묶음이 있는가”**에 가깝다. 따라서 action 좌표를 PCA 한 축에 투영하는 것보다 responsibility 행렬 자체를 사용하는 정렬·co-clustering을 먼저 권한다. 아래는 제안이며 새 학습에 적용하지 않았다.

## 1. 군집 수를 정하지 않고 그림부터 정렬

Candidate j의 특징을 posterior responsibility vector로 둔다.

$$v_j=(\gamma_{1j},\ldots,\gamma_{Nj}),\qquad \sum_i\gamma_{ij}=1.$$

두 candidate가 같은 student들에게 설명되면 이 벡터가 비슷하다. 예를 들어 Hellinger 거리로 비교할 수 있다.

$$d^2(j,l)=\frac12\sum_i(\sqrt{\gamma_{ij}}-\sqrt{\gamma_{il}})^2.$$

열은 이 거리, 행은 row-conditional R의 같은 형태의 거리로 계층적 정렬한다. Dendrogram을 특정 cluster 수로 자르지 않고 leaf ordering만 사용하면 **mode 수를 강제하지 않고도** 가까운 assignment 패턴을 이웃하게 볼 수 있다. 그림의 색은 원래 R을 유지하고 순서만 바꾼다. 정렬은 늘 그럴듯한 구조를 만들 수 있으므로 원본 그림도 함께 보존한다. [SciPy optimal leaf ordering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.optimal_leaf_ordering.html)

## 2. 학습에 쓸 묶음은 student–candidate 공동 grouping

독립적인 행·열 정렬은 시각화다. 학습에서 candidate label이 필요하면, effective assignment를 가중 bipartite graph로 보고 행과 열을 함께 묶는 spectral co-clustering을 검토할 수 있다.

$$J_{ij}=w_j\gamma_{ij},\qquad \alpha_i=\sum_jJ_{ij},\qquad S_{ij}=\frac{J_{ij}}{\sqrt{\alpha_iw_j}}.$$

양의 질량 부분에서 정의한 S의 비자명한 singular vectors는 normalized bipartite graph 구조를 나타낸다. 연속적인 spectral ordering 단계에서는 cluster 수가 필요하지 않다. 실제 partition을 만들 때는 cluster 수 또는 분리 기준이 필요하며, 알고리즘이 이를 공짜로 해결해 주지는 않는다. [Spectral co-clustering 공식 설명](https://scikit-learn.org/stable/modules/biclustering.html#spectral-co-clustering)

후보 group C_m을 정했다면 기존 oracle basin 대신 이 group을 사용한다.

$$H_{mi}=\frac{\sum_{j\in C_m}J_{ij}}{\alpha_i},\qquad m_i=\arg\max_mH_{mi}.$$

그 후 mode selection처럼 자기 담당 group의 output-gradient contribution만 남긴다. Confidence 곱을 추가하지 않는다. 다만 지금은 이 partition rule을 선택하거나 MuJoCo 학습에 적용하지 않았다.

## 3. 무엇을 검증할 것인가?

- 동일 checkpoint에서 candidate/latent를 재표집해도 그룹이 안정적인가? 공통 probe action을 사용해 표본 index 변화와 구조 변화를 구분한다.
- 작은 teacher 질량이나 거의 사용되지 않는 row를 정규화해서 노이즈를 과장하고 있지는 않은가? J·R과 함께 w·alpha도 본다.
- Toy에서는 **추정된 그룹만 학습에 사용**하고, 정답 basin은 사후 평가에만 사용한다. 이전 oracle 결과와 분리한다.
- 그룹이 잘 보이지 않을 때도 무조건 여러 개로 쪼개지 않는다. 안정적인 분할이 없는 경우 원래 GMM gradient를 유지하는 설계는 별도 ablation으로 검토할 수 있다.

이렇게 찾는 것은 우선 **assignment group**이다. 곧바로 Q의 local maximum이나 Boltzmann density mode라는 보장은 없다. 예를 들어 모든 conditional Gaussian이 같으면 gamma는 모든 열에서 균일하다. Q가 multimodal이어도 J는 행 방향으로 rank-one 구조가 되어 이 정보만으로 그 mode를 찾지 못한다. 그런 경우에는 action의 이웃 관계나 Q landscape 정보를 보완해야 한다.

따라서 먼저 책임 패턴으로 정렬한 그림과 재표집 안정성을 확인하고, 그룹이 실제로 존재할 때 그 분할을 training routing에 사용하는 순서가 적절하다. t-SNE/UMAP 그림의 섬을 곧바로 hard label로 쓰는 것보다 이번 알고리즘의 관심 대상과 더 직접적으로 연결된다.
