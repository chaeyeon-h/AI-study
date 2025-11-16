결정 트리 *Decision tree*는 분류/ 회귀 / 다중 출력 작업까지 가능한 다목적 알고리즘이다. 
이번 장에서는 결정 트리 *Decision Tree*의 훈련 , 시각화, 예측 방법, 규제, 제약 사항 등을 다루고자 한다.

- 화이트 박스 모델 
  - ex) 결정트리
  - 결정트리는 필요하다면 수동으로 직접 따라 해볼 수 있는 간단, 명확한 분류 방법 사용
- 블랙박스 모델 
  - ex) 랜덤 포레스트, 신경망
  - 신경망이 어떤 사람이 사진에 있다고 판단했을 때 어떤 이유로 예측을 했는지 파악하기 어렵


## 결정 트리 학습과 시각화
- 결정 트리 모델 `DecisionTreeClassifier` 훈련
- `export_graphviz()` 함수로 그래프 정의를 `iris_tree.dot` 파일로 출력해 훈련된 결정 트리 시각화 가능 (사이킷런에서 `.dot` 파일을 만들지 않고 트리를 그리는 `plot_tree()` 함수 또한 제공됨)
- `graphviz.Source.from_file()` 통해 시각화, `graphviz`는 오픈 소스 그래프 시각화 소프트웨어 패키지
  - dot 파일을 pdf나 png 등과 같은 다양한 형식으로 변환하는 도구가 포함되어 있음

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source

iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)

export_graphviz(
    tree_clf,
    out_file="iris_tree.dot",
    feature_names=["꽃잎 길이(cm)", "꽃잎 너비(cm)"],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

Source.from_file("iris_tree.dot")
```
<img width="600" alt="image"  src="https://github.com/user-attachments/assets/e1ddb4a0-21f2-45da-871b-47887139ad3a" />

## 예측
- 결정 트리에서의 예측 경로
  - 루트 노드부터 조건에 부합하는지 여부를 확인하다가 **리프노드**에 도달하면, 해당 노드의 클래스로 최종 클래스를 예측하게 된다.
- 결정 트리의 장점으로는 특성의 스케일을 맞추거나 평균을 원점에 맞추는 등의 **데이터 전처리**가 필요하지 않다는 것이다.

### 지니 불순도 *Gini Impurity*
$$Gini=1−\sum_{k=1}^{K}p_{k}^2$$

- 지니 불순도는 노드 안에 있는 샘플들의 클래스 혼합 정도를 나타내는 값
  - 한 노드의 모든 샘플이 같은 클래스에 속한다면 이 노드는 순수(gini=0)하다고 할 수 있다.


- 위 시각화된 그림에서 초록색 노드를 보면 지니 불순도는 아래와 같이 계산할 수 있다.
  - Samples = 54, value = [0, 49, 5]
  - $1 - (0/54)^2 - (49/54)^2 - (5/54)^2$

## CART 훈련 알고리즘
- CART (Classification And Regression Tree) 알고리즘을 통해 사이킷런에서 결정트리를 훈련시키며 True/False 두 개의 자식 노드만을 가진다.
- 훈련 세트를 하나의 특성 k의 임계값 $t_k$를 사용해 두 개의 서브셋으로 나눈다.
 - 가장 순수한 서브셋으론 나눌 수 있는 $(k, t_k)$ 쌍을 찾는다.
- 가장 적합한 k와 임계값을 찾기 위한 CART 비용 함수는 아래와 같다.
 <img width="500" alt="image" src="https://github.com/user-attachments/assets/ed4028c6-f164-4099-99b7-c50ceafcfe82" />

- 최대 깊이가 되거나 불순도를 줄이는 분할을 찾을 수 없을 때 멈춘다.
- CART 알고리즘은 현재 단계에서 즉시 가장 좋은 분할만 선택하기 때문에 Greedy 알고리즘이다.
- 최적의 트리를 찾는 것은 NP-Complete 문제로 $O(exp(m))$ 시간을 필요로 한다. 매우 작은 훈련 세트에도 적용하기 어렵다.

## 계산 복잡도
- **훈련 알고리즘**
  - n개의 특성에 대해 m개의 노드를 비교해야하므로 $O(n \times m \space log_2{m})$
- **탐색 알고리즘**
  - 일반적인 결정 트리는 균형을 이루고 있기 때문에 $O(log_2{m})$
 
## 지니 불순도와 엔트로피
- `DecisionTreeClassifier` 클래스는 기본적으로 지니 불순도 사용
- 가장 많이 차지하는 클래스가 누구냐에 집중 → 특정 클래스가 우세하면 그쪽으로 쉽게 치우침
**엔트로피 수식**
<img width="200" alt="image" src="https://github.com/user-attachments/assets/bcc94fdb-0dd3-493b-8f17-75a852d6e77e" />

- 전체 클래스 분포의 균형을 강조 → 작은 클래스까지 고려하여 분할
- 보통은 둘다 비슷한 트리를 만들어낸다.

_그렇다면 어떤 지표가 더 우세한 지표인가?_
- 제곱 연산은 컴퓨터가 아주 빠르게 계산할 수 있지만, 로그 연산은 제곱보다 계산량이 많고 비용이 크기 때문에 **지니 불순도가 일반적으로 계산이 조금 더 빠름**
- 엔트로피를 사용하면, 소수 클래스까지 반영되어 트리가 조금 **더 균형 잡히는 경향이 있음**


## 규제 매개변수
- 결정 트리는 훈련 데이터에 대한 제약사항(ex. 데이터가 선형)이 거의 없기 때문에 **과대적합되기 쉽다.**
- 결정트리는 훈련 전 파라미터 수가 결정되지 않으므로 비파라미터 모델 nonparametric model
  - 모델 구조가 데이터에 맞춰지므로 고정되지 않고 자유로움 (과대적합될 가능성 높아짐)
 
- 과대적합을 피하기 위해 규제를 가할 수 있다.
```
max_depth : 트리의 최대 깊이를 제어
max_features : 각 노드에서 분할에 사용할 특성의 최대 수
max_leaf_nodes : 리프 노드의 최대 수
min_samples_split : 분할되기 위해 노드가 가져야 하는 최소 샘플 수 
min_samples_leaf : 리프 노드가 생성되기 위해 가지고 이어야 할 최소 샘플 수 
min_weight_fraction_leaf : min_sampels_leaf와 같지만 가중치가 부여된 전체 샘플 수에서의 비율
min으로 시작하는 매개변수를 증가시키거나 max로 시작하는 매개변수를 감소시키면 규제가 커진다.
```

<img width="978" height="378" alt="image" src="https://github.com/user-attachments/assets/d4e6022f-f9c2-4ebb-bb4c-e177453336e4" />

- 테스트 셋에서의 정확도를 확인해보면 왼쪽보다 오른쪽의 정확도가 더 높음을 알 수 있다.
```python
tree_clf1.score(X_moons_test, y_moons_test)
# 0.898
tree_clf2.score(X_moons_test, y_moons_test)
# 0.92
```

## 회귀
- 결정 트리는 회귀 문제에서도 사용된다. `DecisionTreeRegressor` 사용
- 각 노드에서 클래스를 예측하는 대신, 어떠한 값을 예측한다.
```python
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X_quad = np.random.rand(200, 1) - 0.5  # 간단한 랜덤한 입력 특성
y_quad = X_quad ** 2 + 0.025 * np.random.randn(200, 1)

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_quad, y_quad)
```
<img width="700" height="534" alt="image" src="https://github.com/user-attachments/assets/5160232b-8f98-4532-9944-da216043781a" />

- 회귀를 위한 CART 비용함수는 아래와 같다.
<img width="400" height="576" alt="image" src="https://github.com/user-attachments/assets/c4cba9c0-c2df-4fef-8fb8-26868174bbe8" />

- 분류에서는 불순도를 최소화하는 방향으로 분할했다면, 여기서는 MSE를 최소화하도록 분할한다.
- 이후 리프 노드는 그 노드에 속한 모든 훈련 샘플의 실제값의 평균을 자신의 값으로 지정하며 이 평균값이 노드의 예측값이 된다.
- 회귀 결정트리 역시 과대적합되기 쉬우므로 규제가 필요하다.

<img width="1000" height="482" alt="image" src="https://github.com/user-attachments/assets/70614137-fa27-4ff9-b324-71cb5d46e1b6" />


## 축 방향에 대한 민감성
- 결정 트리는 계단 모양의 결정 경계를 만들고 **데이터의 방향에 민감하다.**
<img width="900" alt="image" src="https://github.com/user-attachments/assets/979ea767-0e74-493c-a952-b901fd96d62f" />


- 더 좋은 일반화를 위해 PCA 변환을 적용할 수 있다.
  - 특성 간의 상관관계를 줄이는 방식으로 데이터를 회전하여 결정 트리를 더 쉽게 만들 수 있다. (항상 그런것은 아니며 이 장에서 PCA에 대해서는 자세히 다루지 않음)
  - 데이터의 스케일을 조정하고 PCA로 데이터를 회전시키는 작은 파이프라인을 적용하여 결정 트리를 훈련해본 뒤의 시각화
```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pca_pipeline = make_pipeline(StandardScaler(), PCA())
X_iris_rotated = pca_pipeline.fit_transform(X_iris)
tree_clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf_pca.fit(X_iris_rotated, y_iris)
```
<img width="750" height="441" alt="image" src="https://github.com/user-attachments/assets/5b7adb9f-af6f-4885-9478-893a758e10f2" />


## 결정 트리의 분산 문제
- 일반적으로 결정 트리는** 분산이 상당히 크다.**
- 하이퍼파라미터나 데이터를 조금만 변경해도 매우 다른 모델이 생성될 수 있다.
- `random_state` 매개변수 설정하지 않는 한 동일한 데이터에서 재훈련하더라도 매우 다른 모델이 생성될 수 있다.
  - 여러 결정 트리의 **예측을 평균**하면 **분산을 줄일 수 있으며**, 이러한 결정 트리의 앙상블을 **랜덤 포레스트**라고 한다. (다음 장에서 자세히 다룸)
