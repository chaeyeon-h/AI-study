7장에서는 앙상블 학습에 대해 다룬다.

# 투표 기반 분류기
### Hard voting
-  최종 예측 클래스를 다수결로 따름

```python
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ]
)
voting_clf.fit(X_train, y_train)
```
VotingClassifier를 훈련할 때 모든 추정기를 복제하여 훈련
- list 인 경우 원본 추정기는 estimators, 복제본은 estimators_ 속성에 저장됨.
- dictionary 인 경우 원본 추정기는 named_estimators, name_estimators_사용
```python
# 테스트 세트에서 훈련된 각 분류기의 정확도
for name, clf in voting_clf.named_estimators_.items():
    print(name, "=", clf.score(X_test, y_test))


>>>lr = 0.864
   rf = 0.896
   svc = 0.896
```
-> 이건 Hard Voting이므로 테스트 첫번째 샘플이 1이면 세 분류기 중 두 분류기 이상이 1이라고 예측했음을 알 수 있다 (아래는 관련 코드)
```python
print(voting_clf.predict(X_test[:1]))
print([clf.predict(X_test[:1]) for clf in voting_clf.estimators_])

>>> [1]
[array([1]), array([1]), array([0])]

voting_clf.score(X_test, y_test)
>>> 0.912
```
예상대로 투표 기반 분류기가 다른 개별 분류기보다 성능이 조금 더 높음 (개별은 0.864, 0.896, 0.896 이정도였음)

  
### Soft voting
- 특성 클래스에 속할 확률을 평균내어 가장 높은 평균 확률을 가진 클래스를 선택
- 클래스의 확률을 추정하기 위해 교차 검증을 사용하므로 훈련 속도가 느려짐



**약한 학습기 Weak Learner** : 랜덤 추측보다 조금 더 높은 성능을 내는 분류기
**강한 학습기 Strong Learner** : 높은 정확도를 내는 분류기
각 분류기가 약한 학습기이더라도 앙상블에 있는 약한 학습기가 충분히 많고 다양하다면 강한 학습기가 될 수 있다.

이는 큰 수의 법칙 Law of large numbers 때문이다. 완전 랜덤 추측보다 조금 나은 51% 정확도를 가진 1,000개의 분류기로 앙상블 모델을 구축한다고 가정하면 75%의 정확도를 기대할 수 있다.

이때 모든 분류기가 완벽하게 독립적이고 오차에 상관관계가 없다는 가정이 필수적이다. 앙상블 방법은 각 예측기가 가능한 한 서로 독립적일 때 최고의 성능을 발휘한다.
같은 훈련 데이터로 훈련시킬 경우 모든 분류기가 데이터의 편향을 함께 학습하게 될 수 있으며 유사한 종류의 오류를 저지르기 쉽다.  
이에 대한 해결책으로 각기 다른 알고리즘으로 각각 학습시키면 다른 종류의 오차를 만들 가능성이 높아 앙상블 모델의 정확도가 향상된다.

---
# 배깅과 페이스팅
같은 알고리즘을 사용하고 훈련 세트의 서브셋을 랜덤으로 구성하여 분류기를 각기 다르게 학습시키는 방법도 존재한다.
### 배깅(bagging)
: Bootstrap + Aggregating
- Bootstrap: 원본 데이터에서 복원추출로 여러 개의 학습용 데이터를 만듦
(예: 100개 데이터 중 100개를 복원추출 → 일부는 중복, 일부는 빠짐)
- Aggregating: 각각의 모델이 예측한 결과를 평균 or 다수결로 결합
- 즉, 원본 데이터를 여러 번 “복원추출”해서 여러 모델을 훈련시키고, 그 결과를 종합하는 방식
- 장점: 모델 간 다양성(diversity) 확보, 분산(variance) 감소 → 과적합 방지, 일반화 성능 향상
- 대표 알고리즘: Random Forest (= 여러 Decision Tree의 Bagging)

### 페이스팅(pasting)
- Bagging과 거의 같지만, 차이점은 비복원추출 (without replacement)
- 즉, 원본 데이터를 여러 부분으로 나누고, 겹치지 않게 각 모델이 다른 데이터 부분을 학습하는 방식
- 장점: 데이터 중복 없음, 각 모델이 완전히 다른 샘플을 학습 → 좀 더 빠름
- 단점: 다양성이 줄어들 수 있음 (Bagging보다 불안정할 수 있음)

일반적으로 집계 함수는 분류일 때는 통계적 최빈값, 회귀일 때는 평균을 사용
학습과 예측을 병렬로 수행할 수 있음

## 사이킷런의 배깅과 페이스팅
- BaggingClassifier은 predict_proba() 함수가 있으면 soft voting을 사용
- BaggingClassifier는 특성 샘플링도 지원함
  - 훈련 속도를 높일 수 있기 때문에 고차원 데이터셋 다룰 때 유용
  - 랜덤 패치 방식 : 훈련 특성과 샘플을 모두 샘플링
  - 랜덤 서브스페이스 방식 : 훈련 샘플을 모두 사용하고(즉, bootstrap=False, max_sample=1.0), 특성을 샘플링(bootstrap_features=True, max_freatures=1보다 작게)
  - 특성 샘플링은 더 다양한 예측기를 만들어 편향을 늘리지만, 분산을 줄임
- 다음은 결정 트리 분류기 500개의 앙상블을 훈련시키는 코드
  - 각 분류기는 훈련 세트에서 중복 허용해 랜덤으로 선택된 100개의 샘플로 훈련(배깅)
  - 만약 페이스팅 사용하고 싶으면 bootstrap=False
  - n_jos 매개변수: 사이킷런이 훈련과 예측에 사용할 CPU 코어 수 지정(-1이면 가용한 모든 코어 사용)
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, 
                            max_samples=100, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
```

<img width="1280" height="490" alt="image" src="https://github.com/user-attachments/assets/06f76212-024f-4ad4-b2be-ac8cd371ecea" />
-> 500개의 결정 트리를 사용한 배깅 앙상블을 보면 하나의 결정 트리보다 일반화가 잘 된, 덜 불규칙한 결정경계를 볼 수 있음
-> 앙상블의 예측이 결정 트리 하나의 예측보다 일반화가 훨씬 잘됨(비슷한 편향에서 더 작은 분산을 만듦)

## OOB 평가
- 배깅에서 샘플링할 때 선택되지 않은 나머지 샘플들을 OOB(Out-of-bag) 샘플이라고 부름
- 검증 세트를 사용하지 않고 OOB 샘플을 이용해 평가할 수 있음
```python
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            oob_score=True, n_jobs=-1, random_state=42)

bag_clf.fit(X_train, y_train)
print(bag_clf.oob_score_) #0.896
```
-> OOB 평가 결과: BaggingClassifier 테스트에서 89.6% 정확도

```python
from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred) #0.912
```
-> 91.2%의 정확도

## 랜덤 패치와 랜덤 서프스페이스
- 랜덤 패치 방식 : 훈련 특성과 샘플 모두 샘플링
- 랜덤 서브스페이스 방식 : 훈련 샘플은 모두 사용하고 특성만 샘플링
- 특성 샘플링은 더 다양한 예측기를 만들며 편향을 늘리고 분산을 낮춤

# 랜덤 포레스트
- 일반적으로 배깅(또는 페이스팅)을 적용한 결정 트리의 앙상블
- max_samples로 훈련 세트 크기 지정, 이미 결정 트리에 최적화된 RandomFroestClassifier 사용(회귀는 RandomForestRegressor)

아래는 500개 트리로 이뤄진 랜덤 포레스트 분류기를 가능한 모든 CPU 코어에서 훈련시키는 코드
```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, 
                                n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
```

- 랜덤 포레스트 알고리즘은 트리의 노드를 분할할 때 랜덤으로 선택한 특성 후보 중에서 최적의 특성을 찾는 식으로 무작위성을 주입한다. 
- 기본적으로 n개의 특성에서 $\sqrt{n}$개의 특성을 선택한다. 트리를 더욱 다양하게 만들어 **편향을 손해 보는 대신 분산을 낮춘다.**
- 아래의 코드는 `BaggingClassifier`를 `RandomForestClassifier`와 거의 동일하게 만든 것이다.
```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
    n_estimators=500, n_jobs=-1, random_state=42)
```

## 엑스트라 트리
- 극단적으로 랜덤한 트리의 랜덤포레스트를 extremely randomized tree 엑스트라 트리라고 부름
- 무작위성이 증가함에 따라, 편향이 증가하지만 분산은 감소함
- 최적의 값을 찾지 않기 때문에 훈련속도가 빠름
- 즉, 랜덤 포레스트에서 데이터를 전부 사용하고 분할 기준도 랜덤으로 골라버리는 더 단순하고 더 랜덤한 앙상블 트리 모델
- 반면에 랜덤 포레스트의 각 트리는 최적의 분할 기준을 찾으려고 함
- 둘 중 뭐가 좋다고 할 순 없고 교차검증이 유일한 방법

## 특성 중요도
- 랜덤 포레스트는 특성의 상대적 중요도를 측정하기 쉽다. 
- 특성 중요도는 모델이 예측을 수행할 때, 각 특성이 얼마나 기여했는지를 나타내는 값으로 사이킷런에서는 지니 불순도Gini impurity 를 이용해서 측정한다.
- 각 트리의 노드가 어떤 특성으로 데이터를 분리할 때, 불순도가 얼마나 줄어드는지 계산하고 이 감소량을 모든 트리에서 평균 내면 그 특성의 중요도가 됨.
- 이 불순도 감소량을 모든 트리에 대해 가중치 평균을 내며, 각 노드의 가중치는 훈련 샘플 수와 같다. (각 서브셋에 속한 샘플 수, 더 많은 샘플을 포함하는 노드의 불순도에 더 큰 영향을 받게 하도록)
```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(iris.data, iris.target)
for score, name in zip(rnd_clf.feature_importances_, iris.data.columns):
    print(round(score, 2), name)
    
>>> 0.11 sepal length (cm)
0.02 sepal width (cm)
0.44 petal length (cm)
0.42 petal width (cm)
```
-> 중요도를 보면 꽃잎 길이, 너비가 높고 꽃받침의 길이와 너비는 비교적 덜 중요함
MNIST 데이터셋에 랜덤 포레스트 분류기를 훈련시키고 각 픽셀의 중요도를 그래프로 나타내면 아래와 같다.

<p align="center">
  <img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/ffeceef3-cc38-40d0-bf58-9f7b2b3c992d" />
</p>

=> 랜덤 포레스트는 특히 **특성을 선택해야 할 때** 어떤 특성이 중요한지 빠르게 확인할 수 있어 매우 편리


# 부스팅(Boosting)
: 약한 학습기를 여러 개 연결해 강한 학습기를 만드는 앙상블 방법
왜 약한 걸 연결하는데 강해질까? 큰 수의 법칙 때
<img width="1514" height="675" alt="image" src="https://github.com/user-attachments/assets/45e8a781-b669-4617-8123-a974aeb5c9f3" />


부스팅 방법을 살펴보자

## AdaBoost
- '에이다부스트'라고 읽음
- 이전 예측기를 보완하는 새로운 예측기를 만드는 방법은 이전 모델이 과소적합했던 훈련 샘플의 가중치를 더 높이는 것. 이렇게 하면 새로운 예측기는 학습하기 어려운 샘플에 점점 더 맞춰짐
> AdaBoost는 같은 알고리즘을 반복해서 사용하지만, 틀린 샘플의 가중치를 점점 높여가며 학습을 보완해 나가는 부스팅 기법
<p align="center">
  <img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/d2946ace-a22b-4637-9a3c-f96c8b973063" />
</p>
- 연속된 학습 기법의 단점 : 이전 예측기가 훈련되고 평가된 후에 다음 훈련을 진행 할 수 있기 때문에 훈련을 병렬화할 수 없음

### AdaBoost 알고리즘
1) 각 샘플의 가중치는 $\frac{1}{m} $으로 초기화됨
2) 학습 이후에 오류율($r_j$)이 훈련 세트에 대해 계산됨
2-1) j번째 예측기의 가중치가 적용된 오류율
- 가중치 클수록 오차 영향 더 커짐
   
$$
r_j = \frac{\sum_{i=1}^{m} w^{(i)} I(y_i \ne \hat{y}_j^{(i)})}{\sum_{i=1}^{m} w^{(i)}}
$$
   
2-2) 예측 가중치
- 예측기가 정확할수록 가중치는 높아짐(랜덤보다 낮으면 음수, 랜덤 추측이면 0)
  
$$
a_j = \eta \log \frac{1 - r_j}{r_j}
$$

2-3) 가중치 업데이트

$$
w^{(i)} \leftarrow
\begin{cases}
w^{(i)}, & \hat{y}_j^{(i)} = y^{(i)} \\
w^{(i)} \exp(\alpha_j), & \hat{y}_j^{(i)} \ne y^{(i)}
\end{cases}
$$

2-4) 모든 샘플 가중치를 정규화

$$
w^{(i)} \leftarrow
\frac{w^{(i)}}{\sum_{k=1}^{m} w^{(k)}}
$$

- 가중치를 조정하며 연속적으로 학습하는 과정을 코드로 나타내면 다음과 같다.
  (SVM은 AdaBoost 기반 예측기로 적합하지 않지만 예시를 보여주기 위한 코드)


-> 지정된 예측기 수에 도달하거나 완벽한 예측기가 만들어지면 알고리즘을 중지한다.

예측할 때는 단순히 모든 예측기의 예측을 계산하고 예측기 가중치 $a_j$를 더해 예측 결과를 만든다.
가중치 합이 가장 큰 클래스가 측 결과가 된다.

$$
\hat{y}(\mathbf{x}) = \arg\max_k \sum_{\substack{j=1 \\ \hat{y}_j(\mathbf{x}) = k}}^{N} \alpha_j
$$

- 새로운 샘플 **x**에 대한 AdaBoost 모델의 최종 예측값 **ŷ** 에 대한 식  
- **N**은 예측기 수, **k**는 클래스(class)를 의미  
- 개별 예측기의 가중치를 각 예측기가 선택한 클래스에 더해준 뒤,  
  이 가중치 합이 가장 큰 클래스를 최종 예측 결과로 정한다.

```python
m = len(X_train)
sample_weights = np.ones(m) / m # 모든 샘플에 대해 동일한 가중치
  plt.sca(axes[subplot])
  for i in range(5): # 5개의 분류기
      svm_clf = SVC(C=0.2, gamma=0.6, random_state=42)
      svm_clf.fit(X_train, y_train, sample_weight=sample_weights * m)
      y_pred = svm_clf.predict(X_train)

  # 잘못 예측한 샘플들의 가중치 합을 구함 r은 오류율
      error_weights = sample_weights[y_pred != y_train].sum() 
      r = error_weights / sample_weights.sum()  # equation 2-1
      
      # alpha는 모델 가중치 : 오류율이 낮을 수록 alpha는 커짐
      alpha = learning_rate * np.log((1 - r) / r)  # equation 2-2
      
      # 잘못 예측한 샘플들의 가중치를 alpha배만큼 늘림
      sample_weights[y_pred != y_train] *= np.exp(alpha)  # equation 2-3
      sample_weights /= sample_weights.sum()  # normalization step

      plot_decision_boundary(svm_clf, X_train, y_train, alpha=0.4)

 ```

<img width="1280" height="491" alt="image" src="https://github.com/user-attachments/assets/3db1e998-e193-41ec-98ad-402e14be5e45" />
-> 각 예측기는 이전 예측기가 훈련되고 평가된 뒤에 학습될 수 있으므로 병렬화 불가능
  배깅이나 페이스팅만큼 확장성이 높지 않음

`scikit-learn` 에서 `AdaBoostClassifier` 사용하는 방법

```python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
		DecisionTreeClassifier(max_depth=1), n_estimators=30,
        learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)
```

## 그레이디언트 부스팅
- 앙상블에 이전까지의 오차를 보정하도록 예측기를 순차적으로 추가함. 하지만 AdaBoost처럼 반복마다 샘플의 가중치를 수정하는 대신 이전 예측기가 만든 잔여 오차에 새로운 예측기를 학습시킴
결정 트리를 기반 예측기로 사용하는 간단한 회귀 문제
- Gradient Tree Boosting 또는 Gradient Boosted Regression Tree (GBRT)
-> 먼저 2차 방정식으로 잡음이 섞인 데이터셋을 생성하고 Decision TreeRegressor를 학습시켜보자

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 *np.random.randn(100) # y = 3x^2+가우스잡음
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

# 첫 번째 예측기에서 생긴 잔여 오차에 두 번째 DecisionTreeRegressor를 훈련시킴
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X, y2)

# 두 번째 예측기가 만든 잔여 오차에 세 번째 회귀 모델 훈련
y3 = y2- tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_reg3.fit(X, y3)

# 이 세개의 트리를 포함하는 앙상블 모델 생김
# 새로운 샘플에 대한 에측을 만들려면 모든 트리 예측 더하면 됨
X_new = np.array([[-0.4], [0.], [0.5]])
sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))

>>> array([0.49484029, 0.04021166, 0.75026781])
```
<img width="989" height="1017" alt="image" src="https://github.com/user-attachments/assets/9cad499e-6f8a-46a9-bcbe-a89caa30105a" />
-> 오른쪽 열을 보면 각 예측이 더해지는 것을 확인할 수 있다.  

GradientB∞stingRegressor를 사용해서 GBRT를 간단하게 훈련시킬 수 있다. 
분류를 위한 GradientB∞stingClassifier 클래스도 존재한다.
<img width="1280" height="267" alt="image" src="https://github.com/user-attachments/assets/6aaa8cf0-ff36-480f-997b-200fa761f3a9" />


아래는 위에서 본 알고리즘과 같은 앙상블을 만드는 코드이다.
```python
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3,
                                 learning_rate=1.0, random_state=42)
gbrt.fit(X, y)
```

learning rate는 각 트리의 기여도를 조절한다. 0.05로 낮게 설정하면 학습시키기 위해 많은 트리가 필요하지만 일반적으로 예측의 성능은 좋아지는데 이는 축소 Shrinkage 라고 부르는 규제 방법이다. 
각 트리의 기여도가 작아지므로, 모델이 과적합되는 것을 막는다.
```python
gbrt_best = GradientBoostingRegressor(
    max_depth=2, learning_rate=0.05, n_estimators=500,
    n_iter_no_change=10, random_state=42)
gbrt_best.fit(X, y)
```

<img width="1502" height="645" alt="image" src="https://github.com/user-attachments/assets/a13f959a-d07c-41e3-930f-3b78c962edf1" />
- 왼쪽은 트리 개수가 적을 때, 오른쪽은 적정할 때

+) 각 트리가 훈련할 때 훈련 샘플의 비율을 지정할수도 있는데 편향이 높아지는 대신 분산이 낮아지고 훈련 속도도 상당히 빨라진다. => 확률적 그레이디언트 부스팅(stochastic gradient boosting)

## 히스토그램 기반 그레이디언트 부스팅
- 입력 특성을 구간으로 나누어 정수로 대체하는 방식으로 작동
- 구간의 개수는 하이퍼파라미터 max_bins가 결정 (기본값=255, 더 높게 설정 x)
- 구간 분할은 규제처럼 작동 : 정밀도 손실을 유발해 과대적합을 줄이거나, 과소적합을 유발할 수 있음
- 계산복잡도 : O(b*m) (b: 구간의 개수, m: 훈련 샘플의 개수)
1. bin 할당 : 모든 샘플 m개를 보고, 각 샘플을 해당 bin으로 넣음
                   이때, 시간복잡도 :𝑂(𝑚)
2. 히스토그램 집계 단계 : 각 bin마다 (gradient, sample weight 등)을 합산
                  이때, 시간복잡도 :𝑂(b)
3. 모든 특성 반복 : O(b⋅m) 형태로 표현됨

- scikit-learn에서 HistGradientBoostingRegressor (회귀), HistGradientBoostingClassifier (분류) 제공
  - 인스턴스 수가 10,000개보다 많으면 조기 종료가 자동으로 활성화.
  - subsample 매개변수가 지원되지 않음
  - n_estimators -> max_iter
  - 조정할 수 있는 파라미터는 only : max_leaf_nodes, min_samples_leaf, max_depth
```python
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

hgb_reg = make_pipeline(
	make_column_transformer((OrdinalEncoder(), ["ocean_proximity"]),
    						remainder="passthrough"),
    HistGradientBoostingRegressor(categorical_features=[0], random state=42)
)
# 범주형 열의 인덱스로 `categorical_features`를 설정해야함
hgb_reg.fit(housing, housing_labels)
```
더 최적화된 그래디언트 부스팅으로는 XGBoost, CatBoost, LightGBM이 있다.

# 스태킹(stacking)
- stacked generalization의 줄임말
- 앙상블에 속한 모든 예측기의 예측을 취합하는 모델을 훈련시킬 수는 없을까? 에서 출발
- 새로운 샘플에 회귀작업을 수행하는 앙상블이라고 할 때, 최종 예측을 만드는 마지막 예측기를 블렌더 Blender 또는 메타 학습기 Meta Learner 라고 한다.


### blender를 학습시키자
1. blender를 훈련하기 위한 블렌딩 훈련 세트를 생성 (앙상블의 모든 예측기에서 cross_val_predict() 를 사용해 예측을 얻음)
2. 얻은 예측값을 블렌더 훈련하기 위한 입력 특성으로 사용, 타깃은 원본 훈련 세트에서 복사
3. 원본 훈련 세트의 크기의 관계없이 블렌딩 훈련 세트의 입력은 예측기당 하나임
<p align="center">
  <img width="600" height="700" alt="image" src="https://github.com/user-attachments/assets/98831a22-3e9b-408b-b09e-ae4ac5bc380d" />
</p>

- blending 예측기를 사용한 예측 취합

- 다층 스태킹 앙상블
<p align="center">
  <img width="600" height="700" alt="image" src="https://github.com/user-attachments/assets/736d2c35-0e04-450c-873d-b6c596414a0b" />
</p>
-> 다층 스태킹을 통해 조금 더 정확한 예측이 가능하지만, 비용이 증가함

- scikit-learn에서 StackingClassifier, StackingRegressor 제공

```python
from sklearn.ensemble import StackingClassifier

stacking_clf = Stackingclassifier(
	estimators=[
    			('lr', LogisticRegression(random_state=42)),
                ('rf', RandomForestClassifier(random_state=42)),
                ('scv', SVC(probability=True, random_state=42))
	],
	final_estimator=RandomForestClassifier(random_state=43),
	cv=5
)
stacking_clf.fit(X_train, y_train)
```
---
### Study Notes
[이은정](https://velog.io/@dkan9634/HandsOnML-Chap-7.-%EC%95%99%EC%83%81%EB%B8%94-%ED%95%99%EC%8A%B5%EA%B3%BC-%EB%9E%9C%EB%8D%A4-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-284ce8u1)

[안태현-1](https://armugona.tistory.com/entry/%ED%95%B8%EC%A6%88%EC%98%A8-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-Ch7-%EC%95%99%EC%83%81%EB%B8%94-%ED%95%99%EC%8A%B5%EA%B3%BC-%EB%9E%9C%EB%8D%A4-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-12-Voting-Bagging-OOB-%EB%9E%9C%EB%8D%A4-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-Random-Forest-AdaBoost)
[안태현-2](https://armugona.tistory.com/entry/%ED%95%B8%EC%A6%88%EC%98%A8-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-Ch7-%EC%95%99%EC%83%81%EB%B8%94-%ED%95%99%EC%8A%B5%EA%B3%BC-%EB%9E%9C%EB%8D%A4-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-22-%EA%B7%B8%EB%9E%98%EB%94%94%EC%96%B8%ED%8A%B8-%EB%B6%80%EC%8A%A4%ED%8C%85-Gradient-Boosting-%EC%8A%A4%ED%83%9C%ED%82%B9-Stacking)

[허채연](https://velog.io/@algorithm_cell/%ED%95%B8%EC%A6%88%EC%98%A8-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-ch7.-%EC%95%99%EC%83%81%EB%B8%94-%ED%95%99%EC%8A%B5%EA%B3%BC-%EB%9E%9C%EB%8D%A4-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8#blender%EB%A5%BC-%ED%95%99%EC%8A%B5%EC%8B%9C%ED%82%A4%EC%9E%90)
