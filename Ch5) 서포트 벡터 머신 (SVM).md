5장에서는 SVM (Support Vector Machine)에 대해서 알아본다.
SVM은 선형, 비선형 분류 / 회귀 / 특이치 탐지등에 활용할 수 있는 다목적 머신러닝 모델이다.
특히, 중소규모의 비선형 데이터셋, 분류 작업에 탁월하다.

## 선형 SVM
- 두 개의 클래스를 나눌 수 있으면서도 가장 가까운 훈련 샘플(서포트 벡터 Support Vector)로부터 가능한 한 멀리 떨어져 있는 결정경계를 찾고자 하는 것이다.

- Large Margin Classification이라고도 한다.
- `from sklearn.svm import LinearSVC` 을 사용해 선형 분류가 가능함
  <img width="890" height="219" alt="image" src="https://github.com/user-attachments/assets/82998d96-253d-4683-84b9-1fd19e3eb26f"/>

    
    - 왼쪽 그래프의 경우 임의의 직선을 그은 것인데, 결정 경계들을 살펴보면 훈련 샘플은 잘 구분하지만, 샘플에 너무 가까워 새로운 샘플에 대해서 잘 작동하지 않음
    - 오른쪽 그래프의 경우 실선이 결정 경계인데, 더 많은 훈련 샘플을 추가해도 이 경계는 전적으로 서포트 벡터들에 의해 결정됨

- SVM은 특성 스케일에 민감하다는 특징이 있음


  <img width="841" height="255" alt="image" src="https://github.com/user-attachments/assets/928a1370-0be1-47ef-899c-6107a9cb267f" />
    
    - 왼쪽 그림의 경우 수직축의 scale이 훨씬 커서 그래프가 수평에 가까움
    - 오른쪽 그림처럼 scaling을 해주면 훨씬 경계가 나아짐
<br>

### 하드 마진 분류
- 모든 샘플이 경계 바깥쪽으로 올바르게 분류되어 있는 경우
  <img width="1280" height="323" alt="image" src="https://github.com/user-attachments/assets/df8a3c58-966f-4795-8af0-93c6a84af675" />
  - 데이터가 선형적으로 구분될 수 있어야만 하고 이상치에 민감함
  - 따라서, 일반화되기 어려움
  
### 소프트 마진 분류
- 경계의 폭을 가능한 한 넓게 유지하는 것과 마진 오류 사이에 적절한 균형을 잡은 것
- **규제 하이퍼마라미터 *C***

  <img width="1280" height="310" alt="image" src="https://github.com/user-attachments/assets/5d997b9b-86a2-4f4f-9da0-d143e6fa0d4e" />

  - C가 낮을 수록 Support Vector가 많이 지정됨 이상치(혹은 잘못 분류된 데이터)를 더 허용해서 과대적합의 위험성이 줄어들고 완벽한 분류보다는 더 넓은 마진을 찾고자하는 것임
  - C가 높을 수록 이상치에 민감해져 마진이 좁아지고 과대적합의 위험성이 커질 수 있어, 일반화 성능이 떨어질 수 있음

---
## 비선형 SVM 분류
- 일반적으로는 선형으로 분류할 수 없는 데이터가 더 많음

### 1. 다항 특성 추가하기
- 비선형 데이터셋을 다루기 위해서는 다항 특성과 같이 특성을 더 추가하여 선형적으로 구분될 수 있게끔 만듦
![](https://velog.velcdn.com/images/algorithm_cell/post/1435b930-346e-42b9-95e3-57b52e31873f/image.png)

``` python
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

X, y= make_moons(n_samples=100, noise=0.15, random_state=42)

polynomial_svm_clf = make_pipeline(
	PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearSVC(C=10, max_iter=10_000, random_state=42)
)
polynomial_svm_clf.fit(X, y)
```
<img width="688" height="441" alt="image" src="https://github.com/user-attachments/assets/fd3ac9ad-8b31-4b77-b254-6e15d6bfe472" />

### 2. 다항식 커널
- 다항 특성을 추가하는 것은 비교적 간단하지만, **모델을 느리게 만든다는 단점이 있음**

- **커널 트릭** : 실제 특성을 추가하지 않으면서, 높은 차수의 다항 특성을 많이 추가한 것과 같은 효과를 주는 수학적 기교

```python
from sklearn.svm import SVC

poly_kernel_svm_clf = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=3, coef0=1, C=5))

poly_kernel_svm_clf.fit(X, y)
```

- **모델이 과대적합이라면** 다항식의 차수를 줄여야하고, **과소적합이라면** 차수를 늘려 조절한다.
- `coef0` : 모델이 높은 차수와 낮은 차수에 얼마나 영향을 받을지 조절

<p align="center">
  <img src="https://velog.velcdn.com/images/algorithm_cell/post/b4205e88-0911-4e01-870e-4fdd13994604/image.png" />
</p>

### 3. 유사도 특성

- 랜드마크와의 유사도를 측정하는 **유사도함수**를 활용해 특성을 추가하는 방식

  - 유사도 함수로 정의한다는 의미는 다른 샘플 데이터 포인트가 랜드마크와 얼마나 유사한지를 나타내는 척도로 가우스 RBF 함수를 사용하겠다는 의미
  - RBF 유사도 함수로 새로운 특성을 만들어 데이터셋을 변환시키면 선형적으로 구분할 수 있음
  - 훈련 세트가 매우 클 경우 동일한 크기의 아주 많은 특성이 만들어짐

![](https://velog.velcdn.com/images/algorithm_cell/post/a28c972a-ddf7-4302-82ac-c5784fc9aa6e/image.png)
- γ 값이 크면 곡선이 좁고 뾰족해져서, 데이터 포인트가 랜드마크와 아주 가까워야만 높은 유사도를 가짐 

  - 랜드마크의 영향력 범위가 좁아진다는 의미
  - 결정경계가 불규칙해지고 샘플을 따라 휘어짐
  - 
- 값이 작으면 곡선이 넓고 완만해져서, 데이터 포인트가 랜드마크와 조금 떨어져 있어도 높은 유사도를 가짐

  - 랜드마크의 영향력의 범위가 넓어진다는 의미 
  - 결정경계가 부드러워진다 → 과대적합을 해결 (C와 비슷)
  - **0.3은 적절한 수준**


### 4. 가우스 RBF 커널

- 유사도 특성 방식도 ML 알고리즘에 유용하게 사용됨
- 추가 특성을 모두 계산하려면 연산 비용이 많이 드는데 **커널트릭**을 사용해 유사도 특성을 많이 추가하는 것과 비슷한 결과를 얻을 수 있음

```python
rbf_kernel_svm_clf=make_pipeline(StandardScaler(),
								SVC(kernel="rbf, gamma=5,C=0.001))
rbf_kernel_svm_clf.fit(X, y)
```

- `gamma` : 증가시키면 **결정경계가 불규칙해지고, 샘플을 따라 구불구불 휨**

![](https://velog.velcdn.com/images/algorithm_cell/post/7f06d1af-cecc-48c6-b036-77042426eae8/image.png)

- 과대적합일 때는 감소시켜야하고, 과소적합일 때는 증가시키기
- 거의 **rbf kernel**만 사용 , 텍스트 문서나 DNA 서열 분류 시에 문자열 커널을 이용하기도 함
- 훈련세트가 너무 크지 않다면 가우스 RBF 커널을 시도해보면 좋음

### 5. 계산 복잡도

![](https://velog.velcdn.com/images/algorithm_cell/post/bd5faca3-e6e4-44dd-84fc-3eda519ff802/image.png)

**1) LinearSVC**

- 선형 SVM을 위해 최적화된 알고리즘이 구현된 liblinear 라이브러리를 기반으로 함
- 커널 트릭을 지원하지 않음
- 샘플과 특성 수에 거의 선형적으로 늘어나므로 훈련의 시간 복잡도는 O(m x n)
- 정밀도를 높이면 수행 시간이 길어진다. 이는 허용 오차 파라미터 ϵ 으로 조절한다 (사이킷런에서는 →l)
- 대부분의 경우 허용 오차를 기본값으로 두면 잘 작동함

 

**2) SVC**

- 커널 트릭 알고리즘이 구현된 libsvm 라이브러리를 기반으로 함.
- 시간 복잡도는 O(m^2 x n) 과 O(m^3 x n)사이이다. 훈련 샘플의 수가 커지면 엄청나게 느려지므로 중소규모의 비선형 훈련 세트에 잘 맞음
- 특성 수에 대해서는 희소 특성 Sparse Feature 인 경우에는 잘 확장됨
- 알고리즘 성능이 샘플이 가진 0이 아닌 특성의 평균 수에 거의 비례함

 

**3) SGDClassifier**

- 라지 마진 분류를 수행하며 하이퍼파라미터, 규제 하이퍼파라미터(α, pena<y)와 ≤arn∈grate를 조정하여 선형 SVM과 유사한 결과를 생성할 수 있음
- 훈련을 위해 점진적 학습이 가능, SGD 확률적 경사하강법을 사용하므로 메모리를 효율적으로 사용 가능 → 대규모 데이터셋에서 모델 훈련 가능
- 시간 복잡도 O(m x n) 확장성이 매우 뛰어남

---

## SVM 회귀
**SVM 분류**는 서로 다른 클래스에 속한 데이터 포인트들을 최대한 멀리 떨어뜨릴 수 있는 최적의 결정경계를 찾는 것

**SVM 회귀**의 목표

  : 가능한 한 많은 샘플이 마진 안에 포함되도록하는 최적의 회귀선을 찾는 것
  
  : 제한된 오류 내에서 이 오류를 최소화하고 최대한 많은 샘플이 마진 안에 들어오도록 하는 것
  
- 경계의 폭은 **ε**으로 조절

![](https://velog.velcdn.com/images/algorithm_cell/post/6915a606-ebff-4928-bc40-f8e2f5d7ce4b/image.png)
- ε을 줄이면 서포트 벡터의 수가 늘어나서 모델이 규제됨
- 마진 안에서는 샘플이 추가되어도 모델에 영향을 주지 않기 때문에 모델은 ε에 민감하지 않다고 함

```python
from sklearn.svm import LinearSVR
x, y = [ ... ] # 선형 데이터셋
svm_reg = make_pipeline(StandardScale(),
					LinearSVR(epsilon=0.5, random_state=42))
svm_reg.fit(X, y)
```
- `LinearSVR`은 `LinearSVC`의 회귀 version
![](https://velog.velcdn.com/images/algorithm_cell/post/ef065c2e-d21a-4241-9040-6171c4a1d482/image.png)
- 마진 안에서 훈련 샘플이 추가되어도 모델의 예측에는 영향이 없어서 이 모델을 ϵ에 민감하지 않다고 함 (ϵ-insensitive)

- 모델은 ϵ 내의 오차에 대해 민감하게 반응하지 않음. 즉, 마진 안에 있는 샘플들이 손실함수에 영향을 주지 않음
- SVM 회귀는 ϵ-insensitive Loss Function이라는 손실함수를 사용
  
  - 이 함수는 예측값과 실제값의 차이가 ϵ 범위 안에 있으면 손실을 0으로 간주

---
## SVM 이론
- **SVM 훈련 목표**  
 
1) 제한된 마진 오류에서 마진을 가능한 한 넓게 만드는 가중치 벡터(W)와 편향(b)을 찾는 것 *(소프트마진)*
2) 마진 오류를 일으키지 않는 것 *(하드마진)*

- **제약조건** : 결정 함수가 모든 양성 훔련 샘플에서는 1보다 커야하고, 음성 훈련 샘플에서는 -1보다 작아야함

    - 음성 샘플 ($y^{(i)}=0$)일 때 $t^{(i)}=-1$
    - 양성 샘플 ($y^{(i)}=1$)일 때 $t^{(i)}=1$
    - 제약 조건을 모든 샘플에서 $t^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) \ge 1$로 표현할 수 있음
 
- 학습
학습과정에서는 마진 오류를 줄이며 마진을 가능한 한 넓게 만드는 가중치 벡터 W와 편향 b 를 찾아야 함
<img width="1278" height="442" alt="image" src="https://github.com/user-attachments/assets/1c840fd9-ec0e-4192-a316-145da19efd42" />

- 마진을 더 넓게 만들기 위해서는 W를 가능한 한 작게 유지
- 편향 b 는 마진의 크기에는 영향을 미치지 않고 위치만 이동
- 
### 1. 하드 마진 선형 SVM

$$\underset{\mathbf{w},b}{\text{minimize}} \quad \frac{1}{2}\mathbf{w}^T\mathbf{w}$$

### 2. 소프트 마진 선형 SVM

- 목적 함수를 구성하려면 **슬랙 변수(ξ)**를 도입해야함
- **ξ** : i번째 샘플이 얼마나 마진을 위반할 지 정함
  - 목표
      1) 마진 오류를 최소화 -> ξ를 작게 만들어야함
      2) 마진을 크게 하기 위해 -> $\quad \frac{1}			{2}\mathbf{w}^T\mathbf{w}$ 를 최소화
    	-> C를 통해 두 목표 사이의 trade-off를 정의함


$$
\underset{\mathbf{w},b,\zeta}{\text{minimize}} 
\quad  
\frac{1}{2}\mathbf{w}^T\mathbf{w} 
+ 
C \sum_{i=1}^{m}\zeta^{(i)}
$$

$$
\text{subject to} \quad
t^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) \ge 1-\zeta^{(i)}, \;
\zeta^{(i)} \ge 0, \;
i=1,2,\cdots,m
$$


---
## 쌍대문제 *Dual Problem*
- *primal problem* : 등식과 부등식 제약조건이 있는 본래의 최적화 문제
- 제약이 있는 최적화 문제인 원 문제 Primal Problem 은 쌍대 문제 Dual Problem 라고 하는 다른 문제로 표현할 수 있음
- Primal Problem이 W와 b 를 직접 찾아내는 것이라면, Dual Problem은 라그랑주 승수법으로 라그랑주 승수를 찾아내는 문제로 변환됨

	- **특정 조건 하에서 Dual Problem의 해가 Primal Problem의 해와 동일하며, SVM이 이에 해당함**
 	- 즉, **Dual Problem으로 풀어도 원하는 W와 b 를 정확히 얻을 수 있음**

- 훈련 sample수가 feature수보다 작다면, dual problem을 푸는 것이 더 빠름
- primal problem에는 커널 트릭이 적용 안되지만, dual problem에서는 적용이 됨
	




---
### Study Notes
[이은정](https://velog.io/@dkan9634/HandsOnML-Chap-5.-%EC%84%9C%ED%8F%AC%ED%8A%B8-%EB%B2%A1%ED%84%B0-%EB%A8%B8%EC%8B%A0)

[안태현-1](https://armugona.tistory.com/entry/%ED%95%B8%EC%A6%88%EC%98%A8-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-Ch5-SVM-12-%EC%84%A0%ED%98%95-SVM-%EB%B9%84%EC%84%A0%ED%98%95-SVM-%EC%BB%A4%EB%84%90-%ED%8A%B8%EB%A6%AD-%EA%B0%80%EC%9A%B0%EC%8A%A4-RBF-%ED%95%A8%EC%88%98-SVM-%ED%9A%8C%EA%B7%80?category=1256688)
[안태현-2](https://armugona.tistory.com/entry/%ED%95%B8%EC%A6%88%EC%98%A8-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-Ch5-SVM-22-%EC%8A%AC%EB%9E%99-%EB%B3%80%EC%88%98-%EC%8C%8D%EB%8C%80-%EB%AC%B8%EC%A0%9C-%EC%BB%A4%EB%84%90-%ED%8A%B8%EB%A6%AD-%EB%9D%BC%EA%B7%B8%EB%9E%91%EC%A3%BC-%EC%8A%B9%EC%88%98%EB%B2%95)

[허채연](https://velog.io/@algorithm_cell/%ED%95%B8%EC%A6%88%EC%98%A8-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-ch5.-%EC%84%9C%ED%8F%AC%ED%8A%B8-%EB%B2%A1%ED%84%B0-%EB%A8%B8%EC%8B%A0)

