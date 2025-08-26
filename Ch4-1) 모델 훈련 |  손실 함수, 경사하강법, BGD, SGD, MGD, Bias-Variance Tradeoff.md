4장은 양이 많기 때문에 2번에 걸쳐서 다룬다.
1부에서는 4.1 ~ 4.4
2부에서는 4.5 ~ 4.6

모델을 훈련시키는 방법으로 손실 함수의 개념과 이 손실 함수를 최소화하기 위한 대표적인 기법으로 경사하강법에 대해서 소개한다.
데이터에 적합한 복잡도의 모델을 찾는 관점에서 손실 함수(오차)를 활용할 수 있음을 보이며, 이 손실 함수의 변화를 관찰함으로써 과대적합 Overfitting, 과소적합 Underfitting을 판단하고자 한다.



## 선형회귀  _Linear Regression_
회귀에 가장 널리 사용되는 성능 측정 지표는 평균 제곱근 오차(RMSE)이다.

<img width="200" height="100" alt="image" src="https://github.com/user-attachments/assets/8e10a070-9c14-4886-974f-b57b08948ea3" />

어떤 함수를 최소화하는 것은 그 함수의 제곱근을 최소화하는 것과 같으므로 RMSE보다 평균 제곱 오차(MSE)를 최소화하는 것이 같은 결과를 내면서 더 간단하다.

<img width="200" height="100" alt="image" src="https://github.com/user-attachments/assets/3b9b850c-d152-49fb-b0c3-cd80e58341ee" />


#### 정규방정식 _Normal Equation_
<img width="150" height="100" alt="image" src="https://github.com/user-attachments/assets/a2f104c0-e210-4977-b05d-f857d39d7acf" />
비용함수를 최소화하는 세타 값을 찾기 위한 수학 공식이다.

$${X^T}X$$의 역행렬이 존재해야 한다.
$${X^T}X$$가 역행렬을 가질 수 없는 경우(예: 특이행렬, 열 독립성이 없는 경우 등)에는 위 수식을 사용할 수 없다.

이럴 때는 Moore-Penrose 유사역행렬(Pseudo-inverse)을 사용해 다음과 같이 구한다:

$$\hat{θ}={X^+}y$$ 
여기서 $$X^+$$는 X의 유사역행렬이다. 이 유사역행렬은 특잇값 분해(Singular Value Decomposition, SVD)라는 행렬 분해 기술을 통해 계산된다.

```np.linalg.pinv()``` : 유사역행렬 구하는 함수

정규 방정식은 (n+1) * (n+1) 크기의 역행렬을 계산하므로 계산 복잡도는 O(n^3)이 된다. 따라서 특성 수가 늘어나면 시간이 대폭 증가한다.

##  경사하강법 _Gradient descent_
<img width="740" height="400" alt="image" src="https://github.com/user-attachments/assets/30325b41-acc6-40e6-bad6-e04139fa32e4" />

경사 하강법은 일반적인 최적화 알고리즘으로 반복적인 연산을 통해 파라미터를 조정하는 것이다.

model parameter는 학습 시작 시에 랜덤하게 초기화되고 반복적으로 수정되어 MSE(비용함수)를 최소화한다.

그래디언트는 항상 함숫값이 가장 가파르게 증가하는 방향을 가리킨다.
현재 위치에서 그래디언트를 구한 후, 그 반대 방향으로 이동하면 Loss를 줄일 수 있다.

(이것이 Gradient **Descent**라고 불리는 이유)

학습률 Learning Rate는 그래디언트의 반대방향으로 이동할 때 보폭 step을 조절하는 역할을 한다.

학습률이 작으면 수렴을 위해 너무 많은 횟수를 반복하고, 크면 알고리즘을 발산하게 만들어 적절한 답을 찾지 못한다.

<img width="740" height="400" alt="image" src="https://github.com/user-attachments/assets/c33d82f9-037d-433d-849a-e4314901dbc6" />

단점으로는 위 그림에서 볼 수 있듯이 랜덤 초기화 때문에 그림의 왼쪽에서 시작하면 전역 최솟값(global minimum)이 아닌 지역 최솟값(local minimum)에 수렴한다.

그림의 오른쪽에서 시작하면 평탄한 지역을 지나기 위해 시간이 오래 걸리고 일찍 멈추어 전역 최솟값에 도달하지 못한다.

계산 속도가 느리다.

- 경사하강법의 반복 횟수를 지정하는 방법은 반복 횟수를 아주 크게 지정하고 gradient vector가 아주 작아지면, 즉 벡터의 노름이 어떤 값(허용 오차)보다 작아지면 경사 하강법이 (거의) 최솟값에 도달한 것이므로 알고리즘을 중지한다.

### 배치 경사 하강법 _Batch Gradient Descent Method, BGD_
모든 훈련 샘플을 기준으로 한 번에 계산하므로 배치 경사 하강법이라고 부른다.
```python
# 경사하강법 구현
eta = 0.1
n_epochs = 1000
m = len(X_b)
theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients
```
<img width="780" height="300" alt="image" src="https://github.com/user-attachments/assets/873b451d-b714-4e0b-a7f6-3073f19c5dd2" />


### 확률적 경사하강법 _Stochastic Gradient Descent_
매 반복마다 하나의 훈련 샘플만 무작위로 선택하여 그레이디언트를 계산한다.

메모리 효율이 뛰어나고 속도도 빠르지만, 진동이 심하고 수렴이 불안정할 수 있다.

무작위성은 local minimum 탈출에는 도움이 되지만, global minimum에 도달하지 못하게 할 수 있다.

이에 대한 해결책으로 초기에는 학습률을 크게 해서 지역 최솟값에 빠지지 않도록 하고, 점차 학습률을 줄여서 전역 최솟값에 도달하도록 한다.

학습률을 조절하는 것을 학습 스케쥴 Learning Schedule 이라고 한다.
```python
# SGD 구현
 def learning_schedule(t):
 		return t0 / (t + t1)
     n_epochs = 50
				t0, t1 = 5, 50  
  
theta = np.random.randn(2, 1)  # 랜덤 초기화

for epoch in range(n_epochs):
    for iteration in range(m):  # m = 샘플 수
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients

# scikit-learn 활용
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01, random_state=42)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.intercept_, sgd_reg.coef_)
```


### 미니배치 경사하강법 _Mini-Batch Gradient Descent_
작은 샘플 묶음(미니배치)을 사용하여 그레이디언트를 계산한다.

행렬 연산에 최적화된 하드웨어인 GPU를 사용해서 성능을 향상시킬 수 있다는 장점이 있다.

미니배치 사이즈를 크게하면 SGD보다 덜 불규칙하지만 지역 최솟값에 빠질 수도 있다.

SGD보다 더 부드럽게 수렴하지만, 때로는 지역 최솟값에서 빠져나오기 어렵기도 하다.

미니배치 크기는 성능과 수렴 안정성에 큰 영향을 준다.

<img width="550" height="300" alt="image" src="https://github.com/user-attachments/assets/31642a7f-f09f-4ba0-8ea3-5379133b7573" />



## 다항회귀와 학습 곡선 분석
비선형 데이터를 학습할 수 있도록, 각 특징의 거듭제곱, 특징들의 곱으로 상호작용 항 Interaction Term을 새로운 특징으로 사용해 선형 모델을 훈련 시키는 것을 다항 회귀라고 한다.

일반적인 선형 회귀 모델은 찾을 수 없는 특징들 사이에 숨겨진 관계를 찾아낼 수 있다.

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/2ca89712-d9a3-42d2-a5b5-2f408910be3e" />

고차 다항 회귀는 훨씬 훈련 데이터에 적합한 곡선을 찾겠지만, 일반화가 어렵다. (즉,과대적합(overfitting) 될 가능성이 높음)

선형 모델은 과소적합(underfitting)될 가능성이 높다.

모델의 성능은 교차 검증 방법과 학습 곡선을 톨해 확인할 수 있다.
- 교차 검증에서
  - 훈련 데이터에서는 성능이 좋은데, 교차검증 점수가 낮으면 과대적합, 둘 다 좋지 않으면 과소적합
- 학습 곡선 
  - 보통 Train set과 Validation(test) set에 대해서 각각 loss와 metric을 훈련 중간중간 마다 체크한 곡선

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/3be7d16f-fd1a-44ba-a288-ca2718706ed4" />

- 언더피팅된 그래프
  - 훈련 세트와 검증 세트의 RMSE가 모두 높고 비슷,
 
과소적합된 경우에는, 훈련 샘플을 추가해도 효과가 없으며 더 복잡한 모델을 쓰거나 다른 특징을 선택해야 한다

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/48392544-a5ef-4fcc-beaa-8bc2df8dfa3f" />

-오버피팅된 그래프
  - 두 곡선 사이에 공간이 있는데, 훈련 데이터에서의 모델 성능이 검증 데이터에서보다 낫다는 뜻, 그러나 더 큰 훈련 세트를 사용하면 두 곡선이 점점 가까워짐

과대적합 모델을 개선하는 한 가지 방법은 검증 오차가 훈련 오차에 근접할 때까지 더 많은 훈련 데이터를 추가하는 것이다.

### 일반화 오차
오차를 편향, 분산, 그리고 데이터 자체가 내재하고 있어 어떤 모델링으로도 줄일수 없는 오류 Irreducible Error의 합으로 보는 것이다.

편향은 잘못된 가정으로 인한 오차이다. 예시로는 2차원의 데이터인데 선형으로 가정하는 경우이다.
- 편향이 큰 모델은 과소적합되기 쉽다.

분산은 작은 변동에도 모델이 민감하게 반응하는 것이다. 자유도가 높은 모델은 높은 분산을 가지기 쉽다.
- 분산이 큰 모델은 과대적합되기 쉽다.
 

줄일 수 없는 오류는 데이터 자체에 있는 잡음으로 인해 발생한다. 이상치를 제거하는 것과 같은 데이터의 잡음을 줄임으로써 오차를 줄일 수 있다.
 

모델의 복잡도가 커지면 분산은 커지고 편향은 작아지지만, 복잡도가 작아지면 분산은 작아지고, 편향은 커지는 Tradeoff 관계이다.

---
### Study Notes
[이은정](https://velog.io/@dkan9634/HandsOnML-Chap-4.-모델훈련1)
[안태현](https://armugona.tistory.com/entry/%ED%95%B8%EC%A6%88%EC%98%A8-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-Ch4-%EB%AA%A8%EB%8D%B8-%ED%9B%88%EB%A0%A8-12-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98-%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95-BGD-SGD-MGD-Bias-Variance-Tradeoff?category=1256688) 
[허채연](https://velog.io/@algorithm_cell/%ED%95%B8%EC%A6%88%EC%98%A8-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-ch4-1.-%EB%AA%A8%EB%8D%B8-%ED%9B%88%EB%A0%A8)
