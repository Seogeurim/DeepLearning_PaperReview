---
description: '#Model #BatchNormalization'
---

# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

#### 논문제목 : Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift 논문저자 : Sergey Ioffe, Christian Szegedy

이 논문은 **Batch Normalization** 에 대한 논문이다.   
이를 통해 학습 비용을 줄였다. 

### Abstract

Internal Covariate Shift 현상

* 각 layer마다 input distribution이 다르다.
* layer가 뒤로 갈수록 분포가 변화되어 결국 출력층에 안좋은 영향을 끼칠 수 있다.
* 적은 learning rate를 요구한다.
* parameter initialization이 조심스러워진다.
* saturating nonlinearity로 학습시키기 힘들게 만든다.
* 이는 학습을 느리게 만든다. 

Batch Normalization

* 각 training mini-batch에 normalization
* 높은 learning rate를 쓸 수 있다. 
* parameter initialization에 대해 조심스럽지 않아도 된다. 
* regularizer 역할을 하여, Dropout의 필요성을 없애기도 한다.

Results

* state-of-the-art 이미지 분류 모델에서,  Batch Normalization을 사용했을 때, 14배 적은 training step으로 같은 정확도를 달성하였다. 
* Batch Normalization 네트워크로 앙상블했을 때,  ImageNet classification의 가장 높은 성과를 달성했다. \(top-5 validation error : 4.9%, test error : 4.8% &lt;- 사람보다 높은 정확도\)











