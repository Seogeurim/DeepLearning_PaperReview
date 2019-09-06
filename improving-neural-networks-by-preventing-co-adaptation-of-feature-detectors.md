---
description: '#Model #Dropout'
---

# Improving neural networks by preventing co-adaptation of feature detectors

#### 논문제목 : Improving neural networks by preventing co-adaptation of feature detectors 논문저자 : G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever and R. R. Salakhutdinov

이 논문은 **Dropout** 에 대한 논문이다.   
Dropout 기법을 통해 Overfitting을 방지하고,   
error rate를 대폭 감소시켰다. 

### Abstract

> Random "dropout" gives big improvements on many benchmark tasks and sets new records for speech and object recognition.

Feedforward 뉴럴넷이 작은 training set에서 학습되었을 때, test data에서 안 좋은 성능을 보일 수 있다 - Overfitting \(과적합\)  
이는 training 과정에서 feature detector 중 절반을 random하게 빠뜨림으로써 해결될 수 있다 - Dropout

* prevents complex co-adaptation : feature detector
* correct answer를 내기 위해 필요한 feature들을 detect 가능

이는 각 training case의 feature detector들 중 절반을 random하게 빠뜨림으로써 해결될 수 있다. 이는 complex co-adaptation\(feature detector가 동조화되어 다른 특별한 feature들을 못 잡음\)을 방지할 수 있다. 그랬으 ㄹ때 correct output을 내기 위한 필요한 feature들을 detect할 수 있게 된다. 

### 

