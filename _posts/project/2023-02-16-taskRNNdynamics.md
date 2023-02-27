---
title:  "뇌인지 기능의 신경망 모델 - Part 2"
mathjax: true
layout: post
categories: Project
---
2023년도 겨울학기 과학계산 트레이닝 세션의 네 번째 주제는 저번 주차에서 만든 **인지과제를 수행하는 인공신경망의 neural state dynamics 분석** 입니다.
이번 주차에는 인공신경망의 neural state dynamics를 분석한 다양한 선행 연구들을 조금씩 소개하고,
선행 연구들의 접근 방법과 의의를 배우고, 우리 문제에 적용해 보겠습니다.

저번 주차에서 우리는 간단한 시각 판별 과제를 구현하고, 이를 수행할 수 있는 RNN 모델을 구현하였습니다.
이후 학습된 RNN 모델을 이용하여 해당 RNN 모델의 행동 데이터와 짧은 꼬리 원숭이의 행동 데이터를 비교해 보았습니다.
하지만 이러한 접근 방법에는 몇 가지 문제점이 존재합니다.

첫번째 문제는 우리가 해당 인지과제를 풀기 위해 사용한 신경망 구조가 Biological neural network와 비교하기에 적합하지 않을 수 있다는 것 입니다.
저번 주차에서 우리는 `PyTorch`에서 제공하는 LSTM 신경망을 사용하였습니다.
하지만 이렇게 구현한 LSTM 신경망은 우리가 해당 신경망에 다양한 제약 조건을 넣어 Biological neural network와 더 
유사한 동작을 하게 만드는 데에 여러 어려움이 있습니다. 예를 들어 실제 동물의 뇌에서 관찰되는 조건이 추가된 신경망을 구현하기가 어렵습니다.

두번째 문제는 실제 사람이나 동물이 수행하는 인지 과제의 time steps과 RNN의 time steps을 맞추기가 어렵다는 것 입니다.
RNN 모델에 실제 사람이나 동물이 수행하는 인지 과제와 동일한 time steps를 갖는 데이터를 입력하기 위해서 
우리는 연속된 데이터를 이산화(discretize) 해야 하는데, 저번 주차에서는 그런 부분이 구현되지 않았습니다.

따라서 이번 주차에는 여러 계산신경과학 선행 연구에서 인지 과제를 수행하는 RNN 모델을 만들기 위해 사용한 방법들을 알아보고,
이를 따라서 구현해 보도록 하겠습니다. 
다만 여기서 제가 강조하고 싶은 점은 우리가 RNN 모델을 실제 동물 뇌의 신경망과 유사하게 만들기 위해 하는 방법들에는 명확한 정답이 있기 힘들다는 말씀을 드리고 싶습니다.
이는 저번 주차에서 이야기 한 모델 시스템의 추상화를 어떤 수준에서 할 것인가에 대한 문제이며, 넓게는 자연과학, 좁게는 계산과학이 가지고 있는 근본적인 한계라고 생각 합니다.
모델 시스템을 만들고 이를 분석하며 늘 그런 모델 시스템이 실제 현상과 얼마나 유사한가? 실제 현상을 모델링 하는데 있어서 **충분히** 유사한가? 와 같은 질문을 얼마든지 던져 볼 수 있습니다.
모든 동료 연구자들이 이건 실제 현상과 완전히 같다고 동의 할 수 있는 수준의 모델 시스템은 없다고 생각 합니다. 만약 그런 연구가 가능하다 하더라도, 그건 모델 시스템을 만들 의미가 없는 연구이기도 하구요.
그래서 우리가 RNN 모델에 어떤 제약 조건을 추가하는 것으로 우리가 실제 뇌의 신경망을 얼마나 잘 모사하고 있는가를 이야기 하기는 어렵다고 생각 합니다.
**여기엔 정답은 없고, 다만 선행 연구들이 있을 뿐이죠.** 선행 연구들이 설정한 추상화의 수준과 그 모델 시스템을 통해 이야기 할 수 있었던 결론들이
현재 우리 학계가 RNN 모델을 사용한 연구가 신경과학을 연구하는데 있어 유용한 모델 시스템이라고 어느정도 인정하고 있는 수준이다... 정도로 말 할 수 있을것 같습니다.

## 학습 목표 
이번 주차의 학습 목표는 다음과 같습니다. 
- 여러 선행 연구에서 제안된 RNN 모델을 구현해 봅니다.
- RNN 모델의 neural state dynamics를 분석하고 실제 뇌에서 측정된 데이터와 비교 합니다.

## 문제를 정의하기
모든 문제 풀이의 시작은 풀고자 하는 문제를 잘 정의하는 것 입니다. 우리는 위에서 저번 주차에서 구현한 RNN 모델의 두 가지 문제점을 지적하였습니다.
이제 저번 주차에서 RNN 모델을 정의하기 위해 작성한 코드를 보도록 하겠습니다.

{% highlight python %}
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(device)

        output, _ = self.lstm_layer(x, (h0, c0))
        output = self.fc_layer(output)
        return output
{% endhighlight %}

위 코드를 보시면 모델에 입력 텐서 `x`가 들어 올 때 `.forward()` 메서드가 실행되며 hidden state `h`와 cell state `c`를 초기화 한 후,
LSTM 층에 입력 텐서 `x`가 들어가는 것을 볼 수 있습니다.
이렇게 `PyTorch`에서 제공하는 LSTM 신경망을 사용하면 우리가 이 신경망을 연구의 목적에 맞게 제약 조건을 걸 수 있는 옵션이 많이 부족합니다.

어떤 RNN 모델을 사용할 것인가? 역시도 수 많은 계산신경과학 논문에서 서로 다 다른 방법을 취하고 있습니다. 저는 최근 저명한 신경과학 저널인 Neuron에 출판된
Guangyu Robert Yang과 Xiao-Jing Wang의 "Artificial Neural Networks for Neuroscientists: A Primer" 라는 논문을 중심으로
Xiao-Jing Wang 그룹에서 출판된 다양한 논문에서 공통적으로 제시된 방법을 따라서 구현해 보겠습니다.
여러 논문들을 살펴 본 결과 각 논문들이 초점을 맞추고 있는 제약 조건은 조금씩 다르지만 공통적으로 사용하는 몇 가지 제약 조건을 발견하였습니다.
다만 역시 위에서 말씀 드린 대로 모델 시스템을 정의하고 추상화하는데엔 정답이 없고, 연구자에 따라 서로 다른 방법을 사용하고 있다는 점을 꼭 염두해 두셨으면 좋겠습니다.

1. Continous-time RNN을 만든 후, 이를 근사하여 이산화(discretize) 합니다. 이는 실제 사람이나 동물이 수행하는 인지과제의 time steps과 모델의 이산화된 time steps를 맞추기 위함 입니다.
2. RNN 모델의 neural activity를 양수(positive value)로 제한합니다. 이는 실제 동물 뇌의 신경세포의 발화율이 음수로 표현되지 않기 때문입니다.
3. RNN 모델의 recurrent unit이 가질 수 있는 neural activity의 크기를 정규화합니다. 이는 RNN 모델이 지나치게 많은 계산 자원을 사용하지 못하도록 제약 조건을 거는 것으로, 실제 동물 뇌의 신경 활동은 에너지를 소모하여 일어나기 때문에 합리적인 제약 조건이라고 말할 수 있겠습니다.
4. RNN 모델의 network weight가 희박한(sparse) 연결 구조를 갖게끔 제한합니다. 이는 실제 동물 뇌의 신경세포에서 관찰되는 synaptic connection이 fully-connected 되어있지 않기 때문입니다. [포유류의 대뇌 피질에서 관찰되는 synaptic connection의 비율은 약 12% 라는 선행 연구](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0030068)가 존재하고, 우리는 이 비율에 근사하도록 인공신경망에게 제약 조건을 걸 수 있습니다.

이외에도 여러 전기생리학 연구에서 측정된 대뇌 피질의 신경 세포의 특징에 기반한 다양한 제약 조건을 생각 해 볼 수 있겠습니다.
그리고 그런 제약 조건 하에서 정의된 인공신경망 모델의 state dynamics와 행동 데이터가 어떻게 실제 뇌의 neuronal dynamics와 mental state를 더 잘 반영하는지 연구하는 것도 계산신경과학의 주요한 연구 방향이 아닐까 싶습니다.
이제 위 네 가지 제약 조건을 건 RNN 모델을 구현해 보겠습니다. 먼저 RNN 모델의 single-unit의 dynamics를 고려해 보겠습니다.
Single-unit의 recurrent neural activity $\mathbf{r}(t)$를 시간에 대한 미분방정식으로 표현한 Dynamical system을 가정해 보겠습니다.

$$ \tau \frac{d\mathbf{r}}{dt} = -\mathbf{r}(t) + f(W_r \mathbf{r}(t) + W_x \mathbf{x}(t) + \mathbf{b}_r) $$

위 미분방정식은 신경망의 recurrent neural activity인 $\mathbf{r}(t)$가 시간에 따라 어떻게 변화하는지 나타냅니다.
먼저 시간에 따라 신경망에 입력되는 input sequence $\mathbf{x}(t)$는 신경망의 recurrent neural activity $\mathbf{r}(t)$와 가중합(weighted sum) 되고,
이는 신경망이 정보를 비선형 변환하기 위해 고안된 활성화 함수(activation function) $f(\cdot)$을 거치게 됩니다. 
