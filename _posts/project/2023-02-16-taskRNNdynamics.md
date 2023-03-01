---
title:  "뇌인지 기능의 신경망 모델 - Part 2"
mathjax: true
layout: post
categories: Project
---
2023년도 겨울학기 과학계산 트레이닝 세션의 네 번째 주제는 저번 주차에서 만든 **인지과제를 수행하는 인공신경망의 neural state dynamics 분석** 입니다.
이번 주차에는 인공신경망의 neural state dynamics를 분석한 다양한 선행 연구들을 조금씩 소개하고,
선행 연구들의 접근 방법과 의의를 배우고, 우리 문제에 적용해 보겠습니다. 저번 주차에서 우리는 간단한 시각 판별 과제를 구현하고, 이를 수행할 수 있는 RNN 모델을 구현하였습니다.
이후 학습된 RNN 모델을 이용하여 해당 RNN 모델의 행동 데이터와 짧은 꼬리 원숭이의 행동 데이터를 비교해 보았습니다.
하지만 이러한 접근 방법에는 몇 가지 문제점이 존재합니다.

- 첫번째 문제는 우리가 해당 인지과제를 풀기 위해 사용한 신경망 구조가 Biological neural network와 비교하기에 적합하지 않을 수 있다는 것 입니다.
저번 주차에서 우리는 `PyTorch`에서 제공하는 LSTM 신경망을 사용하였습니다.
하지만 이렇게 구현한 LSTM 신경망은 우리가 해당 신경망에 다양한 제약 조건을 넣어 Biological neural network와 더 
유사한 동작을 하게 만드는 데에 여러 어려움이 있습니다. 예를 들어 실제 동물의 뇌에서 관찰되는 조건이 추가된 신경망을 구현하기가 어렵습니다.

- 두번째 문제는 실제 사람이나 동물이 수행하는 인지 과제의 time steps과 RNN의 time steps을 맞추기가 어렵다는 것 입니다.
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
Guangyu Robert Yang과 Xiao-Jing Wang의 [Artificial Neural Networks for Neuroscientists: A Primer](https://www.cell.com/neuron/fulltext/S0896-6273(20)30705-4#%20) 라는 논문을 중심으로
Xiao-Jing Wang 그룹에서 출판된 다양한 논문에서 공통적으로 제시된 방법을 따라서 구현해 보겠습니다.
여러 논문들을 살펴 본 결과 각 논문들이 초점을 맞추고 있는 제약 조건은 조금씩 다르지만 공통적으로 사용하는 몇 가지 제약 조건을 발견하였습니다.
다만 역시 위에서 말씀 드린 대로 모델 시스템을 정의하고 추상화하는데엔 정답이 없고, 연구자에 따라 서로 다른 방법을 사용하고 있다는 점을 꼭 염두해 두셨으면 좋겠습니다.

1. **RNN 모델의 neural activity를 양수(positive value)로 제한합니다.** 이는 실제 동물 뇌의 신경세포의 발화율이 음수로 표현되지 않기 때문입니다.
2. **RNN 모델의 recurrent unit이 가질 수 있는 neural activity의 크기를 정규화합니다.** 이는 RNN 모델이 지나치게 많은 계산 자원을 사용하지 못하도록 제약 조건을 거는 것으로, 실제 동물 뇌의 신경 활동은 에너지를 소모하여 일어나기 때문에 합리적인 제약 조건이라고 말할 수 있겠습니다.
3. **RNN 모델의 network weight가 희박한(sparse) 연결 구조를 갖게끔 제한합니다.** 이는 실제 동물 뇌의 신경세포에서 관찰되는 synaptic connection은 신경세포 사이의 물리적인 거리에 의한 제약 조건이 존재하고, 따라서 fully-connected 되어있지 않기 때문입니다. [포유류의 대뇌 피질에서 관찰되는 synaptic connection의 비율은 약 12% 라는 선행 연구](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0030068)가 존재하고, 우리는 이 비율에 근사하도록 인공신경망에게 제약 조건을 걸 수 있습니다.

이외에도 여러 전기생리학 연구에서 측정된 대뇌 피질의 신경 세포의 특징에 기반한 다양한 제약 조건을 생각 해 볼 수 있겠습니다.
그리고 그런 제약 조건 하에서 정의된 인공신경망 모델의 state dynamics와 행동 데이터가 어떻게 실제 뇌의 neuronal dynamics와 mental state를 더 잘 반영하는 모델 시스템인지 연구하는 것도 계산신경과학의 주요한 연구 방향이 아닐까 싶습니다.
이제 위 세 가지 제약 조건을 건 Continous-time RNN(CTRNN) 모델을 구현해 보겠습니다.
먼저 RNN 모델의 single-unit의 recurrent neural activity $\mathbf{r}(t)$을 시간에 대한 미분방정식으로 표현하면 다음과 같습니다.

$$ \tau \frac{d\mathbf{r}}{dt} = -\mathbf{r}(t) + f(W_r \mathbf{r}(t) + W_x \mathbf{x}(t) + \mathbf{b}_r) $$

위 미분방정식은 신경망의 recurrent neural activity인 $\mathbf{r}(t)$가 시간에 따라 어떻게 변화하는지 나타냅니다.
먼저 시간에 따라 신경망에 입력되는 input sequence $\mathbf{x}(t)$는 신경망의 recurrent neural activity $\mathbf{r}(t)$와 가중합(weighted sum) 되고,
이는 신경망이 정보를 비선형 변환하기 위해 고안된 활성화 함수(activation function)인 $f(\cdot)$을 거치게 됩니다. 사용되는 활성화 함수 $f(\cdot)$의 종류에 대해서는 이후에 논의해 보도록 하겠습니다.
$ -\mathbf{r}(t) $ 항의 경우 미분방정식에서 Leaky integration이라 불리는 항으로 이전 neural activity $\mathbf{r}(t)$에 비례하여 다음 neural activity의 크기를 감소시키는 역할을 합니다.
해당 항이 없는 경우 신경망의 상태가 시간에 따라 무한히 증가하여 불안정한 네트워크 상태를 가지게 됩니다. 이제 이 미분방정식을 Euler method를 이용하여 이산화 합니다.
이 부분을 구현하기 위하여 위 논문과 Guangyu Robert Yang의 [튜토리얼](https://github.com/gyyang/nn-brain/blob/master/RNN_tutorial.ipynb)을 참고하였습니다.

$$  
\begin{aligned}
    \mathbf{r}(t+\Delta t) \approx \mathbf{r}(t) + \Delta \mathbf{r} &= \mathbf{r}(t) + \frac{\Delta t}{\tau}[-\mathbf{r}(t) + f(W_r \mathbf{r}(t) + W_x \mathbf{x}(t) + \mathbf{b}_r)] \\
    &= (1 - \frac{\Delta t}{\tau})\mathbf{r}(t) + \frac{\Delta t}{\tau}f(W_r \mathbf{r}(t) + W_x \mathbf{x}(t) + \mathbf{b}_r)
\end{aligned}
$$

## RNN 모델의 구현과 학습

이제 이렇게 이산화 된 RNN 모델을 `PyTorch`를 이용하여 구현해 보겠습니다.
RNN 모델의 neural activity를 positive value로 제한한다는 1번 제약 조건을 걸기 위해 모델에서 사용되는 활성화 함수 $f(\cdot)$는
렐루 함수(Rectified Linear Unit, ReLU)를 사용해 보도록 하겠습니다. 

{% highlight python %}
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dt):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        self.tau = 100

        self.i2h = nn.Linear(input_dim, hidden_dim)
        self.h2h = nn.Linear(hidden_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = torch.zeros(x.size(0), x.size(1), self.hidden_dim)
        output = torch.zeros(x.size(0), x.size(1), self.output_dim)

        h = torch.zeros(x.size(0), self.hidden_dim).to(device)
        for t in range(x.size(1)):
            h = h * (1 - self.dt/self.tau) + (self.dt/se
            output[:,t,:] = o
        return output, hidden
{% endhighlight %}

위 코드를 보시면 입력 텐서를 hidden layer로 넘겨주는 Linear transformation `.i2h()`와
시간 $t$에서의 hidden state 텐서를 시간 $t+1$에서의 hidden state 텐서로 변환하는 `.h2h()`,
마지막으로 hidden state 텐서를 출력 텐서로 변환하기 위한 `.h2o()` 선형 변환을 정의해 주었습니다.
이후 모델이 실제로 정보를 다음 time step으로 넘겨주는 recurrent neural activity는
위 미분방정식을 오일러법으로 근사한 수식을 `.forward()` 메서드에 직접 구현해 주었습니다.
위 수식에서 시간 상수 $\tau$는 선행 연구에 따라 100ms로 설정하였고,
$\mathbf{b}_r$항의 경우 선형 변환에 이미 포함 되어 있기 때문에 따로 구현해 줄 필요가 없습니다.
이제 마찬가지로 RNN 모델의 학습과 테스트에 사용되는 시각 판별 과제를 생성하는 `VisualDiscrimination()` 클래스를 수정하여 
미소 시간 변수 `.dt`에 따라 다운샘플링을 하도록 수정합니다. 대부분 저번 주차에서 했던 코드와 크게 다르지 않으며 샘플링 구간을 나타내는 변수 `.dt` 만 추가가 되었습니다.

{% highlight python %}
class VisualDiscrimination(Dataset):
    def __init__(self, task_dict):
        self.target_dim = task_dict['target_dim'] # Red and Green
        self.color_dim = task_dict['color_dim']
        self.output_dim = task_dict['output_dim'] # Left and Right
        self.dt = task_dict['dt']
        self.target_onset_range = task_dict['target_onset_range']
        self.decision_onset_range = task_dict['decision_onset_range']
        self.coherence_range = task_dict['coherence_range']
        self.trial_length = task_dict['trial_length']
        assert np.max(self.decision_onset_range) < self.trial_length

        self.trial_steps = int(self.trial_length/self.dt)

    def __getitem__(self, idx):
        target_onset = int(np.random.randint(self.target_onset_range[0], self.target_onset_range[1])/self.dt)
        decision_onset = int(np.random.randint(self.decision_onset_range[0], self.decision_onset_range[1])/self.dt)
        coherence = np.random.uniform(low=self.coherence_range[0], high=self.coherence_range[1])

        input_seq = np.zeros((self.trial_steps, self.target_dim+self.color_dim))
        output_seq = np.zeros((self.trial_steps, self.output_dim))
        checkerboard_color = np.sign(np.random.normal())          # -1(Red) or +1(Green)
        target_idx = np.random.randint(0, self.output_dim)      # 0(Red-Green) or 1(Green-Red)

        # Target cue
        input_seq[target_onset:, target_idx] = 1

        # Color checkerboard
        input_seq[:, self.target_dim:] = np.random.normal(loc=0, size=(self.trial_steps, self.color_dim))
        input_seq[decision_onset:, self.target_dim:] = np.random.normal(loc=checkerboard_color*coherence,
                                                                       size=(self.trial_steps-decision_onset, self.color_dim))

        # Desired output
        color_idx = 1 if checkerboard_color > 0 else 0         # (0: Red, 1: Green)
        output_direction = 0 if color_idx == target_idx else 1  # (0: Left, 1: Right)

        output_seq[decision_onset:, output_direction] = 1

        return {'input_seq': input_seq, 'output_seq': output_seq,
                'checkerboard_color': checkerboard_color, 'coherence': coherence,
                'target_idx': target_idx, 'output_direction': output_direction, 'decision_onset': decision_onset}
{% endhighlight %}

이제 2번과 3번 제약 조건을 추가하여 RNN 학습을 진행합니다. 2번 제약 조건은 RNN 모델의 recurrent unit의 활성화, 즉 firing rate를 정규화 하는 것이고, 
3번 제약 조건은 RNN 모델의 connection을 sparse하게 만드는 것 입니다. 우리는 L1 정규화(regularization)를 통해 RNN 모델에 2번과 3번 제약 조건을 걸 수 있습니다.

$$ L_{\text{L1, rate}} = \beta_{\text{rate}} \sum_{i,t} \lvert \mathbf{r_{i,t}} \rvert  \\ L_{\text{L1, weight}} = \beta_{\text{weight}}  \sum_{l} \sum_{i,j} \lvert \mathbf{w_{i,j}^{l}} \rvert $$



{% highlight python %}
task_params = {'target_dim': 2,
               'color_dim': 10,
               'output_dim': 2,
               'dt': 20,
               'target_onset_range': (400, 900),
               'decision_onset_range': (1200, 1800),
               'trial_length': 2000,
               'coherence_range': (0.0, 1.0)}

input_dim = task_params['target_dim']+task_params['color_dim']
trial_steps = int(task_params['trial_length']/ task_params['dt'])
hidden_dim = 128
output_dim = task_params['output_dim']
dt = task_params['dt']
batch_size = 128
num_epochs = 4000
learning_rate = 0.001
beta_rate = 10e-7
beta_weight = 10e-5

model = RNN(input_dim, hidden_dim, output_dim, dt).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_history = []
L1_rate_history = []
L1_weight_history = []

dataset = VisualDiscrimination(task_params)
for epoch in range(num_epochs):
    inputs = torch.zeros((batch_size, trial_steps, input_dim))
    targets = torch.zeros((batch_size, trial_steps, output_dim))
    for i in range(batch_size):
        data = dataset[0]
        inputs[i], targets[i] = torch.tensor(data['input_seq']), torch.tensor(data['output_seq'])
    inputs.to(device)
    targets.to(device)

    optimizer.zero_grad()
    outputs, hidden_state = model(inputs)
    loss = criterion(outputs, targets)

    L1_rate = beta_rate * torch.norm(hidden_state, 1)
    model_weight = torch.cat([x.view(-1) for x in model.parameters()])
    L1_weight = beta_weight * torch.norm(model_weight, 1)
    total_loss = loss + L1_rate + L1_weight
    total_loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    L1_rate_history.append(L1_rate.item())
    L1_weight_history.append(L1_weight.item())

    if epoch % 100 == 0:
        print (f'Training epoch ({epoch+1}/{num_epochs}), Total loss: {total_loss.item():3.3f}, '
               f'loss: {loss.item():3.3f}, L1 norm(rate): {L1_rate:3.3f}, L1 norm(weight): {L1_weight:3.3f}')
{% endhighlight %}

![Screen-Shot-2023-03-01-at-6-18-21-PM.png](https://i.postimg.cc/5N5MKZkn/Screen-Shot-2023-03-01-at-6-18-21-PM.png)


