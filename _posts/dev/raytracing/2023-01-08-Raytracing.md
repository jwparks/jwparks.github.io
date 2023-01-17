---
title:  "레이트레이싱: 빛과 그래픽스"
mathjax: true
layout: post
categories: Dev
---
2023년도 겨울학기 과학계산 트레이닝 세션의 두번째 주제는 컴퓨터를 이용해 빛을 추적하는 기법인 **Ray tracing** 입니다. 
이번 주제는 첫 주차 주제보다 수학적인 요소도 많고, 이를 컴퓨터를 이용하여 구현하는 것도 조금 더 어려운데요. 
대신 그만큼 더 재미있는 주제라고 생각 합니다. 
이번 주차를 수행하며 여러분은 실제로 현대 컴퓨터 그래픽스에서 널리 쓰이는 렌더링 방법중 하나인 레이트레이싱을 파이썬을 이용해 직접 구현하게 됩니다.
이를 구현하는 과정에서 여러분은 공간 벡터의 개념과 구현 및 응용 그리고 객체지향 프로그래밍을 연습 할 수 있습니다.

## 학습 목표
이번 주차의 학습 목표는 다음과 같습니다.
- 그래픽 렌더링 방법인 레이트레이싱의 개념을 이해합니다.
- 컴퓨터 프로그래밍을 통해 빛과 객체의 요소들을 수학적으로 정의한 공간 벡터를 구현합니다.

또한 공식적인 학습 목표는 아니지만 이번 주차의 학습을 진행하며 여러분들은 공간, 벡터 등 다양한 개념들의 수학적 정의(수식 등)와 
이를 구현하기 위한 프로그래밍 코드 사이의 관계에 조금더 익숙해지실 거라고 기대하고 있습니다.  


## 레이트레이싱

우리가 시각 시스템을 통해 물체를 볼 수 있는 원리는 무엇일까요?
많은 분들이 답을 잘 알고 계시겠지만 우리는 눈으로 들어오는 빛, 즉 광선을 통해서 물체를 볼 수 있습니다. 
태양, 전구 등 광원(Light source)에서 나온 빛이 물체에 반사되고, 물체가 가진 물리적인 특성에 따라 광선의 파장이 바뀌고, 바뀐 광선의 파장을 통해 우리는 물체를 볼 수 있습니다.

레이트레이싱, 우리말로 광선추적 기법은 컴퓨터 그래픽의 렌더링 기법중 하나로, 
우리가 빛을 통해 물체를 볼 수 있는 원리를 컴퓨터를 이용해 직관적으로 구현하는 기법입니다.
실제로 다른 점이 있다면 광원에서 나온 빛을 추적하는 것이 아니라 이미지로 들어오는 빛을 역추적 하는 기법이라는 차이가 있습니다.
다른 렌더링 기법 대비 레이트레이싱의 장점은 **매우 현실적인 그래픽**을 렌더링 할 수 있다는 장점이 있습니다.
당연히 실제 현실의 물리계에서 일어나는 빛의 반사를 가장 직관적으로 구현한 기법이기 때문에 현실적인 그래픽을 만들 수 있습니다.
반면 단점은 이미지를 구성하기 위해 필요한 수 많은 광선 다발을 추적해야 하기 때문에 극도로 높은 컴퓨터 자원을 필요로 합니다. 
그래서 레이트레이싱은 1980년대에 제안된 컴퓨터 그래픽스 기법이지만, 그동안은 매우 제한된 환경에서만 활용되어 왔습니다. 
그래픽을 실시간으로 렌더링해야 하는 비디오 게임 등에서는 거의 사용되지 않았고, 미리 렌더링을 해서 보여 줄 수 있는 영화, 애니메이션 등에서 주로 활용되어 왔습니다.
하지만 현대 컴퓨터 하드웨어의 발전을 통해 가정용 컴퓨터에 들어가는 그래픽카드들이 레이트레이싱 전용 직접회로인 RT 코어를 탑재하며
실시간 렌더링을 필요로 하는 비디오 게임 등에서도 제한적으로 레이트레이싱 기법이 활용되고 있습니다. 
아마 컴퓨터를 잘 모르시는 분들도 그래픽카드 모델명이 RTX3070이다, RTX4080이다 이런 말들을 들어 보셨을 텐데요, 이렇게 RTX 이름을 달고 있는 그래픽카드들이 바로 RT 코어를 내장하고 있는 그래픽카드들 입니다.

![1](https://i.ibb.co/m6cnhDy/Control-RTX-Comparison-6.jpg)
Figure 1. Control(2019)에서 구현된 실시간 레이트레이싱  
{: style="color:gray; font-size: 90%; text-align: center;"}

초당 수십장의 프레임을 렌더링 하는 비디오 게임에서 사용되는 실시간 레이트레이싱 기법은 아직 부족한 점이 많음에도 불구하고,
2019년 출시된 비디오 게임인 "Control"에서 기존 렌더링 기법(왼쪽)과 실시간 레이트레이싱(오른쪽)을 비교해 보면 빛의 반사와 사물의 그림자에서 확연한 차이가 나는 것을 볼 수 있습니다.
이처럼 레이트레이싱은 빛이 반사되어 생기는 상, 가려서 생기는 그림자 등을 매우 현실적으로 렌더링 할 수 있는 기법입니다. 

최근엔 많은 게임 엔진과 렌더링 엔진 등에서 레이트레이싱을 기본 옵션으로 제공하고 있어 손쉽게 구현이 가능합니다. 하지만 우리는 `python`과 `numpy`등 기본적인 과학계산 라이브러리만을 이용하여 실제 레이트레이싱을 밑바닥부터 구현해 보도록 하겠습니다. 

> Quiz 1. 레이트레이싱 기법은 왜 광원으로부터 나오는 광선을 직접 추적하지 않고, 이미지로 들어오는 빛을 역추적 할까요?

## 문제와 좌표계를 정의하기
흥미로운 주제가 정해 졌으니 이제 문제와 좌표계를 정의할 차례 입니다. 우리는 궁극적으로 레이트레이싱 기법을 이용하여 2D 이미지를 렌더링 하기를 원합니다. 
2D 이미지를 렌더링 한다는 의미는, 이미지를 구성하는 픽셀들의 색을 "적절히" 칠한다는 것을 의미합니다. 
그리고 우리는 각 픽셀들의 색이 그 픽셀로 들어온 광선으로부터 정의될 것이라는 사실을 알고 있습니다.

![2](https://i.ibb.co/nBbSDkp/1.png)
Figure 2. 레이트레이싱을 이용한 이미지 렌더링
{: style="color:gray; font-size: 90%; text-align: center;"}

위 그림에는 2D 이미지(프레임)인 `Scene`의 픽셀 $ P(a) $, $P(b)$, $P(c)$를 각각 통과하는 세개의 광선 $R(a)$, $R(b)$, $R(c)$이 그려져 있습니다.
광선 $R(a)$의 경우 정의된 물체인 파란색 구에서 반사되어 광원을 향하고 있습니다. 이 경우 우리는 픽셀 $P(a)$를 어떤 색으로 칠해야 할까요? 
광선 $R(a)$가 파란색 구에 닿는 부분은 상대적으로 밝을 것이기 때문에 우리는 밝은 색을 칠해주면 됩니다. 그림에서는 $P(a)$의 색이 흰색으로 표기 되어 있습니다. 
마찬가지로 픽셀 `$(b)$를 통과하는 광선 $R(b)$는 빛이 물체에 닿고 빈 공간 어딘가로 반사가 될 것입니다. 광원을 직접적으로 향하지 않기 때문에 픽셀 $P(b)$는 물체의 색인 파란색을 띄게 될 것입니다.
마지막으로 픽셀 $P(c)$를 지나는 광선 $R(c)$는 바닥 면에서 반사되어 어딘가로 반사가 될 것이지만, 광원이 물체에 가려져 있습니다. 
이 경우에 물체의 그림자가 생기게 되는데, 광선 $R(c)$는 그림자가 생기는 위치를 향하고 있기 때문에 픽셀 $P(c)$는 어두운 색을 띄게 될 것입니다.

이렇게 레이트레이싱을 통한 렌더링을 구현하기 위해선 우리의 시점(카메라)로부터 `Scene`의 픽셀을 통과하는 각 광선들이 결국 물체의 어떤 부분에 부딪히는지를 계산하고, 
이를 통해 각 픽셀의 색을 결정하면 됩니다. 문제를 정의해 보니 정말 간단하죠?

## 광선 벡터
우리는 이미지를 구성하는 픽셀의 수 만큼 광선을 추적 해야 합니다. 
만약 `1920×1080` 해상도의 이미지(`Scene`) 한 장을 렌더링 한다고 가정하면 총 2,073,600개의 광선을 정의하고 추적해야 합니다.
이러한 광선들은 시작점과 방향이 있는 객체들로 정의하고 구현 할 수 있는데요. 문제는 각 광선들이 향하는 방향이 모두 다르다는 것 입니다.

그래서 우리는 시점(카메라의 위치)으로부터 2D 이미지의 각 픽셀들을 통과하는 광선의 공간 벡터의 방향을 계산해야 합니다.
카메라의 위치 $\vec{C_0}(x,y,z)$와 방향 $\vec{C_d}(x,y,z)$, 카메라의 시야각($ \text{FOV}(\theta) $)이 정해지면 각 픽셀을 통과하는 광선들의 공간 벡터의 방향을 계산 할 수 있습니다. 
물론 카메라는 렌즈를 통과하는 회전축을 중심으로 회전(`roll`)이 가능한데, 이 경우는 카메라의 위쪽이 항상 $\hat{z}(0,0,1)$ 방향을 향한다고 가정하여 카메라의 `roll` 방향 회전은 고려하지 않겠습니다.

![3](https://i.ibb.co/XsWVXV5/2.png)
Figure 3. 이미지의 각 픽셀을 통과하는 광선 벡터의 방향 계산
{: style="color:gray; font-size: 90%; text-align: center;"}

카메라로부터 뻗어나가는 광선 벡터를 프로그래밍으로 구현하기 전에 먼저 사고 실험을 통해 어떤 모습의 이미지가 렌더링 될지 예상 해 보겠습니다. 
**텅 빈 공간 가운데 아주 넓은 평면이 있고, 우리가 그 평면 위에 서서 카메라로 사진을 찍어 본다고 가정해 보겠습니다.**
그럼 어떤 사진이 찍혔을까요? 아마 예시로 가져온 오른쪽 달 사진처럼 멀리까지 뻗어있는 지평선이 보이는 사진이 촬영되었을 것입니다. 
수평선이 사진의 정 가운데에 있다고 해 보면 이미지의 절반 아래쪽의 픽셀을 통과하는 광선 벡터들은 평면(예시에선 달 표면)을 향할테고, 
이미지의 절반 위쪽의 픽셀을 통과하는 광선 벡터들은 어디에도 부딪히지 않은채 텅 빈 공간(예시에선 우주 공간)으로 날아가게 될 것입니다.

그렇다면 각 픽셀들은 어떤 색을 가질까요? 평면에 부딪히는 빛에 해당하는 픽셀 $P_1$은 평면의 색(흰색)을 가지게 될 것이고, 
어디에도 부딪히지 않고 텅 빈 공간으로 날아가는 빛에 해당하는 픽셀 ($P_2$)는 도달한 빛이 없으므로 검은 색을 가지게 될 것입니다. 
이제 실제로 이 사고 실험을 그대로 컴퓨터 프로그래밍을 통해 구현해 보겠습니다. 결과는 다음과 같습니다.

![4](https://i.ibb.co/whHDrDx/3.png)
Figure 4. 레이트레이싱을 통해 렌더링한 평면
{: style="color:gray; font-size: 90%; text-align: center;"}

그냥 검은 이미지 아래 절반 흰색으로 칠한거 아니냐구요? 놀랍게도 아닙니다... 😅

지금 보고 계신 이미지는 `1920×1080` 해상도를 렌더링하기 위해 카메라로부터 뻗어나가는 2,073,600개의 광선을 정의하고, 
해당 광선들이 각각 어디에 부딪히는지를 전부 추적하여 계산된 이미지입니다. 
이 이미지 한 장을 렌더링 하는데 제 노트북으로 1분 20초 정도가 소요되었습니다. 이제 코드를 보면서 레이트레이싱 기법을 이해해 보겠습니다.

먼저 `Scene`을 구성하는 `object`들을 정의해 보겠습니다. 앞으로 다양한 물체들의 클래스를 정의할 것 같지만, 여기서는 평면을 나타내는 `Plane()` 클래스를 만들어 보겠습니다.
구현 방식에 따라 다른데 저는 계산을 담당하는 레이트레이싱 알고리즘은 모두 `Scene()` 클래스에 넣을 것이기 때문에, 
`Plane()` 클래스에서는 이게 어떤 평면인지를 나타내 줄 수 있는 정보만 넣어 주면 충분 할 것 같습니다.
3차원 공간에서 평면은 한 점과 평면에 수직인 방향벡터(이를 보통 노말벡터 라고 합니다)로 정의 할 수 있습니다. 
그래서 `Plane()` 클래스 속성으로 `.position`, `.normal` 그리고 평면의 색을 정의하는 `.color` 속성을 정의합니다. 

{% highlight python %}
class Plane():
    def __init__(self, position, normal, color):
        self.type = 'plane'
        self.position = np.array(position)
        self.normal = normalize(np.array(normal))
        self.color = np.array(color)
{% endhighlight %}

이제 `Plane()` 클래스를 이용하여 점 $P=(0,0,-1)$을 지나고, $\hat{z}$ 방향에 수직(즉, 노말벡터가 $N=(0,0,1)$)인 흰색 평면을 만들어 보겠습니다.
만들어진 흰색 평면 `white_plane`을 `1920×1080` 해상도의 `Scene`에 넣고, `Scene`을 렌더링 하기 위한 시점(카메라)을 정의합니다.
카메라는 공간상에서 원점($C_0=(0,0,0)$)에 존재하고, $\hat{y}$ 축을 바라보고($C_d=(0,1,0)$) 있습니다.  

{% highlight python %}
white_plane = Plane(position=(0,0,-1), normal=(0,0,1), color=(1,1,1))

scene = Scene(width=1920, height=1080, objects=[white_plane])
scene.add_camera(camera_position=(0,0,0), camera_direction=(0,1,0))
scene.render()
scene.draw()
{% endhighlight %}

이제 직관적으로 각 객체들이 어떻게 어울리는지 이해하셨을 거라고 생각 됩니다. `scene` 객체를 만든 후, 
해당 `scene` 내에 물체(`white_plane`)와 카메라를 배치하고(`.add_camera()`), 해당 카메라로 사진을 찍고(`.render()`), 이미지를 출력한다고(`.draw()`) 생각하시면 직관적일것 같습니다. 
이미 섹션이 너무 길어졌네요. 이번 섹션에서는 광선 벡터의 방향($ D $)을 이해하는 것이 목적이니 `Scene()` 클래스에서 `.add_camera()`와 `.render()` 메서드를 살펴 보도록 하겠습니다.

`.add_camera()` 메서드는 `Figure 3`에서 정의한 카메라의 기본적인 속성들을 갖고 있습니다. 아직 `depth_limit`에 대해선 설명하지 않았는데, 이후에 하도록 하겠습니다.

{% highlight python %}
def add_camera(self,
               camera_position,
               camera_direction,
               depth_limit = 3):
    self.Co = np.array(camera_position)
    self.Cd = normalize(np.array(camera_direction) - self.Co)
    self.Cu = np.array([0,0,1])
    self.Cr = normalize(np.cross(self.Cd, self.Cu))
    self.hFOV = 75
    self.depth = depth_limit
    self.pixel_w = 2*np.tan(np.radians(self.hFOV/2)) / self.w
    self.pixel_h = self.pixel_w
{% endhighlight %}

우리는 `.render()` 메서드를 통해 `scene`에 추가된 카메라의 원점(`Co`)에서 출발하여 이미지의 픽셀 `(x,y)`를 향하는 광선의 방향 벡터 $ D $ 를 계산합니다.
카메라의 렌즈가 향하는 방향(`Cd`)으로부터 변위 `(dx, dy)`를 계산하여 광선의 방향 벡터 $ D $를 결정한다고 이해하시면 되겠습니다. 

{% highlight python %}
def render(self):
    for x in range(self.w):
        for y in range(self.h):
            dx = self.pixel_w * (x - self.w/2)
            dy = - self.pixel_h * (y - self.h/2)

            O = self.Co # Origin of ray
            D = normalize(self.Cd + dx*self.Cr + dy*self.Cu) # Direction of ray

            color = self.trace(O, D)
            self.image[y,x] = np.clip(color, 0, 1)
{% endhighlight %}

이렇게 뻗어나간 광선은 시작점($O$)과 방향($D$)을 갖고 있습니다. 다음 섹션에서는 이제 `.trace()` 메서드를 통해 해당 광선이 어떤 물체에 충돌하였는지 계산해 보겠습니다.

## 광선 벡터와 물체의 상호작용
먼저 우리는 카메라의 중심점 $O$로부터 방향 $D$를 갖고 뻗어나가는 각 광선이 처음으로 만나는 물체가 무엇인지 추적해야 합니다. 
`.trace()` 메서드는 다시 `.intersection()` 메서드를 호출하여 광선과 `scene`에 존재하는 각 물체들의 교차 좌표(`intersection`)와 거리(`distance`)를 계산하고,
가장 가까이 있는 물체와 그 거리를 계산합니다. 만약 최소 거리(`min_distance`)가 `np.inf`일 경우 해당 광선이 어떤 물체와도 부딪히지 않는다고 볼 수 있습니다.

{% highlight python %}
def trace(self, ray_origin, ray_direction):
    # Step 1: Find the closest object
    min_distance = np.inf
    closest_object = None
    M = None # closest intersection point
    for o, obj in enumerate(self.objects):
        distance, intersection = self.intersection(ray_origin, ray_direction, obj)
        if distance < min_distance:
            min_distance = distance
            closest_object = obj
            M = intersection
    if min_distance == np.inf: # no object
        return np.zeros(3)

    # Step 2: Get properties of the closest object
    color = closest_object.color
    # Step 3:
    return color
{% endhighlight %}

이렇게 각 광선이 처음으로 만나는 물체의 특징이 `scene`을 렌더링 할 때의 색을 결정합니다. 
이후 우리가 다양한 기법을 적용하여 더 현실적은 그래픽을 렌더링 할 수 있지만, 
여기서는 그냥 가장 가까운 물체의 색(`closest_object.color`)을 바로 적용해 보겠습니다.

`.trace()` 메서드의 핵심은 `.intersection()` 메서드를 호출하여 광선과 물체의 거리와 좌표를 계산하는데 있습니다.
그런데 조금 생각해 보시면 금방 이 부분이 그다지 간단하지 않다는 것을 알 수 있습니다. 
같은 위치에 있다고 하더라도 평면, 구, 정육면체 등 물체의 모양이 다 다르고, 빛을 받는 면의 각도에 따라 거리와 좌표가 달라질 수 있기 때문입니다.
이제 우리는 이 문제를 풀기 위해 공간벡터를 이용하여 각 물체에 대해 정의하고, 벡터 계산을 통해 거리와 좌표를 계산해 보겠습니다.


### Ray-Plane intersection
먼저 광선이 평면에 닿는 경우를 생각해 보겠습니다. 점 $O$로부터 방향 $D$를 갖고 나아가는 광선은 한쪽 방향으로 나아가는 직선으로 표현 할 수 있습니다.
직선에 속한 수많은 점들을 매개변수식으로 표현하면 다음과 같습니다.

$$ O + tD$$

방향벡터 $D$의 크기는 1이므로 $t$는 점 $O$로부터의 거리가 될 것입니다. 광선은 한쪽 방향으로 뻗어나가니 $t>0$인 조건에서만 고려하면 되겠습니다.

이제 평면을 정의해 보겠습니다. 
수학적으로 공간에 위치한 평면은 평면 상의 점 $P_0$를 포함하고, 노말 벡터 $N$에 수직인 점들로 표현 할 수 있습니다.
즉 평면 위의 임의의 벡터($\vec{PP_0}$)는 노말 벡터 $N$에 수직이므로, 평면을 매개변수식으로 표현하면 다음과 같습니다.

$$ (P-P_0) \cdot N = 0 $$

직선 $O+tD$와 평면 $ (P-P_0) \cdot N = 0 $ 사이의 관계를 그림으로 나타내면 다음과 같습니다.
직선과 평면의 교차점을 $I$라고 할 때, I는 두 매개변수 방정식 $I=O+t_{I}D$와 $(I-P_0) \cdot N = 0$을 만족하게 됩니다.

![5](https://i.ibb.co/jWmwkbj/5.png)

직선이 평면에 포함되거나 평행하지 않을 때(즉, $D \cdot N \neq 0$), 이를 만족하는 $t$는 다음과 같습니다.

$$ t = \frac{(P_0 - O) \cdot N}{D \cdot N} $$

위 방정식을 통해 우리는 교차점의 좌표 $I$와 교차점까지의 거리 $t$를 계산 할 수 있습니다. `.intersection()` 메서드의 코드는 다음과 같습니다. 
$t$가 매우 큰 경우($ t > 10^4$), 우리는 광선이 평면을 지나지 않는다고(평면과 평행하다고) 볼 수 있습니다. 이런 경우 교차점까지의 거리는 `np.inf`가 됩니다.
마찬가지로 $t$ 가 음수인 경우($ t < 0 $) 역시 광선이 반대 방향으로 날아가기 때문에 평면과 만나지 않는다고 볼 수 있습니다.

{% highlight python %}
def intersection(self, ray_origin, ray_direction, object):
    if object.type == 'plane':
        # Ray-Plane intersection
        O = ray_origin
        D = ray_direction
        P0 = object.position
        N = object.normal

        t = np.dot((P0-O), N) / np.dot(D, N)
        if t > 10e4 or t < 0:
            distance = np.inf
        else:
            distance = t
        intersection = O + distance*ray_direction
        return distance, intersection
{% endhighlight %}

이렇게 광선과 평면의 공간 벡터를 이용하여 평면이 카메라 상에서 어떻게 보이는지 계산해 보았습니다. 
위 코드를 이용해 `scene`을 렌더링 하면 Figure 4와 같은 결과를 볼 수 있습니다. 
하지만 Figure 4에서 보실 수 있다 시피 단색 컬러를 입힌 평면은 그렇게 공간감이 느껴지지 않습니다. 
위에서 만든 `Plane()` 클래스를 조금 응용하여 체커보드 객체를 생성하는 `Checkerboard()` 클래스를 만들 수 있습니다.

`Checkerboard()` 클래스는 `xy` 평면 상에서 검은색과 흰색 타일이 반복적으로 놓인 구조입니다. 
다시 말해 평면 위 점의 위치에 따라 객체의 `.color` 속성이 달라져야 합니다. 
따라서 `.color` 속성을 동적으로 바꿔 줄 수 있는 `.colorize()` 메서드를 추가하여 `Checkerboard()` 클래스를 구성합니다.

{% highlight python %}
class Checkerboard():
    def __init__(self, position, normal):
        self.type = 'checkerboard'
        self.position = np.array(position)
        self.normal = normalize(np.array(normal))

    def colorize(self, color):
        self.color = color
{% endhighlight %}

이제 `.intersection()` 메서드에 광선이 물체에 닿는 교차점의 좌표 $I$에 따라 `Checkerboard()` 클래스의 `.colorize()` 메서드를 호출 할 수 있는 흐름 제어를 넣습니다. 
교차점(`intersection`)의 `xy` 좌표에 따라 `Checkerboard()` 클래스의 `.color` 속성을 흰색(`np.ones(3)`) 혹은 검정색(`np.zeros(3)`)으로 업데이트 해 줍니다.

{% highlight python %}
def intersection(self, ray_origin, ray_direction, object):
    if object.type == 'plane' or object.type == 'checkerboard':
        # Ray-Plane intersection
        O = ray_origin
        D = ray_direction
        P = object.position
        N = object.normal

        distance = np.dot((P-O), N) / np.dot(D, N)
        if distance > 10e4 or distance < 0:
            distance = np.inf
        intersection = O + distance*ray_direction

        if object.type == 'checkerboard':
            if distance != np.inf:
                if np.floor(intersection[0]) % 2 == np.floor(intersection[1]) % 2:
                    color = np.ones(3)
                else:
                    color = np.zeros(3)
                object.colorize(color)

        return distance, intersection
{% endhighlight %}

이제 Figure 4에서 그린 흰색 평면 대신 같은 위치에 `checkerboard` 객체를 생성하여 `scene`을 렌더링 합니다.

{% highlight python %}
checkerboard = Checkerboard(position=(0,0,-1), normal=(0,0,1))

scene = Scene(width=1920, height=1080, objects=[checkerboard])
scene.add_camera(camera_position=(0,0,0), camera_direction=(0,1,0))
scene.render()
scene.draw()
{% endhighlight %}

결과는 다음과 같습니다. 어떤가요? 실제로 카메라를 통해 공간 상에 위치한 평면을 렌더링 했다는 것이 잘 느껴지시죠? 
이제 더 다양한 물체를 구현하여 `scene`에 추가해 보도록 하겠습니다.

![6](https://i.ibb.co/D83KKV2/6.png)

### Ray-Sphere intersection

평면 다음으로 구현해 볼 물체는 바로 구(Sphere) 입니다. 공간 상에서 구는 구의 중점($P_0$)과 반지름($R$)로 표현 됩니다.
평면과 마찬가지로 구의 점들을 매개변수 방정식으로 표현 할 수 있습니다. 구에 속한 임의의 점($P$)은 구의 중점 $P_0$로부터 반지름 $R$ 만큼 떨어져 있으므로 다음 식을 만족합니다. 

$$ \lvert P-P_0 \rvert ^2 - R^2 = 0 $$

만약 광선($O+tD$)가 구를 지난다면, 조건에 따라 다음과 같은 경우가 발생 할 수 있습니다. 
광선은 구를 뚫고 지나가는 경우가 있을 수 있고, 이 경우엔 광선과 구 사이의 교차점은 2개가 발생합니다. 이를 각각 $I_1$과 $I_2$라 할 수 있습니다.
광선이 구의 한 점에 겹치는 $I_1=I_2$인 경우와 광선이 구를 지나지 않는 경우가 있을 수 있습니다. 이 경우엔 교차점을 찾을 수 없습니다.

![7](https://i.ibb.co/PNdrQvQ/7.png)

오른쪽 그림에서 광선의 매개변수 $t$의 조건에 따라 교차점이 어떤 식으로 정의 되는지 볼 수 있습니다. 
이제 매개변수 방정식을 풀어 실제 교차점 $I$의 좌표와 교차점 까지의 거리 $t$를 계산해 보겠습니다. 
광선이 구를 지나는 경우 구 위의 점 $P$를 광선의 매개변수 식인 $O+tD$로 치환 할 수 있습니다.

$$ \lvert O+tD -P_0 \rvert ^2 - R^2 = 0 $$

위 식을 풀어 쓰면 다음과 같습니다.

$$  D^2 t^2 + 2D(O-P_0) t +  \lvert O-P_0 \rvert ^2 - R^2 = 0  $$


