# 클러스터링 소개

클러스터링은 데이터 세트가 라벨이 없거나 입력이 사전에 정의된 출력과 일치하지 않는다고 가정하는 [비지도 학습](https://wikipedia.org/wiki/Unsupervised_learning)의 한 유형입니다. 다양한 알고리즘을 사용하여 라벨이 없는 데이터를 분류하고 데이터에서 인식한 패턴에 따라 그룹을 제공합니다.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 위 이미지를 클릭하면 동영상을 볼 수 있습니다. 클러스터링으로 머신러닝을 공부하면서 나이지리아 댄스홀 트랙을 즐겨보세요 - 이 곡은 2014년 PSquare의 매우 인기 있는 노래입니다.

## [강의 전 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

### 소개

[클러스터링](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124)은 데이터 탐색에 매우 유용합니다. 나이지리아 관객이 음악을 소비하는 방식을 통해 트렌드와 패턴을 발견할 수 있는지 살펴보겠습니다.

✅ 클러스터링의 용도에 대해 1분 정도 생각해 보세요. 실제 생활에서는 빨래 더미를 가족 구성원의 옷으로 분류해야 할 때 클러스터링이 발생합니다 🧦👕👖🩲. 데이터 과학에서는 사용자의 선호도를 분석하거나 라벨이 없는 데이터 세트의 특성을 파악할 때 클러스터링이 사용됩니다. 클러스터링은 소음이 많은 양말 서랍처럼 혼란을 이해하는 데 도움을 줍니다.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 위 이미지를 클릭하면 동영상을 볼 수 있습니다: MIT의 John Guttag가 클러스터링을 소개합니다.

전문적인 환경에서 클러스터링은 시장 세분화, 예를 들면 어떤 연령대가 어떤 아이템을 구매하는지 결정하는 데 사용할 수 있습니다. 또 다른 용도는 이상 탐지로, 신용카드 거래 데이터에서 사기를 탐지하는 경우가 있습니다. 또는 의료 스캔 배치에서 종양을 판단하는 데 클러스터링을 사용할 수 있습니다.

✅ 은행, 전자상거래 또는 비즈니스 환경에서 클러스터링을 '현실에서' 접한 적이 있는지 1분 정도 생각해 보세요.

> 🎓 흥미롭게도 클러스터 분석은 1930년대 인류학과 심리학 분야에서 시작되었습니다. 어떻게 사용되었을지 상상해 보세요.

또는 검색 결과를 쇼핑 링크, 이미지, 리뷰별로 그룹화하는 데 사용할 수도 있습니다. 클러스터링은 큰 데이터 세트를 줄이고 더 세밀한 분석을 수행하기 원하는 경우에 유용하므로, 다른 모델을 구축하기 전에 데이터에 대해 배우는 데 사용할 수 있습니다.

✅ 데이터가 클러스터로 정리되면 클러스터 ID를 할당하며, 이 방법은 데이터 세트의 프라이버시를 유지하는 데 유용할 수 있습니다. 데이터 포인트를 더 노출되는 식별 가능한 데이터 대신 클러스터 ID로 참조할 수 있습니다. 클러스터의 다른 요소 대신 클러스터 ID로 식별하는 다른 이유가 있을까요?

이 [학습 모듈](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)에서 클러스터링 기법에 대한 이해를 심화해 보세요.
## 클러스터링 시작하기

[Scikit-learn은 다양한](https://scikit-learn.org/stable/modules/clustering.html) 클러스터링 수행 방법을 제공합니다. 선택하는 유형은 사용 사례에 따라 다릅니다. 문서에 따르면 각 방법은 다양한 장점을 가지고 있습니다. 다음은 Scikit-learn에서 지원하는 방법과 적합한 사용 사례를 간략히 정리한 표입니다:

| 방법 이름                    | 사용 사례                                                            |
| :--------------------------- | :------------------------------------------------------------------ |
| K-Means                      | 범용, 귀납적                                                       |
| Affinity propagation         | 다수, 불균일 클러스터, 귀납적                                       |
| Mean-shift                   | 다수, 불균일 클러스터, 귀납적                                       |
| Spectral clustering          | 소수, 균일 클러스터, 전이적                                         |
| Ward hierarchical clustering | 다수, 제약된 클러스터, 전이적                                       |
| Agglomerative clustering     | 다수, 제약 및 비유클리드 거리, 전이적                               |
| DBSCAN                       | 비평면 형상, 불균일 클러스터, 전이적                               |
| OPTICS                       | 비평면 형상, 가변 밀도의 불균일 클러스터, 전이적                   |
| Gaussian mixtures            | 평면 형상, 귀납적                                                  |
| BIRCH                        | 큰 데이터 세트 및 이상치, 귀납적                                     |

> 🎓 클러스터를 만드는 방법은 데이터를 어떻게 그룹으로 모으느냐와 밀접한 관련이 있습니다. 몇 가지 용어를 살펴봅시다:
>
> 🎓 ['전이적' vs. '귀납적'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> 전이적 추론은 특정 테스트 케이스에 매핑되는 관측된 훈련 사례에서 파생됩니다. 귀납적 추론은 일반 규칙에 매핑되는 훈련 사례에서 파생되며, 이후에 테스트 케이스에 적용됩니다. 
> 
> 예시: 데이터 세트가 일부만 라벨이 달려 있다고 가정해 보세요. 일부는 '레코드', 일부는 'CD', 일부는 빈칸입니다. 빈칸에 라벨을 부여하는 것이 과제입니다. 귀납적 접근법을 선택하면 '레코드'와 'CD'를 찾는 모델을 훈련시키고 해당 라벨을 라벨 없는 데이터에 적용합니다. 이 방법은 실제로는 '카세트'인 것을 분류하는 데 어려움이 있습니다. 반면 전이적 접근법은 비슷한 항목을 그룹화하여 레이블을 그룹에 적용하므로 이 알려지지 않은 데이터를 더 효과적으로 처리합니다. 이 경우 클러스터는 '둥근 음악 물건'과 '네모난 음악 물건'을 나타낼 수 있습니다.
> 
> 🎓 ['비평면' vs. '평면' 형상](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> 수학 용어에서 파생된 비평면 대 평면 형상은 점들 사이 거리 측정을 평면([유클리드](https://wikipedia.org/wiki/Euclidean_geometry)) 또는 비평면(비유클리드) 기하학적 방법으로 하는 것을 말합니다.
>
> 이 문맥에서 '평면'은 유클리드 기하학을 의미하며(일부는 '평면' 기하학으로 가르침), 비평면은 비유클리드 기하학을 뜻합니다. 기하학이 머신러닝과 무슨 관련이 있나요? 두 학문 모두 수학을 기반으로 하므로 클러스터 내 점들 간 거리를 측정하는 공통 방법이 필요하며, 데이터 특성에 따라 '평면' 또는 '비평면' 방식으로 측정할 수 있습니다. [유클리드 거리](https://wikipedia.org/wiki/Euclidean_distance)는 두 점 간의 선분 길이로 측정합니다. [비유클리드 거리](https://wikipedia.org/wiki/Non-Euclidean_geometry)는 곡선을 따라 측정합니다. 시각화된 데이터가 평면 위에 존재하지 않는 것처럼 보이면 이를 처리할 특별한 알고리즘이 필요할 수 있습니다.
>
![평면 대 비평면 형상 인포그래픽](../../../../translated_images/ko/flat-nonflat.d1c8c6e2a96110c1.webp)
> 인포그래픽 제공: [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['거리'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> 클러스터는 각 점들 사이 거리 매트릭스로 정의됩니다. 거리는 여러 방법으로 측정할 수 있습니다. 유클리드 클러스터는 점값 평균과 중심점('센트로이드')을 기준으로 정의되며, 거리는 이 중심점까지의 거리로 측정합니다. 비유클리드 거리는 '클러스트로이드', 즉 다른 점들에 가장 가까운 점을 기준으로 합니다. 클러스트로이드는 다시 여러 방법으로 정의될 수 있습니다.
> 
> 🎓 ['제약'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [제약 클러스터링](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf)은 이 비지도 방식에 반지도 학습을 도입합니다. 점들 간 관계가 '연결 금지(cannot link)' 또는 '반드시 연결(must-link)'로 지정되어 데이터 세트에 규칙이 적용됩니다.
>
> 예시: 알고리즘이 라벨이 없거나 부분적으로 라벨링된 데이터에 적용될 때 생성된 클러스터 품질이 낮을 수 있습니다. 앞 예시에서 클러스터가 '둥근 음악 물건', '네모난 음악 물건', '삼각형 물건', '쿠키'로 나뉠 수 있습니다. "이 항목은 플라스틱 재질이어야 한다", "음악을 낼 수 있어야 한다" 등의 제약이 주어진다면 알고리즘이 더 나은 선택을 하도록 '제약'할 수 있습니다.
> 
> 🎓 '밀도'
> 
> '노이즈가 많은' 데이터는 '밀집'되어 있다고 간주됩니다. 각 클러스터 내 점들 사이 거리는 밀집도나 혼잡도로 평가될 수 있으며, 이에 맞는 클러스터링 방법으로 분석해야 합니다. [이 글](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html)은 시끄러운 데이터에 K-평균 클러스터링과 HDBSCAN을 각각 적용했을 때 차이를 보여줍니다.

## 클러스터링 알고리즘

100개가 넘는 클러스터링 알고리즘이 있으며, 사용은 데이터 특성에 따라 달라집니다. 주요 알고리즘 몇 가지를 살펴보겠습니다:

- **계층적 클러스터링**. 객체가 먼 객체가 아니라 근처 객체와의 근접도에 따라 분류되며, 구성원과 다른 객체 간 거리 기준으로 클러스터가 형성됩니다. Scikit-learn의 응집형 클러스터링이 계층적입니다.

   ![계층적 클러스터링 인포그래픽](../../../../translated_images/ko/hierarchical.bf59403aa43c8c47.webp)
   > 인포그래픽 제공: [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **센트로이드 클러스터링**. 인기 있는 이 알고리즘은 클러스터 수 'k'를 선택한 후 클러스터 중심점을 결정하며 주변 데이터를 모읍니다. [K-평균 클러스터링](https://wikipedia.org/wiki/K-means_clustering)은 인기 있는 센트로이드 클러스터링의 한 버전입니다. 중심점은 가장 가까운 평균으로 결정되어 명칭이 붙었습니다. 클러스터로부터의 제곱 거리 합을 최소화합니다.

   ![센트로이드 클러스터링 인포그래픽](../../../../translated_images/ko/centroid.097fde836cf6c918.webp)
   > 인포그래픽 제공: [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **분포 기반 클러스터링**. 통계 모델링에 기반하며, 데이터 점이 클러스터에 속할 확률을 판단해 할당합니다. 가우시안 혼합 방법이 포함됩니다.

- **밀도 기반 클러스터링**. 데이터 점들은 서로 얼마나 밀집되어 모여 있는지 밀도에 따라 클러스터에 할당됩니다. 군집에서 멀리 떨어진 점은 이상치나 노이즈로 간주됩니다. DBSCAN, Mean-shift, OPTICS가 이에 해당합니다.

- **격자 기반 클러스터링**. 다차원 데이터 세트의 경우 격자가 생성되고 데이터가 격자 셀에 분배되어 클러스터를 만듭니다.

## 연습 - 데이터 클러스터링하기

클러스터링은 올바른 시각화가 크게 도움되므로 음악 데이터 시각화로 시작해 봅시다. 이 연습은 이 데이터 특성에 가장 효과적인 클러스터링 방법을 결정하는 데 도움이 됩니다.

1. 이 폴더 내 [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) 파일을 엽니다.

1. 좋은 데이터 시각화를 위해 `Seaborn` 패키지를 가져옵니다.

    ```python
    !pip install seaborn
    ```

1. [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv)에서 노래 데이터를 추가합니다. 노래에 관한 일부 데이터를 포함하는 데이터프레임을 로드합니다. 라이브러리들을 불러와 데이터를 출력하여 탐색 준비를 합니다:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    데이터 앞 몇 줄을 확인하세요:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | 인디 r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | 나이지리아 팝     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | 아프로팝          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. 데이터프레임에 대한 정보를 얻기 위해 `info()`를 호출하세요:

    ```python
    df.info()
    ```

   출력결과는 다음과 같습니다:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 530 entries, 0 to 529
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   name              530 non-null    object 
     1   album             530 non-null    object 
     2   artist            530 non-null    object 
     3   artist_top_genre  530 non-null    object 
     4   release_date      530 non-null    int64  
     5   length            530 non-null    int64  
     6   popularity        530 non-null    int64  
     7   danceability      530 non-null    float64
     8   acousticness      530 non-null    float64
     9   energy            530 non-null    float64
     10  instrumentalness  530 non-null    float64
     11  liveness          530 non-null    float64
     12  loudness          530 non-null    float64
     13  speechiness       530 non-null    float64
     14  tempo             530 non-null    float64
     15  time_signature    530 non-null    int64  
    dtypes: float64(8), int64(4), object(4)
    memory usage: 66.4+ KB
    ```

1. `isnull()`을 호출하고 합계가 0인지 확인하여 널 값이 있는지 다시 확인하세요:

    ```python
    df.isnull().sum()
    ```

    이상 없습니다:

    ```output
    name                0
    album               0
    artist              0
    artist_top_genre    0
    release_date        0
    length              0
    popularity          0
    danceability        0
    acousticness        0
    energy              0
    instrumentalness    0
    liveness            0
    loudness            0
    speechiness         0
    tempo               0
    time_signature      0
    dtype: int64
    ```

1. 데이터를 설명합니다:

    ```python
    df.describe()
    ```

    |       | release_date | length      | popularity | danceability | acousticness | energy   | instrumentalness | liveness | loudness  | speechiness | tempo      | time_signature |
    | ----- | ------------ | ----------- | ---------- | ------------ | ------------ | -------- | ---------------- | -------- | --------- | ----------- | ---------- | -------------- |
    | count | 530          | 530         | 530        | 530          | 530          | 530      | 530              | 530      | 530       | 530         | 530        | 530            |
    | mean  | 2015.390566  | 222298.1698 | 17.507547  | 0.741619     | 0.265412     | 0.760623 | 0.016305         | 0.147308 | -4.953011 | 0.130748    | 116.487864 | 3.986792       |
    | std   | 3.131688     | 39696.82226 | 18.992212  | 0.117522     | 0.208342     | 0.148533 | 0.090321         | 0.123588 | 2.464186  | 0.092939    | 23.518601  | 0.333701       |
    | min   | 1998         | 89488       | 0          | 0.255        | 0.000665     | 0.111    | 0                | 0.0283   | -19.362   | 0.0278      | 61.695     | 3              |
    | 25%   | 2014         | 199305      | 0          | 0.681        | 0.089525     | 0.669    | 0                | 0.07565  | -6.29875  | 0.0591      | 102.96125  | 4              |
    | 50%   | 2016         | 218509      | 13         | 0.761        | 0.2205       | 0.7845   | 0.000004         | 0.1035   | -4.5585   | 0.09795     | 112.7145   | 4              |
    | 75%   | 2017         | 242098.5    | 31         | 0.8295       | 0.403        | 0.87575  | 0.000234         | 0.164    | -3.331    | 0.177       | 125.03925  | 4              |
    | max   | 2020         | 511738      | 73         | 0.966        | 0.954        | 0.995    | 0.91             | 0.811    | 0.582     | 0.514       | 206.007    | 5              |

> 🤔 우리가 레이블이 필요 없는 비지도 학습 방법인 군집화를 다루고 있다면, 왜 이렇게 레이블이 있는 데이터를 보여주고 있을까요? 데이터 탐색 단계에서는 이 레이블들이 유용하지만, 군집화 알고리즘이 작동하는 데는 필요하지 않습니다. 열 머리글을 제거하고 열 번호로 데이터를 참조해도 무방합니다.

데이터의 일반적인 값을 살펴보세요. 인기(popularity)가 '0'인 경우가 있는데, 이는 순위가 없는 노래임을 뜻합니다. 곧 이 값을 제거할 것입니다.

1. 가장 인기 있는 장르를 알아내기 위해 막대 그래프를 사용하세요:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/ko/popular.9c48d84b3386705f.webp)

✅ 더 많은 상위 값을 보고 싶으면 상위 `[:5]`를 더 큰 값으로 변경하거나 제거하여 모두 볼 수 있습니다.

참고로, 상위 장르가 'Missing'으로 표시된다면 Spotify가 분류하지 않은 경우이므로, 이 항목은 제거합시다.

1. 누락된 데이터를 필터링하여 제거하세요:

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    이제 장르를 다시 확인하세요:

    ![most popular](../../../../translated_images/ko/all-genres.1d56ef06cefbfcd6.webp)

1. 이 데이터셋에서 상위 세 장르가 압도적으로 지배적입니다. `afro dancehall`, `afropop`, `nigerian pop`에 집중하고, 0 인기 값을 가진 데이터(즉, 데이터셋 내에서 인기가 분류되지 않아 목적에 따라 노이즈로 간주될 수 있는 데이터)를 추가로 필터링하세요:

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. 데이터가 특별히 강한 상관관계를 보이는지 빠르게 테스트하세요:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/ko/correlation.a9356bb798f5eea5.webp)

    `energy`와 `loudness` 사이에만 강한 상관관계가 있습니다. 이는 시끄러운 음악이 보통 에너지가 넘친다는 것을 고려하면 놀랍지 않습니다. 그 외 상관관계는 상대적으로 약합니다. 이 데이터를 군집화 알고리즘이 어떻게 해석할지 흥미롭습니다.

    > 🎓 상관관계가 인과관계를 의미하지는 않는다는 점에 유의하세요! 상관관계는 증명되었지만 인과관계는 증명되지 않았습니다. [재미있는 웹사이트](https://tylervigen.com/spurious-correlations)는 이 점을 강조하는 시각 자료를 제공합니다.

이 데이터셋에서 노래의 인지도(popularity)와 댄서빌리티(danceability)가 어떤 수렴을 보일까요? FacetGrid는 장르에 상관없이 동심원형이 정렬된 모습을 보여줍니다. 나이지리아 취향이 이 장르에서 특정 댄서빌리티 수준에 수렴할 수도 있을까요?

✅ 다양한 데이터 포인트(energy, loudness, speechiness)와 더 많은 또는 다른 음악 장르를 시도해 보세요. 어떤 점을 발견할 수 있나요? `df.describe()` 표를 참고하여 데이터 포인트의 전반적인 분포를 확인하세요.

### 연습 - 데이터 분포

이 세 장르는 인지도에 기반하여 댄서빌리티 인식에서 통계적으로 유의한 차이가 있나요?

1. 상위 세 장르별 인지도와 댄서빌리티 데이터 분포를 지정된 x, y 축에 따라 조사하세요.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    일반적인 수렴 지점 주위에 동심원을 발견할 수 있습니다. 이는 점들의 분포를 보여줍니다.

    > 🎓 이 예제는 커널 밀도 추정(Kernel Density Estimate, KDE) 그래프를 사용합니다. 이는 연속 확률 밀도 곡선을 통해 데이터를 나타내며 여러 분포 다룰 때 데이터를 해석하는 데 도움 됩니다.

    전반적으로 세 장르는 인지도와 댄서빌리티 면에서 느슨하게 정렬되어 있습니다. 이 느슨하게 정렬된 데이터에서 군집을 결정하는 것은 도전 과제가 될 것입니다:

    ![distribution](../../../../translated_images/ko/distribution.9be11df42356ca95.webp)

1. 산점도를 만드세요:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    동일 축을 가진 산점도는 비슷한 수렴 양상을 보여줍니다.

    ![Facetgrid](../../../../translated_images/ko/facetgrid.9b2e65ce707eba1f.webp)

일반적으로, 군집화를 위해 산점도를 사용하여 데이터 군집을 시각화할 수 있으니, 이러한 유형의 시각화 숙달이 매우 유용합니다. 다음 강의에서는 이 필터링된 데이터를 사용해 k-평균 군집화를 적용하고, 흥미로운 방식으로 겹치는 데이터 그룹을 발견할 것입니다.

---

## 🚀도전 과제

다음 강의를 준비하며, 생산 환경에서 사용할 수 있는 다양한 군집화 알고리즘에 대해 차트를 만들어 보세요. 군집화가 해결하려는 문제 유형은 무엇인가요?

## [강의 후 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

## 복습 및 자습

군집화 알고리즘을 적용하기 전에 데이터셋의 본질을 이해하는 것이 좋습니다. 이에 대한 자세한 내용은 [여기](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)에서 읽어보세요.

[이 유용한 기사](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/)는 다양한 데이터 형태에 따른 여러 군집화 알고리즘의 동작 방식을 안내합니다.

## 과제

[군집화에 대한 다른 시각화 연구하기](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**면책 조항**:
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 기하기 위해 노력하고 있으나, 자동 번역은 오류나 부정확한 부분이 있을 수 있음을 유의하시기 바랍니다. 원본 문서의 원어본이 권위 있는 자료로 간주되어야 합니다. 중요한 정보의 경우, 전문가의 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->