# Cuisine classifiers 1

이 강의에서는 앞서 강의에서 저장한 균형 잡히고 깔끔한 요리 데이터셋을 사용합니다.

이 데이터셋을 다양한 분류기와 함께 사용하여 _주어진 재료 그룹을 기반으로 특정 국가 요리를 예측하는 작업_을 수행할 것입니다. 이를 통해 알고리즘이 분류 작업에 어떻게 활용될 수 있는지에 대해 더 많이 배우게 됩니다.

## [사전 강의 퀴즈](https://ff-quizzes.netlify.app/en/ml/)
# 준비

[Lesson 1](../1-Introduction/README.md)을 완료한 상태라면, 이 네 번의 강의를 위해 루트 `/data` 폴더에 _cleaned_cuisines.csv_ 파일이 존재하는지 확인하세요.

## 연습 - 국가 요리 예측

1. 이번 강의의 _notebook.ipynb_ 폴더에서 Pandas 라이브러리와 해당 파일을 불러옵니다:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    데이터는 다음과 같습니다:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. 이제 몇 가지 라이브러리를 더 불러옵니다:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. X와 y 좌표를 두 개의 데이터프레임으로 나눠 훈련에 사용하세요. `cuisine`은 레이블 데이터프레임이 될 수 있습니다:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    이렇게 보일 것입니다:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. `Unnamed: 0` 열과 `cuisine` 열을 `drop()`으로 삭제하세요. 나머지 데이터를 학습 가능한 특징(feature)으로 저장합니다:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    특징 데이터는 다음과 같습니다:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

이제 모델을 학습시킬 준비가 되었습니다!

## 분류기 선택

데이터가 깨끗하고 학습할 준비가 되었으므로, 어떤 알고리즘을 사용할지 결정해야 합니다.

Scikit-learn은 분류를 감독 학습(Supervised Learning) 아래에 그룹화하며, 그 안에서 다양한 분류 방법을 찾을 수 있습니다. [다양한 방법들](https://scikit-learn.org/stable/supervised_learning.html)은 처음 보면 꽤 복잡해 보일 수 있습니다. 다음 방법들이 모두 분류 기술을 포함합니다:

- 선형 모델 (Linear Models)
- 서포트 벡터 머신 (Support Vector Machines)
- 확률적 경사 하강법 (Stochastic Gradient Descent)
- 최근접 이웃 알고리즘 (Nearest Neighbors)
- 가우시안 프로세스 (Gaussian Processes)
- 결정 트리 (Decision Trees)
- 앙상블 방법 (voting Classifier)
- 다중 클래스 및 다중 출력 알고리즘 (multiclass and multilabel classification, multiclass-multioutput classification)

> [신경망으로도 분류가 가능합니다](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification). 하지만 이는 이 강의 범위를 벗어납니다.

### 어떤 분류기를 쓸까?

그렇다면 어떤 분류기를 선택해야 할까요? 보통은 여러 분류기를 실행해보고 좋은 결과가 나오는 것을 선택하는 방법이 있습니다. Scikit-learn은 생성한 데이터셋 위에서 KNeighbors, 두 가지 방법의 SVC, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB, QuadraticDiscrinationAnalysis를 비교하는 [나란히 비교](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)를 제공합니다. 결과는 시각화되어 있습니다:

![comparison of classifiers](../../../../translated_images/ko/comparison.edfab56193a85e7f.webp)
> 스캐킷런 문서에서 생성된 플롯

> AutoML은 이 문제를 클라우드에서 여러 비교를 실행해 가장 적합한 알고리즘을 선택할 수 있도록 깔끔하게 해결합니다. [여기서 시도해 보세요](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### 더 나은 접근법

무작위로 추측하는 것보다는 다운로드 가능한 [ML 치트 시트](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott)의 아이디어를 따르는 것이 더 좋습니다. 여기서 다중 클래스 문제에 대해 다음과 같은 선택지를 알 수 있습니다:

![cheatsheet for multiclass problems](../../../../translated_images/ko/cheatsheet.07a475ea444d2223.webp)
> 마이크로소프트 알고리즘 치트 시트 일부, 다중 클래스 분류 옵션 설명

✅ 이 치트 시트를 다운로드하여 출력한 후 벽에 붙여 두세요!

### 이유 분석

제약 조건을 고려해 서로 다른 접근법을 논리적으로 살펴봅시다:

- **신경망은 너무 무겁다**. 깔끔하지만 최소한의 데이터셋이고 노트북에서 로컬 학습을 진행하므로 신경망은 과도한 무게를 가집니다.
- **2클래스 분류기를 사용하지 않는다**. 2클래스 분류기가 아니므로 one-vs-all 방식은 제외됩니다.
- **결정 트리나 로지스틱 회귀가 가능하다**. 결정 트리나 다중 클래스 데이터용 로지스틱 회귀를 사용할 수 있습니다.
- **다중 클래스 부스팅 결정 트리는 다른 문제에 적합하다**. 다중 클래스 부스팅 결정 트리는 주로 순위 생성과 같은 비모수 작업에 적합하여 여기에는 적합하지 않습니다.

### Scikit-learn 사용하기

데이터 분석을 위해 Scikit-learn을 사용할 것입니다. 하지만 Scikit-learn에서 로지스틱 회귀를 사용하는 방법은 다양합니다. [전달 가능한 파라미터](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)를 확인해 보세요.

사실상 중요한 파라미터는 `multi_class`와 `solver` 두 가지입니다. 로지스틱 회귀를 수행할 때 이 두 가지를 지정해야 합니다. `multi_class` 값은 특정 동작을 적용하고, `solver`는 사용할 알고리즘을 지정합니다. 모든 solver가 모든 `multi_class` 값과 호환되는 것은 아닙니다.

문서에 따르면, 다중 클래스의 경우 학습 알고리즘은:

- **`multi_class` 옵션이 `ovr`인 경우 one-vs-rest (OvR) 방식을 사용**합니다.
- **`multi_class` 옵션이 `multinomial`인 경우 교차 엔트로피 손실(cross-entropy loss)를 사용**합니다. (`multinomial` 옵션은 현재 'lbfgs', 'sag', 'saga', 'newton-cg' solver만 지원)

> 🎓 여기서 'scheme'은 'ovr'(one-vs-rest) 또는 'multinomial'일 수 있습니다. 로지스틱 회귀는 기본적으로 이진 분류용이지만 이 방식들이 다중 클래스 분류에 적합하도록 돕습니다. [소스](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver'는 "최적화 문제에서 사용하는 알고리즘"으로 정의됩니다. [소스](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn은 소버가 서로 다른 데이터 구조에서 어떻게 문제를 처리하는지 설명하는 표를 제공합니다:

![solvers](../../../../translated_images/ko/solvers.5fc648618529e627.webp)

## 연습 - 데이터 분할

최근 이전 강의에서 배운 로지스틱 회귀를 첫 번째 학습 시도에 집중하여 사용합시다.

`train_test_split()`을 호출하여 데이터를 학습 그룹과 평가 그룹으로 나눕니다:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## 연습 - 로지스틱 회귀 적용

다중 클래스 설정을 사용하기 때문에 어떤 _scheme_과 _solver_를 쓸지 선택해야 합니다. `multi_class`를 설정하고 **liblinear** solver를 사용해 LogisticRegression을 학습하세요.

1. multi_class를 `ovr`로, solver를 `liblinear`로 설정해 로지스틱 회귀를 만듭니다:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ `lbfgs`와 같은 다른 solver도 시도해 보세요. 보통 기본값으로 설정되어 있습니다.

    > 필요할 때 Pandas의 [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) 함수를 사용해 데이터를 평탄화하세요.

    정확도는 <strong>80% 이상</strong>으로 좋습니다!

1. 데이터 한 행 (#50)으로 테스트하여 이 모델이 어떻게 동작하는지 확인할 수 있습니다:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    결과가 출력됩니다:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ 다른 행 번호로 바꿔 결과를 확인해 보세요.
1. 좀 더 자세히 살펴보면, 이 예측의 정확성을 확인할 수 있습니다:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    결과가 출력됩니다 - 인도 요리가 모델의 가장 좋은 추측이며, 높은 확률입니다:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ 모델이 왜 이 음식이 인도 요리라고 꽤 확신하는지 설명할 수 있나요?

1. 회귀 수업에서 했던 것처럼, 분류 보고서를 출력하여 더 자세한 정보를 얻으세요:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | chinese      | 0.73      | 0.71   | 0.72     | 229     |
    | indian       | 0.91      | 0.93   | 0.92     | 254     |
    | japanese     | 0.70      | 0.75   | 0.72     | 220     |
    | korean       | 0.86      | 0.76   | 0.81     | 242     |
    | thai         | 0.79      | 0.85   | 0.82     | 254     |
    | accuracy     |           |        | 0.80     | 1199    |
    | macro avg    | 0.80      | 0.80   | 0.80     | 1199    |
    | weighted avg | 0.80      | 0.80   | 0.80     | 1199    |

## 🚀도전 과제

이번 수업에서는 정리된 데이터를 사용해 일련의 재료를 기반으로 국가 요리를 예측하는 머신러닝 모델을 만들었습니다. Scikit-learn이 제공하는 다양한 데이터 분류 옵션을 살펴볼 시간을 가지세요. 'solver' 개념을 더 깊게 파고들어 내부에서 어떤 일이 일어나는지 이해해 보세요.

## [강의 후 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

## 복습 및 자기 주도 학습

[이 수업](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)에서 로지스틱 회귀의 수학적 배경을 좀 더 깊이 탐구해 보세요.
## 과제

[솔버 학습](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 노력하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있음을 유의하시기 바랍니다. 원본 문서가 권위 있는 출처로 간주되어야 합니다. 중요한 정보의 경우 전문가의 인적 번역을 권장합니다. 본 번역 사용으로 인한 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->