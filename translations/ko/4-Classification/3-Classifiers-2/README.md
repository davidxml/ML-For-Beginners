# Cuisine classifiers 2

이 두 번째 분류 수업에서는 숫자 데이터를 분류하는 더 많은 방법을 탐색합니다. 또한 한 분류기를 선택하는 것의 영향에 대해서도 배우게 됩니다.

## [사전 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

### 전제 조건

이전 수업을 완료했고, 4개 수업 폴더의 루트에 있는 `data` 폴더 내에 _cleaned_cuisines.csv_라는 정리된 데이터셋이 있다고 가정합니다.

### 준비

정리된 데이터셋으로 _notebook.ipynb_ 파일을 로드했고, 모델 빌딩 과정에 대비해 X와 y 데이터프레임으로 분할해 두었습니다.

## 분류 지도

이전에 Microsoft의 치트 시트를 사용해 데이터를 분류할 때 선택할 수 있는 여러 옵션에 대해 배웠습니다. Scikit-learn은 이와 유사하지만 더 세분화된 치트 시트를 제공하여 추정기(분류기라고도 함)를 더 좁히는 데 도움이 됩니다:

![ML Map from Scikit-learn](../../../../translated_images/ko/map.e963a6a51349425a.webp)
> 팁: [이 지도를 온라인으로 방문](https://scikit-learn.org/stable/tutorial/machine_learning_map/)하여 경로를 클릭하며 문서를 읽어보세요.

### 계획

이 지도는 데이터를 명확히 이해했을 때 매우 유용하며, 경로를 따라 '걸어가며' 결정을 내릴 수 있습니다:

- 샘플 수가 >50개임
- 범주를 예측하고자 함
- 레이블이 지정된 데이터가 있음
- 샘플 수가 10만 개 미만임
- ✨ 선형 SVC를 선택할 수 있음
- 만약 작동하지 않으면, 숫자 데이터이므로
    - ✨ KNeighbors Classifier를 시도해 볼 수 있음
      - 이것도 작동하지 않으면 ✨ SVC와 ✨ Ensemble Classifiers를 시도

이 경로를 따라가면 매우 도움이 됩니다.

## 연습 - 데이터 분할

이 경로를 따라가려면 먼저 몇 가지 라이브러리를 가져오는 것으로 시작해야 합니다.

1. 필요한 라이브러리를 가져오세요:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. 훈련 데이터와 테스트 데이터를 분할하세요:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## 선형 SVC 분류기

서포트 벡터 클러스터링(SVC)은 서포트 벡터 머신 계열의 ML 기법 중 하나입니다(아래에서 자세히 배움). 이 방법에서는 라벨을 어떻게 클러스터링할지 결정하는 '커널'을 선택할 수 있습니다. 'C' 매개변수는 '정규화'를 나타내며, 이는 매개변수의 영향을 조절합니다. 커널은 [여러 종류](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) 중 하나로 설정할 수 있으며, 여기서는 선형 SVC를 사용하기 위해 'linear'로 설정했습니다. 확률은 기본값이 'false'인데, 여기서는 확률 추정을 수집하기 위해 'true'로 설정했습니다. 무작위 상태는 확률을 얻기 위해 데이터를 섞기 위해 '0'으로 설정했습니다.

### 연습 - 선형 SVC 적용

클래스 분류기 배열을 먼저 만드세요. 테스트하면서 점차 이 배열에 분류기를 추가할 것입니다.

1. 선형 SVC로 시작하세요:

    ```python
    C = 10
    # 다양한 분류기를 만듭니다.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. 선형 SVC를 사용해 모델을 훈련하고 보고서를 출력하세요:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    결과가 꽤 좋습니다:

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## K-이웃 분류기

K-이웃은 감독 학습 및 비감독 학습 모두에 사용할 수 있는 "이웃" 계열의 ML 기법입니다. 이 방법에서는 미리 정의된 수의 점이 생성되고 데이터가 이 점들 주변에 모여 일반화된 레이블을 예측할 수 있습니다.

### 연습 - K-이웃 분류기 적용

앞의 분류기는 좋았고 데이터와 잘 작동했지만, 아마도 더 나은 정확도를 얻을 수 있을 것입니다. K-이웃 분류기를 시도해보세요.

1. 분류기 배열에 한 줄을 추가하세요(선형 SVC 항목 뒤에 쉼표를 추가):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    결과가 약간 나쁩니다:

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ [K-이웃](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)에 대해 배우기

## 서포트 벡터 분류기

서포트 벡터 분류기는 분류 및 회귀 작업에 사용되는 [서포트 벡터 머신](https://wikipedia.org/wiki/Support-vector_machine) 계열 ML 기법입니다. SVM은 "훈련 예제를 공간의 점에 매핑"하여 두 범주 사이 거리를 최대화합니다. 이후 데이터가 이 공간에 매핑되어 그 범주를 예측합니다.

### 연습 - 서포트 벡터 분류기 적용

서포트 벡터 분류기로 좀 더 나은 정확도를 시도해봅시다.

1. K-이웃 항목 뒤에 쉼표를 추가하고 다음 줄을 추가하세요:

    ```python
    'SVC': SVC(),
    ```

    결과가 꽤 좋습니다!

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ [서포트 벡터](https://scikit-learn.org/stable/modules/svm.html#svm)에 대해 배우기

## 앙상블 분류기

앞의 테스트가 꽤 좋았지만, 최종 경로를 따라가 보겠습니다. '앙상블 분류기', 특히 랜덤 포레스트와 AdaBoost를 시도합시다:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

결과가 매우 좋습니다. 특히 랜덤 포레스트는 더욱 그렇습니다:

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ [앙상블 분류기](https://scikit-learn.org/stable/modules/ensemble.html)에 대해 배우기

이 머신러닝 기법은 "여러 기본 추정기의 예측을 결합"하여 모델 품질을 향상시킵니다. 우리 예제에서는 랜덤 트리와 AdaBoost를 사용했습니다.

- [랜덤 포레스트](https://scikit-learn.org/stable/modules/ensemble.html#forest)는 과적합 방지를 위해 무작위성을 주입한 '결정 트리' 모음(숲)을 구축하는 평균화 방법입니다. n_estimators 매개변수는 트리 수를 나타냅니다.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)는 분류기를 데이터셋에 맞추고, 그 분류기의 복사본을 같은 데이터셋에 맞춥니다. 잘못 분류된 항목의 가중치에 집중하여 다음 분류기의 적합을 조정합니다.

---

## 🚀도전 과제

각 기법에는 조정할 수 있는 많은 매개변수가 있습니다. 각 기법의 기본 매개변수를 조사하고 이러한 매개변수를 조정하는 것이 모델 품질에 어떤 의미인지 생각해 보세요.

## [사후 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

## 복습 및 자기 공부

이 수업들에는 많은 전문 용어가 있으니, [이 목록](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott)을 잠시 검토해 보세요!

## 과제 

[매개변수 연습](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 노력하고 있으나, 자동 번역은 오류나 부정확성이 포함될 수 있음을 유의하시기 바랍니다. 원본 문서가 권위 있는 출처로 간주되어야 합니다. 중요한 정보의 경우에는 전문적인 인간 번역을 권장합니다. 본 번역의 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->