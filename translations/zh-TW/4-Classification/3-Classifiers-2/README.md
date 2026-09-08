# Cuisine classifiers 2

在這第二堂分類課程中，您將探索更多分類數值資料的方法。您也將了解選擇不同分類器可能帶來的影響。

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

### 先備條件

我們假設您已完成前面的課程，並在您的 `data` 資料夾中擁有一個清理過的資料集，檔名為 _cleaned_cuisines.csv_，位於此 4 課程資料夾的根目錄。

### 準備

我們已經將您的 _notebook.ipynb_ 檔案載入清理過的資料集，並將其分割成 X 和 y 兩個資料框，準備進行模型建立流程。

## 分類地圖

先前，您已學習過使用 Microsoft 的作弊表來分類資料的各種選項。Scikit-learn 提供了類似但更細緻的作弊表，可協助您進一步縮小估計器（分類器）的選擇範圍：

![ML Map from Scikit-learn](../../../../translated_images/zh-TW/map.e963a6a51349425a.webp)
> 提示: [線上檢視此地圖](https://scikit-learn.org/stable/tutorial/machine_learning_map/) 並點擊路徑以閱讀文件。

### 計畫

此地圖在您對資料有明確理解後非常有幫助，您可以「沿著路徑走」以做決策：

- 我們有超過 50 筆樣本
- 我們想預測一個類別
- 我們有標記資料
- 我們小於 10 萬筆樣本
- ✨ 可以選擇 Linear SVC
- 若此方法無效，因為我們有數值資料
    - 可以嘗試 ✨ KNeighbors Classifier
      - 若此方法無效，嘗試 ✨ SVC 和 ✨ Ensemble Classifiers

這是一條非常有幫助的路徑。

## 練習 - 分割資料

依照這路徑，我們應先匯入要用的函式庫。

1. 匯入所需函式庫：

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. 分割您的訓練與測試資料：

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC 分類器

支持向量分群 (SVC) 是支持向量機器系列機器學習技術的子集合（以下可了解更多）。此方法中，您可以選擇一個「核函數」來決定如何分群標籤。'C' 參數指的是「正則化」，控制參數的影響力。核函數可為[多種選項](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)之一；這裡我們設定為 'linear' 以使用線性 SVC。預設 probability 為 'false'，這裡我們設定為 'true' 以收集概率估計。我們設定 random state 為 '0' 以洗牌資料取得概率。

### 練習 - 使用線性 SVC

先建立一個分類器陣列。在測試的過程中您將漸進增加該陣列。

1. 先從 Linear SVC 開始：

    ```python
    C = 10
    # 建立不同的分類器。
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. 使用 Linear SVC 訓練模型並輸出報告：

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    結果相當不錯：

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

## K 最近鄰分類器

K-Neighbors 是「鄰居」系列機器學習方法的子集合，可用於監督式與非監督式學習。此方法預先建立了一定數量的點，並將資料聚集於這些點附近，以便為資料預測一般化的標籤。

### 練習 - 使用 K 最近鄰分類器

前一個分類器效果很好並且適合資料，但也許我們能得到更佳的準確率。試試 K-Neighbors 分類器。

1. 在您的分類器陣列中新增一行（在 Linear SVC 項目後加逗號）：

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    結果稍微差一點：

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

    ✅ 了解 [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## 支持向量分類器

支持向量分類器隸屬於[支持向量機器](https://wikipedia.org/wiki/Support-vector_machine)系列機器學習方法，適用於分類和回歸任務。SVM「將訓練範例映射到空間中的點」，以最大化兩個類別之距離。隨後的資料被映射到此空間中，以便預測它們的類別。

### 練習 - 使用支持向量分類器

讓我們嘗試使用支持向量分類器尋求更好的準確率。

1. 在 K-Neighbors 項目後加逗號，然後新增這行：

    ```python
    'SVC': SVC(),
    ```

    結果相當不錯！

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

    ✅ 了解 [支持向量](https://scikit-learn.org/stable/modules/svm.html#svm)

## 集成分類器

讓我們走到最終路徑，雖然先前測試已經很好。我們來嘗試「集成分類器」，特別是隨機森林和 AdaBoost：

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

結果非常好，尤其是隨機森林：

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

✅ 了解 [集成分類器](https://scikit-learn.org/stable/modules/ensemble.html)

這種機器學習方法「結合多個基礎估計器的預測」以提升模型品質。在本例中，我們使用隨機樹和 AdaBoost。

- [隨機森林](https://scikit-learn.org/stable/modules/ensemble.html#forest)，為平均方法，構建「決策樹森林」，並注入隨機性以避免過擬合。n_estimators 參數設定為樹的數量。

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) 對資料集擬合一個分類器，然後擬合同樣分類器的多個副本。它會聚焦於錯誤分類項目的權重，並調整下一個分類器的擬合以做修正。

---

## 🚀挑戰

每種技術都有大量可以調整的參數。研究各自的預設參數，並思考調整這些參數會對模型品質有何影響。

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 複習與自學

這些課程中有很多行話，花點時間複習[此列表](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott)的實用術語！

## 作業 

[參數實作](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：  
本文件係使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們致力於精確翻譯，但請注意自動翻譯可能包含錯誤或不準確之處。原始語言版本之文件應視為權威來源。對於關鍵資訊，建議使用專業人工翻譯。本公司不對因使用本翻譯文件所導致之任何誤解或誤釋負責。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->