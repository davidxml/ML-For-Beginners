# Cuisine classifiers 2

在第二課分類課程中，你將探索更多分類數值資料的方法。你也會了解選擇不同分類器的影響。

## [課前小測驗](https://ff-quizzes.netlify.app/en/ml/)

### 先備知識

我們假設你已完成先前課程並已在此 4 課資料夾根目錄的 `data` 資料夾中，準備好一份名為 _cleaned_cuisines.csv_ 的清理後資料集。

### 準備工作

我們已在你的 _notebook.ipynb_ 檔案中載入清理後的資料集，並將其分割成 X 和 y 的資料框，準備進行模型建立流程。

## 分類地圖

之前，你已透過 Microsoft 的流程圖學習如何分類資料。Scikit-learn 提供了一個類似但更加細緻的流程圖，能協助你縮小估計器（另一種稱呼為分類器）的選擇範圍：

![ML Map from Scikit-learn](../../../../translated_images/zh-HK/map.e963a6a51349425a.webp)
> 提示: [線上造訪此地圖](https://scikit-learn.org/stable/tutorial/machine_learning_map/) 並沿路徑點擊以閱讀文件。

### 計畫

當你清楚掌握資料時，這張地圖非常有用，你可以沿著路徑“走”到決定點：

- 我們有超過 50 筆樣本
- 我們想要預測一個類別
- 我們有標記資料
- 樣本數少於 10 萬筆
- ✨ 我們可以選擇 Linear SVC
- 若不行，因為我們有數值資料
    - 我們可以嘗試 ✨ KNeighbors 分類器 
      - 若還不行，再試 ✨ SVC 和 ✨ 集成分類器

這是一條非常實用的路徑可循。

## 練習 - 分割資料

沿著這條路徑，我們應先匯入一些庫。

1. 匯入所需的庫：

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. 分割你的訓練與測試資料：

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## 線性 SVC 分類器

支持向量聚類 (SVC) 是支持向量機家族的子集（下方可查看更多關於 SVM 介紹）。此方法中，您可選擇「核函數」決定標籤如何聚類。「C」參數指的是「正則化」，用來調節參數的影響力。核函數可從多種選擇中設定（詳見[多種核函數](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)）；這裡我們設為 'linear' 確保使用線性 SVC。機率預設為 'false'，這裡設為 'true' 以獲得機率估計。random_state 設為 '0' 用來打亂資料以取得機率。

### 練習 - 使用線性 SVC

首先建立一個分類器陣列，隨著測試會逐漸加入分類器。

1. 先從 Linear SVC 開始：

    ```python
    C = 10
    # 建立不同的分類器。
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. 使用 Linear SVC 訓練模型，並輸出報告：

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

## K-Neighbors 分類器

K-Neighbors 屬於「鄰居」系列的機器學習方法，可用於監督式與非監督式學習。此方法會建立預定數量的點，再根據這些點聚集資料，從而對資料做出一般化標籤預測。

### 練習 - 套用 K-Neighbors 分類器

前一個分類結果不錯並且與資料相符，但也許還能得到更佳準確度。試試 K-Neighbors 分類器。

1. 在分類器陣列加入一行（在 Linear SVC 之後加逗號）：

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

    ✅ 了解更多 [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## 支持向量分類器

支持向量分類器屬於[支持向量機](https://wikipedia.org/wiki/Support-vector_machine)家族的機器學習方法，用於分類和回歸任務。SVM 透過「將訓練範例映射到空間點」的方式，最大化兩類別之間的距離。後續資料映射到此空間，以便預測其類別。

### 練習 - 使用支持向量分類器

讓我們試著用支持向量分類器提升準確度。

1. 在 K-Neighbors 項目後加逗號，再加入此行：

    ```python
    'SVC': SVC(),
    ```

    結果相當好！

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

    ✅ 了解更多 [支持向量](https://scikit-learn.org/stable/modules/svm.html#svm)

## 集成分類器

即使前面測試結果相當不錯，讓我們一路沿著路徑嘗試集成分類器，特別是隨機森林和 AdaBoost：

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

✅ 了解更多 [集成分類器](https://scikit-learn.org/stable/modules/ensemble.html)

此機器學習方法「結合多個基礎估計器的預測」，以提升模型品質。範例中，我們使用了隨機森林與 AdaBoost。

- [隨機森林](https://scikit-learn.org/stable/modules/ensemble.html#forest) 是一種平均法，建立多棵帶有隨機性的「決策樹」森林以避免過擬合。n_estimators 參數設定樹的數量。

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) 會先將分類器擬合到資料集，然後複製該分類器並再次擬合同一資料集。它會關注錯誤分類項的權重，並調整下一個分類器以糾正。

---

## 🚀挑戰

這些技術都有大量可調整的參數。請研究它們的預設參數，並思考參數微調會如何影響模型品質。

## [課後小測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學

這些課程中有許多術語，花點時間複習[這份有用的術語列表](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott)！

## 作業

[參數調整遊戲](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：  
本文件乃透過人工智能翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻譯而成。雖然我們致力於確保準確性，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應視為權威來源。對於重要資訊，建議使用專業人工翻譯。我們不對因使用本翻譯而引起的任何誤解或曲解承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->