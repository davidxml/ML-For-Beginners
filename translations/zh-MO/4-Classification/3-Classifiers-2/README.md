# Cuisine classifiers 2

在第二堂分類課中，您將探索更多對數值數據進行分類的方法。您亦會了解選擇不同分類器的影響。

## [課前小測驗](https://ff-quizzes.netlify.app/en/ml/)

### 先決條件

假設您已完成之前的課程，並在本4堂課資料夾根目錄的 `data` 資料夾中有一個名為 _cleaned_cuisines.csv_ 的已清理數據集。

### 準備工作

我們已載入您的 _notebook.ipynb_ 檔案，並將清理後的數據集劃分為 X 和 y 資料框，準備進行模型建構。

## 一張分類地圖

之前，您學習了使用 Microsoft 的速查表對數據進行分類的各種選項。Scikit-learn 提供了類似但更細緻的速查表，能進一步協助您篩選估算器（另一種稱呼分類器）：

![ML Map from Scikit-learn](../../../../translated_images/zh-MO/map.e963a6a51349425a.webp)
> 提示：[線上瀏覽此地圖](https://scikit-learn.org/stable/tutorial/machine_learning_map/) 並沿著路徑點擊以閱讀文件。

### 計劃

一旦您對數據有清晰的理解，這張地圖非常有用，因為您可沿著路徑作出決定：

- 我們有超過50個樣本
- 我們想預測一個類別
- 我們有標籤數據
- 我們的樣本少於10萬
- ✨ 我們可以選擇 Linear SVC
- 如果那不起作用，因為我們有數值數據
    - 我們可以嘗試 ✨ KNeighbors 分類器
      - 如果那也不行，試試 ✨ SVC 和 ✨ 集成分類器

這是一條非常實用的路徑。

## 練習 - 分割數據

沿著這條路徑開始，我們應該先導入一些所需的庫。

1. 導入必要的庫：

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. 分割訓練及測試數據：

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## 線性 SVC 分類器

支持向量分類（SVC）是支持向量機器（SVM）機器學習技術家族的一員（稍後會進一步了解）。此方法允許選擇「核函數」來決定如何聚類標籤。「C」參數指的是「正則化」，用來調節參數的影響力。核函數可從[多個選項中選擇](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)；這裡我們設為「linear」以使用線性 SVC。預設的 probability 為 false，此處設為 true 以收集概率估計。我們將 random_state 設為 0 以便對數據進行隨機打亂獲得概率。

### 練習 - 應用線性 SVC

開始創建一個分類器陣列。隨著測試進行，逐步將更多分類器加入此陣列。

1. 從 Linear SVC 開始：

    ```python
    C = 10
    # 建立不同的分類器。
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. 使用 Linear SVC 來訓練模型並打印報告：

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

K-Neighbors 是「鄰居」機器學習方法的成員，可用於監督式及非監督學習。此方法先創建指定數量的點，然後將數據聚集於這些點周圍，從而可以對數據做出泛化標籤的預測。

### 練習 - 應用 K-Neighbors 分類器

之前的分類器表現良好且與數據配合度高，但或許我們能獲得更好的準確率。試試 K-Neighbors 分類器。

1. 在分類器陣列中加一行（在 Linear SVC 項目後加逗號）：

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    結果稍差一些：

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

    ✅ 學習更多關於 [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## 支持向量分類器

支持向量分類器是[支持向量機](https://wikipedia.org/wiki/Support-vector_machine)機器學習方法家族中的一員，用於分類和回歸任務。SVM「將訓練樣本映射到空間中的點」，使兩類別間距最大化。後續資料會被映射到這個空間以預測其類別。

### 練習 - 應用支持向量分類器

讓我們試試利用支持向量分類器取得稍好的準確率。

1. 在 K-Neighbors 項目後添加逗號，然後加入此行：

    ```python
    'SVC': SVC(),
    ```

    結果相當優異！

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

    ✅ 學習更多關於 [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## 集成分類器

讓我們沿路徑走到最後，儘管前面的測試已經相當好。我們試試「集成分類器」，具體為隨機森林和 AdaBoost：

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

✅ 學習更多關於 [集成分類器](https://scikit-learn.org/stable/modules/ensemble.html)

這種機器學習方法「結合多個基學估算器的預測」，以提升模型品質。在本例中，我們使用隨機樹和 AdaBoost。

- [隨機森林](https://scikit-learn.org/stable/modules/ensemble.html#forest) 是一種平均方法，建立一個包含隨機元素的「決策樹森林」以避免過擬合。n_estimators 參數設定為樹的數量。

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) 先擬合一個分類器到數據集，然後對同一數據集多次擬合該分類器複製本，著重於錯誤分類項的權重，調整下一個分類器的擬合以修正錯誤。

---

## 🚀挑戰

每種技術都有大量參數可以調整。研究各自的預設參數，並思考調整這些參數對模型品質會有什麼影響。

## [課後小測驗](https://ff-quizzes.netlify.app/en/ml/)

## 複習與自學

這些課程用到許多術語，花點時間複習[這個詞彙清單](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott)吧！

## 作業

[參數遊戲](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：  
本文件使用人工智能翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們努力確保準確性，但請注意，自動翻譯可能包含錯誤或不準確之處。原始語言版本的文件應視為權威來源。對於重要資訊，建議採用專業人工翻譯。我們不對因使用本翻譯而引起的任何誤解或誤譯承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->