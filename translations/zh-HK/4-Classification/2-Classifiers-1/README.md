# Cuisine classifiers 1

在本課程中，您將使用上一課保存的平衡且乾淨的關於不同料理類型的數據集。

您將使用該數據集搭配各種分類器，_根據一組食材預測指定的國家料理_。在此過程中，您將了解更多算法如何用於分類任務的方法。

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)
# 準備工作

假設您已完成[課程 1](../1-Introduction/README.md)，請確認在這四課的根目錄 `/data` 資料夾中存在一個名為 _cleaned_cuisines.csv_ 的檔案。

## 練習 - 預測一個國家料理

1. 在本課的 _notebook.ipynb_ 中，導入該文件以及 Pandas 函式庫：

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    數據看起來如下：

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. 現在，導入更多函式庫：

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. 將 X 和 y 分別劃分成兩個資料框，用於訓練。`cuisine` 可作為標籤資料框：

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    看起來將會是這樣：

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. 丟棄 `Unnamed: 0` 欄位和 `cuisine` 欄位，呼叫 `drop()`。將剩餘資料作為可訓練特徵：

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    您的特徵看起來像這樣：

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

現在，您已準備好訓練模型！

## 選擇你的分類器

既然資料已乾淨且準備好做訓練，您得決定用哪個算法來執行這個任務。

Scikit-learn 將分類歸類於監督學習，您會在該類別下看到多種分類方法。[種類](https://scikit-learn.org/stable/supervised_learning.html)乍看之下令人眼花繚亂。以下方法都包含分類技術：

- 線性模型
- 支援向量機
- 隨機梯度下降
- 最近鄰居
- 高斯過程
- 決策樹
- 集成方法（投票分類器）
- 多類與多輸出算法（多類與多標籤分類，多類多輸出分類）

> 你也可以使用[神經網絡來對數據分類](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification)，但那超出本課範圍。

### 選擇哪個分類器？

那麼，應該選哪個分類器呢？通常，可以嘗試多個方法，看看哪個結果最好是檢測的好方法。Scikit-learn 提供了一個對比[範例](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)，在一個自己製作的數據集上比較 KNeighbors、SVC 兩種方式、GaussianProcessClassifier、DecisionTreeClassifier、RandomForestClassifier、MLPClassifier、AdaBoostClassifier、GaussianNB 及 QuadraticDiscrinationAnalysis，並以視覺化展示結果：

![comparison of classifiers](../../../../translated_images/zh-HK/comparison.edfab56193a85e7f.webp)
> 由 Scikit-learn 文件生成的圖表

> AutoML 通過在雲端運行這些比較來巧妙解決此問題，讓您可選擇最適合自己數據的算法。試試看[這裡](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### 一個更好的方法

比起盲目猜測，跟隨這個可下載的 [ML 速查表](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) 的想法會更好。在這裏，我們發現針對我們的多類問題，有以下幾種選擇：

![cheatsheet for multiclass problems](../../../../translated_images/zh-HK/cheatsheet.07a475ea444d2223.webp)
> 微軟算法速查表的其中一部分，詳列多類分類的選項

✅ 下載這張速查表，列印出來，並掛在桌旁！

### 推理分析

讓我們看看能否依照我們的限制條件，推斷出不同方法的合理性：

- <strong>神經網絡太重</strong>。基於我們潔淨但有限的資料集，以及我們在筆記本中本地訓練的事實，神經網絡對此任務來說過於龐大。
- <strong>沒有雙類分類器</strong>。我們沒有使用雙類分類器，因此排除 one-vs-all 的方法。
- <strong>決策樹或邏輯迴歸可能奏效</strong>。決策樹可能有效，或者使用適合多類資料的邏輯迴歸。
- <strong>多類提升決策樹解決不同問題</strong>。多類提升決策樹適合非參數任務，例如用來建立排序的任務，因此對我們無用。

### 使用 Scikit-learn

我們將使用 Scikit-learn 來分析資料。但 Scikit-learn 中有許多使用邏輯迴歸的方法。看看[必須傳入的參數](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)。

主要有兩個重要參數 - `multi_class` 和 `solver` - 需要指定，當您要求 Scikit-learn 執行邏輯迴歸時。`multi_class` 設定套用特定行為。`solver` 定義使用的算法。並非所有 solver 都能搭配所有 `multi_class` 參數。

根據文件，對多類情況而言，訓練算法：

- **使用 one-vs-rest (OvR) 機制**，當 `multi_class` 選項設定為 `ovr`
- <strong>使用交叉熵損失</strong>，當 `multi_class` 選項設定為 `multinomial`。（目前 `multinomial` 僅支援 ‘lbfgs’, ‘sag’, ‘saga’ 和 ‘newton-cg’ solver。)"

> 🎓 此處的 'scheme' 可設為 'ovr'（一對多）或 'multinomial'（多項式）。因邏輯迴歸本質設計為二分類，這些機制幫助其更好地處理多類分類任務。 [來源](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' 定義為「優化問題中所用的算法」。[來源](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)

Scikit-learn 以此表格解釋不同 solver 如何處理不同數據結構的挑戰：

![solvers](../../../../translated_images/zh-HK/solvers.5fc648618529e627.webp)

## 練習 - 分割數據

我們可聚焦於邏輯迴歸，作為首次訓練嘗試，因為您之前課程中剛學習過它。
呼叫 `train_test_split()` 將資料分割為訓練組和測試組：

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## 練習 - 應用邏輯迴歸

因為您使用多類情況，您需要選擇使用的 _機制_ 與設定的 _solver_。用 multi_class 設為 `ovr` 且 solver 設為 **liblinear** 的 LogisticRegression 做訓練。

1. 建立一個 multi_class 設為 `ovr` 且 solver 是 `liblinear` 的邏輯迴歸：

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ 試試其他 solver，例如經常作為預設的 `lbfgs`

    > 注意，如有需要，請使用 Pandas 的 [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) 函式來壓平資料。

    準確率相當好，超過 **80%**！

1. 透過測試一筆資料（第 50 行）來看看此模型在運作時的表現：

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    列印輸出結果：

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ 更換不同的行號來檢查結果
1. 挖得更深，你可以檢查這個預測的準確性：

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    結果列印出來 - 印度料理是其最佳猜測，且機率相當高：

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ 你能解釋為什麼模型相當確定這是印度料理嗎？

1. 如同迴歸課程一樣，透過列印分類報告取得更多細節：

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

## 🚀挑戰

在本課程中，你使用清理過的資料建構了一個機器學習模型，可以根據一系列的食材預測國家料理。花點時間閱讀 Scikit-learn 提供的多種資料分類選項。深入了解「solver」的概念，以理解幕後的運作原理。

## [課後小測驗](https://ff-quizzes.netlify.app/en/ml/)

## 複習與自學

深入探究邏輯斯迴歸背後的數學原理，請參閱[此課程](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## 作業 

[研究 solver](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：  
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們致力於準確性，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應視為權威來源。對於重要資訊，建議採用專業人工翻譯。我們不承擔因使用此翻譯而引起的任何誤解或曲解之責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->