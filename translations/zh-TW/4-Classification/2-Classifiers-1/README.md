# Cuisine classifiers 1

在本課程中，您將使用上課節保存的完整且平衡的乾淨資料集，這些資料集是關於各種料理的。

您將使用這個資料集搭配各種分類器，_根據一組食材預測特定國家的料理_。在此過程中，您將進一步了解算法如何應用於分類任務的各種方式。

## [課前小測驗](https://ff-quizzes.netlify.app/en/ml/)
# 準備工作

假設您已完成[第1課](../1-Introduction/README.md)，請確保根目錄下的 `/data` 資料夾中有一個 _cleaned_cuisines.csv_ 檔案，這四堂課皆會使用。

## 練習 - 預測國家料理

1. 在本課的 _notebook.ipynb_ 資料夾中，匯入該檔案及 Pandas 函式庫：

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    資料看起來如下：

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. 現在，匯入更多函式庫：

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. 將 X 和 y 坐標分為兩個資料框做訓練，`cuisine` 為標籤資料框：

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    它看起來如下：

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. 丟掉 `Unnamed: 0` 欄和 `cuisine` 欄，呼叫 `drop()`。並將剩下資料保存為可訓練特徵：

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

現在你已準備好訓練你的模型！

## 選擇分類器

現在您的資料已經清理完畢且準備好進行訓練，您必須決定使用哪個算法。

Scikit-learn 把分類歸類於監督式學習(Supervised Learning)，在該類別中會發現許多分類方式。 [種類繁多](https://scikit-learn.org/stable/supervised_learning.html) 首見時會令人眼花撩亂。以下方法均包含分類技術：

- 線性模型
- 支持向量機
- 隨機梯度下降
- 最近鄰
- 高斯過程
- 決策樹
- 集成方法（投票分類器）
- 多類別與多輸出算法（多類別和多標籤分類，多類別-多輸出分類）

> 您也可以使用[神經網路來分類資料](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification)，但這超出本課範圍。

### 選擇何種分類器？

所以，應該選哪種分類器？通常嘗試幾種分類器並尋找好的結果是個測試方法。 Scikit-learn 提供一個[並排比較](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)的範例，該範例在一個生成的資料集中，並列比較 KNeighbors、兩種方式的 SVC、GaussianProcessClassifier、DecisionTreeClassifier、RandomForestClassifier、MLPClassifier、AdaBoostClassifier、GaussianNB 及 QuadraticDiscrinationAnalysis，並以視覺化方式呈現結果：

![comparison of classifiers](../../../../translated_images/zh-TW/comparison.edfab56193a85e7f.webp)
> 圖為 Scikit-learn 文件中產生的圖表

> AutoML 並行在雲端執行這些比較，幫助您為資料選擇最佳算法，解決此問題。請在[此處嘗試](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### 更佳的做法

比起盲目猜測，更好方式是參考可下載的[機器學習速查表](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) 中的建議。這裡我們發現，針對我們的多類別問題，我們有一些選擇：

![cheatsheet for multiclass problems](../../../../translated_images/zh-TW/cheatsheet.07a475ea444d2223.webp)
> 微軟演算法速查表的一段，詳細介紹多類別分類選項

✅ 下載此速查表，列印並掛在您的牆壁上！

### 推理

讓我們看看在現有限制下，如何對各種方法進行推理：

- **神經網路太龐大。** 以我們乾淨但有限的資料集，且訓練在本地端的 notebook 上運行，神經網路對此任務過於沉重。
- **沒有二元分類器。** 我們不用二元分類器，因此排除一對多(one-vs-all)方法。
- **決策樹或邏輯迴歸都可行。** 決策樹可能有效，或針對多類別資料的邏輯迴歸也可。
- **多類別提升決策樹解決不同問題。** 多類別提升決策樹適合非參數任務，例如設計排名的任務，因此不適用於我們。

### 使用 Scikit-learn 

我們將用 Scikit-learn 來分析資料。但 Scikit-learn 中有多種方式使用邏輯迴歸。請參考[可傳入的參數](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) 。

本質上，有兩個重要參數 - `multi_class` 和 `solver` － 需要在要求 Scikit-learn 執行邏輯迴歸時指定。`multi_class` 設定採用的行為方式；`solver` 是指定解法算法。並非每個 solver 都能與所有的 `multi_class` 值搭配使用。

文件中說明，在多類別情形下，訓練演算法：

- **若將 `multi_class` 設為 `ovr`，將採用一對多 (one-vs-rest, OvR) 方案**
- **若將 `multi_class` 設為 `multinomial`，將使用交叉熵損失函數 (cross-entropy loss)**（目前 `multinomial` 選項只支援 ‘lbfgs’, ‘sag’, ‘saga’ 與 ‘newton-cg’ solver）

> 🎓 這裡的「scheme」可以是 ‘ovr’（一對多） 或 ‘multinomial’。由於邏輯迴歸本是設計給二元分類，這些方案可使其更好地處理多類別分類任務。 [來源](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 「solver」定義為「用於優化問題的算法」。 [來源](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)。

Scikit-learn 提供這張表格說明不同 solver 如何處理不同結構的挑戰：

![solvers](../../../../translated_images/zh-TW/solvers.5fc648618529e627.webp)

## 練習 - 分割資料

我們可以專注於邏輯迴歸進行第一次訓練，畢竟您在上節課剛學過它。
呼叫 `train_test_split()` 將資料分為訓練與測試組：

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## 練習 - 使用邏輯迴歸

由於您選用多類別情境，須決定使用什麼 _scheme_ 與 _solver_。請用 `LogisticRegression` 指定多類別設定，並搭配 **liblinear** solver 進行訓練。

1. 建立邏輯迴歸模型，設定 multi_class 為 `ovr`，solver 為 `liblinear`：

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ 嘗試不同 solver，如常設定為預設的 `lbfgs`

    > 注意，需時請用 Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) 函數將資料攤平成一維。

    準確率超過 **80%**！表現良好！

1. 您也可以透過測試一筆資料（第 50 筆）來觀察此模型實際效果：

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    結果會印出：

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ 嘗試不同筆數並檢查結果
1. 深入探討，您可以檢查此預測的準確性：

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    結果已列印 — 印度料理是其最佳猜測，且有很高的機率：

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ 您能解釋為什麼模型相當確定這是印度料理嗎？

1. 藉由列印分類報告來獲得更多細節，就像你在迴歸課程中所做的那樣：

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

在本課程中，您利用清理過的資料建立一個機器學習模型，能根據一系列配料預測國家料理。花些時間閱讀 Scikit-learn 提供的各種分類資料的選項。深入了解「解算器」的概念，以理解背後運作的原理。

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 複習與自學

深入探討邏輯迴歸背後的數學，詳見[本課程](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## 作業

[研究解算器](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：  
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們致力於確保準確性，請注意自動翻譯可能包含錯誤或不準確之處。原文的母語版本應視為權威來源。對於重要資訊，建議尋求專業人工翻譯。我們不對因使用本翻譯而引起的任何誤解或誤譯負責。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->