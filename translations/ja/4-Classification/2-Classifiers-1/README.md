# Cuisine classifiers 1

このレッスンでは、前回のレッスンで保存した、さまざまな国の料理に関するバランスの取れたクリーンなデータで構成されたデータセットを使用します。

このデータセットを使って、様々な分類器を用いて_特定の材料のグループに基づいて国の料理を予測_します。その過程で、アルゴリズムが分類タスクにどのように活用できるかについてさらに学びます。

## [事前講義クイズ](https://ff-quizzes.netlify.app/en/ml/)
# 準備

[Lesson 1](../1-Introduction/README.md)を完了していることを前提に、この4つのレッスンのためにルートの`/data`フォルダーに_cleaned_cuisines.csv_ファイルが存在することを確認してください。

## 演習 - 国の料理を予測する

1. このレッスンの_notebook.ipynb_フォルダーで、Pandasライブラリと共にそのファイルをインポートします：

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    データは以下のようになっています：

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. 次に、いくつかのライブラリをインポートします：

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. トレーニング用にXとyの座標を2つのデータフレームに分割します。`cuisine`はラベルのデータフレームにできます：

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    これで以下のようになります：

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. `Unnamed: 0`列と`cuisine`列を`drop()`で削除し、残りのデータをトレーニング用の特徴量として保存します：

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    特徴量は以下のようになります：

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

さあ、モデルをトレーニングする準備ができました！

## 分類器の選択

データがクリーンになり、トレーニングの準備が整ったので、この仕事にどのアルゴリズムを使うか決める必要があります。

Scikit-learnは分類を監視学習（Supervised Learning）の下にまとめており、そのカテゴリには多くの分類方法があります。[バラエティ](https://scikit-learn.org/stable/supervised_learning.html)は最初はかなり圧倒されるかもしれません。次の方法はすべて分類技術を含みます：

- 線形モデル
- サポートベクターマシン
- 確率的勾配降下法
- 最近傍法
- ガウス過程
- 決定木
- アンサンブル法（投票分類器）
- 多クラス・多出力アルゴリズム（多クラス・多ラベル分類、多クラス多出力分類）

> [ニューラルネットワークを使ってデータを分類する](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification)こともできますが、それはこのレッスンの範囲外です。

### どの分類器を選ぶ？

どの分類器を選ぶべきでしょうか？多くの場合、いくつか試してみて良い結果を探すのがテスト方法の一つです。Scikit-learnは[横並びの比較](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)を作成済みデータセット上で提供しており、KNeighbors、SVC２種類、GaussianProcessClassifier、DecisionTreeClassifier、RandomForestClassifier、MLPClassifier、AdaBoostClassifier、GaussianNB、QuadraticDiscrinationAnalysisを比較し、その結果を視覚化しています：

![comparison of classifiers](../../../../translated_images/ja/comparison.edfab56193a85e7f.webp)
> Scikit-learnのドキュメントで生成されたプロット

> AutoMLはこれらの比較をクラウド上で実行し、データに最適なアルゴリズムを選択できるようにして、この問題をうまく解決します。[こちらで試せます](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### より良いアプローチ

しかし、ランダムに推測するより良い方法として、このダウンロード可能な[ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott)にあるアイデアに従うことが挙げられます。ここでは、多クラス問題に対していくつかの選択肢が示されています：

![cheatsheet for multiclass problems](../../../../translated_images/ja/cheatsheet.07a475ea444d2223.webp)
> MicrosoftのAlgorithm Cheat Sheetの一部、多クラス分類のオプションを詳細に示す

✅ このチートシートをダウンロードして印刷し、壁に貼りましょう！

### 理由付け

制約条件を踏まえて異なるアプローチを検討してみましょう：

- <strong>ニューラルネットワークは重すぎる</strong>。クリーンだが最小限のデータセットであり、ローカルのノートブックでトレーニングを実行するため、ニューラルネットワークはこのタスクには重すぎます。
- <strong>二クラス分類器は使わない</strong>。二クラス分類器は使わないため、one-vs-allは除外されます。
- <strong>決定木やロジスティック回帰は候補</strong>。決定木や多クラスデータのロジスティック回帰は有効かもしれません。
- <strong>多クラスブースト決定木は別の問題向け</strong>。多クラスブースト決定木は非パラメトリックなタスク（ランキング構築など）向けであり、今回の用途には適しません。

### Scikit-learnの使用

Scikit-learnを使ってデータを分析しますが、Scikit-learnでロジスティック回帰を使う方法は複数あります。[パラメーター](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)を見てみましょう。

基本的に重要なパラメーターは`multi_class`と`solver`の2つで、ロジスティック回帰の実行時に指定する必要があります。`multi_class`の値は挙動を決め、`solver`は使用するアルゴリズムを決定します。すべての`multi_class`値とすべての`solver`は組み合わせられるわけではありません。

ドキュメントによると、多クラスの場合、

- `multi_class`が`ovr`に設定されている場合は<strong>one-vs-rest (OvR) スキーム</strong>を使います
- `multi_class`が`multinomial`に設定されている場合は<strong>クロスエントロピー損失</strong>を使います。（現在、`multinomial`は‘lbfgs’, ‘sag’, ‘saga’, ‘newton-cg’ソルバーのみサポートされています）"

> 🎓 ここでの'scheme'は'ovr'（one-vs-rest）か'multinomial'のいずれかです。ロジスティック回帰は本質的に二クラス分類向けのため、これらのスキームは多クラス分類タスクに対応しやすくします。[出典](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver'は「最適化問題で使用するアルゴリズム」と定義されています。[出典](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)

Scikit-learnは以下の表でソルバーが異なるデータ構造の課題にどう対応するか説明しています：

![solvers](../../../../translated_images/ja/solvers.5fc648618529e627.webp)

## 演習 - データの分割

最近のレッスンでロジスティック回帰を学んだことに基づき、最初のトレーニング試行にはロジスティック回帰に注目します。`train_test_split()`を使ってデータをトレーニング用とテスト用に分割しましょう：

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## 演習 - ロジスティック回帰を適用する

多クラスの場合を使うため、どの_スキーム_を使い、どの_ソルバー_を設定するか選択する必要があります。ロジスティック回帰を使い、`multi_class`設定を指定し、<strong>liblinear</strong>ソルバーを使ってトレーニングします。

1. `multi_class`を`ovr`に、ソルバーを`liblinear`に設定してロジスティック回帰を作成します：

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ `lbfgs`のような別のソルバーも試してみてください。これはよくデフォルトになっています。

    > 必要に応じて、データを平坦化するためにPandasの[`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html)関数を使ってください。

    精度は80%以上と良好です！

1. モデルを実際に動かして、50行目のデータでテストしてみましょう：

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    結果が表示されます：

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ 別の行番号でも試して結果を確認してください。
1. さらに深掘りして、この予測の正確さを確認できます：

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    結果が表示されます - インド料理が最も可能性が高い推測です：

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ なぜモデルがこれをかなり確信してインド料理と判断しているのか説明できますか？

1. 回帰のレッスンで行ったように、分類レポートを表示して詳細を取得します：

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

## 🚀チャレンジ

このレッスンでは、クリーンアップしたデータを使って、一連の材料に基づき国の料理を予測できる機械学習モデルを構築しました。Scikit-learnが提供する多くのデータ分類オプションをゆっくり読んでみてください。舞台裏で何が行われているのかを理解するために「solver」の概念をさらに掘り下げましょう。

## [講義後クイズ](https://ff-quizzes.netlify.app/en/ml/)

## レビュー & 自主学習

[このレッスン](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)でロジスティック回帰の数学的背景をさらに掘り下げましょう。
## 課題

[ソルバーの勉強](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責事項**:  
本書類は AI 翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確さを期しておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があることをご了承ください。原文の母国語版が正式な情報源とみなされるべきです。重要な情報については専門の人間による翻訳をお勧めします。本翻訳の使用により生じるいかなる誤解や解釈違いについても、一切の責任を負いかねます。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->