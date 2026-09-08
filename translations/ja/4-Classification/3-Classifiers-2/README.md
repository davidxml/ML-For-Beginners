# Cuisine classifiers 2

この第2の分類レッスンでは、数値データを分類するさらなる方法を探ります。また、異なる分類器の選択がもたらす影響についても学びます。

## [事前講義クイズ](https://ff-quizzes.netlify.app/en/ml/)

### 前提条件

前回のレッスンを完了し、この4レッスンフォルダーのルートにある`data`フォルダーにクリーンなデータセット _cleaned_cuisines.csv_ があることを前提とします。

### 準備

クリーンデータセットを読み込んだ_notebook.ipynb_ ファイルを用意しており、モデル構築プロセスのためにXとyのデータフレームに分割しています。

## 分類マップ

前回は、Microsoftのチートシートを使った分類のさまざまな選択肢について学びました。Scikit-learnにも類似でより詳細なチートシートがあり、それによって推定器（分類器の別称）をさらに絞り込むことができます。

![ML Map from Scikit-learn](../../../../translated_images/ja/map.e963a6a51349425a.webp)
> ヒント: [このマップをオンラインでご覧ください](https://scikit-learn.org/stable/tutorial/machine_learning_map/) そして、パスをクリックしてドキュメントを読んでみましょう。

### 計画

データに明確な理解があれば、このマップはとても役立ちます。パスを「歩く」ことで決断に導きます：

- サンプル数は50以上
- カテゴリを予測したい
- ラベル付きデータがある
- 10万サンプル未満
- ✨ Linear SVCを選択できる
- それでうまくいかなければ、数値データなので
    - ✨ KNeighbors Classifierを試すことができる
      - それでもうまくいかなければ、✨ SVC と ✨ Ensemble Classifiers を試す

これは非常に役立つ道筋です。

## 演習 - データの分割

このパスに従い、最初にいくつかのライブラリをインポートしましょう。

1. 必要なライブラリをインポートします：

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. トレーニングデータとテストデータを分割します：

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC分類器

サポートベクタークラスタリング（SVC）は、サポートベクターマシンのファミリーに属するML手法の一つです（以下で詳しく説明します）。この方法では、「カーネル」を選択してラベルのクラスタリング方法を決めます。パラメータ 'C' は「正則化」を意味し、パラメータの影響力を制御します。カーネルは[複数](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)の中から選べますが、ここでは線形SVCを使うために'linear'に設定しています。確率はデフォルトで 'false' ですが、ここでは確率推定を取得するために 'true' に設定しています。データをシャッフルして確率を得るために乱数状態を'0'に設定しています。

### 演習 - 線形SVCを適用する

まず、分類器の配列を作成します。テストしながら順次分類器をこの配列に追加します。

1. Linear SVCで始めましょう：

    ```python
    C = 10
    # 異なる分類器を作成します。
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Linear SVCを使ってモデルを訓練し、レポートを表示します：

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    結果はかなり良好です：

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

## K-Nearest Neighbors分類器

K-Nearest Neighborsは「隣接点」ファミリーのML手法で、教師あり・教師なし学習の両方に使われます。この方法は、予め定めた数の点を作成し、その周辺にデータを集めてデータの一般化したラベルを予測します。

### 演習 - K-Nearest Neighbors分類器を適用する

前の分類器は良好に機能しましたが、精度向上を狙いましょう。K-Nearest Neighbors分類器を試します。

1. 分類器配列に行を追加します（Linear SVCの要素の後ろにカンマを追加）：

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    結果は少し悪くなります：

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

    ✅ [K-Neighborsについて学ぶ](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## サポートベクター分類器

サポートベクター分類器は、分類や回帰タスクに用いられる[サポートベクターマシン](https://wikipedia.org/wiki/Support-vector_machine)ファミリーの一部です。SVMは「訓練サンプルを空間上の点にマップし」、2つのカテゴリ間の距離を最大化します。その空間に新たなデータをマッピングしてカテゴリを予測します。

### 演習 - サポートベクター分類器を適用する

サポートベクター分類器で少し精度向上を目指しましょう。

1. K-Neighborsの項目の後ろにカンマを追加し、次の行を加えます：

    ```python
    'SVC': SVC(),
    ```

    結果はかなり良いです！

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

    ✅ [サポートベクターについて学ぶ](https://scikit-learn.org/stable/modules/svm.html#svm)

## アンサンブル分類器

前回の結果は良好でしたが、一番最後のパスに従って「アンサンブル分類器」、特にRandom ForestとAdaBoostを試しましょう：

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

結果は非常に良好で、特にRandom Forestが優れています：

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

✅ [アンサンブル分類器について学ぶ](https://scikit-learn.org/stable/modules/ensemble.html)

この機械学習の方法は「複数の基本推定器の予測を組み合わせる」ことでモデルの品質を向上させます。例ではRandom TreesとAdaBoostを使いました。

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest) は平均化手法で、「決定木」の「森林」を作り、過学習を避けるためにランダム性を注入します。n_estimatorsは木の数に設定します。

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) はデータセットに対して分類器を訓練し、その分類器のコピーを同じデータセットに適用します。誤分類された項目の重みを重視し、次の分類器の適合を調整してエラーを修正します。

---

## 🚀チャレンジ

これらの手法は多くのパラメータを持っており、調整が可能です。それぞれのデフォルトパラメータを調べ、パラメータの調整がモデルの品質にどのような意味を持つか考えてみましょう。

## [講義後クイズ](https://ff-quizzes.netlify.app/en/ml/)

## 復習＆自主学習

このレッスンには多くの専門用語が登場しますので、[こちらの用語集](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott)をざっと確認してみてください！

## 課題

[パラメータ遊び](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責事項**：  
本書類はAI翻訳サービス「[Co-op Translator](https://github.com/Azure/co-op-translator)」を使用して翻訳されました。正確性を期しておりますが、自動翻訳には誤りや不正確な部分が含まれる場合があります。原文の母国語による文書が正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用により生じたいかなる誤解や誤訳に関しても、当方は責任を負いかねます。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->
