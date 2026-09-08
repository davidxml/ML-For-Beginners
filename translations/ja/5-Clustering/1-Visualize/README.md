# クラスタリング入門

クラスタリングは、データセットにラベルが付いていないか、その入力と事前定義された出力が対応付けられていないことを想定する [教師なし学習](https://wikipedia.org/wiki/Unsupervised_learning) の一種です。さまざまなアルゴリズムを用いてラベルのないデータを分類し、データ中に認識されたパターンに基づいてグループ分けを行います。

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 上記の画像をクリックすると動画が再生されます。クラスタリングで機械学習を学習しながら、ナイジェリアのダンスホールトラックをお楽しみください。これは PSquare による2014年の高評価曲です。

## [事前講義クイズ](https://ff-quizzes.netlify.app/en/ml/)

### はじめに

[クラスタリング](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124)はデータ探索に非常に有用です。ナイジェリアの聴衆が音楽を消費する傾向やパターンの発見に役立つか見てみましょう。

✅ クラスタリングの用途について1分考えてみましょう。実生活では、洗濯物の山があって家族の服を分ける必要があるときにクラスタリングが起きます 🧦👕👖🩲。データ科学では、ユーザーの好みを分析したり、ラベルのないデータセットの特徴を判別したりするときにクラスタリングが使われます。クラスタリングはある意味で混沌を理解する助けとなり、靴下の引き出しの整理のようなものです。

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 上記の画像をクリックすると動画が再生されます：MITのジョン・ガタッグがクラスタリングを紹介します

専門的な環境では、市場セグメンテーションの決定、例えばどの年齢層がどの製品を購入するかを明らかにするためにクラスタリングを使うことができます。もう一つの用途は異常検知で、クレジットカード取引のデータセットから不正を検出する場合などです。あるいは医療画像の一括スキャンの中で腫瘍を特定するためにクラスタリングを使うこともあります。

✅ 銀行業界、電子商取引、ビジネスの現場でクラスタリングがどのように使われているかについて、1分考えてみましょう。

> 🎓 興味深いことに、クラスタ分析は1930年代に人類学や心理学の分野から始まりました。どのように使われていたか想像できますか？

または、検索結果をショッピングリンク、画像、レビューなどでグループ化するために使うこともできます。クラスタリングは大規模なデータセットを縮約し、より詳細な分析を行いたい場合に便利なため、他のモデルを構築する前にデータについて学ぶための技術として使われます。

✅ データがクラスタに整理されたらクラスタIDを割り当てます。この技術はデータセットのプライバシー保護に役立つことがあります。個別特定しやすいデータではなくクラスタIDで参照できるからです。クラスタIDを使って他のクラスタの要素ではなく識別する理由を考えてみてください。

クラスタリング技術の理解を深めるには、この [Learn モジュール](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott) をご覧ください。

## クラスタリングの始め方

[Scikit-learn は豊富なクラスタリング手法](https://scikit-learn.org/stable/modules/clustering.html)を提供しています。どのタイプを選ぶかはユースケースによります。ドキュメントによると、それぞれの手法に様々な利点があります。Scikit-learnでサポートされている手法と適切な適用例の簡単な表は以下のとおりです。

| メソッド名                  | 使用例                                                             |
| :--------------------------- | :----------------------------------------------------------------- |
| K-Means                      | 一般用途、帰納的                                                   |
| Affinity propagation         | クラスタ数が多く不均等、帰納的                                     |
| Mean-shift                   | クラスタ数が多く不均等、帰納的                                     |
| Spectral clustering          | クラスタ数が少なく均等、演繹的                                     |
| Ward hierarchical clustering | クラスタ数が多く制約あり、演繹的                                   |
| Agglomerative clustering     | クラスタ数が多く制約あり、非ユークリッド距離、演繹的              |
| DBSCAN                       | 非平坦な幾何、クラスタが不均等、演繹的                            |
| OPTICS                       | 非平坦な幾何、密度が不均一なクラスタ、演繹的                      |
| Gaussian mixtures            | 平坦な幾何、帰納的                                                 |
| BIRCH                        | 大規模データセットに外れ値あり、帰納的                             |

> 🎓 クラスタの作成方法は、データポイントをどうまとめるかに大きく関係しています。用語を解説しましょう：
>
> 🎓 ['演繹的'vs. '帰納的'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> 演繹推論は、特定のテストケースに対応する観測された訓練ケースから導出されます。帰納推論は、訓練ケースから一般化された規則を導き、それをテストケースに適用します。
> 
> 例：部分的にラベル付きのデータセットがあります。一部は 'records'、一部は 'cds'、そして一部は空白です。空白のラベルを付けるのが仕事です。帰納的手法なら、'records' と 'cds'のラベルの付いたモデルを訓練し、それを未ラベルデータに適用しますが、実際は 'cassettes' であるものを分類するのが難しいです。一方、演繹的手法は不明のデータをより効果的に扱い、似たアイテムをまとめてグループにラベルを付けます。この場合、例えば「丸い楽器」と「四角い楽器」のようなクラスタが形成されます。
> 
> 🎓 ['非平坦' vs. '平坦' 幾何学](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> 数学用語に由来し、非平坦と平坦の幾何学は、点間の距離を平坦な([ユークリッド](https://wikipedia.org/wiki/Euclidean_geometry))か非平坦な(非ユークリッド)幾何学的手法で測る区別です。
>
> ここでの『平坦』はユークリッド幾何学（「平面」幾何学として教えられる部分もある）を指し、『非平坦』は非ユークリッド幾何学を指します。機械学習で幾何学が関係するのは、数学を基盤とする両分野で、クラスタ内の点間距離を共通の方法で測る必要があるためです。距離は『平坦』または『非平坦』で可能です。 [ユークリッド距離](https://wikipedia.org/wiki/Euclidean_distance)は2点間の直線距離、[非ユークリッド距離](https://wikipedia.org/wiki/Non-Euclidean_geometry)は曲線に沿った距離となります。データが平面上にないように見える場合は、特殊なアルゴリズムを使用する必要があります。
>
![平坦と非平坦の幾何学インフォグラフィック](../../../../translated_images/ja/flat-nonflat.d1c8c6e2a96110c1.webp)
> インフォグラフィック作成： [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['距離'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> クラスタは距離行列、つまり点間距離で定義されます。この距離は幾通りかに測定可能です。ユークリッドクラスタは点の値の平均によって定義され、中心点（セントロイド）を持ちます。距離はこのセントロイドへの距離で測定されます。非ユークリッド距離は近接点に最も近い点（クラストロイド）を用います。クラストロイドは様々な定義があります。
> 
> 🎓 ['制約あり'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [制約付きクラスタリング](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf)は、この教師なし手法に半教師あり学習を導入します。点間の関係は『リンク禁止』や『リンク必須』とフラグ付けされ、データセットに規則が課されます。
>
> 例：アルゴリズムを未ラベルまたは半ラベルデータに自由に適用すると、クラスタ品質は低下します。上の例では「丸い音楽物体」「四角い音楽物体」「三角形の物」「クッキー」が混ざるかもしれません。制約（「アイテムはプラスチック製である」「音楽を再生できる必要がある」など）を与えれば、アルゴリズムの選択が改善します。
> 
> 🎓 『密度』
> 
> 『ノイズの多い』データは『密度が高い』とされます。各クラスタ内の点間距離が検討され、よりまたはそうでない密度、つまり混み具合が測定されます。このため適切なクラスタリング法で解析する必要があります。 [この記事](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) は、K-MeansとHDBSCANを用いたノイズの多い不均一密度クラスタのデータセット解析の違いを示しています。

## クラスタリングアルゴリズム

100を超えるクラスタリングアルゴリズムがあり、利用はデータの性質によって異なります。代表的なものをいくつか紹介しましょう。

- <strong>階層的クラスタリング</strong>。オブジェクトが遠くより近くのオブジェクトへの近接性で分類される場合、メンバーの他オブジェクトとの距離に基づきクラスタが形成されます。Scikit-learn の凝集型クラスタリングは階層的です。

   ![階層的クラスタリング インフォグラフィック](../../../../translated_images/ja/hierarchical.bf59403aa43c8c47.webp)
   > インフォグラフィック作成： [Dasani Madipalli](https://twitter.com/dasani_decoded)

- <strong>セントロイドクラスタリング</strong>。人気のあるアルゴリズムで、まず 'k'、すなわち形成するクラスタ数を選択し、その後アルゴリズムがクラスタの中心点を決定してデータを集めます。[K-means クラスタリング](https://wikipedia.org/wiki/K-means_clustering)は有名なセントロイドクラスタリングの一例です。中心は最も近い平均で決定され、名前の由来となっています。クラスタからの二乗距離が最小化されます。

   ![セントロイドクラスタリング インフォグラフィック](../../../../translated_images/ja/centroid.097fde836cf6c918.webp)
   > インフォグラフィック作成： [Dasani Madipalli](https://twitter.com/dasani_decoded)

- <strong>分布ベースクラスタリング</strong>。統計モデリングに基づき、データ点がクラスタに属する確率を求めて割り当てます。ガウス混合法はこのタイプに含まれます。

- <strong>密度ベースクラスタリング</strong>。データポイントは密度、すなわち互いの周囲での集まりに基づきクラスタに割り当てられます。グループから離れた点は外れ値やノイズとされます。DBSCAN、Mean-shift、OPTICSがこのタイプです。

- <strong>グリッドベースクラスタリング</strong>。多次元データセット用で、グリッドを作成し、データをグリッドのセルに割り当ててクラスタを形成します。

## 演習 - データをクラスタリングしよう

クラスタリングは適切な可視化と組み合わせることで効果が高まるため、まず音楽データの可視化から始めましょう。この演習は、このデータの性質に最適なクラスタリング手法を決めるのに役立ちます。

1. このフォルダの[_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb)ファイルを開いてください。

1. 良好なデータ可視化のために `Seaborn` パッケージをインポートします。

    ```python
    !pip install seaborn
    ```

1. [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv) から曲データを追加します。曲に関するデータフレームを作成して読み込みます。ライブラリをインポートし、データを表示してこのデータの探索に備えましょう。

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    最初の数行を確認します。

    |     | name                     | album                    | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ------------------------ | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle       | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | インディーR&B   | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | ナイジェリアンポップ | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | アフロポップ    | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. `info()` を呼び出してデータフレームの情報を取得します。

    ```python
    df.info()
    ```

   結果は以下のようになります:

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

1. `isnull()` を使って null 値を二重チェックし、合計が0であることを確認します。

    ```python
    df.isnull().sum()
    ```

    問題ありません:

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

1. データを記述します。

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

> 🤔 クラスタリングはラベルを必要としない教師なし学習法ですが、なぜラベル付きでこのデータを示しているのでしょうか？データ探索段階ではラベルが役立ちますが、クラスタリングアルゴリズムの動作にラベルは不要です。列名を削除し、列番号でデータを参照しても問題ありません。

データの一般的な値を見てみましょう。人気度が '0' の場合はランキングに入っていない曲を示します。あとでこれらを除去しましょう。

1. バープロットを使って最も人気のあるジャンルを見つけます。

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/ja/popular.9c48d84b3386705f.webp)

✅ より多くの上位値を見たい場合は、`[:5]` の数値を大きくするか、すべてを見るために外してください。

トップジャンルが「Missing」と表示される場合、それはSpotifyが分類していないことを意味します。これを取り除きましょう。

1. 欠損データを取り除くためにフィルタリングします。

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ジャンルを再確認します:

    ![most popular](../../../../translated_images/ja/all-genres.1d56ef06cefbfcd6.webp)

1. 断然、トップ3のジャンルがこのデータセットの大部分を占めています。`afro dancehall`、`afropop`、`nigerian pop` に注目し、人気度が0のデータ（データセットで人気度が割り当てられていないためノイズと見なせるデータ）を除外しましょう。

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. データが特に強い相関を持つかどうかを簡単にテストします。

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/ja/correlation.a9356bb798f5eea5.webp)

    唯一強い相関は `energy` と `loudness` の間にあり、音量が大きい音楽は通常エネルギッシュであることを考えれば驚くにはあたりません。それ以外の相関は比較的弱いです。このデータにクラスタリングアルゴリズムを適用するとどのような結果になるか興味深いでしょう。

    > 🎓 相関は因果関係を示すものではありません！相関の証拠はありますが因果の証拠はありません。 [面白いウェブサイト](https://tylervigen.com/spurious-correlations) にはその点を強調するビジュアルがあります。

このデータセットで楽曲の人気度とダンス適性の間に収束はありますか？FacetGrid ではジャンルに関係なく同心円が並んでいる様子が見られます。ナイジェリアの趣味はこのジャンルのダンス適性で一定のレベルに収束しているのかもしれません。

✅ 異なるデータポイント（energy、loudness、speechiness）や異なる音楽ジャンルも試してみてください。何が発見できるでしょうか？一般的なデータの広がりを見るには `df.describe()` テーブルを参照してください。

### 演習 - データ分布

これら3つのジャンルは、人気度に基づくダンス適性の認識において有意差がありますか？

1. 人気度とダンス適性について、3大ジャンルのデータ分布を x軸と y軸に沿って調べてみます。

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    一般的な収束点のまわりに同心円が見られ、ポイントの分布が示されます。

    > 🎓 この例は KDE（カーネル密度推定）グラフを用いており、連続確率密度曲線でデータを表現しています。複数分布を扱うときに役立ちます。

    全体的に、これら3ジャンルは人気度とダンス適性でゆるやかに一致しています。このゆるい一致データでクラスタを判断するのは難題です。

    ![distribution](../../../../translated_images/ja/distribution.9be11df42356ca95.webp)

1. 散布図を作成します。

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    同じ軸の散布図は似た収束パターンを示します。

    ![Facetgrid](../../../../translated_images/ja/facetgrid.9b2e65ce707eba1f.webp)

一般的に、クラスタリングのために散布図を用いてデータのクラスタを示すことができます。この種の可視化を習得することは非常に有用です。次のレッスンでは、このフィルターされたデータを使い、k-means クラスタリングで興味深い重なりを持つグループを発見します。

---

## 🚀チャレンジ

次のレッスンに向けて、さまざまなクラスタリングアルゴリズムについてのチャートを作成してください。これらのクラスタリングはどのような問題の解決を目指しているでしょうか？

## [講義後クイズ](https://ff-quizzes.netlify.app/en/ml/)

## 復習 & 自主学習

クラスタリングアルゴリズムを適用する前に、データセットの性質を理解することは重要です。詳しくはこちらをご覧ください: [こちら](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[この役立つ記事](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) は、異なるデータ形状に対して各種クラスタリングアルゴリズムがどのように動作するかをわかりやすく解説しています。

## 課題

[クラスタリングの他の可視化方法について調査する](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責事項**：
本書類は AI 翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を期していますが、自動翻訳には誤りや不正確な部分が含まれる可能性があることをご承知おきください。原文の原語版が正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用により生じたいかなる誤解や解釈違いについても、当方は責任を負いかねます。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->