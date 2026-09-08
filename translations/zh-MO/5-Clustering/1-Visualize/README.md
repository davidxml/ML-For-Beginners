# 叢集分析簡介

叢集分析是一種假設資料集沒有標籤或其輸入未與預定義輸出相匹配的[無監督學習](https://wikipedia.org/wiki/Unsupervised_learning)。它使用各種演算法來篩選未標記的資料，並根據資料中辨識出的模式提供分組。

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 點擊上方圖片觀看影片。在您學習使用叢集進行機器學習時，享受一些奈及利亞舞廳曲目 — 這是 PSquare 於 2014 年的高評價歌曲。

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

### 簡介

[叢集分析](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) 對於資料探索非常有用。讓我們看看它是否能幫助發掘奈及利亞觀眾聽音樂的趨勢和模式。

✅ 花一分鐘思考叢集分析的用途。在現實生活中，叢集分析就像當你有一堆待整理的衣物，得為家人的衣服分類🧦👕👖🩲。在資料科學中，叢集分析出現在嘗試分析使用者喜好或決定任何未標籤資料集的特徵時。某種程度上，叢集分析幫助整理混亂，像整理襪子抽屜一樣。

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 點擊上方圖片觀看影片：MIT 的 John Guttag 介紹叢集分析

在專業環境中，叢集可用於決定像市場細分，例如判斷不同年齡層購買哪些商品。另一種用途是異常檢測，可能用於從信用卡交易資料中偵測詐騙。您也可以使用叢集分析判定醫學掃描中的腫瘤。

✅ 想想您可能在哪些銀行、電子商務或商業情境中遇過叢集分析。

> 🎓 有趣的是，叢集分析起源於 1930 年代的人類學和心理學領域。您能想像它當時是如何被使用的嗎？

另外，您也可以用來分群搜尋結果 — 例如依購物連結、圖片或評論分組。當您擁有龐大資料集，想要降低維度並對其進行更細緻分析時，叢集分析十分有用，因此該技術可以在建構其他模型前，用於了解資料。

✅ 將資料組織成叢集後，您會賦予其叢集 ID。這技術在保護資料集隱私時很有用；您可以用叢集 ID 來指代資料點，而不是用更具揭露性的可識別資料。您能想到其他為何會用叢集 ID 來識別叢集，而非其他元素的理由嗎？

透過此 [學習模組](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott) 深入瞭解叢集技術。

## 叢集分析入門

[Scikit-learn 提供多種](https://scikit-learn.org/stable/modules/clustering.html)進行叢集分析的方法。您選擇的類型將取決於您的使用案例。根據文件，每種方法都有不同的好處。以下是一個簡化的 Scikit-learn 支援方法及其適用案例的表格：

| 方法名稱                      | 使用場景                                                               |
| :--------------------------- | :--------------------------------------------------------------------- |
| K-Means                      | 一般用途，歸納式                                                       |
| 親和力傳播 (Affinity propagation) | 多個、不均等叢集，歸納式                                               |
| 均值漂移 (Mean-shift)         | 多個、不均等叢集，歸納式                                               |
| 光譜叢集 (Spectral clustering) | 少量、均等叢集，傳導式                                                 |
| Ward 階層式叢集              | 多個、受限叢集，傳導式                                                 |
| 凝聚叢集 (Agglomerative clustering) | 多個、受限、非歐幾里得距離，傳導式                                     |
| DBSCAN                       | 非平面幾何、不均整叢集，傳導式                                         |
| OPTICS                       | 非平面幾何、多密度不均整叢集，傳導式                                   |
| 高斯混合 (Gaussian mixtures)  | 平面幾何，歸納式                                                       |
| BIRCH                        | 大型資料集含異常值，歸納式                                             |

> 🎓 我們如何創建叢集與如何將資料點組成群組息息相關。讓我們解釋一些詞彙：
>
> 🎓 [‘傳導式’ vs. ‘歸納式’](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> 傳導推理是從觀察到的訓練案例直接映射到特定測試案例。歸納推理則是從訓練案例推導一般規則，然後將規則應用於測試案例。
> 
> 例如：假設您有一個資料集只有部分標註。某些是“黑膠唱片”，某些是“CD”，有些沒標。您的工作是給沒標的資料打標。採用歸納法，您會訓練出一個模型來尋找“黑膠唱片”和“CD”，並將這些標籤套用到未標記資料上。這種方法會難以辨認“卡帶”。傳導法則較有效，因為它透過將相似物品群聚再將標籤賦予群集。在這個案例中，叢集可能會反映“圓形樂器”和“方形樂器”等群組。
> 
> 🎓 [‘非平面’ vs. ‘平面’幾何](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> 取自數學術語，非平面與平面幾何指的是透過“平面”（[歐幾里得](https://wikipedia.org/wiki/Euclidean_geometry)）或“非平面”（非歐幾里得）幾何方法測量點間距離。
>
> 這裡的“平面”指的是歐幾里得幾何（部分一樣是“平面”幾何），而非平面指非歐幾里得幾何。幾何與機器學習相關的原因是，它們都根植於數學，必須有共通方式來測量叢集中點間距離，而這可用平面或非平面方式決定，視資料性質而定。[歐幾里得距離](https://wikipedia.org/wiki/Euclidean_distance) 是測量兩點間線段長度。[非歐幾里得距離](https://wikipedia.org/wiki/Non-Euclidean_geometry) 是沿曲線測量。若您的視覺化資料似乎不在平面上，您可能需要用專門演算法處理它。
>
![Flat vs Nonflat Geometry Infographic](../../../../translated_images/zh-MO/flat-nonflat.d1c8c6e2a96110c1.webp)
> 圖解由 [Dasani Madipalli](https://twitter.com/dasani_decoded) 提供
> 
> 🎓 [‘距離’](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> 叢集定義依其距離矩陣，例如點之間的距離。這距離可有多種測量方式。歐幾里得叢集以點值的平均數來定義，包含“質心”或中心點。距離即為每點到質心的距離。非歐幾里得距離則指“叢集中心點”（clustroid），即與其他點距離最接近的點，而clustroid可用不同方式定義。
> 
> 🎓 [‘受限’](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [受限叢集](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) 將“半監督”學習引入此無監督方法。點與點之間關係被標記為“不可連結”或“必須連結”，對資料集施加規則。
>
> 例如：若演算法被放開在無標或半標資料上，產生的叢集質量可能不佳。以上例，叢集可能會分組“圓形音樂物品”、“方形音樂物品”、“三角物品”及“餅乾”。若給予限制或規則（如“物品必須為塑膠材質”、“物品需能產生音樂”），有助於‘受限’演算法作出更好選擇。
> 
> 🎓 密度
> 
> “嘈雜”資料被視為“密集”。叢集中各點間距離受密度影響可能較為稠密或擁擠，故需用適當叢集方法分析。[這篇文章](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) 比較使用 K-Means 與 HDBSCAN 演算法探討有不均密度的嘈雜資料集的差異。

## 叢集演算法

叢集演算法有超過 100 種，其使用依賴於資料特性。我們來介紹幾種主要的：

- <strong>階層式叢集</strong>。若物件以與鄰近物件的距離分類，而非較遠物件，則根據成員與其他物件的距離形成叢集。Scikit-learn 的凝聚叢集即為階層式。

   ![Hierarchical clustering Infographic](../../../../translated_images/zh-MO/hierarchical.bf59403aa43c8c47.webp)
   > 圖解由 [Dasani Madipalli](https://twitter.com/dasani_decoded) 提供

- <strong>質心叢集</strong>。此流行演算法需選擇“k”，即要形成的叢集數，之後演算法決定叢集中心點並將資料聚集於此。 [K-means 叢集](https://wikipedia.org/wiki/K-means_clustering) 是質心叢集的流行版本。中心由最近的平均數決定，因此名稱。使叢集平方距離最小化。

   ![Centroid clustering Infographic](../../../../translated_images/zh-MO/centroid.097fde836cf6c918.webp)
   > 圖解由 [Dasani Madipalli](https://twitter.com/dasani_decoded) 提供

- <strong>分布式叢集</strong>。基於統計模型，此叢集法側重於決定資料點屬於叢集的機率，並相應指派。高斯混合方法屬於此類。

- <strong>密度式叢集</strong>。資料點依其密度或彼此分佈群組指派至叢集。遠離群組的資料點視為離群值或雜訊。DBSCAN、均值漂移和 OPTICS 屬此類。

- <strong>格狀叢集</strong>。針對多維資料集創建格狀，資料依格格劃分進而形成叢集。

## 練習 - 叢集您的資料

叢集分析技術在視覺化時特別有幫助，現在開始視覺化我們的音樂資料。這練習將協助我們決定針對此資料最有效的叢集方法。

1. 開啟此資料夾內的 [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) 檔案。

1. 匯入 `Seaborn` 套件以便良好資料視覺化。

    ```python
    !pip install seaborn
    ```

1. 附加 [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv) 中的歌曲資料。載入一些關於歌曲的資料框（dataframe）。準備透過匯入圖書館並列印資料來探索它：

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    查看前幾行資料：

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. 獲取 dataframe 的一些資訊，呼叫 `info()`：

    ```python
    df.info()
    ```

   輸出看起來像這樣：

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

1. 透過呼叫 `isnull()` 並驗證總和為 0 來再次檢查是否有缺失值：

    ```python
    df.isnull().sum()
    ```

    情況良好：

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

1. 描述資料：

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

> 🤔 如果我們正在使用叢集分析，一種無監督且不需要標籤資料的方法，為什麼要顯示帶標籤的資料？在資料探索階段，它們很有用，但叢集演算法本身並不需要這些標籤。你也可以直接移除欄位名稱，並以欄位編號來引用資料。

觀察資料的一般數值。注意 popularity 可以是 '0'，代表歌曲沒有排名。稍後讓我們先移除這些。

1. 使用條形圖找出最受歡迎的音樂類型：

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/zh-MO/popular.9c48d84b3386705f.webp)

✅ 如果你想看更多熱門值，請將上方的 `[:5]` 改成更大的數字，或者移除它來查看全部。

注意：當熱門類型顯示為 'Missing'，代表 Spotify 沒有給它分類，因此我們將剔除這些。

1. 移除缺失資料，透過篩選掉

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    現在重新檢查類型：

    ![most popular](../../../../translated_images/zh-MO/all-genres.1d56ef06cefbfcd6.webp)

1. 截至目前，前三名音樂類型主導這份資料集。我們將專注於 `afro dancehall`、`afropop` 和 `nigerian pop`，並進一步篩選資料，移除 popularity 為 0 的資料（意即在資料集中未被分類為有排名的歌曲，可視為噪音）：

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. 快速檢測資料是否有特別強的相關性：

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/zh-MO/correlation.a9356bb798f5eea5.webp)

    唯一明顯相關的是 `energy` 和 `loudness`，這並不令人意外，因為音量大的音樂通常充滿活力。其餘的相關性則較弱。看看叢集演算法如何處理這些資料會很有趣。

    > 🎓 請注意，相關性不代表因果關係！我們有相關性的證據，但沒有因果的證明。這個[有趣的網站](https://tylervigen.com/spurious-correlations)有許多視覺範例強調這點。

在這份資料中，歌曲的受歡迎度與舞蹈感是否呈現任何收斂趨勢？FacetGrid 展示了無論類型如何，皆有同心圓排列的情況。難道尼日利亞音樂品味在舞蹈感上達到某個共識層次？

✅ 嘗試不同的資料點（energy、loudness、speechiness）及更多或不同的音樂類型。你能發現什麼？請查看 `df.describe()` 表格了解資料點的一般分布。

### 練習 - 資料分布

這三種音樂類型在其受歡迎度的舞蹈感認知上是否有顯著差異？

1. 檢視前三大類型在受歡迎度及舞蹈感的資料分布，分別以 x 與 y 軸表示。

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    你會發現圍繞一個一般共識點的同心圓，顯示點的分布。

    > 🎓 請注意，此範例以 KDE（核密度估計）圖呈現資料，使用連續機率密度曲線表示。這允許我們處理多重分布的資料。

    總體來說，這三個類型在受歡迎度及舞蹈感上的分布是鬆散對齊的。在這種鬆散對齊的資料中辨識叢集將會是挑戰：

    ![distribution](../../../../translated_images/zh-MO/distribution.9be11df42356ca95.webp)

1. 建立散點圖：

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    同軸的散點圖展現相似的收斂圖樣

    ![Facetgrid](../../../../translated_images/zh-MO/facetgrid.9b2e65ce707eba1f.webp)

一般而言，叢集分析可以使用散點圖來顯示資料群，所以掌握此類視覺化非常實用。下一節課，我們將使用這份篩選後的資料，透過 k-means 叢集演算法找出看來有趣重疊群組。

---

## 🚀挑戰

為準備下一節課，畫一張圖表說明你可能會發現並用於生產環境中的各種叢集演算法。叢集嘗試解決的是什麼樣的問題？

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 複習與自學

在你應用叢集演算法之前，正如我們學到的，了解資料集的性質是個好主意。請參考更多相關內容[這裡](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[這篇有用的文章](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/)帶你了解不同叢集演算法在不同資料形狀下的表現差異。

## 作業

[研究其他叢集視覺化方法](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們力求準確，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要資訊，建議尋求專業人工翻譯。我們不對因使用本翻譯而引起的任何誤解或曲解承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->