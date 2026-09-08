# 叢集介紹

叢集是一種[無監督學習](https://wikipedia.org/wiki/Unsupervised_learning)，假設資料集是未標記的，或其輸入並未對應到預定義的輸出。它使用各種演算法來分類未標記的資料，並根據其在資料中辨識的模式提供分組。

[![PSquare 的 No One Like You](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "PSquare 的 No One Like You")

> 🎥 點擊上方圖片觀看影片。當你在學習帶有叢集的機器學習時，欣賞一些奈及利亞舞廳曲目——這是 PSquare 於2014年推出的高評價歌曲。

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

### 介紹

[叢集](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124)對資料探索非常有用。讓我們看看它能否幫助發現奈及利亞聽眾消費音樂的趨勢與模式。

✅ 花一分鐘思考叢集的用途。在現實生活中，當你有一堆洗好的衣服，需要分類家人衣物時就會用到叢集🧦👕👖🩲。在資料科學中，叢集用於分析使用者喜好或確定任何未標記資料集的特性。某種程度上，叢集有助於理解混亂，就像整理襪子抽屜。

[![ML 介紹](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "叢集介紹")

> 🎥 點擊上方圖片觀看影片：MIT 的 John Guttag 介紹叢集

在專業環境中，叢集可用於如市場區隔，判斷哪些年齡層購買哪些商品。例如，也可用於異常偵測，可能用於從信用卡交易資料集偵測詐欺。你也可能使用叢集來判斷醫療影像中的腫瘤。

✅ 花一分鐘思考你是否在銀行、電子商務或商業環境中遇過「叢集」的應用。

> 🎓 有趣的是，叢集分析起源於1930年代的人類學和心理學領域。你能想像它當時可能怎麼被使用嗎？

或者，你也可以用於將搜尋結果分組，例如以購物連結、圖片或評價分類。當你有大型資料集想要降維並進行更細緻分析時，叢集很有用，因此此技術可用於在構建其他模型之前先學習資料。

✅ 一旦資料被組成叢集，你會分配叢集ID，這技術在保護資料隱私時很實用；你可以用叢集ID來代替更揭露個人資料的識別資料。你還能想出其他為什麼要用叢集ID而非叢集中其他元素來識別的原因嗎？

在這個 [Learn 模組](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)中，深入了解叢集技術。

## 入門叢集

[Scikit-learn 提供了多種](https://scikit-learn.org/stable/modules/clustering.html)叢集方法。選擇的類型視使用案例而定。根據文件說明，每種方法有不同優勢。以下為 Scikit-learn 支援的方法及適用案例簡化表：

| 方法名稱                     | 使用案例                                                               |
| :-------------------------- | :--------------------------------------------------------------------- |
| K-Means                     | 通用，歸納式（inductive）                                              |
| 親和傳播(Affinity propagation) | 眾多、不均等叢集，歸納式                                               |
| 平均漂移(Mean-shift)           | 眾多、不均等叢集，歸納式                                               |
| 光譜叢集(Spectral clustering)    | 少數、均等叢集，傳遞式（transductive）                                |
| Ward 階層叢集                 | 多數、有限制叢集，傳遞式                                              |
| 凝聚叢集(Agglomerative clustering) | 多數、有限制、非歐氏距離，傳遞式                                      |
| DBSCAN                      | 非平面幾何、不均等叢集，傳遞式                                        |
| OPTICS                      | 非平面幾何、不均等且密度可變叢集，傳遞式                              |
| 高斯混合(Gaussian mixtures)     | 平面幾何，歸納式                                                     |
| BIRCH                       | 大型資料集有離群值，歸納式                                            |

> 🎓 我們如何創建叢集很大程度上取決於如何將資料點歸成群組。讓我們拆解一些詞彙：
>
> 🎓 [『傳遞式(transductive)與歸納式(inductive)』](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> 傳遞推理是由觀察到的訓練案例映射至特定測試案例所得；歸納推理是由訓練案例推導出一般規則，然後才應用於測試案例。
> 
> 例子：假設你有部分標記的資料集。有些是『黑膠唱片』，一些是『CD』，有些是空白。你的任務是為空白標籤提供標籤。如果你採取歸納方法，你會訓練模型識別『黑膠唱片』和『CD』，並將這些標籤套用於未標記資料。此方法對其實是『錄音帶』的分類會很吃力。相反地，傳遞方法更有效處理未知資料，因為它先將相似項目分組，再為群組賦標籤；在此情況下，叢集可能會是『圓形音樂物品』和『方形音樂物品』。
> 
> 🎓 [『非平面(non-flat)與平面(flat)幾何』](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> 源自數學術語，非平面與平面幾何指以『平面』（[歐氏幾何](https://wikipedia.org/wiki/Euclidean_geometry)）或『非平面』（非歐氏）幾何方法測量點與點間距離。
>
>此處『平面』指歐氏幾何（部分稱為『平面』幾何），非平面則指非歐氏幾何。幾何和機器學習有何關係？兩者都根源於數學，必須用共同方法量測叢集內點距離，可依資料性質採用平面或非平面度量方式。 [歐氏距離](https://wikipedia.org/wiki/Euclidean_distance)是指兩點連線長度。 [非歐氏距離](https://wikipedia.org/wiki/Non-Euclidean_geometry)則沿曲線測量。若你視覺化資料發現它似乎不在平面上，可能需要使用特殊演算法處理。
>
![平面與非平面幾何資訊圖](../../../../translated_images/zh-TW/flat-nonflat.d1c8c6e2a96110c1.webp)
> 資訊圖作者：[Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 [『距離』](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> 叢集由距離矩陣定義，例如點與點之間的距離。此距離可用幾種方式測量。歐氏叢集由點值平均定義，含有『質心』或中心點，所以距離是以該中心點為依據。非歐氏距離則是指『clustroid』，即最接近其他點的點，而clustroid的定義又可有多種變化。
> 
> 🎓 [『有限制(Constrained)』](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [有限制叢集](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf)在此無監督方法中引入『半監督學習』。點間關係會被標記為『不可連結』或『必須連結』，因此對資料集加上有限制規則。
>
>舉例：若演算法自由運作於一批未標記或半標記資料，生成的叢集品質可能不佳。上例中，叢集可能僅分為『圓形音樂物品』、『方形音樂物品』、『三角形物品』及『餅乾』。若給予一些限制（「必須為塑膠製成」、「必須能發出音樂」），可協助『限制』演算法做出更佳選擇。
> 
> 🎓 『密度』
> 
> 資料若『雜訊多』即被視為『密集』。叢集中點與點之距離，經過檢視後可能發現或多或少密集或擁擠，因此需要使用適合的叢集方法分析此類資料。 [此文](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html)展示使用 K-Means 叢集法與 HDBSCAN 演算法探索帶有不均密叢集的雜訊資料集的差異。

## 叢集演算法

叢集演算法超過百種，使用取決於資料性質。讓我們說明一些主要演算法：

- <strong>階層式叢集</strong>。若一物件的分類依據是其與附近物件的相似度，而非更遠物件，叢集即根據其成員間距離形成。Scikit-learn 的凝聚叢集即為階層式。

   ![階層式叢集資訊圖](../../../../translated_images/zh-TW/hierarchical.bf59403aa43c8c47.webp)
   > 資訊圖作者：[Dasani Madipalli](https://twitter.com/dasani_decoded)

- <strong>質心叢集</strong>。此流行演算法需選擇「k」即叢集數量，接著演算法判斷叢集中心點，將資料聚集於該點周圍。 [K-means叢集](https://wikipedia.org/wiki/K-means_clustering)為質心叢集的常見方法。中心由最近均值決定，因此得名。叢集的平方距離被最小化。

   ![質心叢集資訊圖](../../../../translated_images/zh-TW/centroid.097fde836cf6c918.webp)
   > 資訊圖作者：[Dasani Madipalli](https://twitter.com/dasani_decoded)

- <strong>基於分布的叢集</strong>。基於統計建模，分布式叢集著重判定資料點屬於叢集的概率，並依此指派。高斯混合方法即屬此類。

- <strong>基於密度的叢集</strong>。根據資料點間密度或群聚程度指派叢集，遠離群體的點被視為離群或噪聲。DBSCAN、平均漂移和 OPTICS 屬於此類。

- <strong>基於格點的叢集</strong>。對多維資料集建立格子，將資料分配至格點內，進而形成叢集。

## 練習 - 對你的資料進行叢集

叢集技術大大受惠於良好視覺化，讓我們從視覺化音樂資料開始。此練習將助我們決定對此資料性質應最有效使用哪種叢集方法。

1. 開啟此資料夾內的 [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) 檔案。

1. 匯入 `Seaborn` 套件以進行良好資料視覺化。

    ```python
    !pip install seaborn
    ```

1. 附加 [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv) 的歌曲資料。載入包含歌曲資料的 dataframe。準備引入函式庫並匯出資料以探索它：

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

1. 取得一些關於資料框的資訊，呼叫 `info()`：

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

1. 再次檢查是否有空值，呼叫 `isnull()` 並確認總和是 0：

    ```python
    df.isnull().sum()
    ```

    看起來不錯：

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

> 🤔 如果我們使用的是聚類，一種不需要標記資料的無監督方法，為什麼我們要顯示帶標籤的資料？在資料探索階段，它們很有用，但對聚類演算法來說不一定是必要的。你也可以刪除欄位標頭，直接用欄位號碼參考資料。

觀察資料的一般數值。注意人氣（popularity）可以是 '0'，表示歌曲沒有排名。稍後我們將移除這些。

1. 使用長條圖找出最受歡迎的類型：

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![最受歡迎](../../../../translated_images/zh-TW/popular.9c48d84b3386705f.webp)

✅ 如果你想看更多頂尖值，將前五筆的 `[:5]` 改成更大數字，或移除它以查看全部。

注意，當最熱門類型顯示為「Missing」時，表示 Spotify 沒有分類這類型，所以我們將其移除。

1. 透過篩選排除缺失資料

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    現在再檢查這些類型：

    ![最受歡迎](../../../../translated_images/zh-TW/all-genres.1d56ef06cefbfcd6.webp)

1. 前三大類型明顯主導此資料集。我們將專注於 `afro dancehall`、`afropop` 與 `nigerian pop`，並進一步篩選資料，移除人氣值為 0 的資料（表示資料集中未分類的人氣，可以視為噪音）：

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. 快速測試資料是否存在明顯強相關性：

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![相關性](../../../../translated_images/zh-TW/correlation.a9356bb798f5eea5.webp)

    唯一強烈相關的是 `energy` 和 `loudness`，這不意外，因為響亮的音樂通常很有能量。其他相關性較弱。看看聚類演算法對此資料會做出什麼樣的判斷會很有趣。

    > 🎓 注意，相關性不代表因果關係！我們只有相關證明，沒有因果證明。一個 [有趣的網站](https://tylervigen.com/spurious-correlations) 有一些視覺化例子強調這點。

這個資料集中，歌曲的人氣感受與舞蹈感是否有任何收斂現象？FacetGrid 顯示無論類型如何都有一圈圈的同心圓。難道奈及利亞口味對這類型的舞蹈感有某個共同比例？

✅ 嘗試不同的數據點（能量、音量、說話性）和更多或不同音樂類型。你會發現什麼？看看 `df.describe()` 表格了解資料的整體分布。

### 練習 - 數據分布

這三種類型在舞蹈感的認知上是否根據人氣有顯著差異？

1. 觀察我們前三大類型在人氣和舞蹈感的資料分布，以特定 x 軸和 y 軸繪圖。

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    你可以看出圍繞一個總體偏好的同心圓點，顯示資料點分布。

    > 🎓 本範例使用 KDE（核密度估算）圖，通過連續的機率密度曲線來表示資料。這讓我們在處理多種分布時能更好詮釋資料。

    三種類型在人氣和舞蹈感上大致呈鬆散對齊。判斷這鬆散對齊資料中的群聚會是一個挑戰：

    ![分布](../../../../translated_images/zh-TW/distribution.9be11df42356ca95.webp)

1. 繪製散佈圖：

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    相同維度的散點圖顯示相似的收斂模式

    ![Facetgrid](../../../../translated_images/zh-TW/facetgrid.9b2e65ce707eba1f.webp)

一般而言，用於聚類分析時，你可以使用散點圖來顯示資料群聚，因此掌握這類可視化非常重要。下一課我們將使用此過濾後的資料，透過 k-means 聚類找出資料中有趣的重疊群聚。

---

## 🚀挑戰

為下一課做好準備，畫出你可能會發現並在生產環境中使用的各種聚類演算法的圖表。聚類嘗試解決什麼類型的問題？

## [課後小考](https://ff-quizzes.netlify.app/en/ml/)

## 複習與自學

在應用聚類演算法前，我們如同學習到的，理解資料集本質是很重要的。更多資訊請參考 [這裡](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)。

[這篇有幫助的文章](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) 引導你瞭解不同聚類演算法在不同資料形態下的行為。

## 作業

[研究其他聚類視覺化](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
此文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們努力追求準確性，但請注意自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應視為權威來源。對於關鍵資訊，建議採用專業人工翻譯。我們不對因使用此翻譯所產生的任何誤解或誤譯承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->