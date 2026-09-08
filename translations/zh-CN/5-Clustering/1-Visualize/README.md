# 聚类介绍

聚类是一种[无监督学习](https://wikipedia.org/wiki/Unsupervised_learning)，假设数据集是无标签的，或者其输入未与预定义输出匹配。它使用各种算法对无标签数据进行排序，并根据数据中察觉到的模式提供分组。

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 点击上方图片观看视频。在学习机器学习中的聚类时，欣赏一些尼日利亚舞厅音乐 —— 这是PSquare 2014年广受好评的一首歌曲。

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

### 介绍

[聚类](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124)对于数据探索非常有用。让我们看看它是否能帮助发现尼日利亚观众消费音乐的趋势和模式。

✅ 花一分钟思考聚类的用途。在现实生活中，当你有一堆洗好的衣服需要分拣家人衣物时，就发生了聚类 🧦👕👖🩲。在数据科学中，当试图分析用户偏好，或确定任何无标签数据集的特征时，就会用到聚类。从某种意义上说，聚类帮助理清混乱，就像整理袜子抽屉一样。

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 点击上方图片观看视频：MIT的John Guttag介绍聚类

在专业环境中，聚类可以用于确定市场细分，例如确定不同年龄段购买的商品。另一个应用是异常检测，比如从信用卡交易数据集中检测欺诈。或者你可以用聚类来确定一批医学扫描中的肿瘤。

✅ 花一分钟思考你在银行、电商或商业环境中如何遇到过聚类技术。

> 🎓 有趣的是，聚类分析最早源于20世纪30年代的人类学和心理学领域。你能想象它当时是如何被使用的吗？

或者，你可以用聚类来对搜索结果进行分组，比如购物链接、图片或评论。当你拥有一个大型数据集并希望减少维度、进行更细致的分析时，聚类非常有用。这种技术可以用来在构建其他模型之前了解数据。

✅ 一旦你的数据被组织成簇（clusters），你就会分配簇ID，这项技术在保护数据隐私时很有用；你可以用簇ID而不是更具揭示性的可识别数据来引用数据点。你能想到其他使用簇ID而非簇中其他元素来标识的原因吗？

通过这个[学习模块](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)加深对聚类技术的理解。

## 聚类入门

[Scikit-learn提供](https://scikit-learn.org/stable/modules/clustering.html)多种执行聚类的方法。选择哪种取决于你的使用场景。根据文档，每种方法各有优点。这里是Scikit-learn支持的方法及其合适用例的简化表格：

| 方法名称                      | 适用场景                                                               |
| :--------------------------- | :--------------------------------------------------------------------- |
| K-Means                      | 通用，归纳式                                                         |
| 亲和传播（Affinity propagation） | 多个、不均匀簇，归纳式                                                 |
| Mean-shift                   | 多个、不均匀簇，归纳式                                                 |
| 谱聚类（Spectral clustering）    | 少数、均匀簇，传导式                                                   |
| Ward层次聚类                  | 多个、受约束簇，传导式                                                 |
| 凝聚式聚类（Agglomerative clustering） | 多个、受约束、非欧几里得距离，传导式                                   |
| DBSCAN                       | 非平坦几何、不均匀簇，传导式                                           |
| OPTICS                       | 非平坦几何、不均匀且密度可变簇，传导式                                 |
| 高斯混合（Gaussian mixtures）      | 平坦几何，归纳式                                                       |
| BIRCH                        | 大型数据集带异常值，归纳式                                             |

> 🎓 聚类的创建方式与如何将数据点归集到组密切相关。让我们解析一些术语：
>
> 🎓 ['传导式' vs. '归纳式'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> 传导推断源于观察训练案例直接映射到特定测试案例。归纳推断源于训练案例对应的通用规则，再应用于测试案例。
> 
> 举例：假设你的数据集只有部分标签，有的标记为“唱片”，有的标为“CD”，有的为空白。你的任务是给空白项赋标签。若用归纳方法，训练模型识别“唱片”和“CD”，并把这些标签应用于无标签数据。此方法难以识别“磁带”等新类别。传导方法则通过聚类将相似项分组，再将标签应用于组，比如“圆形音乐物品”和“方形音乐物品”。
> 
> 🎓 ['非平坦' vs. '平坦'几何](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> 源自数学术语，非平坦和平坦几何指用“平坦”（[欧氏](https://wikipedia.org/wiki/Euclidean_geometry)）或“非平坦”（非欧氏）几何方法测量点间距离。
>
> 此处“平坦”指欧几里得几何（部分内容称为“平面”几何），非平坦则指非欧氏几何。几何如何与机器学习相关？作为数学基础的两个领域，必须有统一方式测量簇中点距，根据数据特性选择“平坦”或“非平坦”测量方式。[欧氏距离](https://wikipedia.org/wiki/Euclidean_distance)是点间线段长度，[非欧氏距离](https://wikipedia.org/wiki/Non-Euclidean_geometry)是沿曲线测量距离。若数据可视化为非平面空间，需用专门算法处理。
>
![平坦与非平坦几何信息图](../../../../translated_images/zh-CN/flat-nonflat.d1c8c6e2a96110c1.webp)
> 信息图由[Dasani Madipalli](https://twitter.com/dasani_decoded)提供
> 
> 🎓 ['距离'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> 簇由其距离矩阵定义，如点间距离。距离可通过多种方式测量。欧氏簇基于点值平均，含有“质心”或中心点，距离即为点到质心的距离。非欧氏距离指“簇核”（clustroid），是距离其他点最近的点。簇核定义多样。
> 
> 🎓 ['受约束'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [受约束聚类](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf)将“半监督”学习引入无监督方法，点间关系标记为“不能链接”或“必须链接”，从而对数据集施加规则。
>
> 举例：若算法自由处理无标签或半标签数据，生成的簇质量可能较差。例如上例中簇可能分为“圆形音乐物品”、“方形音乐物品”、“三角形物品”和“饼干”。如果添加约束如“物品必须由塑料制成”，“物品必须能够发出音乐”，可帮助算法做出更优选择。
> 
> 🎓 密度
> 
> 噪声数据被视为“密集”的。其内部点间距离审查可显示簇的稠密度或“拥挤”程度，因此需用适当聚类方法分析。[这篇文章](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html)演示了用K-Means与HDBSCAN算法探索具有不均簇密度的噪声数据的差异。

## 聚类算法

聚类算法有100多种，选用取决于数据特性。以下是一些主要算法：

- <strong>层次聚类</strong>。若对象通过与近邻对象的接近性分类，而非远处对象，簇则基于其成员间距离形成。Scikit-learn的凝聚式聚类即为层次聚类。

   ![层次聚类信息图](../../../../translated_images/zh-CN/hierarchical.bf59403aa43c8c47.webp)
   > 信息图由[Dasani Madipalli](https://twitter.com/dasani_decoded)提供

- <strong>质心聚类</strong>。此流行算法需指定“k”（簇数），算法随后确定每簇中心点，收集周边数据。[K-means聚类](https://wikipedia.org/wiki/K-means_clustering)是质心聚类的常见形式。中心点通过最邻近均值确定，故名。簇的平方距离被最小化。

   ![质心聚类信息图](../../../../translated_images/zh-CN/centroid.097fde836cf6c918.webp)
   > 信息图由[Dasani Madipalli](https://twitter.com/dasani_decoded)提供

- <strong>基于分布的聚类</strong>。基于统计建模，通过确定数据点属于某簇的概率并进行分配。高斯混合方法属于此类。

- <strong>基于密度的聚类</strong>。通过密度或点间集结区分簇。远离簇的点被视为异常或噪声。DBSCAN、Mean-shift和OPTICS属于此类聚类。

- <strong>基于网格的聚类</strong>。对多维数据集构建网格，将数据划分到网格单元中，从而形成簇。

## 练习 - 对你的数据进行聚类

聚类技术大受益于合适的可视化，让我们先可视化音乐数据开始。本练习帮助我们决定哪种聚类方法最适合此数据特质。

1. 打开本文件夹中的[_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb)文件。

1. 导入`Seaborn`包以实现良好的数据可视化。

    ```python
    !pip install seaborn
    ```

1. 追加来自[_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv)的歌曲数据。加载包含歌曲信息的数据框。准备好导入库并导出数据进行探索：

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    查看前几行数据：

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. 获取关于数据框的一些信息，调用 `info()`：

    ```python
    df.info()
    ```

   输出如下所示：

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

1. 通过调用 `isnull()` 并验证和为0，进行空值的双重检查：

    ```python
    df.isnull().sum()
    ```

    一切正常：

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

1. 描述数据：

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

> 🤔 如果我们正在使用聚类，作为一种不需要标签数据的无监督方法，为什么我们要用带标签的数据来展示呢？在数据探索阶段，标签很有用，但对于聚类算法的运行并非必需。你完全可以移除列标题，仅用列号来引用数据。

观察数据的一般值。注意，popularity 可能是‘0’，这说明歌曲没有排名。我们稍后会去除这些数据。

1. 使用条形图查找最受欢迎的流派：

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/zh-CN/popular.9c48d84b3386705f.webp)

✅ 如果你想查看更多的顶级值，可以将顶部的 `[:5]` 改为更大的值，或者移除它来查看全部。

注意，当顶级流派被描述为‘Missing’时，表示 Spotify 没有对其进行分类，我们应该去除这些数据。

1. 通过过滤去除缺失数据

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    现在重新检查流派：

    ![most popular](../../../../translated_images/zh-CN/all-genres.1d56ef06cefbfcd6.webp)

1. 到目前为止，前三大流派主导着这个数据集。我们重点关注 `afro dancehall`，`afropop` 和 `nigerian pop`，另外过滤掉流行度为0的记录（表示该数据未被分类为有流行度，可以视为噪声）：

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. 快速测试数据是否存在特别强的相关性：

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/zh-CN/correlation.a9356bb798f5eea5.webp)

    唯一显著的强相关是 `energy` 和 `loudness`，这并不令人惊讶，因为响亮的音乐通常很有能量。其他相关性相对较弱。观察聚类算法如何挖掘这些数据将会很有趣。

    > 🎓 请注意，相关性不代表因果关系！我们有相关性的证明，但没有因果关系的证明。这个[有趣的网站](https://tylervigen.com/spurious-correlations)中有一些图示强调了这一观点。

该数据集中是否存在在歌曲感知流行度和舞蹈性的某种趋同？一个 FacetGrid 显示，无论流派如何，都有同心圆的排列。难道尼日利亚人的口味对于这个流派的某个舞蹈性水平存在趋同？

✅ 尝试不同的数据点（energy，loudness，speechiness）和更多或不同的音乐流派。你能发现什么？看一下 `df.describe()` 表了解数据点的整体分布。

### 练习 - 数据分布

这三个流派在舞蹈性的感知上是否因流行度不同而显著不同？

1. 检查我们前三大流派的流行度和舞蹈性的分布，分别用给定的x轴和y轴表示。

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    你可以发现围绕一个整体汇聚点存在同心圆，显示出点的分布。

    > 🎓 此示例使用 KDE（核密度估计）图表，使用连续概率密度曲线表示数据。这使我们能够解释多个分布的数据。

    总体来说，三个流派在流行度和舞蹈性方面的大致趋势是相似的。要确定这些松散对齐的数据的聚类将是一项挑战：

    ![distribution](../../../../translated_images/zh-CN/distribution.9be11df42356ca95.webp)

1. 绘制散点图：

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    使用相同坐标轴的散点图显示了相似的趋同模式

    ![Facetgrid](../../../../translated_images/zh-CN/facetgrid.9b2e65ce707eba1f.webp)

总的来说，对于聚类，你可以使用散点图来展示数据群集，掌握这种可视化方法非常有用。下一节课，我们将使用过滤后的数据并用 k-means 聚类方法发现数据中有趣重叠的群组。

---

## 🚀挑战

为下一节课做准备，制作一个关于各种聚类算法的图表，这些算法你可能会在生产环境中发现并使用。聚类试图解决什么样的问题？

## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 复习与自学

在应用聚类算法之前，正如我们所学，了解数据集的性质是个好主意。可在[这里](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)阅读更多相关内容。

[这篇有用的文章](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/)带你了解在不同数据形状下各种聚类算法的不同表现方式。

## 作业

[研究其他聚类可视化方法](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免责声明**：
本文件由 AI 翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻译完成。尽管我们力求准确，但请注意，自动翻译可能包含错误或不准确之处。原始语言版文件应视为权威来源。对于重要信息，建议使用专业人工翻译。我们对因使用本翻译而产生的任何误解或误释不承担责任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->