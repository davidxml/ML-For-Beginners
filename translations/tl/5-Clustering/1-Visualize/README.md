# Panimula sa clustering

Ang clustering ay isang uri ng [Unsupervised Learning](https://wikipedia.org/wiki/Unsupervised_learning) na ipinagpapalagay na ang isang dataset ay walang label o ang mga input nito ay hindi tumutugma sa mga paunang natukoy na output. Gumagamit ito ng iba't ibang mga algorithm upang ayusin ang mga unlabeled na data at magbigay ng mga grupo ayon sa mga pattern na nakikita nito sa data.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 I-click ang larawan sa itaas para sa isang video. Habang nag-aaral ka ng machine learning gamit ang clustering, mag-enjoy sa ilang Nigerian Dance Hall tracks - ito ay isang mataas na na-rating na kanta mula 2014 ng PSquare.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Panimula

[Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) ay napaka-kapaki-pakinabang para sa paggalugad ng data. Tingnan natin kung makakatulong ito upang matuklasan ang mga trend at pattern sa paraan ng pagkonsumo ng musika ng mga Nigerian audience.

✅ Maglaan ng isang minuto upang pag-isipan ang mga gamit ng clustering. Sa tunay na buhay, nangyayari ang clustering tuwing may bunton ka ng labada at kailangan mong ayusin ang mga damit ng iyong mga kapamilya 🧦👕👖🩲. Sa data science, nangyayari ang clustering kapag sinusubukang suriin ang mga hilig ng isang gumagamit, o tukuyin ang mga katangian ng anumang unlabeled na dataset. Sa isang paraan, tumutulong ang clustering na maunawaan ang kaguluhan, tulad ng drawer ng medyas.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 I-click ang larawan sa itaas para sa isang video: Inilalarawan ni John Guttag ng MIT ang clustering

Sa isang propesyonal na setting, maaaring gamitin ang clustering upang tukuyin ang mga bagay tulad ng segmentasyon ng merkado, pagtukoy kung anong pangkat ng edad ang bumibili ng anong mga item, halimbawa. Isa pang gamit ay ang pagtuklas ng anomalya, marahil para tuklasin ang pandaraya mula sa isang dataset ng mga transaksyon sa credit card. O maaari mong gamitin ang clustering upang tukuyin ang mga tumor sa isang batch ng mga medikal na scan.

✅ Mag-isip ng isang minuto kung paano mo maaaring naranasan ang clustering 'sa totoong buhay', sa banking, e-commerce, o business na setting.

> 🎓 Kapansin-pansin, nagmula ang cluster analysis sa mga larangan ng Antropolohiya at Sikolohiya noong 1930s. Maiisip mo ba kung paano ito ginamit noon?

Bilang alternatibo, maaari mo itong gamitin para sa pangkat ng mga resulta ng paghahanap - sa pamamagitan ng mga link sa pamimili, mga imahe, o mga pagsusuri, halimbawa. Kapaki-pakinabang ang clustering kapag mayroon kang malaking dataset na nais mong paliitin at kung saan nais mong magsagawa ng mas detalyadong pagsusuri, kaya maaaring gamitin ang teknik na ito upang matuto tungkol sa data bago gumawa ng ibang mga modelo.

✅ Kapag naayos mo na ang iyong data sa mga cluster, bibigyan mo ito ng cluster Id, at maaaring maging kapaki-pakinabang ang teknik na ito sa pagpapanatili ng privacy ng sebuah dataset; maaari kang tumukoy sa isang data point gamit ang cluster id nito, sa halip na sa mga mas nagpapakilalang data. Maiisip mo ba ang iba pang mga dahilan kung bakit mo tatawagin ang isang cluster Id kaysa sa iba pang mga elemento ng cluster upang tukuyin ito?

Palalimin ang iyong pagkaunawa sa mga teknik ng clustering sa [Learn module](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)
## Pagsisimula sa clustering

[Inaalok ng Scikit-learn ang malawak na hanay](https://scikit-learn.org/stable/modules/clustering.html) ng mga pamamaraan upang magsagawa ng clustering. Ang pipiliin mo ay depende sa iyong use case. Ayon sa dokumentasyon, bawat pamamaraan ay may iba't ibang benepisyo. Narito ang isang pinasimpleng talahanayan ng mga metodong sinusuportahan ng Scikit-learn at angkop na mga gamit:

| Pangalan ng Metodo             | Gamit                                                                   |
| :--------------------------- | :--------------------------------------------------------------------- |
| K-Means                      | pangkalahatang gamit, inductive                                        |
| Affinity propagation         | marami, hindi pantay na cluster, inductive                            |
| Mean-shift                   | marami, hindi pantay na cluster, inductive                            |
| Spectral clustering          | kakaunti, pantay na cluster, transductive                              |
| Ward hierarchical clustering | marami, may mga limitasyong cluster, transductive                      |
| Agglomerative clustering     | marami, may limitasyon, non Euclidean distances, transductive          |
| DBSCAN                       | hindi patag na geometry, hindi pantay na cluster, transductive         |
| OPTICS                       | hindi patag na geometry, hindi pantay na cluster na may variable density, transductive |
| Gaussian mixtures            | patag na geometry, inductive                                           |
| BIRCH                        | malaking dataset na may outliers, inductive                           |

> 🎓 Paano tayo gumagawa ng mga cluster ay malaki ang kinalaman sa kung paano natin pinagsasama-sama ang mga data point sa mga grupo. Tatalakayin natin ang ilang bokabularyo:
>
> 🎓 ['Transductive' vs. 'inductive'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Ang transductive inference ay nagmula sa mga naobserbahang training cases na tumutugma sa mga tiyak na test cases. Ang inductive inference ay nagmula sa mga training cases na tumutugma sa mga pangkalahatang patakaran na pagkatapos lamang ay inilalapat sa mga test cases.
> 
> Isang halimbawa: Isipin mo na mayroon kang dataset na bahagyang may label lamang. Ang ilang bagay ay 'records', ang ilan ay 'cds', at ang ilan ay walang label. Ang iyong trabaho ay magbigay ng mga label para sa mga walang label. Kung pipiliin mo ang inductive na paraan, magsasanay ka ng modelo na naghahanap ng 'records' at 'cds', at ilalapat mo ang mga label na iyon sa iyong unlabeled na data. Magkakaroon ito ng problema sa pag-uri ng mga bagay na talagang 'cassettes'. Ang transductive na paraan, sa kabilang banda, ay mas epektibong humahawak sa di-kilalang data habang nagsisikap maggrupo ng magkatulad na mga item at pagkatapos ay maglagay ng label sa isang grupo. Sa kasong ito, ang mga cluster ay maaaring kumatawan sa 'mga bilog na musikang bagay' at 'mga parisukat na musikang bagay'.
>
> 🎓 ['Non-flat' vs. 'flat' geometry](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
>
> Nagmula sa terminolohiyang matematika, ang non-flat vs. flat na geometry ay tumutukoy sa pagsukat ng distansya sa pagitan ng mga punto gamit ang 'flat' ([Euclidean](https://wikipedia.org/wiki/Euclidean_geometry)) o 'non-flat' (non-Euclidean) na mga pamamaraan ng geometry.
>
>'Flat' sa kontekstong ito ay tumutukoy sa Euclidean geometry (na bahagi nito ay tinuturo bilang 'plane' geometry), at ang non-flat ay tumutukoy sa non-Euclidean geometry. Ano ang kinalaman ng geometry sa machine learning? Bilang dalawang larangan na naka-ugat sa matematika, dapat mayroong karaniwang paraan upang sukatin ang distansya sa pagitan ng mga punto sa mga cluster, at magagawa iyon sa paraang 'flat' o 'non-flat', depende sa likas ng data. Ang [Euclidean distances](https://wikipedia.org/wiki/Euclidean_distance) ay sinusukat bilang haba ng linya sa pagitan ng dalawang punto. Ang [Non-Euclidean distances](https://wikipedia.org/wiki/Non-Euclidean_geometry) ay sinusukat sa kahabaan ng isang kurba. Kung ang iyong data, kapag na-visualize, ay tila hindi umiiral sa isang eroplano, maaaring kailanganin mong gumamit ng isang espesyal na algorithm upang hawakan ito.
>
![Flat vs Nonflat Geometry Infographic](../../../../translated_images/tl/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infographic ni [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Distances'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Ang mga cluster ay tinutukoy ng kanilang distance matrix, halimbawa ang mga distansya sa pagitan ng mga punto. Ang distansyang ito ay maaaring sukatin sa ilang paraan. Ang mga Euclidean cluster ay tinutukoy ng average ng mga halaga ng punto, at mayroong 'centroid' o gitnang punto. Ang mga distansya ay sinusukat mula sa distansya patungo sa centroid na iyon. Ang mga non-Euclidean distances ay tumutukoy sa mga 'clustroids', ang punto na pinakamalapit sa ibang mga punto. Ang mga clustroid ay maaaring tukuyin sa iba't ibang paraan.
> 
> 🎓 ['Constrained'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> Ang [Constrained Clustering](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) ay naglalagay ng 'semi-supervised' learning sa pamamaraang unsupervised na ito. Ang mga relasyon sa pagitan ng mga punto ay minamarkahan bilang 'cannot link' o 'must-link' kaya may ilang mga patakaran na ipinapataw sa dataset.
>
> Isang halimbawa: Kung hayagan mong pinakawalan ang isang algorithm sa isang batch ng unlabeled o semi-labeled na data, maaaring maging mababa ang kalidad ng mga cluster na nilikha nito. Sa halimbawa sa itaas, ang mga cluster ay maaaring maggrupo ng 'mga bilog na musikang bagay' at 'mga parisukat na musikang bagay' at 'mga tatsulok na bagay' at mga 'cookies'. Kung bibigyan ng mga constraints, o patakaran na sundin ("ang item ay dapat gawa sa plastik", "ang item ay kailangang makalikha ng musika") makakatulong ito upang 'limitahan' ang algorithm upang gumawa ng mas mahusay na mga pagpipilian.
> 
> 🎓 'Density'
> 
> Ang data na 'maingay' ay itinuturing na 'dense'. Ang mga distansya sa pagitan ng mga punto sa bawat cluster nito ay maaaring masuri, upang malaman kung masikip o 'siksikan' ang mga ito kaya ang data na ito ay kailangang i-analyze gamit ang angkop na pamamaraan ng clustering. [Itong artikulo](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) ay nagpapakita ng pagkakaiba sa paggamit ng K-Means clustering kumpara sa HDBSCAN algorithms upang tuklasin ang isang maingay na dataset na may hindi pantay na density ng cluster.

## Mga clustering algorithm

Mayroong mahigit 100 clustering algorithm, at ang paggamit nito ay depende sa uri ng data na hawak. Talakayin natin ang ilan sa mga pangunahing ito:

- **Hierarchical clustering**. Kapag ang isang bagay ay kinilala ayon sa pagiging malapit nito sa ibang bagay na malapit, sa halip na sa isang bagay na malayo, nabubuo ang mga cluster base sa distansya ng kanilang mga kasapi papunta at mula sa ibang mga bagay. Ang agglomerative clustering ng Scikit-learn ay hierarchical.

   ![Hierarchical clustering Infographic](../../../../translated_images/tl/hierarchical.bf59403aa43c8c47.webp)
   > Infographic ni [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroid clustering**. Ang kilalang algorithm na ito ay nangangailangan ng pagpili ng 'k', o ang bilang ng mga cluster na bubuuin, kung saan tinutukoy ng algorithm ang punto ng sentro ng cluster at kinokolekta ang data sa paligid ng puntong iyon. Ang [K-means clustering](https://wikipedia.org/wiki/K-means_clustering) ay isang kilalang bersyon ng centroid clustering. Ang sentro ay tinutukoy ng pinakamalapit na mean, kaya ang pangalan. Ang squared distance mula sa cluster ay minimal.

   ![Centroid clustering Infographic](../../../../translated_images/tl/centroid.097fde836cf6c918.webp)
   > Infographic ni [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Distribution-based clustering**. Nakabatay sa statistical modeling, ang distribution-based clustering ay nakatuon sa pagtukoy ng posibilidad na ang isang data point ay kabilang sa isang cluster, at pagkatapos ay itinalaga ito nang naaayon. Kasama sa uri na ito ang Gaussian mixture methods.

- **Density-based clustering**. Ang mga data point ay itinalaga sa mga cluster base sa kanilang density, o sa kanilang pagsasama-sama. Ang mga punto na malayo sa grupo ay tinuturing na outliers o ingay. Kabilang sa uri na ito ng clustering ang DBSCAN, Mean-shift, at OPTICS.

- **Grid-based clustering**. Para sa multi-dimensional na mga dataset, nagagawa ang isang grid at hinahati ang data sa mga cell ng grid, kaya nabubuo ang mga cluster.

## Ehersisyo - i-cluster ang iyong data

Malaki ang naitutulong ng clustering bilang isang teknik kapag may maayos na visualization, kaya magsimula tayo sa pag-visualize ng ating data ng musika. Tutulungan tayo ng ehersisyong ito na magpasya kung alin sa mga pamamaraan ng clustering ang pinakamainam gamitin para sa uri ng data na ito.

1. Buksan ang file na [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) sa folder na ito.

1. I-import ang `Seaborn` package para sa magandang data visualization.

    ```python
    !pip install seaborn
    ```

1. Idagdag ang data ng mga kanta mula sa [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Mag-load ng dataframe na may ilang data tungkol sa mga kanta. Maghanda upang suriin ang data na ito sa pamamagitan ng pag-import ng mga library at pag-check ng data:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Tingnan ang unang ilang linya ng data:

    |     | pangalan                  | album                        | artist              | artist_top_genre | release_date | haba   | kasikatan | danceability | acousticness | enerhiya | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | -------- | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b   | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42     | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683    | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Kumuha ng ilang impormasyon tungkol sa dataframe, pagtawag sa `info()`:

    ```python
    df.info()
    ```

   Ang output ay ganito ang hitsura:

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

1. I-double-check ang mga null values, sa pamamagitan ng pagtawag sa `isnull()` at pagsusuri na ang sum ay 0:

    ```python
    df.isnull().sum()
    ```

    Mukhang maayos:

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

1. Ilarawan ang data:

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

> 🤔 Kung tayo ay nagtatrabaho gamit ang clustering, isang unsupervised na pamamaraan na hindi nangangailangan ng labeled data, bakit ipinapakita natin ang data na ito na may mga labels? Sa yugto ng paggalugad ng data, ito ay kapaki-pakinabang, ngunit hindi ito kinakailangan para gumana ang mga clustering algorithm. Maaari mo ring tanggalin ang mga column headers at tukuyin ang data batay sa numero ng column. 

Tingnan ang mga pangkalahatang halaga ng data. Tandaan na ang popularity ay maaaring '0', na nagpapakita ng mga kanta na walang ranggo. Tanggalin natin ang mga iyon sa lalong madaling panahon.

1. Gumamit ng barplot upang malaman ang pinaka-popular na mga genre:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/tl/popular.9c48d84b3386705f.webp)

✅ Kung nais mong makita ang mas maraming top values, palitan ang top `[:5]` ng mas malaking halaga, o alisin ito upang makita ang lahat.

Tandaan, kapag ang top genre ay inilalarawan bilang 'Missing', ibig sabihin nito ay hindi na-classify ng Spotify ang genre, kaya't alisin natin ito.

1. Alisin ang mga nawawalang data sa pamamagitan ng pag-filter nito

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Ngayon, suriin muli ang mga genre:

    ![most popular](../../../../translated_images/tl/all-genres.1d56ef06cefbfcd6.webp)

1. Sa ngayon, nangingibabaw ang tatlong nangungunang genre sa dataset na ito. Magpokus tayo sa `afro dancehall`, `afropop`, at `nigerian pop`, dagdag pang i-filter ang dataset upang alisin ang anumang may 0 na popularidad (na nangangahulugang hindi ito na-classify na may popularidad sa dataset at maaaring ituring na ingay para sa ating layunin):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Gumawa ng mabilis na pagsusuri upang tingnan kung ang data ay may matibay na ugnayan sa anumang partikular na paraan:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/tl/correlation.a9356bb798f5eea5.webp)

    Ang tanging malakas na ugnayan ay sa pagitan ng `energy` at `loudness`, na hindi nakakagulat, dahil ang maingay na musika ay karaniwang masigla. Sa ibang bagay, mahina ang mga ugnayan. Magiging interesante kung ano ang maaaring makuha ng clustering algorithm mula sa data na ito.

    > 🎓 Tandaan na ang correlation ay hindi nangangahulugang causation! May patunay tayo ng correlation ngunit walang patunay ng causation. Isang [nakakatuwang web site](https://tylervigen.com/spurious-correlations) ang nagpapakita ng mga visual na nagdidiin dito.

Mayroon bang convergence sa dataset na ito tungkol sa inaakalang popularidad ng kanta at danceability? Ipinapakita ng isang FacetGrid na may mga concentric circle na nag-aayos, kahit ano pa man ang genre. Maaaring ang Nigerian na panlasa ay nagtatagpo sa isang tiyak na antas ng danceability para sa genre na ito?  

✅ Subukan ang iba't ibang datapoints (energy, loudness, speechiness) at mas marami o iba't ibang musikang genre. Ano ang maaari mong matuklasan? Tingnan ang `df.describe()` table para makita ang pangkalahatang pagkakalat ng mga data point.

### Pagsasanay - pamamahagi ng data

Malaki ba ang pagkakaiba ng tatlong genre na ito sa persepsyon ng kanilang danceability, batay sa kanilang popularidad?

1. Suriin ang pamamahagi ng data ng ating tatlong nangungunang genre para sa popularity at danceability sa isang ibinigay na x at y axis.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Makikita mo ang mga concentric circle sa paligid ng isang pangkalahatang punto ng convergence, na nagpapakita ng pamamahagi ng mga punto.

    > 🎓 Tandaan na ang halimbawang ito ay gumagamit ng KDE (Kernel Density Estimate) graph na kumakatawan sa data gamit ang isang tuloy-tuloy na kurba ng probabilidad. Pinapahintulutan nito tayo na mag-interpret ng data kapag nagtatrabaho sa maraming pamamahagi.

    Sa pangkalahatan, ang tatlong genre ay bahagyang nagkakatugma sa aspeto ng kanilang popularity at danceability. Ang pagtukoy ng mga cluster sa ganitong bahagyang nagkakatugmang data ay magiging hamon:

    ![distribution](../../../../translated_images/tl/distribution.9be11df42356ca95.webp)

1. Gumawa ng scatter plot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Ang scatterplot ng parehong axes ay nagpapakita ng katulad na pattern ng convergence

    ![Facetgrid](../../../../translated_images/tl/facetgrid.9b2e65ce707eba1f.webp)

Sa pangkalahatan, para sa clustering, maaari mong gamitin ang scatterplots upang ipakita ang mga cluster ng data, kaya ang pag-master sa ganitong uri ng visualization ay napaka-kapaki-pakinabang. Sa susunod na aralin, gagamitin natin ang na-filter na data na ito at gamitin ang k-means clustering upang tuklasin ang mga grupo sa data na mukhang nag-o-overlap sa mga kawili-wiling paraan.

---

## 🚀Hamón

Bilang paghahanda para sa susunod na aralin, gumawa ng isang tsart tungkol sa iba't ibang clustering algorithm na maaari mong tuklasin at gamitin sa isang production environment. Anong uri ng mga problema ang sinusubukang lutasin ng clustering?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Sariling Pag-aaral

Bago mo ilapat ang clustering algorithms, gaya ng ating natutunan, magandang ideya na maunawaan ang kalikasan ng iyong dataset. Magbasa pa tungkol sa paksang ito [dito](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Ang kapaki-pakinabang na artikulong ito](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) ay naglalakad sa iyo sa iba't ibang paraan ng pag-uugali ng iba't ibang clustering algorithm, batay sa iba't ibang hugis ng data.

## Takdang-Aralin

[Mag-research tungkol sa iba pang visualizations para sa clustering](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Pagtatanggi**:
Ang dokumentong ito ay isinalin gamit ang serbisyo ng AI translation na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't nagsusumikap kami para sa katumpakan, pakatandaan na ang awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa orihinal nitong wika ang dapat ituring na pangunahing sanggunian. Para sa mahahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang maling pagkakaintindi o maling interpretasyon na nagmula sa paggamit ng pagsasaling ito.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->