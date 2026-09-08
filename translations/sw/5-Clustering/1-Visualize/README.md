# Utangulizi wa uundaji makundi

Uundaji makundi ni aina ya [Mafunzo yasiyoongozwa](https://wikipedia.org/wiki/Unsupervised_learning) ambayo inadhani dataset haina lebo au kwamba pembejeo zake hazilingani na matokeo yaliyowekwa awali. Inatumia algoriti mbalimbali kuchambua data isiyolebwa na kutoa makundi kulingana na mifumo inayoiona katika data.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Bofya picha hapo juu kwa video. Unapojifunza mashine ya kujifunza na uundaji makundi, furahia baadhi ya nyimbo za Nigerian Dance Hall - hii ni wimbo uliopewa alama kubwa kutoka 2014 na PSquare.

## [Mtihani wa kabla ya duru](https://ff-quizzes.netlify.app/en/ml/)

### Utangulizi

[Uundaji makundi](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) ni muhimu sana kwa uchunguzi wa data. Tuchunguze ikiwa inaweza kusaidia kugundua mwenendo na mifumo katika njia ambayo hadhira wa Nigeria huchukua muziki.

✅ Chukua dakika moja kufikiria matumizi ya uundaji makundi. Katika maisha halisi, uundaji makundi hutokea wakati wowote unapokuwa na mfululizo wa nguo za kulazimika kupangilia nguo za wanak family wako 🧦👕👖🩲. Katika sayansi ya data, uundaji makundi hutokea wakati wa kujaribu kuchambua upendeleo wa mtumiaji, au kubaini sifa za dataset yoyote isiyo na lebo. Uundaji makundi, kwa namna fulani, husaidia kufasiri machafuko, kama vile kivuli cha soksi.

[![Utangulizi wa ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Utangulizi wa Uundaji Makundi")

> 🎥 Bofya picha hapo juu kwa video: John Guttag wa MIT anaanzisha uundaji makundi

Katika mazingira ya kazi, uundaji makundi unaweza kutumika kuamua vitu kama segmentation ya soko, kuamua ni makundi ya umri gani yanayonunua vitu gani, kwa mfano. Matumizi mengine ni kugundua kasoro, labda kugundua udanganyifu kutoka kwa dataset ya miamala ya kadi ya mkopo. Au unaweza kutumia uundaji makundi kubaini uvimbe katika kundi la skani za matibabu.

✅ Fikiria kwa dakika moja jinsi ulivyoweza kukutana na uundaji makundi 'katika mazingira halisi', katika benki, e-commerce, au mazingira ya biashara.

> 🎓 Kwa kufurahisha, uchambuzi wa makundi ulizaliwa katika nyanja za Anthropology na Psychology katika miaka ya 1930. Unaweza kufikiria jinsi ulivyotumika?

Vinginevyo, unaweza kuitumia kwa kuandaa matokeo ya utafutaji - kwa viungo vya kununua, picha, au mapitio, kwa mfano. Uundaji makundi ni wa manufaa wakati una dataset kubwa unayotaka kupunguza na kufanya uchambuzi wa kina zaidi, hivyo mbinu hii inaweza kutumika kujifunza kuhusu data kabla ya kujengwa modeli nyingine.

✅ Mara dataset yako inapowekwa katika makundi, unampa kitambulisho cha kundi, na mbinu hii inaweza kuwa muhimu wakati wa kuhifadhi faragha ya dataset; badala yake unaweza kurejelea kipengele kwa kitambulisho cha kundi, badala ya data inayofichua zaidi. Unaweza kufikiria sababu nyingine kwanini utarejelea kitambulisho cha kundi badala ya vipengele vingine vya kundi ili kukitambulisha?

Inua uelewa wako wa mbinu za uundaji makundi katika [moduli ya Kujifunza](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)
## Kuanzisha na uundaji makundi

[Scikit-learn inatoa mbinu nyingi](https://scikit-learn.org/stable/modules/clustering.html) za kufanya uundaji makundi. Aina unayochagua itategemea matumizi yako. Kulingana na nyaraka, kila mbinu ina faida mbalimbali. Hapa kuna jedwali lililorahisishwa la mbinu zinazoungwa mkono na Scikit-learn na matumizi yake yanayofaa:

| Jina la mbinu                | Matumizi                                                             |
| :--------------------------- | :------------------------------------------------------------------- |
| K-Means                      | matumizi ya jumla, inductive                                         |
| Affinity propagation         | makundi mengi, yasiyo sawa, inductive                               |
| Mean-shift                   | makundi mengi, yasiyo sawa, inductive                               |
| Spectral clustering          | makundi machache, sawa, transductive                                |
| Ward hierarchical clustering | makundi mengi, yanayozuiliwa, transductive                          |
| Agglomerative clustering     | makundi mengi, yanayozuiliwa, umbali usiotegemea Euclid, transductive |
| DBSCAN                       | jiometri isiyo sawa, makundi yasiyo sawa, transductive             |
| OPTICS                       | jiometri isiyo sawa, makundi yasiyo sawa yenye msongamano tofauti, transductive |
| Gaussian mixtures            | jiometri sare, inductive                                            |
| BIRCH                        | dataset kubwa yenye ving'ora, inductive                             |

> 🎓 Jinsi tunavyounda makundi ina uhusiano mkubwa na jinsi tunavyokusanya data katika makundi. Hebu tufafanue baadhi ya msamiati:
>
> 🎓 ['Transductive' dhidi ya 'inductive'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Hitimisho la transductive linatokana na kesi za mafunzo zilizoonekana zinazofanana na kesi maalum za mtihani. Hitimisho la inductive linatokana na kesi za mafunzo zinazofikia sheria za jumla ambazo kisha hutumika kwa kesi za mtihani.
> 
> Mfano: Fikiria una dataset iliyolebwa sehemu tu. Baadhi ni 'rekodi', baadhi ni 'cds', na baadhi hazina kitu. Kazi yako ni kuweka lebo kwa zile zisizo na lebo. Ukichagua njia ya inductive, ungefundisha modeli kutafuta 'rekodi' na 'cds', na kutumia lebo hizo kwa data isiyo na lebo. Njia hii itakumbwa na shida katika kutambua vitu ambavyo kwa kweli ni 'kaseti'. Njia ya transductive, kwa upande mwingine, hushughulikia data isiyojulikana vyema zaidi kwa kujaribu kuunganisha vitu vinavyofanana kisha kuweka lebo kwa kundi. Katika kesi hii, makundi yanaweza kuwakilisha 'vitu vya muziki mviringo' na 'vitu vya muziki vya mraba'.
> 
> 🎓 ['Jiometri isiyo sare' dhidi ya 'jiometri sare'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Inatokana na istilahi za hisabati, jiometri isiyo sare dhidi ya sare inahusu kipimo cha umbali kati ya pointi kwa kutumia mbinu za jiometri 'sare' ([Euclidean](https://wikipedia.org/wiki/Euclidean_geometry)) au 'isiyo sare' (isiyo-Euclidean).
>
> 'Sare' katika muktadha huu inahusu jiometri ya Euclid (sehemu zake hufundishwa kama jiometri ya 'mwelekeo'), na isiyo sare inahusu jiometri isiyo-Euclidean. Jiometri ina mahusiano gani na mashine ya kujifunza? Vizuri, kama nyanja mbili zinazotegemea hisabati, lazima kuwe na njia ya kawaida ya kupima umbali kati ya pointi za makundi, na hiyo inaweza kufanywa kwa njia ya 'sare' au 'isiyo sare', kulingana na asili ya data. [Umbali wa Euclid](https://wikipedia.org/wiki/Euclidean_distance) hupimwa kama urefu wa kipengele cha mstari kati ya pointi mbili. [Umbali usio wa Euclid](https://wikipedia.org/wiki/Non-Euclidean_geometry) hupimwa kwa njia ya mviringo. Ikiwa data yako, ikionyeshwa, inaonekana haitokani na ndege, unaweza kuhitaji kutumia algoriti maalum kushughulikia.
>
![Infograpiki ya Jiometri Sare vs Isiyo Sare](../../../../translated_images/sw/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infograpiki na [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Umbali'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Makundi hufafanuliwa na matriki ya umbali wake, mfano umbali kati ya pointi. Umbali huu unaweza kupimwa kwa njia kadhaa. Makundi ya Euclid hufafanuliwa na wastani wa thamani za pointi, na yana 'centroid' au kituo cha katikati. Umbali hupimwa kwa umbali hadi kwenye centroid hiyo. Umbali wa isiyo ya Euclid unahusu 'clustroids', pointi inayokaribia pointi zingine. Clustroids pia zinaweza kufafanuliwa kwa njia mbalimbali.
> 
> 🎓 ['Yanayozuiliwa'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Uundaji Makundi Yanayozuiliwa](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) huingiza mafunzo ya 'nusu-ongozwa' katika mbinu hii isiyoongozwa. Uhusiano kati ya pointi huwekwa kama 'haipatikani kiunganishi' au 'lazima iunganishwe' hivyo sheria fulani hutumika kwenye dataset.
>
> Mfano: Ikiwa algoriti huru itatumiwa kwenye kundi la data isiyo na lebo au isiyo kamili, makundi itakayozalisha yanaweza kuwa ya ubora mdogo. Katika mfano hapo juu, makundi yanaweza kuunganisha 'vitu vya muziki mviringo' na 'vitu vya muziki vya mraba' na 'vitu vya mviringo wa pembetatu' na 'biskuti'. Ikiwa itapewa vikwazo au sheria za kufuata ("kitu lazima kifanywe kwa plastiki", "kitu kinapaswa kuweza kutoa muziki") hii inaweza kusaidia 'kuzuia' algoriti kuchagua vyema.
> 
> 🎓 'Msongamano'
> 
> Data ambayo ni 'kelele' huchukuliwa kuwa na 'msongamano'. Umbali kati ya pointi katika kila kundi linaweza kuonyesha kuwa na msongamano mkubwa au mdogo, au 'kushikana'. Hivyo data hii inahitaji kuchambuliwa kwa njia inayofaa ya uundaji makundi. [Makala hii](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) inaonyesha tofauti kati ya kutumia K-Means dhidi ya HDBSCAN kuchunguza dataset yenye msongamano usio sawa.

## Algoriti za uundaji makundi

Kuna zaidi ya algoriti 100 za uundaji makundi, na matumizi yao hutegemea asili ya data iliyopo. Tujadili baadhi ya maarufu:

- **Uundaji makundi wa mfuatano**. Ikiwa kitu kinaainishwa kwa ukaribu wake na kitu kilicho karibu badala ya kile kilicho mbali, makundi hudhibitiwa kwa umbali wa wanachama na vitu vingine. Uundaji makundi wa agglomerative wa Scikit-learn ni wa mfuatano.

   ![Infograpiki ya uundaji makundi wa mfuatano](../../../../translated_images/sw/hierarchical.bf59403aa43c8c47.webp)
   > Infograpiki na [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Uundaji makundi wa kituo**. Algoriti hii maarufu inahitaji kuchagua 'k', au idadi ya makundi ya kuunda, kisha algoriti huamua kituo cha katikati cha kundi na kukusanya data karibu na kituo hicho. [K-means clustering](https://wikipedia.org/wiki/K-means_clustering) ni toleo maarufu la uundaji makundi wa kituo. Kituo kinaamuliwa na wastani wa karibu, hivyo jina. Umbali wa mraba kutoka kwenye kundi hupunguzwa.

   ![Infograpiki ya uundaji makundi wa kituo](../../../../translated_images/sw/centroid.097fde836cf6c918.webp)
   > Infograpiki na [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Uundaji makundi wa msingi wa usambazaji**. Msingi wake ni utambuzi wa uwezekano kwamba kipengele cha data kinahusishwa na kundi, na kupewa lebo ipasavyo. Mbinu za mchanganyiko wa Gaussian zinahusiana na aina hii.

- **Uundaji makundi kwa msongamano**. Pointi za data huthibitishwa katika makundi kulingana na msongamano wake, au kuungana kwazo. Pointi zilizo mbali na kundi huonekana kama ving'ora au kelele. DBSCAN, Mean-shift na OPTICS ni aina hii ya uundaji makundi.

- **Uundaji makundi wa msingi wa gridi**. Kwa datasets zenye vipimo vingi, gridi huundwa na data kugawanywa kwa seli za gridi, hivyo kuunda makundi.

## Zoema - unda makundi ya data yako

Uundaji makundi kama mbinu huwasaidia sana na uonyesho mzuri, kwa hivyo tuanze kwa kuonyesha data yetu ya muziki. Zoema hili litatusaidia kuamua Mbinu gani ya uundaji makundi tunapaswa kutumia kwa ufanisi zaidi kulingana na asili ya data hii.

1. Fungua faili la [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) katika saraka hii.

1. Ingiza kifurushi cha `Seaborn` kwa ajili ya uonyesho mzuri wa data.

    ```python
    !pip install seaborn
    ```

1. Ongeza data ya nyimbo kutoka [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Pakia dataframe lenye taarifa kuhusu nyimbo. Jiandae kuchunguza data hii kwa kuingiza maktaba na kuonyesha data:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Angalia mistari michache ya kwanza ya data:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Furahia Maisha Yako          | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Pata baadhi ya taarifa kuhusu dataframe, kwa kuitisha `info()`:

    ```python
    df.info()
    ```

   Matokeo yanaonekana kama ifuatavyo:

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

1. Thibitisha mara mbili kama kuna thamani tupu, kwa kuitisha `isnull()` na kuthibitisha jumla kuwa 0:

    ```python
    df.isnull().sum()
    ```

    Inaonekana nzuri:

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

1. Elezea data:

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

> 🤔 Ikiwa tunafanya kazi na usambazaji wa makundi, njia isiyo ya usaidizi ambayo haitaji data yenye lebo, kwanini tunaonyesha data hii na lebo? Katika hatua ya uchunguzi wa data, huwa ni ya msaada, lakini sio lazima kwa algorithms za ugawaji kufanya kazi. Unaweza pia kuondoa vichwa vya safu na kurejelea data kwa nambari ya safu.

Tazama thamani za jumla za data. Kumbuka kwamba umaarufu unaweza kuwa '0', ambayo inaonyesha nyimbo ambazo hazina nafasi. Wacha tujiondoe kwa muda mfupi.

1. Tumia barplot ili kufahamu aina maarufu zaidi:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![maarufu zaidi](../../../../translated_images/sw/popular.9c48d84b3386705f.webp)

✅ Ikiwa ungependa kuona maadili zaidi ya juu, badilisha top `[:5]` kwa thamani kubwa zaidi, au uiondoe ili uone yote.

Kumbuka, wakati aina kuu imeelezwa kama 'Missing', hiyo ina maana kwamba Spotify hairatibu, hivyo wacha tuiondoe.

1. Ondoa data iliyokosekana kwa kuchuja

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Sasa angalia tena aina za muziki:

    ![maarufu zaidi](../../../../translated_images/sw/all-genres.1d56ef06cefbfcd6.webp)

1. Kwa mbali, aina kuu tatu ndizo zinazoongoza dataset hii. Tujitunze `afro dancehall`, `afropop`, na `nigerian pop`, pia chuja dataset ili kuondoa chochote kilicho na thamani ya umaarufu wa 0 (maana yake haikuainishwa na umaarufu katika dataset na inaweza kuchukuliwa kama kelele kwa malengo yetu):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Fanya jaribio la haraka kuona kama data ina uhusiano wa nguvu kwa namna yoyote:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![uhusiano](../../../../translated_images/sw/correlation.a9356bb798f5eea5.webp)

    Uhusiano pekee wenye nguvu ni kati ya `energy` na `loudness`, ambalo si la kushangaza sana, kwa kuwa muziki wenye sauti kubwa kawaida huwa na nguvu nyingi. Vinginevyo, uhusiano ni dhaifu zaidi. Itakuwa ya kuvutia kuona algorithm ya ugawaji itachunguza data hii vipi.

    > 🎓 Kumbuka kuwa uhusiano hauashirii sababu! Tuna ushahidi wa uhusiano lakini hatuna ushahidi wa sababu. [Tovuti ya kusisimua](https://tylervigen.com/spurious-correlations) ina picha zinazobainisha hili.

Je, kuna muungano wowote katika dataset hii kuhusu umaarufu unaoonekana wa wimbo na danceability? FacetGrid inaonyesha kuwa kuna duara zinazopangwa sawia, bila kujali aina ya muziki. Je, inawezekana ladha za Nigeria zinafanana kwa kiwango fulani cha danceability kwa aina hii ya muziki?  

✅ Jaribu pointi tofauti za data (energy, loudness, speechiness) na aina za muziki tofauti au zaidi. Unaweza kugundua nini? Tazama jedwali la `df.describe()` kuona usambazaji wa jumla wa pointi za data.

### Mazoezi - usambazaji wa data

Je, aina hizi tatu tofauti sana katika mtazamo wa danceability yao, kulingana na umaarufu wao?

1. Chunguza usambazaji wa data wa aina zetu tatu kuu kwa umaarufu na danceability kupitia mhimili uliopewa wa x na y.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Unaweza kugundua duara zinazopangwa sawia karibu na sehemu ya jumla ya muungano, ikionyesha usambazaji wa pointi.

    > 🎓 Kumbuka mfano huu unatumia grafu ya KDE (Kernel Density Estimate) inayowakilisha data kwa kutumia mkoa wa uwezekano unaoendelea. Hii inatuwezesha kufasiri data tunapofanya kazi na usambazaji wengi.

    Kwa ujumla, aina tatu zinaelewana kwa kiwango katika umaarufu na danceability. Kuweka makundi katika data hii isiyopangwa vizuri itakuwa changamoto:

    ![usambazaji](../../../../translated_images/sw/distribution.9be11df42356ca95.webp)

1. Unda grafu ya kuchoragatia:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Grafu ya kuchoragatia ya mihimili ile ile inaonyesha muundo kama huo wa muungano

    ![Facetgrid](../../../../translated_images/sw/facetgrid.9b2e65ce707eba1f.webp)

Kwa ujumla, kwa ugawaji, unaweza kutumia grafu za kuchoragatia kuonyesha makundi ya data, hivyo kumudu aina hii ya uonyeshaji ni muhimu sana. Katika somo lijalo, tutachukua data hii iliyochujwa na kutumia ugawaji wa k-means kugundua makundi katika data hii yanayoonekana kuyopanga kwa njia za kuvutia.

---

## 🚀Changamoto

Kujitayarisha kwa somo lijalo, tengeneza chati kuhusu algorithms mbalimbali za ugawaji ambazo unaweza kugundua na kutumia katika mazingira ya uzalishaji. Ni aina gani za matatizo ugawaji unajaribu kushughulikia?

## [Mtihani wa baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio & Kujifunza Binafsi

Kabla ya kutumia algorithms za ugawaji, kama tulivyojifunza, ni wazo zuri kuelewa asili ya dataset yako. Soma zaidi kuhusu mada hii [hapa](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Makala hii yenye msaada](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) inakuongoza kupitia njia tofauti algorithms za ugawaji zinavyofanya kazi, zikizingatiwa maumbo tofauti ya data.

## Kazi ya Nyumbani

[Tafuta maonyesho mengine ya ugawaji](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Kionyozo**:
Hati hii imetafsiriwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kupata usahihi, tafadhali fahamu kwamba tafsiri za kiotomatiki zinaweza kuwa na makosa au upungufu wa usahihi. Hati ya asili katika lugha yake halisi inapaswa kuchukuliwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu inayofanywa na binadamu inapendekezwa. Hatutojibu kwa kuelewa vibaya au tafsiri potofu zinazotokea kutokana na matumizi ya tafsiri hii.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->