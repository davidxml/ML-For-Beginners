# Bevezetés a klaszterezésbe

A klaszterezés az [felügyelet nélküli tanulás](https://wikipedia.org/wiki/Unsupervised_learning) egy típusa, amely azt feltételezi, hogy az adathalmaz címkézetlen vagy bemenetei nincsenek előre meghatározott kimenetekhez rendelve. Különböző algoritmusokat használ az címkézetlen adatok átvizsgálására és mintázatok alapján csoportokba rendezésére.

[![Nincs senki hozzád hasonló a PSquare-tól](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "Nincs senki hozzád hasonló a PSquare-tól")

> 🎥 Kattints a fenti képre egy videóért. Amíg a klaszterezéssel tanulod a gépi tanulást, élvezd néhány nigériai Dance Hall dalt - ez egy 2014-ben készült, nagyra értékelt dal a PSquare-tól.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

### Bevezetés

A [klaszterezés](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) nagyon hasznos az adatelemzéshez. Nézzük meg, hogy segíthet-e megismerni a nigériai közönség zene fogyasztási szokásainak trendjeit és mintázatait.

✅ Gondolj egy percet a klaszterezés felhasználására. A való életben klaszterezés történik, amikor egy kosár szennyest kell szétválogatni a családtagok ruhái szerint 🧦👕👖🩲. Az adattudományban klaszterezés történik, amikor egy felhasználó preferenciáit elemzed, vagy egy címkézetlen adathalmaz jellemzőit próbálod meghatározni. A klaszterezés valahogy segít rendet rakni a káoszban, mint egy zoknis fiókban.

[![Bevezetés az ML-be](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Bevezetés a klaszterezésbe")

> 🎥 Kattints a fenti képre egy videóért: John Guttag, az MIT-től bemutatja a klaszterezést

Szakmai környezetben a klaszterezés használható például piaci szegmensek meghatározására, például hogy mely korcsoport mely termékeket vásárolja. Egy másik felhasználás az anomália detektálás lehet, például csalások felismerése hitelkártyás tranzakciók adatállományából. Vagy használhatod tumorok azonosítására orvosi felvételek között.

✅ Gondolj egy percet arra, hogy hol találkozhattál a klaszterezéssel „a való életben”, banki, e-kereskedelmi vagy üzleti környezetben.

> 🎓 Érdekesség, hogy a klaszterelemzés a 1930-as évek antropológia és pszichológia területéről ered. El tudod képzelni, hogyan használták?

Alternatívaként csoportosíthatod vele a keresési találatokat - például vásárlási linkek, képek vagy értékelések szerint. Klaszterezés akkor hasznos, amikor nagy adatállományt akarsz csökkenteni, és részletesebb elemzést szeretnél végezni rajta, így a módszert felhasználhatod adatok megismerésére, mielőtt más modelleket építesz.

✅ Ha az adatokat klaszterekbe rendezted, hozzárendelsz egy klaszterazonosítót, és ez a technika hasznos lehet az adatállomány adatvédelmének megőrzésében; egy adatpontot hivatkozhatsz klaszterazonosítóval a nyilvánosabb azonosító adatok helyett. Tudsz más okot is mondani, hogy miért jobb a klaszterazonosító használata a klaszter azonosítására?

Mélyítsd el a klaszterezési technikák ismeretét ebben a [Learn modulban](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)
## Klaszterezés megkezdése

A [Scikit-learn számos](https://scikit-learn.org/stable/modules/clustering.html) módszert kínál klaszterezéshez. A választott típust az eseted határozza meg. A dokumentáció szerint minden módszernek megvannak a maga előnyei. Íme egy egyszerűsített táblázat a Scikit-learn által támogatott módszerekről és azok megfelelő alkalmazásairól:

| Módszer neve                | Alkalmazási terület                                                   |
| :--------------------------- | :-------------------------------------------------------------------- |
| K-Means                      | általános cél, induktív                                              |
| Affinity propagation         | sok, egyenetlen klaszter, induktív                                  |
| Mean-shift                   | sok, egyenetlen klaszter, induktív                                  |
| Spektrális klaszterezés       | kevés, egyenletes klaszter, transzduktív                            |
| Ward hierarchikus klaszterezés | sok, megszorított klaszter, transzduktív                            |
| Agglomeratív klaszterezés     | sok, megszorított, nem euklideszi távolságokat használó, transzduktív |
| DBSCAN                       | nem sík geometria, egyenetlen klaszterek, transzduktív              |
| OPTICS                       | nem sík geometria, változó sűrűségű egyenetlen klaszterek, transzduktív |
| Gauss keverékek             | sík geometria, induktív                                             |
| BIRCH                        | nagy adathalmaz kiugró értékekkel, induktív                         |

> 🎓 A klaszterek létrehozása sokban függ attól, hogyan csoportosítjuk a pontokat. Nézzük meg pár kifejezést:
>
> 🎓 ['Transzduktív' vs. 'induktív'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> A transzduktív következtetés a megfigyelt tanító esetekből származik, amelyek konkrét tesztesetekhez kapcsolódnak. Az induktív következtetés a tanító esetekből általános szabályokat von le, amelyeket aztán alkalmaz a tesztesetekre.
> 
> Példa: Képzelj el egy részben címkézett adathalmazt. Van benne 'lemez', 'cd' és néhány üres címke. Az a dolgod, hogy megtöltsd az üres címkéket. Ha induktív megközelítést választasz, olyat tanítasz, ami 'lemezeket' és 'cd-ket' keres, és ezekkel látod el a címkézetlen adatokat. Ez bajban lesz, ha 'kazetták' is vannak. Egy transzduktív megközelítés jobb, mert az ismeretlen adatokat úgy kezeli, hogy hasonló elemeket csoportosít, majd címkét rendel csoporthoz. Ebben az esetben a klaszterek lehetnek például 'kerek zenei tárgyak' és 'négyzetes zenei tárgyak'.
> 
> 🎓 ['Nem sík' vs. 'sík' geometria](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Matematikai terminológiából eredően a nem sík vs. sík geometria a pontok közti távolság mérésének módját jelenti, amely vagy sík ([euklideszi](https://wikipedia.org/wiki/Euclidean_geometry)) vagy nem sík (nem euklideszi) geometriai módszerek alapján történik.
>
> A 'sík' itt az euklideszi geometriára utal (amit 'síkgemometriaként' is tanítanak), a nem sík pedig a nem euklideszi geometriát jelenti. Mi köze van a geometriának a gépi tanuláshoz? Mindkettő matematikán alapul, ezért közös mód van a pontok közötti távolság mérésére klaszterekben, ami lehet 'sík' vagy 'nem sík', az adat természetétől függően. [Euklideszi távolság](https://wikipedia.org/wiki/Euclidean_distance) egy vonalszakasz hossza két pont között. [Nem euklideszi távolság](https://wikipedia.org/wiki/Non-Euclidean_geometry) görbén mért távolság. Ha az adat vizualizációját nézve nem síkon van, speciális algoritmust kell használni.
>
![Sík és nem sík geometria infografika](../../../../translated_images/hu/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infografika készítője: [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Távolságok'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> A klasztereket a távolságmátrix határozza meg, azaz a pontok közti távolságok. Ezt többféleképpen lehet mérni. Az euklideszi klaszterek a pontok értékeinek átlagát veszik, és van egy 'centruma', vagy középpontja. Távolságok a centrumtól számított távolságok alapján vannak mérve. A nem euklideszi távolságok a 'klusztoidokat' jelentik, amelyek a többi ponthoz legközelebb eső pontok. A klusztoidokat többféleképpen definiálják.
> 
> 🎓 ['Megszervezett'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> A [korlátozott klaszterezés](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) bevezeti a 'félfelügyelt' tanulást ebbe a felügyelet nélküli módszerbe. A pontok közti viszonyokat 'nem összekapcsolható' vagy 'összekapcsolandó' címkével látják el, hogy szabályokat alkalmazzanak az adatokra.
>
> Példa: Ha egy algoritmust szabadon engednek egy címkézetlen vagy részben címkézett adathalmazon, a kész klaszterek gyengék lehetnek. A fenti példánál a klaszterek csoportosíthatnak 'kerek zenei tárgyakat', 'négyzetes zenei tárgyakat', 'háromszög alakú dolgokat' és 'sütiket'. Ha szabályokat adunk hozzájuk ("az elem műanyagból kell, hogy legyen", "az elemnek zenét kell tudnia előállítani"), az segíthet jobb döntéseket hozni.
> 
> 🎓 'Sűrűség'
> 
> A 'zajos' adatot sűrűnek tekintjük. Egy klaszter pontjai közötti távolságok vizsgálata alapján kiderülhet, hogy egy klaszter sűrű vagy ritkás, és ez a megfelelő klaszterezési módszer kiválasztását igényli. [Ez a cikk](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) bemutatja, hogy mi a különbség K-Means és HDBSCAN algoritmusok használata között egy zajos, változó sűrűségű klaszterű adathalmazon.

## Klaszterező algoritmusok

Több mint 100 klaszterező algoritmus létezik, és az alkalmazásuk az adatok természetétől függ. Nézzünk meg néhány nagyobb típust:

- **Hierarchikus klaszterezés**. Ha egy objektumot a hozzá közeli objektum távolsága alapján osztályozunk, nem pedig a távolabbi alapján, akkor a klaszterek tagjaik közti távolság szerint jönnek létre. A Scikit-learn agglomeratív klaszterezője hierarchikus.

   ![Hierarchikus klaszterezés infografika](../../../../translated_images/hu/hierarchical.bf59403aa43c8c47.webp)
   > Infografika készítője: [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroid klaszterezés**. Ez a népszerű algoritmus megköveteli a 'k' érték kiválasztását, vagyis a klaszterek számát, majd az algoritmus meghatározza a klaszter középpontját, és a adatokat ahhoz gyűjti össze. A [K-means klaszterezés](https://wikipedia.org/wiki/K-means_clustering) a centroid klaszterezés ismert változata. A középpontot a legközelebbi átlag határozza meg, innen ered a neve. A klasztertől való négyzetes távolságot minimalizálja.

   ![Centroid klaszterezés infografika](../../../../translated_images/hu/centroid.097fde836cf6c918.webp)
   > Infografika készítője: [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Eloszlás alapú klaszterezés**. Statisztikai modellezésen alapul, ahol megállapítják a valószínűségét, hogy egy adatpont melyik klaszterhez tartozik, és ennek megfelelően sorolja be. A Gauss keverék módszerek ide tartoznak.

- **Sűrűség alapú klaszterezés**. Az adatpontok sűrűségük, vagyis egymáshoz való csoportosulás alapján kerülnek klaszterbe. A csoporttól távoli pontokat kiugrónak vagy zajnak tekintik. Ilyen algoritmusok a DBSCAN, Mean-shift és az OPTICS.

- **Rácsalapú klaszterezés**. Többdimenziós adatállomány esetén rácsot hoz létre, majd az adatokat a cellák között osztja szét, így klasztereket hoz létre.

## Gyakorlat - klaszterezzük az adatokat!

A klaszterezés technikáját nagyon segíti a megfelelő vizualizáció, ezért kezdjük azzal, hogy vizualizáljuk a zenei adatainkat. Ez a gyakorlat segít eldönteni, melyik klaszterezési módot érdemes alkalmazni erre az adatra.

1. Nyisd meg ebben a mappában a [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) fájlt.

1. Importáld a `Seaborn` csomagot a jó adatvizualizációért.

    ```python
    !pip install seaborn
    ```

1. Add hozzá a dal adatokat az [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv) fájlból. Tölts be egy adatkeretet néhány adattal a dalokról. Készülj fel az adatok felfedezésére a könyvtárak importálásával és az adatok kiírásával:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Nézd meg az első néhány adat sort:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigériai pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Szerezzünk némi információt az adattábláról az `info()` meghívásával:

    ```python
    df.info()
    ```

   A kimenet így néz ki:

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

1. Kettős ellenőrzés a hiányzó értékekre, az `isnull()` meghívásával és a nulla összeg ellenőrzésével:

    ```python
    df.isnull().sum()
    ```

    Minden rendben:

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

1. Írjuk le az adatokat:

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

> 🤔 Ha klaszterezéssel dolgozunk, egy olyan felügyelt módszer nélkülivel, amely nem igényel címkézett adatokat, miért mutatjuk meg ezeket az adatokat címkékkel? Az adatfeltárás fázisában jól jönnek, de nem szükségesek a klaszterező algoritmusok működéséhez. Egyszerűen eltávolíthatnánk az oszlopfejléceket, és hivatkozhatnánk az adatokra oszlopszám szerint.

Tekintsük át az adatok általános értékeit. Vegyük észre, hogy a népszerűség lehet '0' is, ami olyan dalokat jelent, amelyeknek nincs rangsorolásuk. Ezeket rövidesen töröljük.

1. Használjunk oszlopdiagramot, hogy megtudjuk melyik a legnépszerűbb műfaj:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![legnépszerűbb](../../../../translated_images/hu/popular.9c48d84b3386705f.webp)

✅ Ha több top értéket szeretnél látni, a top `[:5]` értékét növeld vagy töröld, hogy az összes megjelenjen.

Megjegyzés: ha a legnépszerűbb műfaj "Missing" (hiányzik) megjelöléssel szerepel, az azt jelenti, hogy a Spotify nem sorolta be, így szabaduljunk meg tőle.

1. Szabaduljunk meg a hiányzó adatokról szűréssel:

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Most ellenőrizzük újra a műfajokat:

    ![legnépszerűbb](../../../../translated_images/hu/all-genres.1d56ef06cefbfcd6.webp)

1. Egyértelműen, a három legnépszerűbb műfaj dominálja az adattáblát. Koncentráljunk az `afro dancehall`, `afropop` és `nigerian pop` műfajokra, tovább szűrve az adatokat úgy, hogy eltávolítjuk azokat, amelyek népszerűsége 0 (ami azt jelenti, hogy nem volt besorolva népszerűségi adatként az adathalmazban, és zajként kezelhető a céljaink szempontjából):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Végezzünk egy gyors tesztet, hogy lássuk, az adatok között van-e különösen erős korreláció:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![korrelációk](../../../../translated_images/hu/correlation.a9356bb798f5eea5.webp)

    Az egyetlen erős korreláció az `energia` és a `hangerő` között van, ami nem meglepő, hiszen a hangos zene általában elég energikus. Egyébként a korrelációk viszonylag gyengék. Érdekes lesz látni, mit tud kezdeni a klaszterező algoritmus ezzel az adattal.

    > 🎓 Ne feledd, a korreláció nem jelent oksági kapcsolatot! Bizonyítékunk van korrelációra, de nem az ok-okozatra. Egy [szórakoztató weboldal](https://tylervigen.com/spurious-correlations) vizuális példákat mutat erre.

Van-e konvergencia ebben az adathalmazban a dal népszerűségének érzékelése és táncolhatósága között? Egy FacetGrid azt mutatja, hogy koncentrikus körök vannak, amik sorba rendeződnek, függetlenül a műfajtól. Lehet, hogy a nigériai ízlés egy bizonyos táncolhatósági szintnél konvergál ebben a műfajban?

✅ Próbálj ki különböző adatpontokat (energia, hangerő, beszédesség) és több vagy más zenei műfajt. Mit fedezhetsz fel? Nézd meg a `df.describe()` táblát az adatok általános eloszlásának megértéséhez.

### Gyakorlat - adateloszlás

E három műfaj lényegesen különbözik-e táncolhatóságuk érzékelésében, népszerűségük alapján?

1. Vizsgáld meg a három vezető műfaj adatainak eloszlását népszerűség és táncolhatóság szerint egy adott x és y tengely mentén.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Felfedezhetsz koncentrikus köröket egy általános konvergencia pont körül, amelyek az adatok eloszlását mutatják.

    > 🎓 Ez a példa KDE (Kernel Density Estimate) grafikont használ, ami az adatokat egy folyamatos valószínűségi sűrűség görbével ábrázolja. Ez lehetővé teszi az adatok értelmezését több eloszlás esetén.

    Általánosságban a három műfaj laza összhangban van népszerűség és táncolhatóság tekintetében. Klaszterek meghatározása ebben a laza összhangban lévő adatban kihívás lesz:

    ![eloszlás](../../../../translated_images/hu/distribution.9be11df42356ca95.webp)

1. Készíts egy pontdiagramot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Ugyanazon tengelyek pontdiagramja hasonló konvergencia mintát mutat:

    ![Facetgrid](../../../../translated_images/hu/facetgrid.9b2e65ce707eba1f.webp)

Általánosságban a klaszterezéshez pontdiagramokat használhatsz az adatok klaszterek szerinti megjelenítésére, ezért ennek a vizualizáció típusnak a meglátása nagyon hasznos. A következő leckében ezt a szűrt adatot fogjuk használni k-móduszú klaszterezéssel, hogy csoportokat fedezzünk fel, amelyek érdekes módon fedik egymást.

---

## 🚀Kihívás

A következő lecke előkészítéseként készíts egy ábrát a különböző klaszterező algoritmusokról, amelyeket felfedezhetsz és használhatsz egy éles környezetben. Milyen problémákat próbál megoldani a klaszterezés?

## [Leckezáró kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés & Önálló tanulás

Mielőtt klaszterező algoritmusokat alkalmaznál, ahogy tanultuk, jó ötlet megérteni az adathalmazod természetét. Olvass többet erről a témáról [itt](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Ez a hasznos cikk](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) végigvezet a különböző klaszterező algoritmusok viselkedésén, eltérő adat alakok esetén.

## Feladat

[Keresd a klaszterezés egyéb vizualizációit](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Jogi nyilatkozat**:
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével készült. Bár az pontosságra törekszünk, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az anyanyelvén tekintendő hiteles forrásnak. Fontos információk esetén professzionális emberi fordítást javasolunk. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely ebből a fordításból ered.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->