# Úvod do zhlukovania

Zhlukovanie je typ [neurčeného učenia](https://wikipedia.org/wiki/Unsupervised_learning), ktorý predpokladá, že dátová množina nie je označená alebo že jej vstupy nie sú spárované s preddefinovanými výstupmi. Používa rôzne algoritmy na pretriedenie neoznačených dát a poskytuje zoskupenia podľa vzorov, ktoré v dátach rozpozná.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Kliknite na obrázok vyššie pre video. Kým študujete strojové učenie so zhlukovaním, vychutnajte si nigerijské tanečné skladby - toto je veľmi hodnotená pieseň z roku 2014 od PSquare.

## [Prednáškový kvíz](https://ff-quizzes.netlify.app/en/ml/)

### Úvod

[Zhlukovanie](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) je veľmi užitočné pri prieskume dát. Pozrime sa, či môže pomôcť objaviť trendy a vzory v spôsobe, akým nigerijské publikum konzumuje hudbu.

✅ Venujte minútu zamysleniu sa nad použitím zhlukovania. V reálnom živote k zhlukovaniu dochádza, keď máte hromadu bielizne a potrebujete roztriediť oblečenie členov rodiny 🧦👕👖🩲. V dátovej vede k zhlukovaniu dochádza pri analýze preferencií používateľa alebo pri určovaní charakteristík akejkoľvek neoznačenej množiny dát. Zhlukovanie akoby pomáha dávať zmysel chaosu, ako zásuvka na ponožky.

[![Úvod do ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Úvod do zhlukovania")

> 🎥 Kliknite na obrázok vyššie pre video: John Guttag z MIT predstavuje zhlukovanie

V profesionálnom prostredí sa zhlukovanie môže použiť na určovanie vecí ako segmentácia trhu, napríklad určenie, ktoré vekové skupiny kupujú ktoré produkty. Ďalším použitím môže byť detekcia anomálií, napríklad na odhalenie podvodov v množine dát o kreditných kartách. Alebo môžete použiť zhlukovanie na určenie nádorov v sérii lekárskych snímok.

✅ Zamyslite sa minútu nad tým, ako ste sa mohli stretnúť so zhlukovaním „v divočine“, v bankovníctve, e-commerce alebo biznisovom prostredí.

> 🎓 Zaujímavosťou je, že analýza zhlukov vznikla v oblastiach antropológie a psychológie v 30. rokoch 20. storočia. Viete si predstaviť, ako mohla byť používaná?

Alternatívne by ste ju mohli použiť na zoskupovanie výsledkov vyhľadávania - napríklad podľa nákupných odkazov, obrázkov alebo recenzií. Zhlukovanie je užitočné, keď máte veľkú množinu dát, ktorú chcete zredukovať a na ktorej chcete vykonať podrobnejšiu analýzu, takže technika sa dá použiť na spoznanie dát predtým, než sa vybudujú iné modely.

✅ Akonáhle sú vaše dáta zorganizované do zhlukov, priradíte im ID zhluku a táto technika môže byť užitočná pri zachovaní súkromia množiny dát; môžete namiesto identifikácie dátového bodu podľa odhalujúcich identifikovateľných údajov použiť jeho ID zhluku. Viete si predstaviť iné dôvody, prečo by ste radšej odkazovali na ID zhluku ako na iné prvky zhluku na jeho identifikáciu?

Prehĺbte svoje znalosti o technikách zhlukovania v tomto [moduly Learn](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)
## Začíname so zhlukovaním

[Scikit-learn ponúka veľkú škálu](https://scikit-learn.org/stable/modules/clustering.html) metód na zhlukovanie. Typ, ktorý si zvolíte, bude závisieť od vášho použitia. Podľa dokumentácie má každá metóda rôzne výhody. Tu je zjednodušená tabuľka metód podporovaných Scikit-learn a ich vhodných prípadov použitia:

| Názov metódy                | Prípad použitia                                                       |
| :--------------------------- | :------------------------------------------------------------------- |
| K-Means                      | všeobecné použitie, induktívne                                      |
| Affinity propagation         | veľa, nerovnomerných zhlukov, induktívne                            |
| Mean-shift                   | veľa, nerovnomerných zhlukov, induktívne                            |
| Spektrálne zhlukovanie       | málo, rovnomerných zhlukov, transduktívne                           |
| Ward hierarchické zhlukovanie | veľa, obmedzených zhlukov, transduktívne                           |
| Agglomeratívne zhlukovanie  | veľa, obmedzených, ne-euklidovských vzdialeností, transduktívne    |
| DBSCAN                       | neplochá geometria, nerovnomerné zhluky, transduktívne             |
| OPTICS                       | neplochá geometria, nerovnomerné zhluky s premenlivou hustotou, transduktívne |
| Gaussovské zmesi            | plochá geometria, induktívne                                        |
| BIRCH                        | veľká množina dát s odľahlými hodnotami, induktívne                 |

> 🎓 Ako vytvárame zhluky veľmi súvisí s tým, ako zhromažďujeme dátové body do skupín. Pozrime sa na niektorú slovnú zásobu:
>
> 🎓 ['Transduktívne' verzus 'induktívne'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktívny záver sa vyvodzuje z pozorovaných trénovacích prípadov, ktoré mapujú konkrétne testovacie prípady. Induktívny záver sa odvodzuje z trénovacích prípadov, ktoré mapujú všeobecné pravidlá, ktoré sa až potom aplikujú na testovacie prípady.
> 
> Príklad: Predstavte si, že máte dataset, ktorý je len čiastočne označený. Niektoré veci sú 'platne', niektoré 'cdčka' a niektoré sú prázdne. Vašou úlohou je priradiť štítky prázdnym záznamom. Ak zvolíte induktívny prístup, vytrénujete model na hľadanie 'platní' a 'cdčiek' a použijete tieto štítky na neoznačené dáta. Tento prístup bude mať problém s klasifikáciou vecí, ktoré sú v skutočnosti 'kazety'. Transduktívny prístup, na druhej strane, lepšie spracováva tieto neznáme dáta, pretože pracuje na zoskupení podobných položiek a potom priradí štítok skupine. V tomto prípade by zhluky mohli odrážať "guľaté hudobné veci" a "štvorcové hudobné veci."
> 
> 🎓 ['Neplochá' verzus 'plochá' geometria](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Vychádzajúc z matematickej terminológie, neplochá verzus plochá geometria sa vzťahuje na meranie vzdialeností medzi bodmi pomocou buď 'plochej' ([euklidovskej](https://wikipedia.org/wiki/Euclidean_geometry)) alebo 'neplochej' (ne-euklidovskej) geometrickej metódy.
>
> 'Plochá' v tomto kontexte odkazuje na euklidovskú geometriu (jej časť sa učí ako 'rovinná' geometria), a neplochá na ne-euklidovskú geometriu. Čo má geometria spoločné so strojovým učením? No, ako dva odbory založené na matematike, musí existovať spoločný spôsob, ako merať vzdialenosti medzi bodmi v zhlukoch, a to sa môže robiť plochým alebo neplochým spôsobom v závislosti od povahy dát. [Euklidovské vzdialenosti](https://wikipedia.org/wiki/Euclidean_distance) sa merajú ako dĺžka úseku medzi dvoma bodmi. [Ne-euklidovské vzdialenosti](https://wikipedia.org/wiki/Non-Euclidean_geometry) sa merajú pozdĺž krivky. Ak vaše dáta, zobrazené vizuálne, vyzerajú, že neexistujú na rovine, možno budete potrebovať použiť špecializovaný algoritmus na ich spracovanie.
>
![Infografika plochá verzus neplochá geometria](../../../../translated_images/sk/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Vzdialenosti'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Zhluky sú definované ich maticou vzdialeností, napríklad vzdialenosťami medzi bodmi. Táto vzdialenosť sa môže merať niekoľkými spôsobmi. Euklidovské zhluky sú definované priemerom hodnôt bodov a obsahujú 'centroid' alebo stredový bod. Vzdialenosti sa teda merajú podľa vzdialenosti k tomuto centriodu. Ne-euklidovské vzdialenosti sa týkajú 'clustroidov', bodu najbližšieho k iným bodom. Clustroidy môžu byť definované rôznymi spôsobmi.
> 
> 🎓 ['Obmedzené'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Obmedzené zhlukovanie](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) zavádza do metódy nesupervidovaného učenia prvky semi-supervidovaného učenia. Vzťahy medzi bodmi sú označené ako 'nemôže byť spojené' alebo 'musí byť spojené', takže na množinu dát sú vynútené niektoré pravidlá.
>
>Príklad: Ak algoritmus necháme voľne pracovať na množine neoznačených alebo čiastočne označených dát, zhluky, ktoré vytvorí, môžu byť nekvalitné. V predchádzajúcom príklade by zhluky mohli zoskupiť „guľaté hudobné veci“, „štvorcové hudobné veci“, „trojuholníkové veci“ a „keksíky“. Ak mu zadáme niektoré obmedzenia alebo pravidlá ("položka musí byť z plastu", "položka musí vedieť produkovať hudbu"), môže to pomôcť algoritmu robiť lepšie rozhodnutia.
> 
> 🎓 'Hustota'
> 
> Dáta, ktoré sú 'šumové', sa považujú za 'husté'. Vzdialenosti medzi bodmi v jednotlivých zhlukoch môžu byť po preskúmaní viac alebo menej husté, alebo 'preplnené', a preto tieto dáta treba analyzovať vhodnou metódou zhlukovania. [Tento článok](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) ukazuje rozdiel medzi použitím K-Means a HDBSCAN algoritmov na skúmanie šumovej množiny dát s nerovnomernou hustotou zhlukov.

## Algoritmy zhlukovania

Existuje viac ako 100 algoritmov na zhlukovanie a ich použitie závisí od povahy spracovávaných dát. Pozrime sa na niektoré hlavné:

- **Hierarchické zhlukovanie**. Ak je objekt klasifikovaný podľa svojej blízkosti k susednému objektu, namiesto k vzdialenejšiemu, vytvárajú sa zhluky na základe vzdialenosti členov k iným objektom. Agglomeratívne zhlukovanie v Scikit-learn je hierarchické.

   ![Infografika hierarchického zhlukovania](../../../../translated_images/sk/hierarchical.bf59403aa43c8c47.webp)
   > Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroidové zhlukovanie**. Tento obľúbený algoritmus vyžaduje zvoliť počet „k“ zhlukov, ktoré sa majú vytvoriť, potom algoritmus určí stredový bod zhluku a zhromaždí údaje okolo neho. [K-means zhlukovanie](https://wikipedia.org/wiki/K-means_clustering) je populárna verzia centroidového zhlukovania. Stred je určený najbližším priemerom, odtiaľ názov. Štvorcová vzdialenosť od zhluku je minimalizovaná.

   ![Infografika centroidového zhlukovania](../../../../translated_images/sk/centroid.097fde836cf6c918.webp)
   > Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Distribučné zhlukovanie**. Založené na štatistickom modelovaní, distribučné zhlukovanie sa zameriava na určenie pravdepodobnosti, že dátový bod patrí do zhluku, a podľa toho ho priraďuje. Metódy gaussovských zmesí patria do tohto typu.

- **Hustotné zhlukovanie**. Dátové body sa priraďujú ku zhlukom na základe ich hustoty alebo ich zoskupenia okolo seba. Dátové body vzdialené od skupiny sa považujú za odľahlé hodnoty alebo šum. DBSCAN, Mean-shift a OPTICS patria do tohto typu zhlukovania.

- **Mriežkové zhlukovanie**. Pre viacrozmerné dáta sa vytvára mriežka a dáta sa rozdeľujú medzi bunky mriežky, čím vznikajú zhluky.

## Cvičenie - zhlukujte svoje dáta

Zhlukovanie ako technika je výrazne podporované správnou vizualizáciou, takže začnime vizualizáciou našich hudobných dát. Toto cvičenie nám pomôže rozhodnúť, ktorú z metód zhlukovania by sme mali najefektívnejšie použiť pre povahu týchto dát.

1. Otvorte súbor [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) v tomto priečinku.

1. Importujte balík `Seaborn` pre dobrú vizualizáciu dát.

    ```python
    !pip install seaborn
    ```

1. Pridajte dáta piesní zo súboru [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Načítajte dataframe s niektorými informáciami o piesňach. Pripravte sa na prieskum týchto dát importom knižníc a vyvedením dát:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Skontrolujte prvé riadky dát:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Získajte niektoré informácie o dátovom rámci zavolaním `info()`:

    ```python
    df.info()
    ```

   Výstup vyzerá takto:

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

1. Dvakrát si overte null hodnoty zavolaním `isnull()` a overením, že súčet je 0:

    ```python
    df.isnull().sum()
    ```

    Vyzerá to dobre:

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

1. Popíšte dáta:

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

> 🤔 Ak pracujeme so zhlukovaním, nepozorovanou metódou, ktorá nevyžaduje označené dáta, prečo tento údaj zobrazujeme s označeniami? V fáze prieskumu dát sú užitočné, no pre zhlukovacie algoritmy nie sú nevyhnutné. Môžete rovno odstrániť názvy stĺpcov a odkazovať sa na dáta podľa čísel stĺpcov. 

Pozrite sa na všeobecné hodnoty dát. Všimnite si, že popularita môže byť '0', čo ukazuje skladby bez rebríčkovania. Tie čoskoro odstránime.

1. Použite barplot, aby ste zistili najpopulárnejšie žánre:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/sk/popular.9c48d84b3386705f.webp)

✅ Ak si chcete pozrieť viac top hodnôt, zmeňte horný limit `[:5]` na vyššiu hodnotu alebo ho odstráňte a zobrazíte všetko.

Dávajte pozor, keď je top žáner popísaný ako 'Missing', znamená to, že ho Spotify neklasifikovalo, preto ho odstránime.

1. Odstráňte chýbajúce dáta ich filtrovaním

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Teraz opätovne skontrolujte žánre:

    ![most popular](../../../../translated_images/sk/all-genres.1d56ef06cefbfcd6.webp)

1. Zďaleka najdominantnejšie v tomto datasete sú tri žánre. Zamerajme sa na `afro dancehall`, `afropop` a `nigerian pop` a zároveň filtrovať dataset, aby sa odstránilo všetko s hodnotou popularity 0 (čo znamená, že v datasete nebolo zaradené do rebríčka a môže to byť považované za šum):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Vykonajte rýchly test, či je v dátach nejaká silná korelácia:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/sk/correlation.a9356bb798f5eea5.webp)

    Jediná silná korelácia je medzi `energy` a `loudness`, čo nie je prekvapujúce, pretože hlasitá hudba je zvyčajne dosť energická. Inak sú korelácie relatívne slabé. Bude zaujímavé vidieť, čo s týmito dátami dokáže urobiť zhlukovací algoritmus.

    > 🎓 Upozorňujeme, že korelácia neimplikuje príčinnosť! Máme dôkaz korelácie, ale nie dôkaz príčiny. [Zábavná webová stránka](https://tylervigen.com/spurious-correlations) obsahuje vizuály zdôrazňujúce túto tému.

Existuje nejaká konvergencia v tomto datasete okolo vnímania popularity a tanečnosti skladby? FacetGrid ukazuje, že existujú sústredné kružnice, ktoré sa zhodujú bez ohľadu na žáner. Mohlo by to znamenať, že nigerijské chute sa zbiehajú pri určitej úrovni tanečnosti pre tento žáner?  

✅ Vyskúšajte rôzne údaje (energy, loudness, speechiness) a viac alebo iné hudobné žánre. Čo môžete objaviť? Pozrite si tabuľku `df.describe()` a všimnite si všeobecné rozloženie dát.

### Cvičenie – distribúcia dát

Sú tieto tri žánre výrazne odlišné vnímaním ich tanečnosti na základe popularity?

1. Preskúmajte rozdelenie dát top troch žánrov podľa popularity a tanečnosti na osi x a y.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Môžete objaviť sústredné kružnice okolo všeobecného bodu zbiehania, ktoré ukazujú rozloženie bodov.

    > 🎓 Upozorňujeme, že tento príklad používa KDE (odhad hustoty jadra), ktorý reprezentuje dáta pomocou spojitej krivky pravdepodobnosti. To nám umožňuje interpretovať dáta pri práci s viacerými distribúciami.

    Všeobecne sa tri žánre voľne zhodujú v popularite a tanečnosti. Určiť klastry v týchto voľne usporiadaných dátach bude výzvou:

    ![distribution](../../../../translated_images/sk/distribution.9be11df42356ca95.webp)

1. Vytvorte scatter plot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Scatterplot pre rovnaké osi ukazuje podobný vzor zbiehania

    ![Facetgrid](../../../../translated_images/sk/facetgrid.9b2e65ce707eba1f.webp)

Všeobecne, pre zhlukovanie môžete použiť scatterploty na znázornenie klastrov dát, takže ovládanie tohto typu vizualizácie je veľmi užitočné. V ďalšej lekcii použijeme tieto filtrované dáta a k-means zhlukovanie na objavenie skupín, ktoré sa zdajú priťahovať zaujímavým spôsobom.

---

## 🚀Výzva

Na prípravu na ďalšiu lekciu vytvorte graf o rôznych zhlukovacích algoritmoch, ktoré môžete objaviť a použiť v produkčnom prostredí. Aké problémy sa zhlukovanie snaží riešiť?

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium

Pred použitím zhlukovacích algoritmov, ako sme sa naučili, je dobré porozumieť povahy vášho datasetu. Viac o tejto téme si prečítajte [tu](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Tento užitočný článok](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) vás prevedie rôznymi spôsobmi správania sa zhlukovacích algoritmov vzhľadom na rôzne tvary dát.

## Zadanie

[Preskúmajte ďalšie vizualizácie pre zhlukovanie](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vyhlásenie o zodpovednosti**:
Tento dokument bol preložený pomocou AI prekladateľskej služby [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, vezmite prosím na vedomie, že automatické preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho natívnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za žiadne nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->