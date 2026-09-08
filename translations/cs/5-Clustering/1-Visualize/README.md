# Úvod do shlukování

Shlukování je typ [učení bez učitele](https://wikipedia.org/wiki/Unsupervised_learning), které předpokládá, že dataset je neoznačený nebo že jeho vstupy nejsou párovány s předdefinovanými výstupy. Používá různé algoritmy k rozdělení neoznačených dat a vytvoření skupin podle vzorů, které v datech rozpozná.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Klikněte na obrázek výše pro video. Zatímco se učíte strojové učení pomocí shlukování, užijte si nigerijské dance hall skladby – toto je velmi vysoce hodnocená píseň z roku 2014 od PSquare.

## [Přednáškový kvíz](https://ff-quizzes.netlify.app/en/ml/)

### Úvod

[Shlukování](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) je velmi užitečné pro průzkum dat. Podívejme se, zda může pomoci odhalit trendy a vzory v tom, jak nigerijské publikum konzumuje hudbu.

✅ Věnujte minutu přemýšlení o využití shlukování. V reálném životě shlukování nastává, kdykoli máte hromadu prádla a potřebujete roztřídit oblečení členů rodiny 🧦👕👖🩲. Ve světě datové vědy se shlukování používá při analýze preferencí uživatele nebo při určování charakteristik jakéhokoli neoznačeného datasetu. Shlukování tak trochu pomáhá dát smysl chaosu, jako zásuvka na ponožky.

[![Introdukce do ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Klikněte na obrázek výše pro video: John Guttag z MIT představuje shlukování

V profesionálním prostředí lze shlukování využít k určení věkových skupin, které kupují jaké zboží například na segmentaci trhu. Dalším využitím může být detekce anomálií, třeba odhalení podvodu v datasetu transakcí kreditní kartou. Nebo byste mohli použít shlukování k rozpoznání nádorů ve skupině lékařských snímků.

✅ Zamyslete se chvíli, kdy jste se mohli setkat se shlukováním "v terénu", v bankovnictví, e-commerce nebo v podnikání.

> 🎓 Zajímavé je, že analýza shluků pochází z antropologie a psychologie ze 30. let 20. století. Dokážete si představit, jak byla tehdy využívána?

Alternativně ji můžete použít ke skupinovému seskupení výsledků vyhledávání – například podle nákupních odkazů, obrázků či recenzí. Shlukování je užitečné, pokud máte velký dataset, který chcete zjednodušit a na kterém chcete provést podrobnější analýzu, takže tuto techniku lze použít k pochopení dat dříve, než se vytvoří jiné modely.

✅ Jakmile máte data uspořádaná ve shlucích, přiřadíte jim ID shluku a tato technika může být užitečná při uchovávání soukromí datasetu; místo konkrétního bodu dat můžete odkazovat na jeho ID shluku, nikoli na více odhalující identifikovatelná data. Napadá vás nějaký jiný důvod, proč byste raději odkazovali na ID shluku než na jiné prvky shluku k jeho identifikaci?

Prohlubte své znalosti o technikách shlukování v tomto [Learn modulu](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott).

## Začínáme se shlukováním

[Scikit-learn nabízí širokou škálu](https://scikit-learn.org/stable/modules/clustering.html) metod pro provádění shlukování. Typ, který zvolíte, závisí na vašem použití. Podle dokumentace má každá metoda různé výhody. Zde je zjednodušená tabulka metod podporovaných Scikit-learnem a jejich vhodné použití:

| Název metody                | Použití                                                              |
| :--------------------------- | :------------------------------------------------------------------- |
| K-Means                      | obecné použití, induktivní                                          |
| Affinity propagation         | mnoho, nerovnoměrné shluky, induktivní                             |
| Mean-shift                   | mnoho, nerovnoměrné shluky, induktivní                             |
| Spektrální shlukování        | málo, rovnoměrné shluky, transduktivní                             |
| Wardovo hierarchické shlukování | mnoho, omezené shluky, transduktivní                              |
| Agglomerativní shlukování    | mnoho, omezené, ne-Eukleidovské vzdálenosti, transduktivní         |
| DBSCAN                       | neplochá geometrie, nerovnoměrné shluky, transduktivní             |
| OPTICS                       | neplochá geometrie, nerovnoměrné shluky s proměnnou hustotou, transduktivní |
| Gaussian mixtures            | plochá geometrie, induktivní                                       |
| BIRCH                        | velký dataset s odlehlými hodnotami, induktivní                   |

> 🎓 Jak vytváříme shluky, hodně souvisí s tím, jak shromažďujeme datové body do skupin. Pojďme rozebrat slovní zásobu:
>
> 🎓 ['Transduktivní' vs. 'induktivní'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktivní inferenci získáváme z pozorovaných tréninkových případů, které odpovídají konkrétním testovacím případům. Induktivní inference vychází z tréninkových případů, které vedou k obecným pravidlům, jež jsou pak aplikována na testovací případy.
> 
> Příklad: Představte si, že máte dataset, který je pouze částečně označený. Některá data jsou 'desky', některá 'CD' a některá jsou prázdná. Vaším úkolem je přiřadit štítky prázdným položkám. Pokud zvolíte induktivní přístup, vytrénujete model na hledání 'desek' a 'CD' a tyto štítky aplikujete na neoznačená data. Tento přístup bude mít potíže klasifikovat věci, které jsou ve skutečnosti 'kazety'. Transduktivní přístup naopak efektivněji pracuje s neznámými daty, protože se snaží seskupit podobné položky a poté přiřadit označení skupině. V tomto případě by shluky mohly reflektovat „kulaté hudební věci“ a „čtvercové hudební věci“.
> 
> 🎓 ['Neplochá' vs. 'plochá' geometrie](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Původ v matematické terminologii, neplochá vs. plochá geometrie se týká měření vzdáleností mezi body pomocí buď 'plochých' ([Eukleidovských](https://wikipedia.org/wiki/Euclidean_geometry)) nebo 'neplochých' (ne-Eukleidovských) geometrických metod.
>
> 'Plochá' zde znamená Eukleidovskou geometrii (její část je vyučována jako 'rovinná' geometrie), zatímco neplochá označuje ne-Eukleidovskou geometrii. Co má geometrie společného se strojovým učením? Jako obory založené na matematice je potřeba mít společný způsob měření vzdáleností mezi body ve shlucích, což lze provést plochým nebo neplochým způsobem v závislosti na povaze dat. [Eukleidovské vzdálenosti](https://wikipedia.org/wiki/Euclidean_distance) se měří jako délka úseku mezi dvěma body. [Ne-Eukleidovské vzdálenosti](https://wikipedia.org/wiki/Non-Euclidean_geometry) se měří podél křivky. Pokud vaše data vizualizovaná nevypadají, že by byla na rovině, budete možná potřebovat speciální algoritmus, který s tím pracuje.
>
![Flat vs Nonflat Geometry Infographic](../../../../translated_images/cs/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Vzdálenosti'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Shluky jsou definovány jejich maticí vzdáleností, tj. vzdálenostmi mezi body. Tato vzdálenost může být měřena různými způsoby. Eukleidovské shluky jsou definovány průměrem hodnot bodů a obsahují 'centroid' neboli středový bod. Vzdálenosti se pak měří k tomuto centroidu. Ne-Eukleidovské vzdálenosti označují 'klustroidy', body nejbližší ostatním bodům. Klustroidy mohou být definovány různými způsoby.
> 
> 🎓 ['Omezené'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Omezené shlukování](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) zavádí do této metody učení bez učitele „polodohledové“ učení. Vztahy mezi body jsou označeny jako 'nelze spojit' nebo 'musí být spojeny', takže jsou do datasetu vynucena určitá pravidla.
>
> Příklad: Pokud je algoritmus spuštěn na neoznačených či poloznačených datech bez omezení, může produkovat shluky nízké kvality. Ve výše uvedeném příkladu by shluky mohly skupovat 'kulaté hudební věci' a 'čtvercové hudební věci' a 'trojúhelníkové věci' a 'sušenky'. Pokud jsou ale dána omezení či pravidla („položka musí být z plastu“, „položka musí umět vydávat hudbu“), pomáhá to algoritmu dělat lepší volby.
> 
> 🎓 'Hustota'
> 
> Data, která jsou 'hluková', jsou považována za 'hustá'. Vzdálenosti mezi body v každém shluku mohou být po zkoumání více či méně husté, tedy „přeplněné“, a proto je třeba data analyzovat vhodnou metodou shlukování. [Tento článek](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) demonstruje rozdíl mezi použitím K-Means shlukování versus HDBSCAN algoritmy k prozkoumání hlučného datasetu s nerovnoměrnou hustotou shluků.

## Algoritmy shlukování

Existuje přes 100 algoritmů shlukování a jejich použití závisí na povaze dostupných dat. Pojďme probrat některé z hlavních:

- **Hierarchické shlukování**. Pokud je objekt klasifikován podle blízkosti k sousednímu objektu spíše než k vzdálenějšímu, shluky jsou tvořeny podle vzdáleností svých členů k ostatním objektům. Agglomerativní shlukování ve Scikit-learn je hierarchické.

   ![Hierarchical clustering Infographic](../../../../translated_images/cs/hierarchical.bf59403aa43c8c47.webp)
   > Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroidové shlukování**. Tento populární algoritmus vyžaduje volbu 'k', tedy počtu shluků, které se mají vytvořit, poté algoritmus určí středový bod shluku a shromažďuje data kolem tohoto bodu. [K-means shlukování](https://wikipedia.org/wiki/K-means_clustering) je oblíbenou variantou centroidového shlukování. Střed se určuje podle nejbližšího průměru, což vysvětluje název. Čtvercová vzdálenost od shluku je minimalizována.

   ![Centroid clustering Infographic](../../../../translated_images/cs/centroid.097fde836cf6c918.webp)
   > Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Shlukování založené na rozdělení**. Zakládá se na statistickém modelování, kdy se určuje pravděpodobnost, že datový bod patří do určitého shluku, a tomuto shluku se přiřadí. Metody Gaussovské směsi patří do tohoto typu.

- **Shlukování založené na hustotě**. Datové body jsou přiřazovány ke shlukům podle hustoty, tj. podle jejich vzájemného seskupení. Body vzdálené od skupiny jsou považovány za odlehlé hodnoty nebo šum. Do tohoto typu patří DBSCAN, Mean-shift a OPTICS.

- **Mřížkové shlukování**. Pro vícerozměrné datasety se vytvoří mřížka a data se rozdělí mezi buňky mřížky, čímž vznikají shluky.

## Cvičení – seskupte svá data

Shlukování jako technika je velmi usnadněno vhodnou vizualizací, tak pojďme začít vizualizováním našich hudebních dat. Toto cvičení nám pomůže rozhodnout, kterou metodu shlukování by bylo pro charakter dat nejefektivnější použít.

1. Otevřete soubor [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) v této složce.

1. Importujte balíček `Seaborn` pro kvalitní vizualizaci dat.

    ```python
    !pip install seaborn
    ```

1. Připojte údaje o písních z [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Načtěte dataframe s údaji o písních. Připravte se prozkoumat data importem knihoven a vypsáním dat:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Zkontrolujte prvních několik řádků dat:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Získejte nějaké informace o dataframe, zavoláním `info()`:

    ```python
    df.info()
    ```

   Výstup vypadá takto:

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

1. Dvojitě zkontrolujte hodnoty null, zavoláním `isnull()` a ověřte, že součet je 0:

    ```python
    df.isnull().sum()
    ```

    Vypadá to dobře:

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

1. Popište data:

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

> 🤔 Pokud pracujeme s shlukováním, neřízenou metodou, která nevyžaduje označená data, proč ukazujeme tato data s popisky? Ve fázi průzkumu dat jsou užitečná, ale nejsou nezbytná pro fungování shlukovacích algoritmů. Můžete také jednoduše odstranit záhlaví sloupců a odkazovat se na data podle čísla sloupce.

Podívejte se na obecné hodnoty dat. Všimněte si, že popularita může být '0', což ukazuje písně, které nemají hodnocení. Tyto brzy odstraníme.

1. Použijte barplot k zjištění nejoblíbenějších žánrů:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/cs/popular.9c48d84b3386705f.webp)

✅ Pokud chcete vidět více nejlepších hodnot, změňte horní `[:5]` na větší hodnotu, nebo jej odstraňte pro zobrazení všech.

Poznámka, když je nejvyšší žánr označen jako 'Missing', znamená to, že Spotify ho neklasifikoval, takže se ho zbavme.

1. Zbavte se chybějících dat jejich filtrováním

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Nyní znovu zkontrolujte žánry:

    ![most popular](../../../../translated_images/cs/all-genres.1d56ef06cefbfcd6.webp)

1. Zdaleka dominují tomuto datasetu tři hlavní žánry. Zaměřme se na `afro dancehall`, `afropop` a `nigerian pop`, navíc odfiltrujeme dataset, abychom odstranili cokoli s hodnotou popularity 0 (to znamená, že nebyl v datasetu zařazen do hodnocení popularity a může být pro naše účely považován za šum):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Proveďte rychlý test, zda data nějak zvlášť silně korelují:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/cs/correlation.a9356bb798f5eea5.webp)

    Jediná silná korelace je mezi `energy` a `loudness`, což není příliš překvapivé, protože hlasitá hudba je obvykle docela energická. Jinak jsou korelace relativně slabé. Bude zajímavé sledovat, co z těchto dat vytvoří shlukovací algoritmus.

    > 🎓 Upozorňujeme, že korelace neznamená kauzalitu! Máme důkaz korelace, ale ne důkaz příčinné souvislosti. [Zábavná webová stránka](https://tylervigen.com/spurious-correlations) obsahuje vizualizace, které tento bod zdůrazňují.

Existuje v tomto datasetu nějaká konvergence mezi vnímanou popularitou písně a tanečností? FacetGrid ukazuje, že se vytvářejí soustředné kruhy, a to bez ohledu na žánr. Může to být, že nigerijské chutě se sblíží na určité úrovni tanečnosti pro tento žánr?

✅ Vyzkoušejte různé datové body (energy, loudness, speechiness) a více nebo jiné hudební žánry. Co můžete objevit? Podívejte se do tabulky `df.describe()`, abyste viděli obecné rozložení datových bodů.

### Cvičení - rozložení dat

Jsou tyto tři žánry významně odlišné ve vnímání své tanečnosti, na základě jejich popularity?

1. Prozkoumejte rozložení dat našich tří hlavních žánrů podle popularity a tanečnosti podél os x a y.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Můžete objevit soustředné kruhy kolem obecného bodu konvergence, ukazující rozložení bodů.

    > 🎓 Upozorňujeme, že tento příklad používá graf KDE (Kernel Density Estimate), který reprezentuje data pomocí spojité křivky pravděpodobnostní hustoty. To nám umožňuje interpretovat data při práci s více rozděleními.

    Obecně se tři žánry volně shodují, pokud jde o jejich popularitu a tanečnost. Určení shluků v těchto volně se překrývajících datech bude výzvou:

    ![distribution](../../../../translated_images/cs/distribution.9be11df42356ca95.webp)

1. Vytvořte scatter plot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Scatterplot stejných os ukazuje podobný vzor konvergence

    ![Facetgrid](../../../../translated_images/cs/facetgrid.9b2e65ce707eba1f.webp)

Obecně lze pro shlukování použít scatterploty k zobrazení shluků dat, proto je zvládnutí tohoto typu vizualizace velmi užitečné. V příští lekci použijeme tento filtrovaný dataset a aplikujeme k-means shlukování, abychom objevili skupiny v těchto datech, které se zdají zajímavě překrývat.

---

## 🚀Výzva

Na přípravu na další lekci vytvořte graf o různých shlukovacích algoritmech, které můžete objevit a použít v produkčním prostředí. Jaké problémy se shlukování snaží řešit?

## [Kvíz po lekci](https://ff-quizzes.netlify.app/en/ml/)

## Revize a samostudium

Než použijete shlukovací algoritmy, jak jsme se naučili, je dobré pochopit povahu vašeho datasetu. Více na toto téma si přečtěte [zde](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html).

[Tento užitečný článek](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) vás provede různými způsoby chování shlukovacích algoritmů při různých tvarech dat.

## Zadání

[Prozkoumejte další vizualizace pro shlukování](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Prohlášení o omezení odpovědnosti**:
Tento dokument byl přeložen pomocí AI překladatelské služby [Co-op Translator](https://github.com/Azure/co-op-translator). Přestože usilujeme o co největší přesnost, mějte prosím na paměti, že automatizované překlady mohou obsahovat chyby nebo nepřesnosti. Originální dokument v jeho mateřském jazyce by měl být považován za autoritativní zdroj. Pro kritické informace se doporučuje profesionální lidský překlad. Nejsme odpovědní za jakékoli nedorozumění nebo nesprávné interpretace vzniklé použitím tohoto překladu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->