# Introductie tot clustering

Clustering is een type [Ongecontroleerd Leren](https://wikipedia.org/wiki/Unsupervised_learning) dat ervan uitgaat dat een dataset niet gelabeld is of dat de inputs niet gekoppeld zijn aan vooraf gedefinieerde outputs. Het gebruikt verschillende algoritmen om ongelabelde data te sorteren en groepen te maken op basis van patronen die het in de data waarneemt.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Klik op de afbeelding hierboven voor een video. Terwijl je machine learning met clustering bestudeert, geniet van wat Nigeriaanse Dance Hall tracks - dit is een zeer gewaardeerd nummer uit 2014 van PSquare.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Introductie

[Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) is erg nuttig voor data verkenning. Laten we zien of het kan helpen bij het ontdekken van trends en patronen in de manier waarop Nigeriaanse publieken muziek consumeren.

✅ Neem een minuut om na te denken over de toepassingen van clustering. In het echte leven gebeurt clustering altijd wanneer je een stapel wasgoed hebt en de kleren van je familieleden moet sorteren 🧦👕👖🩲. In datawetenschap gebeurt clustering wanneer je probeert de voorkeuren van een gebruiker te analyseren, of de kenmerken van een ongelabelde dataset te bepalen. Clustering helpt op een bepaalde manier om chaos te begrijpen, zoals een sokkenla.

[![Introductie tot ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introductie tot Clustering")

> 🎥 Klik op de afbeelding hierboven voor een video: John Guttag van MIT introduceert clustering

In een professionele omgeving kan clustering gebruikt worden om dingen te bepalen zoals marksegmentatie, vaststellen welke leeftijdsgroepen welke artikelen kopen, bijvoorbeeld. Een andere toepassing is anomaliedetectie, bijvoorbeeld om fraude te detecteren in een dataset van creditcardtransacties. Of je zou clustering kunnen gebruiken om tumoren te bepalen in een reeks medische scans.

✅ Denk een minuut na over hoe je clustering wellicht bent tegengekomen 'in het wild', in een banking-, e-commerce- of zakelijke omgeving.

> 🎓 Interessant genoeg is clusteranalyse ontstaan in de vakgebieden Antropologie en Psychologie in de jaren 1930. Kun je je voorstellen hoe het toen werd gebruikt?

Alternatief kun je het gebruiken voor het groeperen van zoekresultaten - bijvoorbeeld op winkellinks, afbeeldingen of recensies. Clustering is nuttig wanneer je een grote dataset hebt die je wilt verkleinen en waarop je meer gedetailleerde analyse wilt uitvoeren, zodat de techniek gebruikt kan worden om over data te leren voordat andere modellen worden geconstrueerd.

✅ Zodra je data is georganiseerd in clusters, ken je elk cluster een ID toe, en deze techniek kan handig zijn bij het waarborgen van de privacy van een dataset; je kunt in plaats daarvan naar een datapunt verwijzen via zijn cluster-ID in plaats van met meer onthullende identificeerbare data. Kun je andere redenen bedenken waarom je naar een cluster-ID zou verwijzen in plaats van naar andere elementen van het cluster om het te identificeren?

Verdiep je in clusteringtechnieken in deze [Leer module](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

## Aan de slag met clustering

[Scikit-learn biedt een grote reeks](https://scikit-learn.org/stable/modules/clustering.html) methoden om clustering uit te voeren. De keuze hangt af van je gebruikssituatie. Volgens de documentatie heeft elke methode verschillende voordelen. Hier is een vereenvoudigde tabel van de door Scikit-learn ondersteunde methoden en hun toepassingsgebieden:

| Naam van methode             | Gebruik                                                                 |
| :--------------------------- | :--------------------------------------------------------------------- |
| K-Means                      | algemeen doel, inductief                                                |
| Affiniteit propagatie        | veel, ongelijke clusters, inductief                                    |
| Mean-shift                   | veel, ongelijke clusters, inductief                                    |
| Spectrale clustering         | weinig, gelijke clusters, transductief                                 |
| Ward hiërarchische clustering| veel, beperkende clusters, transductief                               |
| Agglomeratieve clustering    | veel, beperkend, niet-Euclidische afstanden, transductief              |
| DBSCAN                       | niet-vlakke geometrie, ongelijke clusters, transductief               |
| OPTICS                       | niet-vlakke geometrie, ongelijke clusters met variabele dichtheid, transductief |
| Gaussiaanse mengsels         | vlakke geometrie, inductief                                            |
| BIRCH                        | grote dataset met uitschieters, inductief                              |

> 🎓 Hoe we clusters maken heeft veel te maken met hoe we de datapunten in groepen verzamelen. Laten we wat vocabulaire uitleggen:
>
> 🎓 ['Transductief' vs. 'inductief'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transductieve inferentie wordt afgeleid van waargenomen trainingsgevallen die map naar specifieke testgevallen. Inductieve inferentie wordt afgeleid van trainingsgevallen die map naar algemene regels die pas dan worden toegepast op testgevallen.
> 
> Een voorbeeld: Stel je hebt een dataset die slechts gedeeltelijk gelabeld is. Sommige dingen zijn 'platen', andere 'cd's', en sommige zijn leeg. Jouw taak is om labels te geven aan de lege plaatsen. Kies je een inductieve aanpak, dan train je een model op 'platen' en 'cd's' en pas je die labels toe op ongelabelde data. Deze aanpak zal moeite hebben met het classificeren van dingen die eigenlijk 'cassettes' zijn. Een transductieve aanpak daarentegen gaat dit onbekende data effectiever aan omdat het werkt om gelijke items te groeperen en een label toe te passen op de groep. In dit geval kunnen clusters 'ronde muziekdingen' en 'vierkante muziekdingen' reflecteren.
> 
> 🎓 ['Niet-vlak' vs. 'vlakke' geometrie](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Afgeleid van wiskundige terminologie verwijst niet-vlak vs. vlakke geometrie naar de maat van afstanden tussen punten door middel van 'vlakke' ([Euclidische](https://wikipedia.org/wiki/Euclidean_geometry)) of 'niet-vlakke' (niet-Euclidische) geometrische methoden.
>
> 'Vlak' in deze context verwijst naar Euclidische geometrie (onderdelen daarvan worden onderwezen als 'vlakke' geometrie) en niet-vlak verwijst naar niet-Euclidische geometrie. Wat heeft geometrie te maken met machine learning? Nou, aangezien het twee vakgebieden zijn die geworteld zijn in wiskunde, moet er een gemeenschappelijke manier zijn om afstanden tussen punten in clusters te meten, en dat kan op een 'vlakke' of 'niet-vlakke' manier, afhankelijk van de aard van de data. [Euclidische afstanden](https://wikipedia.org/wiki/Euclidean_distance) worden gemeten als de lengte van een lijnstuk tussen twee punten. [Niet-Euclidische afstanden](https://wikipedia.org/wiki/Non-Euclidean_geometry) worden gemeten langs een kromme. Als jouw data, gevisualiseerd, lijkt te bestaan op iets anders dan een vlak, heb je mogelijk een gespecialiseerd algoritme nodig om het aan te kunnen.
>
![Flat vs Nonflat Geometry Infographic](../../../../translated_images/nl/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infographic door [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Afstanden'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Clusters worden gedefinieerd door hun afstandsmatrix, bijvoorbeeld de afstanden tussen punten. Deze afstand kan op verschillende manieren worden gemeten. Euclidische clusters worden bepaald door het gemiddelde van de puntwaarden en bevatten een 'centroid' of middelpunt. Afstanden worden dan gemeten als de afstand tot dat centroid. Niet-Euclidische afstanden verwijzen naar 'clustroids', het punt dat het dichtst bij andere punten ligt. Clustroids kunnen op verschillende manieren worden gedefinieerd.
> 
> 🎓 ['Beperkt'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Beperkte clustering](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introduceert 'semi-gefundeerd' leren in deze ongecontroleerde methode. De relaties tussen punten worden gemarkeerd als 'mag niet linken' of 'moet linken' zodat regels op de dataset worden afgedwongen.
>
> Een voorbeeld: Als een algoritme vrij op een batch ongelabelde of semi-gelabelde data wordt losgelaten, kunnen de geproduceerde clusters van slechte kwaliteit zijn. In het bovenstaande voorbeeld kunnen de clusters 'ronde muziekdingen', 'vierkante muziekdingen', 'driehoekige dingen' en 'koekjes' groeperen. Als je enkele beperkingen of regels toevoegt ("het item moet van plastic zijn", "het item moet muziek kunnen maken") kan dit helpen het algoritme te 'beperken' en betere keuzes te bepalen.
> 
> 🎓 'Dichtheid'
> 
> Data die 'ruis' bevat wordt als 'dicht' beschouwd. De afstanden tussen punten in elk van de clusters kunnen na onderzoek meer of minder dicht zijn, of 'druk' en daarom moet deze data met de juiste clusteringmethode worden geanalyseerd. [Dit artikel](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) toont het verschil aan tussen het gebruik van K-Means clustering versus HDBSCAN-algoritmen om een lawaaierige dataset met ongelijke clusterdichtheid te onderzoeken.

## Clusteringsalgoritmen

Er zijn meer dan 100 clusteringalgoritmen en hun gebruik hangt af van de aard van de data. Laten we enkele belangrijke bespreken:

- **Hiërarchische clustering**. Als een object wordt geclassificeerd op basis van zijn nabijheid tot een naburig object, in plaats van tot een verder weg liggend object, worden clusters gevormd op basis van hun afstand tot en van andere objecten. Agglomeratieve clustering van Scikit-learn is hiërarchisch.

   ![Hierarchische clustering Infographic](../../../../translated_images/nl/hierarchical.bf59403aa43c8c47.webp)
   > Infographic door [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroid clustering**. Dit populaire algoritme vereist het kiezen van 'k', of het aantal clusters dat gevormd moet worden, waarna het algoritme het middelpunt van een cluster bepaalt en data daaromheen verzamelt. [K-means clustering](https://wikipedia.org/wiki/K-means_clustering) is een populaire versie van centroid clustering. Het centrum wordt bepaald door het dichtstbijzijnde gemiddelde, vandaar de naam. De kwadratische afstand tot het cluster wordt geminimaliseerd.

   ![Centroid clustering Infographic](../../../../translated_images/nl/centroid.097fde836cf6c918.webp)
   > Infographic door [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Op distributie gebaseerde clustering**. Gebaseerd op statistische modellering, richt distributiegebaseerde clustering zich op het bepalen van de waarschijnlijkheid dat een datapunt tot een cluster behoort, en wijst het dienovereenkomstig toe. Gaussiaanse mengsels behoren tot dit type.

- **Dichtheidsgebaseerde clustering**. Datapunten worden toegewezen aan clusters op basis van hun dichtheid, of hun groepering rondom elkaar. Datapunten die ver van de groep liggen, worden beschouwd als uitschieters of ruis. DBSCAN, Mean-shift en OPTICS behoren tot dit type clustering.

- **Rastersgebaseerde clustering**. Voor multidimensionale datasets wordt een raster gemaakt en wordt de data verdeeld over de cellen van het raster, waardoor clusters ontstaan.

## Oefening - cluster je data

Clustering als techniek wordt sterk ondersteund door goede visualisatie, dus laten we beginnen met het visualiseren van onze muziekdata. Deze oefening helpt ons te beslissen welke clusteringmethode we het beste kunnen gebruiken voor de aard van deze data.

1. Open het [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) bestand in deze map.

1. Importeer het `Seaborn`-pakket voor goede datavisualisatie.

    ```python
    !pip install seaborn
    ```

1. Voeg de songdata toe van [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Laad een dataframe met wat gegevens over de nummers. Maak je klaar om deze data te verkennen door de bibliotheken te importeren en de data uit te dumpen:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Bekijk de eerste paar datalijnen:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Krijg wat informatie over de dataframe, door `info()` aan te roepen:

    ```python
    df.info()
    ```

   De output ziet er als volgt uit:

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

1. Controleer dubbel op null waarden, door `isnull()` aan te roepen en te verifiëren dat de som 0 is:

    ```python
    df.isnull().sum()
    ```

    Ziet er goed uit:

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

1. Beschrijf de data:

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

> 🤔 Als we werken met clustering, een onbewaakte methode die geen gelabelde data vereist, waarom tonen we deze data dan met labels? In de data-exploratiefase zijn ze handig, maar ze zijn niet noodzakelijk voor de clustering-algoritmes om te werken. Je zou net zo goed de kolomkoppen kunnen verwijderen en naar de data verwijzen op kolomnummer.

Bekijk de algemene waarden van de data. Let op dat populariteit '0' kan zijn, wat nummers toont die geen ranking hebben. Laten we die snel verwijderen.

1. Gebruik een staafdiagram om de meest populaire genres te vinden:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![meest populair](../../../../translated_images/nl/popular.9c48d84b3386705f.webp)

✅ Als je meer topwaarden wilt zien, verander dan de top `[:5]` in een hogere waarde, of verwijder het om alles te zien.

Let op, wanneer het topgenre wordt beschreven als 'Missing', betekent dit dat Spotify het niet heeft geclassificeerd, dus laten we het verwijderen.

1. Verwijder missende data door deze eruit te filteren

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Controleer nu opnieuw de genres:

    ![meest populair](../../../../translated_images/nl/all-genres.1d56ef06cefbfcd6.webp)

1. Bij verreweg domineren de top drie genres deze dataset. Laten we ons concentreren op `afro dancehall`, `afropop` en `nigerian pop`, filter daarnaast de dataset om alles met een populariteitswaarde van 0 te verwijderen (wat betekent dat het niet geclassificeerd werd met een populariteit in de dataset en kan worden beschouwd als ruis voor onze doeleinden):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Doe een snelle test om te zien of de data op een bijzonder sterke manier correleert:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlaties](../../../../translated_images/nl/correlation.a9356bb798f5eea5.webp)

    De enige sterke correlatie is tussen `energy` en `loudness`, wat niet zo verrassend is, aangezien harde muziek meestal vrij energiek is. Verder zijn de correlaties relatief zwak. Het zal interessant zijn om te zien wat een clustering-algoritme van deze data kan maken.

    > 🎓 Let op dat correlatie geen causaliteit impliceert! We hebben bewijs van correlatie maar geen bewijs van causaliteit. Een [grappige website](https://tylervigen.com/spurious-correlations) heeft enkele visuals die dit punt benadrukken.

Is er enige convergentie in deze dataset rond de waargenomen populariteit en dansbaarheid van een nummer? Een FacetGrid laat zien dat er concentrische cirkels zijn die overeenkomen, ongeacht het genre. Zou het kunnen dat Nigeriaanse smaken convergeren op een bepaald niveau van dansbaarheid voor dit genre?

✅ Probeer verschillende datapunten (energy, loudness, speechiness) en meer of andere muziekgenres. Wat kun je ontdekken? Bekijk de `df.describe()` tabel om de algemene spreiding van de datapunten te zien.

### Oefening - datadistributie

Zijn deze drie genres significant verschillend in de perceptie van hun dansbaarheid, gebaseerd op hun populariteit?

1. Onderzoek de datadistributie van onze top drie genres voor populariteit en dansbaarheid langs gegeven x- en y-assen.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Je kunt concentrische cirkels ontdekken rond een algemeen convergentiepunt, die de spreiding van punten laten zien.

    > 🎓 Let op dat dit voorbeeld een KDE (Kernel Density Estimate) grafiek gebruikt die de data vertegenwoordigt met een continue kansdichtheidscurve. Dit stelt ons in staat data te interpreteren wanneer we met meerdere distributies werken.

    Over het algemeen lijnen de drie genres vrij losjes uit in termen van hun populariteit en dansbaarheid. Het bepalen van clusters in deze losjes-uitgelijnde data zal een uitdaging zijn:

    ![distributie](../../../../translated_images/nl/distribution.9be11df42356ca95.webp)

1. Maak een scatterplot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Een scatterplot van dezelfde assen toont een vergelijkbaar patroon van convergentie

    ![Facetgrid](../../../../translated_images/nl/facetgrid.9b2e65ce707eba1f.webp)

Over het algemeen kun je voor clustering scatterplots gebruiken om clusters van data te tonen, dus het beheersen van dit type visualisatie is erg nuttig. In de volgende les zullen we deze gefilterde data nemen en k-means clustering gebruiken om groepen in deze data te ontdekken die op interessante manieren overlappen.

---

## 🚀Uitdaging

Ter voorbereiding op de volgende les, maak een diagram over de verschillende clustering-algoritmes die je kunt ontdekken en gebruiken in een productieomgeving. Welke soorten problemen probeert clustering op te lossen?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Voordat je clustering-algoritmen toepast, is het, zoals we hebben geleerd, een goed idee om de aard van je dataset te begrijpen. Lees meer over dit onderwerp [hier](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Dit nuttige artikel](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) leidt je door de verschillende manieren waarop diverse clustering-algoritmes zich gedragen, gegeven verschillende datavormen.

## Opdracht

[Onderzoek andere visualisaties voor clustering](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:
Dit document is vertaald met behulp van de AI vertaaldienst [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u er rekening mee te houden dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in de oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor kritieke informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->