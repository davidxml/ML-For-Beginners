# Introduktion til klyngedannelse

Klyngedannelse er en type [Unsupervised Learning](https://wikipedia.org/wiki/Unsupervised_learning), som antager, at et datasæt er uden etiketter, eller at dets input ikke er matchet med foruddefinerede output. Det bruger forskellige algoritmer til at sortere gennem uetiketterede data og levere grupperinger i henhold til mønstre, det skelner i dataene.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Klik på billedet ovenfor for en video. Mens du studerer maskinlæring med klyngedannelse, kan du nyde nogle nigerianske Dance Hall-numre - dette er en højt vurderet sang fra 2014 af PSquare.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Introduktion

[Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) er meget nyttig til dataudforskning. Lad os se, om det kan hjælpe med at opdage tendenser og mønstre i den måde, nigerianske publikum forbruger musik på.

✅ Tag et minut til at tænke på anvendelserne af klyngedannelse. I det virkelige liv sker klyngedannelse, når du har en bunke vasketøj og skal sortere dine familiemedlemmers tøj 🧦👕👖🩲. I datavidenskab sker klyngedannelse, når man forsøger at analysere en brugers præferencer eller bestemme karakteristika for et hvilket som helst uetiketteret datasæt. Klyngedannelse hjælper på en måde med at skabe mening i kaos, som en sokkeskuffe.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Klik på billedet ovenfor for en video: MIT's John Guttag introducerer klyngedannelse

I en professionel sammenhæng kan klyngedannelse bruges til at bestemme ting som markedsssegmentation, for eksempel at bestemme, hvilke aldersgrupper der køber hvilke varer. En anden anvendelse kan være anomalidetektion, måske for at opdage svindel i et datasæt af kreditkorttransaktioner. Eller du kan bruge klyngedannelse til at bestemme tumorer i en række medicinske scanninger.

✅ Tænk et minut over, hvordan du måske har mødt klyngedannelse 'i det fri', i en bank-, e-handels- eller forretningssammenhæng.

> 🎓 Interessant nok stammer klyngeanalyse fra antropologi og psykologi i 1930'erne. Kan du forestille dig, hvordan det kunne være blevet brugt?

Alternativt kunne du bruge det til at gruppere søgeresultater - for eksempel efter shoppinglinks, billeder eller anmeldelser. Klyngedannelse er nyttig, når du har et stort datasæt, du ønsker at reducere og foretage en mere detaljeret analyse på, så teknikken kan bruges til at lære om data, før andre modeller konstrueres.

✅ Når dine data er organiseret i klynger, tildeler du dem et klynge-ID, og denne teknik kan være nyttig, når man bevarer et datasæts privatliv; du kan i stedet henvise til et datapunkt ved dets klynge-ID i stedet for ved mere afslørende identificerende data. Kan du komme i tanke om andre grunde til, at du vil henvise til et klynge-ID frem for andre elementer i klyngen for at identificere det?

Dyk dybere ned i din forståelse af klyngedannelsesteknikker i denne [Learn-modul](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

## Kom godt i gang med klyngedannelse

[Scikit-learn tilbyder et stort udvalg](https://scikit-learn.org/stable/modules/clustering.html) af metoder til at udføre klyngedannelse. Den type, du vælger, afhænger af din brugssag. Ifølge dokumentationen har hver metode forskellige fordele. Her er en forenklet tabel over metoderne, som Scikit-learn understøtter, og deres passende anvendelsestilfælde:

| Metodenavn                   | Anvendelsestilfælde                                                    |
| :--------------------------- | :-------------------------------------------------------------------- |
| K-Means                      | generelt formål, induktiv                                             |
| Affinity propagation         | mange, ujævne klynger, induktiv                                       |
| Mean-shift                   | mange, ujævne klynger, induktiv                                       |
| Spektral klyngedannelse      | få, lige klynger, transduktiv                                         |
| Ward hierarkisk klyngedannelse | mange, begrænsede klynger, transduktiv                              |
| Agglomerativ klyngedannelse | mange, begrænsede, ikke-euklidiske afstande, transduktiv              |
| DBSCAN                       | ikke-flad geometri, ujævne klynger, transduktiv                      |
| OPTICS                       | ikke-flad geometri, ujævne klynger med variabel tæthed, transduktiv  |
| Gaussian mixtures            | flad geometri, induktiv                                               |
| BIRCH                        | stort datasæt med udliggere, induktiv                                 |

> 🎓 Hvordan vi skaber klynger har meget at gøre med, hvordan vi samler datapunkterne i grupper. Lad os pakke noget jargon ud:
>
> 🎓 ['Transduktiv' vs. 'induktiv'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktiv inferens er afledt af observerede træningstilfælde, der kortlægger til specifikke testtilfælde. Induktiv inferens er afledt af træningstilfælde, der kortlægger til generelle regler, som først derefter anvendes på testtilfælde.
> 
> Et eksempel: Forestil dig, at du har et datasæt, der kun er delvist mærket. Nogle ting er 'plader', nogle 'cd'er', og nogle er blanke. Din opgave er at give etiketter til de blanke. Hvis du vælger en induktiv tilgang, vil du træne en model, der leder efter 'plader' og 'cd'er', og anvende de etiketter på dine uetiketterede data. Denne tilgang vil have problemer med at klassificere ting, der faktisk er 'kassetter'. En transduktiv tilgang håndterer derimod disse ukendte data mere effektivt, da den arbejder på at gruppere lignende objekter sammen og derefter anvender en etiket til en gruppe. I dette tilfælde kan klynger afspejle 'runde musikgenstande' og 'firkantede musikgenstande'.
> 
> 🎓 ['Ikke-flad' vs. 'flad' geometri](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Afledt af matematisk terminologi henviser ikke-flad vs. flad geometri til måling af afstande mellem punkter ved enten 'flad' ([Euclidisk](https://wikipedia.org/wiki/Euclidean_geometry)) eller 'ikke-flad' (ikke-Euclidisk) geometriske metoder.
>
> 'Flad' i denne sammenhæng refererer til Euclidisk geometri (dele af hvilken læres som 'plan'-geometri), og ikke-flad refererer til ikke-Euclidisk geometri. Hvad har geometri at gøre med maskinlæring? Nå, som to felter der er rodfæstet i matematik, må der være en fælles måde at måle afstande mellem punkter i klynger på, og det kan gøres på en 'flad' eller 'ikke-flad' måde, afhængigt af dataenes natur. [Euclidiske afstande](https://wikipedia.org/wiki/Euclidean_distance) måles som længden af en linjesegment mellem to punkter. [Ikke-Euclidiske afstande](https://wikipedia.org/wiki/Non-Euclidean_geometry) måles langs en kurve. Hvis dine data, visualiseret, ser ud til ikke at eksistere på et plan, kan det være nødvendigt at bruge en specialiseret algoritme til at håndtere det.
>
> ![Flat vs Nonflat Geometry Infographic](../../../../translated_images/da/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infografik af [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Afstande'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Klynger defineres ved deres afstandsmatrix, f.eks. afstandene mellem punkter. Denne afstand kan måles på flere måder. Euclidiske klynger defineres ved gennemsnittet af punkternes værdier, og indeholder et 'centroid' eller midtpunkt. Afstande måles således som afstanden til dette centroid. Ikke-euclidiske afstande refererer til 'clustroids', punktet tættest på andre punkter. Clustroids kan i øvrigt defineres på forskellige måder.
> 
> 🎓 ['Begrænset'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Begrænset klyngedannelse](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introducerer 'semi-supervised' læring i denne unsupervised metode. Relationerne mellem punkter markeres som 'cannot link' eller 'must-link', så nogle regler påtvinges datasættet.
>
> Et eksempel: Hvis en algoritme sættes fri på et parti uetiketteret eller semi-etiketteret data, kan de klynger, den producerer, være af dårlig kvalitet. I eksemplet ovenfor kunne klyngerne gruppere 'runde musikgenstande' og 'firkantede musikgenstande' og 'trekantede ting' og 'kager'. Hvis den får nogle begrænsninger eller regler at følge ("genstanden skal være lavet af plastik", "genstanden skal kunne producere musik"), kan dette hjælpe med at 'begrænse' algoritmen til at træffe bedre valg.
> 
> 🎓 'Tæthed'
> 
> Data, der er 'støjende', betragtes som 'tætte'. Afstandene mellem punkterne i hver af dets klynger kan ved undersøgelse vise sig at være mere eller mindre tætte eller 'tætpakkede', og derfor skal disse data analyseres med den passende klyngedannelsesmetode. [Denne artikel](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) demonstrerer forskellen mellem at bruge K-Means klyngedannelse vs. HDBSCAN algoritmer til at udforske et støjende datasæt med ujævn klyngedensitet.

## Klyngedannelsesalgoritmer

Der findes over 100 klyngedannelsesalgoritmer, og deres anvendelse afhænger af dataenes natur. Lad os diskutere nogle af de vigtigste:

- **Hierarkisk klyngedannelse**. Hvis et objekt klassificeres ud fra sin nærhed til et nærliggende objekt, snarere end til et længere væk, dannes klynger baseret på deres medlemmers afstand til og fra andre objekter. Scikit-learns agglomerative klyngedannelse er hierarkisk.

   ![Hierarchical clustering Infographic](../../../../translated_images/da/hierarchical.bf59403aa43c8c47.webp)
   > Infografik af [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroid klyngedannelse**. Denne populære algoritme kræver valget af 'k', eller antallet af klynger, der skal dannes, hvorefter algoritmen bestemmer klyngens midtpunkt og samler data omkring dette punkt. [K-means klyngedannelse](https://wikipedia.org/wiki/K-means_clustering) er en populær version af centroidklyngedannelse. Midtpunktet bestemmes af det nærmeste gennemsnit, deraf navnet. Den kvadrerede afstand fra klyngen minimeres.

   ![Centroid clustering Infographic](../../../../translated_images/da/centroid.097fde836cf6c918.webp)
   > Infografik af [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Fordelingsbaseret klyngedannelse**. Baseret på statistisk modellering fokuserer fordelingsbaseret klyngedannelse på at bestemme sandsynligheden for, at et datapunkt tilhører en klynge, og tildeler det derpå. Gaussiske blandingsmetoder hører til denne type.

- **Tæthetsbaseret klyngedannelse**. Datapunkter tildeles klynger baseret på deres tæthed eller deres gruppering omkring hinanden. Datapunkter langt fra gruppen betragtes som udliggere eller støj. DBSCAN, Mean-shift og OPTICS hører til denne type klyngedannelse.

- **Gitterbaseret klyngedannelse**. For multidimensionale datasæt oprettes et gitter, og data deles imellem gitterets celler, hvilket danner klynger.

## Øvelse - klyng dine data

Klyngedannelse som teknik hjælpes meget af korrekt visualisering, så lad os komme i gang med at visualisere vores musikdata. Denne øvelse hjælper os med at beslutte, hvilken metode til klyngedannelse vi mest effektivt skal bruge til arten af disse data.

1. Åbn filen [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) i denne mappe.

1. Importer `Seaborn` pakken for god datavisualisering.

    ```python
    !pip install seaborn
    ```


1. Vedhæng sangdataene fra [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Indlæs en dataframe med nogle data om sangene. Gør dig klar til at udforske disse data ved at importere bibliotekerne og udskrive dataene:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```


    Tjek de første par linjer med data:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Få nogle oplysninger om dataframe, ved at kalde `info()`:

    ```python
    df.info()
    ```

   Outputtet ser således ud:

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

1. Dobbelttjek for null-værdier ved at kalde `isnull()` og kontrollere summen for at være 0:

    ```python
    df.isnull().sum()
    ```

    Ser godt ud:

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

1. Beskriv dataene:

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

> 🤔 Hvis vi arbejder med clustering, en usuperviseret metode, der ikke kræver mærkede data, hvorfor viser vi så disse data med labels? I dataundersøgelsesfasen er de nyttige, men de er ikke nødvendige for, at clustering-algoritmerne kan fungere. Du kunne ligeså godt fjerne kolonneoverskrifterne og referere til dataene ved kolonnenummer. 

Se på de generelle værdier for dataene. Bemærk, at popularitet kan være '0', hvilket viser sange, der ikke har nogen rangering. Lad os fjerne dem snart.

1. Brug et søjlediagram til at finde ud af de mest populære genrer:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![mest populære](../../../../translated_images/da/popular.9c48d84b3386705f.webp)

✅ Hvis du vil se flere topværdier, kan du ændre toppen `[:5]` til en større værdi eller fjerne den for at se alle.

Bemærk, når den topgenre, der beskrives som 'Missing', betyder det, at Spotify ikke har klassificeret den, så lad os fjerne den.

1. Fjern manglende data ved at filtrere det ud

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Tjek genrerne igen:

    ![mest populære](../../../../translated_images/da/all-genres.1d56ef06cefbfcd6.webp)

1. De tre øverste genrer dominerer dette datasæt langt. Lad os koncentrere os om `afro dancehall`, `afropop` og `nigerian pop`, og yderligere filtrere datasættet for at fjerne alt med en popularitet på 0 (hvilket betyder, at det ikke blev klassificeret med popularitet i datasættet og kan betragtes som støj for vores formål):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Lav en hurtig test for at se, om dataene korrelerer på en særlig stærk måde:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![korrelationer](../../../../translated_images/da/correlation.a9356bb798f5eea5.webp)

    Den eneste stærke korrelation er mellem `energy` og `loudness`, hvilket ikke er så overraskende, da høj musik normalt er ret energisk. Ellers er korrelationerne relativt svage. Det bliver interessant at se, hvad en clustering-algoritme kan udrette med disse data.

    > 🎓 Bemærk, at korrelation ikke antyder årsagssammenhæng! Vi har bevis for korrelation, men ikke bevis for årsagssammenhæng. En [underholdende hjemmeside](https://tylervigen.com/spurious-correlations) viser nogle visualiseringer, der understreger dette punkt.

Er der nogen konvergens i datasættet omkring en sangs opfattede popularitet og dansbarhed? Et FacetGrid viser, at der er koncentriske cirkler, der går i spænd, uanset genre. Kan det være, at Nigerianske smag samles om et bestemt niveau af dansbarhed for denne genre?  

✅ Prøv forskellige datapunkter (energi, lydstyrke, talthed) og flere eller forskellige musikgenrer. Hvad kan du opdage? Tag et kig på `df.describe()` tabellen for at se den generelle spredning af datapunkterne.

### Øvelse - dataspredning

Er de tre genrer signifikant forskellige i opfattelsen af deres dansbarhed, baseret på deres popularitet?

1. Undersøg vores tre topgenre dataspredning for popularitet og dansbarhed langs en given x- og y-akse.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Du kan opdage koncentriske cirkler omkring et generelt konvergenspunkt, der viser fordelingen af punkter.

    > 🎓 Bemærk, at dette eksempel bruger en KDE (Kernel Density Estimate) graf, der repræsenterer data ved hjælp af en kontinuerlig sandsynlighedstæthetskurve. Dette gør det muligt at fortolke data, når man arbejder med flere fordelinger.

    Generelt hænger de tre genrer løst sammen med hensyn til popularitet og dansbarhed. At bestemme klynger i disse løst justerede data bliver en udfordring:

    ![fordeling](../../../../translated_images/da/distribution.9be11df42356ca95.webp)

1. Lav et spredningsplot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Et spredningsdiagram af de samme akser viser et lignende mønster af konvergens

    ![Facetgrid](../../../../translated_images/da/facetgrid.9b2e65ce707eba1f.webp)

Generelt kan du ved clustering bruge spredningsdiagrammer til at vise klynger af data, så det at mestre denne visualiseringstype er meget nyttigt. I den næste lektion vil vi tage disse filtrerede data og bruge k-means clustering til at opdage grupper i dataene, som synes at overlappe på interessante måder.

---

## 🚀Udfordring

Som forberedelse til den næste lektion, lav en oversigt over de forskellige clustering-algoritmer, du kan støde på og bruge i et produktionsmiljø. Hvilke slags problemer forsøger clustering at løse?

## [Quiz efter lektion](https://ff-quizzes.netlify.app/en/ml/)

## Review & Selvstudium

Før du anvender clustering-algoritmer, som vi har lært, er det en god ide at forstå karakteren af dit datasæt. Læs mere om dette emne [her](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Denne hjælpsomme artikel](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) guider dig gennem de forskellige måder, som forskellige clustering-algoritmer opfører sig på, givet forskellige datatyper.

## Opgave

[Forskning andre visualiseringer for clustering](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfraskrivelse**:
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os intet ansvar for misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->