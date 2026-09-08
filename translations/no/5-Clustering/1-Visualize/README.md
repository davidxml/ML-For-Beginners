# Innføring i klynging

Klynging er en type [Uovervåket læring](https://wikipedia.org/wiki/Unsupervised_learning) som forutsetter at et datasett er umerket eller at dets innganger ikke er knyttet til forhåndsdefinerte utganger. Den bruker ulike algoritmer for å sortere gjennom umerkede data og gi grupperinger i henhold til mønstre den skiller ut i dataene.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Klikk på bildet over for en video. Mens du studerer maskinlæring med klynging, kan du nyte noen nigerianske Dance Hall-spor - dette er en høyt vurdert sang fra 2014 av PSquare.

## [Før-forelesningsquiz](https://ff-quizzes.netlify.app/en/ml/)

### Innledning

[Klynging](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) er veldig nyttig for datautforskning. La oss se om det kan hjelpe med å oppdage trender og mønstre i hvordan nigerianske publikum konsumerer musikk.

✅ Ta et minutt for å tenke på bruken av klynging. I det virkelige liv skjer klynging når du har en haug med skittentøy og må sortere familiemedlemmenes klær 🧦👕👖🩲. I datavitenskap skjer klynging når man prøver å analysere en brukers preferanser, eller bestemme egenskapene til et hvilket som helst umerket datasett. Klynging, på en måte, hjelper med å gi mening til kaos, som en sokkeskuff.

[![Innledning til ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Innledning til klynging")

> 🎥 Klikk på bildet over for en video: MITs John Guttag introduserer klynging

I en profesjonell setting kan klynging brukes til å bestemme ting som markedssegmentering, for eksempel å finne ut hvilke aldersgrupper som kjøper hvilke varer. En annen bruk kan være anomalioppdagelse, kanskje for å oppdage svindel i et datasett med kredittkorttransaksjoner. Eller du kan bruke klynging for å finne svulster i en rekke medisinske skanninger.

✅ Tenk et minutt over hvordan du kan ha møtt klynging 'i det fri', i bank-, e-handel- eller forretningssammenheng.

> 🎓 Interessant nok oppstod klynganalyse innenfor antropologi og psykologi på 1930-tallet. Kan du forestille deg hvordan det kan ha blitt brukt?

Alternativt kan du bruke det for å gruppere søkeresultater - etter handlelenker, bilder eller anmeldelser, for eksempel. Klynging er nyttig når du har et stort datasett som du ønsker å redusere og på hvilket du ønsker å utføre mer granulær analyse, så teknikken kan brukes for å lære om data før andre modeller bygges.

✅ Når dataene dine er organisert i klynger, tildeler du det en klynge-ID, og denne teknikken kan være nyttig for å bevare datamengdens personvern; du kan i stedet referere til et datapunkt ved dets klynge-ID, i stedet for ved mer avslørende identifiserbare data. Kan du tenke på andre grunner til at du vil referere til en klynge-ID fremfor andre elementer i klyngen for å identifisere den?

Fordyp din forståelse av klyngingsteknikker i denne [Lær-modulen](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)
## Komme i gang med klynging

[Scikit-learn tilbyr et stort utvalg](https://scikit-learn.org/stable/modules/clustering.html) av metoder for å utføre klynging. Typen du velger avhenger av ditt brukstilfelle. Ifølge dokumentasjonen har hver metode ulike fordeler. Her er en forenklet tabell over metodene som støttes av Scikit-learn og deres passende bruksområder:

| Metodenavn                   | Bruksområde                                                           |
| :--------------------------- | :-------------------------------------------------------------------- |
| K-Means                      | generell anvendelse, induktiv                                         |
| Affinity propagation         | mange, ujevne klynger, induktiv                                       |
| Mean-shift                   | mange, ujevne klynger, induktiv                                       |
| Spectral clustering          | få, jevne klynger, transduktiv                                        |
| Ward hierarkisk klynging     | mange, begrensede klynger, transduktiv                                |
| Agglomerativ klynging        | mange, begrensede, ikke-euklidiske avstander, transduktiv             |
| DBSCAN                       | ikke-flat geometri, ujevne klynger, transduktiv                       |
| OPTICS                       | ikke-flat geometri, ujevne klynger med variabel tetthet, transduktiv  |
| Gaussiske blandinger         | flat geometri, induktiv                                               |
| BIRCH                        | stort datasett med uteliggere, induktiv                               |

> 🎓 Hvordan vi lager klynger har mye å gjøre med hvordan vi samler datapunktene i grupper. La oss gå gjennom noe vokabular:
> 
> 🎓 ['Transduktiv' vs. 'induktiv'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktiv inferens er avledet fra observerte treningsdata som mappes til spesifikke testtilfeller. Induktiv inferens er avledet fra treningsdata som mappes til generelle regler som først deretter anvendes på testtilfeller.
> 
> Et eksempel: Tenk at du har et datasett som bare delvis er merket. Noe er 'plater', noe 'cd-er', og noe er blankt. Din oppgave er å gi etiketter til blankene. Hvis du velger en induktiv tilnærming, vil du trene en modell som søker etter 'plater' og 'cd-er', og anvender disse etikettene på dine umerkede data. Denne tilnærmingen vil ha problemer med å klassifisere ting som faktisk er 'kassetter'. En transduktiv tilnærming, derimot, håndterer disse ukjente dataene mer effektivt ved å jobbe for å gruppere lignende ting sammen og deretter anvender en etikett på en gruppe. I dette tilfellet kan klyngene reflektere 'runde musikalske ting' og 'firkantede musikalske ting'.
> 
> 🎓 ['Ikke-flat' vs. 'flat' geometri](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Avledet fra matematisk terminologi, refererer ikke-flat vs. flat geometri til måling av avstander mellom punkter enten ved 'flat' ([Euklidisk](https://wikipedia.org/wiki/Euclidean_geometry)) eller 'ikke-flat' (ikke-euklidisk) geometriske metoder.
> 
> 'Flat' i denne sammenheng refererer til euklidisk geometri (deler av dette undervises som 'plan' geometri), og ikke-flat refererer til ikke-euklidisk geometri. Hva har geometri med maskinlæring å gjøre? Vel, som to fagfelt som er forankret i matematikk må det finnes en felles måte å måle avstander mellom punkter i klynger, og det kan gjøres på en 'flat' eller 'ikke-flat' måte, avhengig av datanatur. [Euklidiske avstander](https://wikipedia.org/wiki/Euclidean_distance) måles som lengden av en linjesegment mellom to punkter. [Ikke-euklidiske avstander](https://wikipedia.org/wiki/Non-Euclidean_geometry) måles langs en kurve. Hvis dine data, visualisert, ser ut til ikke å eksistere på et plan, kan du trenge å bruke en spesialisert algoritme for å håndtere det.
>
>![Flat vs Ikke-flat Geometri Infografikk](../../../../translated_images/no/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infografikk av [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Avstander'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Klynger blir definert av deres avstandsmatrise, f.eks. avstandene mellom punkter. Denne avstanden kan måles på flere måter. Euklidiske klynger defineres av gjennomsnittet av punktverdiene, og inneholder et 'sentroid' eller midtpunkt. Avstander måles dermed i forhold til avstanden til det sentroid. Ikke-euklidiske avstander refererer til 'klustroider', det punktet som er nærmest andre punkter. Klustroider kan defineres på ulike måter.
> 
> 🎓 ['Begrenset'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Begrenset klynging](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introduserer 'semi-overvåket' læring i denne uovervåkede metoden. Relasjonene mellom punkter markeres som 'kan ikke kobles' eller 'må kobles' slik at noen regler pålegges datasettet.
>
> Et eksempel: Hvis en algoritme slippes fri på en batch med umerkede eller semi-merkede data, kan kvaliteten på klyngene den produserer bli dårlig. I eksempelet ovenfor kan klyngene gruppert 'runde musikk ting' og 'firkantede musikk ting' og 'trekantede ting' og 'kjeks'. Om man gir noen begrensninger, eller regler å følge ("gjenstanden må være laget av plast", "gjenstanden må kunne produsere musikk") kan dette hjelpe med å 'begrense' algoritmen til å gjøre bedre valg.
> 
> 🎓 'Tetthet'
> 
> Data som er 'støyende' anses som 'tett'. Avstandene mellom punkter i hver klynge kan på undersøkelse vise seg å være mer eller mindre tett, eller 'trangt', og disse dataene trenger å analyseres med passende klyngemetode. [Denne artikkelen](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) demonstrerer forskjellen mellom bruk av K-Means klynging vs. HDBSCAN-algoritmer for å utforske et støyende datasett med ujevn klyngtetthet.

## Klyngingsalgoritmer

Det finnes over 100 klyngingsalgoritmer, og deres bruk avhenger av datas natur. La oss diskutere noen av hovedtypene:

- **Hierarkisk klynging**. Hvis et objekt klassifiseres etter sin nærhet til et nærliggende objekt, heller enn til ett lenger unna, dannes klynger basert på medlemmers avstand til og fra andre objekter. Scikit-learns agglomerative klynging er hierarkisk.

   ![Hierarkisk klynging Infografikk](../../../../translated_images/no/hierarchical.bf59403aa43c8c47.webp)
   > Infografikk av [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Sentroid-klynging**. Denne populære algoritmen krever valg av 'k', eller antall klynger som skal dannes, hvoretter algoritmen bestemmer midtpunktet for en klynge og samler data rundt dette punktet. [K-means klynging](https://wikipedia.org/wiki/K-means_clustering) er en populær versjon av sentroid-klynging. Midtpunktet bestemmes av nærmeste gjennomsnitt, derav navnet. Den kvadrerte avstanden fra klyngen minimeres.

   ![Sentroid-klynging Infografikk](../../../../translated_images/no/centroid.097fde836cf6c918.webp)
   > Infografikk av [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Distribusjonsbasert klynging**. Basert på statistisk modellering, fokuserer distribusjonsbasert klynging på å bestemme sannsynligheten for at et datapunkt tilhører en klynge, og tildeler det deretter. Gaussiske blandingsmetoder tilhører denne typen.

- **Tetthetsbasert klynging**. Datapunkter tildeles klynger basert på deres tetthet, eller deres gruppering rundt hverandre. Datapunkter langt fra gruppen anses som uteliggere eller støy. DBSCAN, Mean-shift og OPTICS tilhører denne typen klynging.

- **Rutenettbasert klynging**. For flerdimensjonale datasett opprettes et rutenett, og dataene deles mellom rutenettcellene, slik at klynger dannes.

## Oppgave - klyng dine data

Klynging som teknikk støttes sterkt av god visualisering, så la oss starte med å visualisere musikkdataene våre. Denne øvelsen vil hjelpe oss med å avgjøre hvilken av klyngemetodene vi bør bruke mest effektivt for denne datatypen.

1. Åpne filen [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) i denne mappen.

1. Importer `Seaborn`-pakken for god datavisualisering.

    ```python
    !pip install seaborn
    ```

1. Legg til sangdata fra [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Last inn en dataframe med noen data om sangene. Gjør deg klar til å utforske disse dataene ved å importere bibliotekene og skrive ut dataene:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Sjekk de første linjene av data:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigeriansk pop   | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Få noe informasjon om dataframen ved å kalle `info()`:

    ```python
    df.info()
    ```

   Utskriften ser slik ut:

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

1. Dobbeltsjekk for nullverdier ved å kalle `isnull()` og verifisere at summen er 0:

    ```python
    df.isnull().sum()
    ```

    Ser bra ut:

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

> 🤔 Hvis vi jobber med klynging, en uovervåket metode som ikke krever merket data, hvorfor viser vi da disse dataene med etiketter? I fase for datautforskning er de nyttige, men de er ikke nødvendige for at klyngealgoritmene skal fungere. Man kan like gjerne fjerne kolonneoverskriftene og referere til dataene via kolonnenummer. 

Se på de generelle verdiene i dataene. Merk at popularitet kan være '0', som viser sanger uten rangering. La oss fjerne disse snart.

1. Bruk et stolpediagram for å finne ut hvilke sjangre som er mest populære:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/no/popular.9c48d84b3386705f.webp)

✅ Hvis du vil se flere toppverdier, endre toppen `[:5]` til et større tall, eller fjern det for å se alle.

Merk at når den mest populære sjangeren er beskrevet som 'Missing', betyr det at Spotify ikke klassifiserte den, så la oss bli kvitt den.

1. Fjern manglende data ved å filtrere det ut

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Sjekk nå sjangrene på nytt:

    ![most popular](../../../../translated_images/no/all-genres.1d56ef06cefbfcd6.webp)

1. De tre mest dominerende sjangrene i dette datasettet er langt overlegne. La oss fokusere på `afro dancehall`, `afropop`, og `nigeriansk pop`, i tillegg filtrere datasettet for å fjerne alt med en popularitet på 0 (som betyr at den ikke ble klassifisert med popularitet i datasettet og kan betraktes som støy for våre formål):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Gjør en rask test for å se om dataene korrelerer på noen spesielt sterk måte:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/no/correlation.a9356bb798f5eea5.webp)

    Den eneste sterke korrelasjonen er mellom `energy` og `loudness`, noe som ikke er overraskende, gitt at høy musikk vanligvis er ganske energisk. Ellers er korrelasjonene relativt svake. Det blir interessant å se hva en klyngealgoritme kan utlede fra disse dataene.

    > 🎓 Merk at korrelasjon ikke innebærer årsakssammenheng! Vi har bevis for korrelasjon, men ikke bevis for årsak. Et [morsomt nettsted](https://tylervigen.com/spurious-correlations) har noen illustrasjoner som understreker dette poenget.

Finnes det noen konvergens i dette datasettet rundt en sangs oppfattede popularitet og dansbarhet? En FacetGrid viser at det finnes konsentriske sirkler som stemmer overens, uavhengig av sjanger. Kan det være at nigeriansk smak konvergerer på et visst nivå av dansbarhet for denne sjangeren?  

✅ Prøv ulike datapunkt (energy, loudness, speechiness) og flere eller forskjellige musikksjangre. Hva kan du oppdage? Ta en titt på `df.describe()`-tabellen for å se generalisert spredning av datapunkt.

### Øvelse – datadistribusjon

Er disse tre sjangrene markant forskjellige i oppfattelsen av deres dansbarhet, basert på deres popularitet?

1. Undersøk datafordelingen til våre tre toppsjangre for popularitet og dansbarhet langs en gitt x- og y-akse.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Du kan oppdage konsentriske sirkler rundt et generelt punkt for konvergens som viser punktenes fordeling.

    > 🎓 Merk at dette eksemplet bruker en KDE (Kernel Density Estimate) graf som representerer data ved hjelp av en kontinuerlig sannsynlighetstetthetkurve. Dette gjør at vi kan tolke data når vi jobber med flere fordelinger.

    Generelt stemmer de tre sjangrene løst overens med tanke på popularitet og dansbarhet. Å bestemme klynger i disse løst justerte dataene vil være en utfordring:

    ![distribution](../../../../translated_images/no/distribution.9be11df42356ca95.webp)

1. Lag et spredningsdiagram:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Et spredningsdiagram av de samme aksene viser et lignende konvergenmønster

    ![Facetgrid](../../../../translated_images/no/facetgrid.9b2e65ce707eba1f.webp)

Generelt kan du bruke spredningsdiagrammer til å vise dataklynger for klynging, så det å mestre denne typen visualisering er veldig nyttig. I neste leksjon skal vi ta denne filtrerte dataen og bruke k-means-klynging for å oppdage grupper i dataene som ser ut til å overlappe på interessante måter.

---

## 🚀Utfordring

I forberedelse til neste leksjon, lag en graf over de ulike klyngealgoritmene du kan oppdage og bruke i en produksjonsmiljø. Hvilke typer problemer forsøker klynging å løse?

## [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang & Selvstudium

Før du anvender klyngingsalgoritmer, som vi har lært, er det en god idé å forstå naturen til datasettet ditt. Les mer om dette emnet [her](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Denne hjelpsomme artikkelen](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) tar deg gjennom de ulike måtene ulike klyngingsalgoritmer oppfører seg på, gitt forskjellige datatyper.

## Oppgave

[Forsk andre visualiseringer for klynging](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfraskrivelse**:
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det opprinnelige dokumentet på originalspråket skal betraktes som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->