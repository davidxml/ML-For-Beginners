# Introduktion till klustring

Klustring är en typ av [Oövervakad inlärning](https://wikipedia.org/wiki/Unsupervised_learning) som förutsätter att en dataset är oetiketterad eller att dess indata inte är kopplade till fördefinierade utdata. Den använder olika algoritmer för att sortera igenom oetiketterad data och skapa grupper baserat på mönster den upptäcker i datan.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Klicka på bilden ovan för en video. Medan du studerar maskininlärning med klustring, njut av några nigerianska Dance Hall-spår – detta är en mycket uppskattad låt från 2014 av PSquare.

## [Förföreläsningsquiz](https://ff-quizzes.netlify.app/en/ml/)

### Introduktion

[Klustring](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) är mycket användbart för datautforskning. Låt oss se om det kan hjälpa till att upptäcka trender och mönster i hur nigerianska publikgrupper konsumerar musik.

✅ Ta en minut och tänk på användningarna av klustring. I verkliga livet sker klustring när du har en hög med tvätt och behöver sortera ut familjemedlemmarnas kläder 🧦👕👖🩲. Inom datavetenskap sker klustring när man försöker analysera en användares preferenser eller bestämma egenskaper för någon oetiketterad dataset. Klustring hjälper på sätt och vis till att skapa ordning i kaos, som en strumplåda.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Klicka på bilden ovan för en video: MIT:s John Guttag introducerar klustring

I en professionell miljö kan klustring användas för att avgöra saker som marknadssegmentering, till exempel för att bestämma vilka åldersgrupper som köper vilka varor. En annan användning kan vara anomalidetektion, kanske för att upptäcka bedrägeri i en dataset med kreditkortstransaktioner. Eller så kan du använda klustring för att identifiera tumörer i en samling medicinska skanningar.

✅ Tänk en minut på hur du kan ha stött på klustring 'i det vilda', i bank-, e-handels- eller affärssammanhang.

> 🎓 Intressant nog härstammar klusteranalys från antropologi och psykologi på 1930-talet. Kan du föreställa dig hur det kunde ha använts?

Alternativt kan du använda det för att gruppera sökresultat – till exempel efter shoppinglänkar, bilder eller recensioner. Klustring är användbart när du har en stor dataset som du vill reducera och på vilken du vill utföra mer granulär analys, så tekniken kan användas för att lära sig om data innan andra modeller konstrueras.

✅ När din data är organiserad i kluster tilldelar du den ett kluster-ID, och denna teknik kan vara användbar för att bevara datas integritet; du kan istället referera till en datapunkt med dess kluster-ID snarare än med mer avslöjande identifierbar data. Kan du komma på andra skäl till varför du skulle använda ett kluster-ID snarare än andra element i klustret för att identifiera det?

Fördjupa din förståelse om klustringstekniker i denna [Learn-modul](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)
## Komma igång med klustring

[Scikit-learn erbjuder ett stort urval](https://scikit-learn.org/stable/modules/clustering.html) av metoder för att utföra klustring. Vilken typ du väljer beror på ditt användningsfall. Enligt dokumentationen har varje metod olika fördelar. Här är en förenklad tabell över metoderna som stöds av Scikit-learn och deras lämpliga användningsområden:

| Metodnamn                   | Användningsfall                                                      |
| :-------------------------- | :------------------------------------------------------------------ |
| K-Means                     | allmänt ändamål, induktiv                                           |
| Affinity propagation        | många, ojämna kluster, induktiv                                     |
| Mean-shift                  | många, ojämna kluster, induktiv                                     |
| Spektral klustring          | få, jämna kluster, transduktiv                                      |
| Ward hierarkisk klustring   | många, begränsade kluster, transduktiv                              |
| Agglomerativ klustring      | många, begränsade, icke-Euklidiska avstånd, transduktiv            |
| DBSCAN                      | icke-plan geometri, ojämna kluster, transduktiv                    |
| OPTICS                      | icke-plan geometri, ojämna kluster med variabel densitet, transduktiv |
| Gaussiska blandningar       | plan geometri, induktiv                                             |
| BIRCH                       | stor dataset med uteliggare, induktiv                               |

> 🎓 Hur vi skapar kluster har mycket att göra med hur vi samlar datapunkterna till grupper. Låt oss gå igenom lite terminologi:
>
> 🎓 ['Transduktiv' vs. 'induktiv'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktiv inferens härleds från observerade träningsfall som kartläggs till specifika testfall. Induktiv inferens härleds från träningsfall som kartläggs till generella regler som först därefter appliceras på testfall.
> 
> Exempel: Föreställ dig att du har en dataset som bara delvis är etiketterad. Några saker är 'skivor', några 'cd-skivor' och några är tomma. Din uppgift är att tilldela etiketter till de tomma. Om du väljer en induktiv metod skulle du träna en modell för att känna igen 'skivor' och 'cd-skivor' och applicera dessa etiketter på din oetiketterade data. Denna metod kommer att ha svårt att klassificera saker som faktiskt är 'kassetter'. En transduktiv metod hanterar däremot denna okända data mer effektivt då den arbetar för att gruppera liknande objekt tillsammans och därefter tilldela en etikett till gruppen. I detta fall kan kluster reflektera 'runda musiksaker' och 'fyrkantiga musiksaker'.
> 
> 🎓 ['Icke-plan' vs. 'plan' geometri](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Härledd från matematisk terminologi, refererar icke-plan vs. plan geometri till mätning av avstånd mellan punkter med antingen 'plan' ([Euklidisk](https://wikipedia.org/wiki/Euclidean_geometry)) eller 'icke-plan' (icke-Euklidisk) geometriska metoder. 
>
> 'Plan' i detta sammanhang avser Euklidisk geometri (delar av vilken lärs ut som 'plan' geometri), och icke-plan avser icke-Euklidisk geometri. Vad har geometri med maskininlärning att göra? Jo, som två områden som är rotade i matematik måste det finnas ett gemensamt sätt att mäta avstånd mellan punkter i kluster, och det kan göras på ett 'plant' eller 'icke-plant' sätt beroende på datans natur. [Euklidiska avstånd](https://wikipedia.org/wiki/Euclidean_distance) mäts som längden på en linjesegment mellan två punkter. [Icke-Euklidiska avstånd](https://wikipedia.org/wiki/Non-Euclidean_geometry) mäts längs en kurva. Om din data, när den visualiseras, verkar inte existera på en plan yta kan du behöva använda en specialiserad algoritm för att hantera detta.
>
![Flat vs Nonflat Geometry Infographic](../../../../translated_images/sv/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infografik av [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Avstånd'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Kluster definieras av deras avståndsmatris, t.ex. avstånden mellan punkterna. Detta avstånd kan mätas på några olika sätt. Euklidiska kluster definieras av medelvärdet av punktvärdena, och innehåller en 'centroid' eller central punkt. Avstånd mäts då som avståndet till den centroiden. Icke-Euklidiska avstånd refererar till 'klustroids', punkten närmast andra punkter. Klustroids kan i sin tur definieras på olika sätt.
> 
> 🎓 ['Begränsad'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Begränsad klustring](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introducerar 'semi-övervakad' inlärning i denna oövervakade metod. Relationerna mellan punkterna markeras som 'får inte länkas' eller 'måste länkas' så vissa regler tvingas på datasetet.
>
>Exempel: Om en algoritm släpps lös på en samling oetiketterad eller semi-etiketterad data, kan de kluster den producerar vara av låg kvalitet. I exemplet ovan kan klustren gruppera 'runda musiksaker', 'fyrkantiga musiksaker', 'triangulära saker' och 'kakor'. Om man ger vissa begränsningar eller regler att följa ("föremålet måste vara av plast", "föremålet måste kunna producera musik") kan detta hjälpa algoritmen att göra bättre val.
> 
> 🎓 'Densitet'
> 
> Data som är 'brusig' anses vara 'tät'. Avstånden mellan punkterna i varje kluster kan efter undersökning visa sig vara mer eller mindre täta, eller 'trånga' och därför behöver denna data analyseras med passande klustringsmetod. [Denna artikel](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) visar skillnaden mellan att använda K-Means klustring vs. HDBSCAN-algoritmer för att utforska en brusig dataset med ojämn klusterdensitet.

## Klustringsalgoritmer

Det finns över 100 klustringsalgoritmer, och deras användning beror på datans natur. Låt oss diskutera några av de största:

- **Hierarkisk klustring**. Om ett objekt klassificeras efter dess närhet till ett närliggande objekt, snarare än ett längre bort, bildas kluster baserat på medlemmarnas avstånd till och från andra objekt. Scikit-learns agglomerativa klustring är hierarkisk.

   ![Hierarchical clustering Infographic](../../../../translated_images/sv/hierarchical.bf59403aa43c8c47.webp)
   > Infografik av [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroidklustring**. Denna populära algoritm kräver val av 'k', alltså antalet kluster som ska bildas, varefter algoritmen bestämmer en klusters mittpunkt och samlar data runt denna punkt. [K-means klustring](https://wikipedia.org/wiki/K-means_clustering) är en populär variant av centroidklustring. Mittpunkten bestäms av närmsta medelvärde, därav namnet. Det kvadrerade avståndet från klustret minimeras.

   ![Centroid clustering Infographic](../../../../translated_images/sv/centroid.097fde836cf6c918.webp)
   > Infografik av [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Distributionsbaserad klustring**. Baserad på statistisk modellering fokuserar distributionsbaserad klustring på att avgöra sannolikheten för att en datapunkt tillhör ett kluster och tilldela den därefter. Gaussiska blandningsmetoder tillhör denna typ.

- **Densitetsbaserad klustring**. Datapunkter tilldelas kluster baserat på deras densitet, eller deras gruppering runt varandra. Datapunkter långt från gruppen betraktas som uteliggare eller brus. DBSCAN, Mean-shift och OPTICS tillhör denna typ av klustring.

- **Rutbaserad klustring**. För flerdimensionella dataset skapas ett rutnät och datan delas upp mellan rutnätets celler, vilket skapar kluster.

## Övning – klustra din data

Klustring som teknik underlättas mycket av bra visualisering, så låt oss börja med att visualisera vår musikdata. Denna övning hjälper oss att avgöra vilken av klustringsmetoderna vi bör använda mest effektivt för denna datas natur.

1. Öppna filen [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) i denna mapp.

1. Importera `Seaborn`-paketet för bra datavisualisering.

    ```python
    !pip install seaborn
    ```

1. Lägg till låtdata från [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Ladda en dataframe med några data om låtarna. Gör dig redo att utforska denna data genom att importera biblioteken och skriva ut datan:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Kolla på de första raderna av data:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Få lite information om dataframe, genom att anropa `info()`:

    ```python
    df.info()
    ```

   Utdata ser ut så här:

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

1. Dubbelkolla för null-värden genom att anropa `isnull()` och verifiera att summan är 0:

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

1. Beskriv datan:

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

> 🤔 Om vi arbetar med klustring, en osuperviserad metod som inte kräver märkta data, varför visar vi då denna data med etiketter? I datautforskningsfasen är de användbara, men de är inte nödvändiga för att klustringsalgoritmer ska fungera. Du kan precis lika gärna ta bort kolumnrubrikerna och referera till datan via kolumnnummer.

Titta på de allmänna värdena i datan. Notera att popularitet kan vara '0', vilket visar låtar som inte har någon ranking. Låt oss ta bort dessa strax.

1. Använd ett stapeldiagram för att ta reda på de mest populära genrerna:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/sv/popular.9c48d84b3386705f.webp)

✅ Om du vill se fler toppvärden, ändra top-`[:5]` till ett större värde, eller ta bort det för att se alla.

Observera, när toppgenren beskrivs som 'Missing', betyder det att Spotify inte klassificerade den, så låt oss ta bort den.

1. Ta bort borttappade data genom att filtrera bort den

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Kontrollera nu genrerna igen:

    ![most popular](../../../../translated_images/sv/all-genres.1d56ef06cefbfcd6.webp)

1. De tre överlägset största genrerna dominerar denna dataset. Låt oss koncentrera oss på `afro dancehall`, `afropop`, och `nigerian pop`, samt filtrera datasetet för att ta bort något med en popularitet på 0 (vilket betyder att det inte klassificerades med en popularitet i datasetet och kan betraktas som brus för våra syften):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Gör ett snabbt test för att se om datan korrelerar på något särskilt starkt sätt:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/sv/correlation.a9356bb798f5eea5.webp)

    Den enda starka korrelationen är mellan `energy` och `loudness`, vilket inte är så förvånande, eftersom hög volym oftast är ganska energiskt. Annars är korrelationerna relativt svaga. Det blir intressant att se vad en klustringsalgoritm kan göra med denna data.

    > 🎓 Notera att korrelation innebär inte orsakssamband! Vi har bevis på korrelation men inget bevis på orsakssamband. En [rolig webbplats](https://tylervigen.com/spurious-correlations) har några visualiseringar som betonar denna poäng.

Finns det någon konvergens i detta dataset kring en låts upplevda popularitet och dansbarhet? Ett FacetGrid visar att det finns koncentriska cirklar som ligger i linje, oavsett genre. Kan det vara så att Nigerianska smaker konvergerar vid en viss nivå av dansbarhet för denna genre?

✅ Prova olika datapunkter (energy, loudness, speechiness) och fler eller andra musikgenrer. Vad kan du upptäcka? Ta en titt på tabellen `df.describe()` för att se den allmänna spridningen av datapunkterna.

### Övning - datadistribution

Är dessa tre genrer signifikant olika i uppfattningen av deras dansbarhet, baserat på deras popularitet?

1. Undersök våra tre toppgenrer med datadistribution för popularitet och dansbarhet längs givna x- och y-axlar.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Du kan upptäcka koncentriska cirklar runt en allmän konvergenspunkt som visar punkternas fördelning.

    > 🎓 Notera att detta exempel använder en KDE-graf (Kernel Density Estimate) som representerar datan med en kontinuerlig sannolikhetstäthetkurva. Detta gör det möjligt för oss att tolka data när vi arbetar med flera fördelningar.

    Generellt ligger de tre genrerna löst i linje med varandra vad gäller popularitet och dansbarhet. Att avgöra kluster i denna löst sammanfogade data blir en utmaning:

    ![distribution](../../../../translated_images/sv/distribution.9be11df42356ca95.webp)

1. Skapa ett scatterplot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Ett scatterplot med samma axlar visar ett liknande mönster av konvergens

    ![Facetgrid](../../../../translated_images/sv/facetgrid.9b2e65ce707eba1f.webp)

Generellt kan du för klustring använda scatterplots för att visa datakluster, så att behärska denna typ av visualisering är mycket användbart. I nästa lektion kommer vi ta denna filtrerade data och använda k-means klustring för att upptäcka grupper i denna data som verkar överlappa på intressanta sätt.

---

## 🚀Utmaning

Som förberedelse för nästa lektion, gör ett diagram över de olika klustringsalgoritmer du kan upptäcka och använda i en produktionsmiljö. Vilka slags problem försöker klustringen lösa?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Granskning & Självstudier

Innan du tillämpar klustringsalgoritmer, som vi har lärt oss, är det en bra idé att förstå naturen hos ditt dataset. Läs mer om detta ämne [här](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Denna hjälpsamma artikel](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) går igenom de olika sätt som olika klustringsalgoritmer beter sig, givet olika datatyper.

## Uppgift

[Forska andra visualiseringar för klustring](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfriskrivning**:
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, var vänlig notera att automatiska översättningar kan innehålla fel eller brister. Det ursprungliga dokumentet på dess modersmål bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för några missförstånd eller feltolkningar som uppstår till följd av användningen av denna översättning.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->