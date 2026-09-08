# Introducere în clustering

Clustering este un tip de [Învățare nesupravegheată](https://wikipedia.org/wiki/Unsupervised_learning) care presupune că un set de date este neetichetat sau că intrările sale nu sunt asociate cu ieșiri predefinite. Utilizează diverse algoritmi pentru a parcurge datele neetichetate și a oferi grupări în funcție de tiparele pe care le distinge în date. 

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Dați clic pe imaginea de mai sus pentru un videoclip. În timp ce studiați învățarea automată cu clustering, bucurați-vă de niște piese de Nigerian Dance Hall - acesta este un cântec foarte apreciat din 2014 de PSquare.

## [Chestionar pre-lectură](https://ff-quizzes.netlify.app/en/ml/)

### Introducere

[Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) este foarte util pentru explorarea datelor. Să vedem dacă poate ajuta la descoperirea tendințelor și tiparelor în modul în care publicul nigerian consumă muzică.

✅ Luați un minut să vă gândiți la utilizările clusteringului. În viața reală, clusteringul se întâmplă ori de câte ori aveți un maldăr de rufe și trebuie să sortați hainele membrilor familiei 🧦👕👖🩲. În știința datelor, clusteringul apare când încercați să analizați preferințele unui utilizator sau să determinați caracteristicile unui set de date neetichetat. Clusteringul, într-un fel, ajută la înțelegerea haosului, ca un sertar cu șosete.

[![Introducere în ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introducere în clustering")

> 🎥 Dați clic pe imaginea de mai sus pentru un videoclip: John Guttag de la MIT introduce clusteringul

Într-un mediu profesional, clusteringul poate fi folosit pentru a determina lucruri precum segmentarea pieței, identificând ce grupuri de vârstă cumpără ce articole, de exemplu. O altă utilizare ar fi detectarea anomaliilor, poate pentru detectarea fraudelor dintr-un set de date cu tranzacții cu carduri de credit. Sau ați putea folosi clusteringul pentru a identifica tumori într-un lot de scanări medicale.

✅ Gândiți-vă un minut cum ați putea să fi întâlnit clustering 'în libertate', într-un mediu bancar, de comerț electronic sau de afaceri.

> 🎓 Interesant, analiza clusterelor a apărut în domeniile Antropologiei și Psihologiei în anii 1930. Vă puteți imagina cum ar fi fost folosită?

Alternativ, ați putea să îl folosiți pentru gruparea rezultatelor căutării - după linkuri de cumpărături, imagini sau recenzii, de exemplu. Clusteringul este util când aveți un set de date mare pe care doriți să-l reduceți și pe care doriți să faceți o analiză mai granulară, astfel încât tehnica poate fi folosită pentru a învăța despre date înainte ca alte modele să fie construite.

✅ Odată ce datele dvs. sunt organizate în clustere, le atribuiți un ID de cluster, iar această tehnică poate fi utilă pentru păstrarea confidențialității unui set de date; puteți să vă referiți la un punct de date prin ID-ul clusterului, în loc de date identificabile mai revelatoare. Puteți să vă gândiți la alte motive pentru care ați folosi ID-ul unui cluster în loc de alte elemente ale clusterului pentru a-l identifica?

Adânciți-vă înțelegerea tehnicilor de clustering în acest [modul Learn](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)
## Primii pași în clustering

[Scikit-learn oferă o gamă largă](https://scikit-learn.org/stable/modules/clustering.html) de metode pentru a efectua clustering. Tipul pe care îl alegeți va depinde de cazul dvs. de utilizare. Conform documentației, fiecare metodă are diferite avantaje. Iată un tabel simplificat al metodelor acceptate de Scikit-learn și cazurile lor de utilizare adecvate:

| Numele metodei               | Caz de utilizare                                                     |
| :--------------------------- | :------------------------------------------------------------------ |
| K-Means                      | scop general, inductiv                                              |
| Propagarea afinității        | multe clustere inegale, inductiv                                    |
| Mean-shift                   | multe clustere inegale, inductiv                                    |
| Clustering spectral          | puține clustere egale, transductiv                                  |
| Clustering ierarhic Ward     | multe clustere constrânse, transductiv                              |
| Clustering aglomerativ       | multe, constrâns, distanțe non euclidiene, transductiv              |
| DBSCAN                       | geometrie non-plană, clustere inegale, transductiv                  |
| OPTICS                       | geometrie non-plană, clustere inegale cu densitate variabilă, transductiv |
| Amestecuri gaussiane         | geometrie plană, inductiv                                           |
| BIRCH                        | set de date mare cu outlieri, inductiv                             |

> 🎓 Modul în care creăm clustere are mult de-a face cu modul în care grupăm punctele de date. Să dezvoltăm puțin vocabularul:
>
> 🎓 ['Transductiv' vs. 'inductiv'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Inferența transductivă este derivată din cazuri de antrenament observate care se mapează pe cazuri de test specifice. Inferența inductivă este derivată din cazuri de antrenament care se mapează pe reguli generale care apoi sunt aplicate cazurilor de test.
> 
> Un exemplu: Imaginați-vă că aveți un set de date parțial etichetat. Unele lucruri sunt 'discuri', altele 'cd-uri', iar unele sunt necompletate. Sarcina dvs. este să atribuiți etichete celor necompletate. Dacă alegeți o abordare inductivă, ați antrena un model să caute 'discuri' și 'cd-uri' și să aplice aceste etichete datelor neetichetate. Această abordare va avea probleme în clasificarea lucrurilor care sunt de fapt 'casete'. O abordare transductivă, pe de altă parte, gestionează mai eficient aceste date necunoscute, deoarece lucrează să grupeze elementele similare împreună și apoi aplică o etichetă grupului. În acest caz, clusterele ar putea reflecta 'lucruri muzicale rotunde' și 'lucruri muzicale pătrate'.
> 
> 🎓 ['Geometrie non-plană vs. plană'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Derivată din terminologia matematică, geometria non-plană vs. plană se referă la măsurarea distanțelor dintre puncte prin metode geometrice 'plane' ([Euclidiană](https://wikipedia.org/wiki/Euclidean_geometry)) sau 'non-plane' (non-Euclidiană).
>
> 'Plană' în acest context se referă la geometria euclidiană (părți din care se predau ca geometrie 'plană'), iar non-plană se referă la geometria non-euclidiană. Ce legătură are geometria cu învățarea automată? Ei bine, ca două domenii care au rădăcini în matematică, trebuie să existe o metodă comună de a măsura distanțele dintre punctele din clustere, iar aceasta se poate face într-un mod 'plan' sau 'non-plan', în funcție de natura datelor. [Distanțele euclidiene](https://wikipedia.org/wiki/Euclidean_distance) sunt măsurate ca lungimea unui segment de linie între două puncte. [Distanțele non-euclidiene](https://wikipedia.org/wiki/Non-Euclidean_geometry) sunt măsurate de-a lungul unei curbe. Dacă datele dvs., vizualizate, par să nu existe pe un plan, este posibil să aveți nevoie să folosiți un algoritm specializat pentru a le gestiona.
>
![Flat vs Nonflat Geometry Infographic](../../../../translated_images/ro/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infografic de [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Distanțe'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Clusterele sunt definite prin matricea lor de distanțe, de exemplu distanțele dintre puncte. Această distanță poate fi măsurată în câteva moduri. Clusterele euclidiene sunt definite prin media valorilor punctelor și conțin un 'centroid' sau punct central. Distanțele sunt astfel măsurate față de acel centroid. Distanțele non-euclidiene se referă la 'clustroizi', punctele cele mai apropiate de alte puncte. Clustroizii, la rândul lor, pot fi definiți în diferite moduri.
> 
> 🎓 ['Constrâns'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Clusteringul constrâns](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introduce învățarea 'semi-supervizată' în această metodă nesupravegheată. Relațiile dintre puncte sunt marcate ca 'nu se pot lega' sau 'trebuie legate', astfel că anumite reguli sunt impuse setului de date.
>
> Un exemplu: Dacă un algoritm este lăsat liber pe un set de date neetichetat sau semi-etichetat, clusterele pe care le produce pot fi de calitate scăzută. În exemplul de mai sus, clusterele ar putea grupa 'lucruri muzicale rotunde' și 'lucruri muzicale pătrate' și 'lucruri triunghiulare' și 'biscuiți'. Dacă i se dau niște constrângeri sau reguli de urmat ("obiectul trebuie să fie făcut din plastic", "obiectul trebuie să poată produce muzică") acestea pot ajuta algoritmul să ia decizii mai bune.
> 
> 🎓 'Densitate'
> 
> Datele considerate 'zgomotoase' sunt considerate 'dense'. Distanțele dintre punctele din fiecare cluster, după examinare, pot fi mai mult sau mai puțin dense sau 'aglomerate', iar aceste date trebuie analizate cu metoda de clustering potrivită. [Acest articol](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) demonstrează diferența dintre utilizarea clusteringului K-Means vs. HDBSCAN pentru explorarea unui set de date zgomotos cu densitate neuniformă a clusterului.

## Algoritmi de clustering

Există peste 100 de algoritmi de clustering, iar utilizarea lor depinde de natura datelor la îndemână. Să discutăm unii dintre cei mai importanți:

- **Clustering ierarhic**. Dacă un obiect este clasificat după proximitatea sa față de un obiect apropiat, în loc de unul mai îndepărtat, clusterele sunt formate pe baza distanței membrilor lor către și de la alte obiecte. Clusteringul aglomerativ din Scikit-learn este ierarhic.

   ![Hierarchical clustering Infographic](../../../../translated_images/ro/hierarchical.bf59403aa43c8c47.webp)
   > Infografic de [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering centroid**. Acest algoritm popular necesită alegerea lui 'k', sau numărul de clustere de format, după care algoritmul determină punctul central al unui cluster și grupează datele în jurul acelui punct. [Clusteringul K-means](https://wikipedia.org/wiki/K-means_clustering) este o versiune populară a clusteringului centroid. Centrul este determinat de media cea mai apropiată, de unde și numele. Distanța pătratică față de cluster este minimizată.

   ![Centroid clustering Infographic](../../../../translated_images/ro/centroid.097fde836cf6c918.webp)
   > Infografic de [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering bazat pe distribuție**. Bazat pe modelarea statistică, clusteringul bazat pe distribuție se concentrează pe determinarea probabilității ca un punct de date să aparțină unui cluster și pe atribuirea sa corespunzătoare. Metodele cu amestec gaussian fac parte din acest tip.

- **Clustering bazat pe densitate**. Punctele de date sunt atribuite clusterelor în funcție de densitatea lor sau de gruparea în jurul unul altuia. Punctele de date departe de grup sunt considerate outlieri sau zgomot. DBSCAN, Mean-shift și OPTICS fac parte din acest tip de clustering.

- **Clustering pe bază de grilă**. Pentru seturi de date multidimensionale, se creează o grilă iar datele sunt împărțite între celulele grilei, creând astfel clustere.

## Exercițiu - clustrează-ți datele

Clusteringul ca tehnică este foarte ajutat de o vizualizare corectă, așa că să începem prin a vizualiza datele noastre muzicale. Acest exercițiu ne va ajuta să decidem care dintre metodele de clustering ar trebui să folosim cel mai eficient pentru natura acestor date.

1. Deschideți fișierul [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) din acest folder.

1. Importați pachetul `Seaborn` pentru o vizualizare bună a datelor.

    ```python
    !pip install seaborn
    ```

1. Adăugați datele melodiilor din [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Încărcați un dataframe cu câteva date despre melodii. Pregătiți-vă să explorați aceste date importând bibliotecile și afișând datele:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Verificați primele câteva linii de date:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Obțineți câteva informații despre dataframe, apelând `info()`:

    ```python
    df.info()
    ```

   Ieșirea arată astfel:

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

1. Verificați din nou pentru valori nule, apelând `isnull()` și verificând dacă suma este 0:

    ```python
    df.isnull().sum()
    ```

    Arată bine:

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

1. Descrieți datele:

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

> 🤔 Dacă lucrăm cu clustering, o metodă nesupravegheată care nu necesită date etichetate, de ce afișăm aceste date cu etichete? În faza de explorare a datelor, acestea sunt utile, dar nu sunt necesare pentru funcționarea algoritmilor de clustering. Ați putea foarte bine să eliminați anteturile coloanelor și să vă referiți la date după numărul coloanei.

Uitați-vă la valorile generale ale datelor. Rețineți că popularitatea poate fi '0', ceea ce arată cântece care nu au nicio clasare. Să le eliminăm în curând.

1. Folosiți un barplot pentru a afla care sunt cele mai populare genuri:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/ro/popular.9c48d84b3386705f.webp)

✅ Dacă doriți să vedeți mai multe valori de top, schimbați `[:5]` la o valoare mai mare sau eliminați-l pentru a vedea tot.

Notă, când genul de top este descris ca „Missing”, asta înseamnă că Spotify nu l-a clasificat, deci să scăpăm de el.

1. Scăpați de datele lipsă filtrându-le

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Acum verificați din nou genurile:

    ![most popular](../../../../translated_images/ro/all-genres.1d56ef06cefbfcd6.webp)

1. De departe, primele trei genuri domină acest set de date. Să ne concentrăm pe `afro dancehall`, `afropop` și `nigerian pop`, de asemenea filtrăm setul de date pentru a elimina orice are valoarea 0 la popularitate (însemnând că nu a fost clasificat cu o popularitate în setul de date și poate fi considerat zgomot pentru scopurile noastre):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Faceți un test rapid pentru a vedea dacă datele corelează în vreun mod deosebit de puternic:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/ro/correlation.a9356bb798f5eea5.webp)

    Singura corelație puternică este între `energy` și `loudness`, ceea ce nu este prea surprinzător, având în vedere că muzica tare este de obicei destul de energică. În rest, corelațiile sunt relativ slabe. Va fi interesant să vedem ce poate face un algoritm de clustering cu aceste date.

    > 🎓 Rețineți că corelația nu implică cauzalitate! Avem dovezi ale corelației, dar nu avem dovezi ale cauzalității. Un [site amuzant](https://tylervigen.com/spurious-correlations) are niște vizualizări care subliniază acest punct.

Există vreo convergență în acest set de date în jurul popularității percepute a unui cântec și a dansabilității? Un FacetGrid arată că există cercuri concentrice care se aliniază, indiferent de gen. Ar putea fi ca gusturile nigeriene să convergă la un anumit nivel de dansabilitate pentru acest gen?

✅ Încercați diferite puncte de date (energie, volum, expresivitate vocală) și mai multe sau alte genuri muzicale. Ce puteți descoperi? Aruncați o privire la tabelul `df.describe()` pentru a vedea răspândirea generală a datelor.

### Exercițiu - distribuția datelor

Sunt aceste trei genuri semnificativ diferite în percepția dansabilității lor, pe baza popularității?

1. Examinați distribuția datelor pentru cele trei genuri de top în ceea ce privește popularitatea și dansabilitatea pe axe x și y date.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Puteți descoperi cercuri concentrice în jurul unui punct general de convergență, arătând distribuția punctelor.

    > 🎓 Rețineți că acest exemplu folosește un grafic KDE (Kernel Density Estimate) care reprezintă datele folosind o curbă de densitate de probabilitate continuă. Aceasta ne permite să interpretăm datele când lucrăm cu distribuții multiple.

    În general, cele trei genuri se aliniază vag în ceea ce privește popularitatea și dansabilitatea. Determinarea clusterelor în aceste date vag aliniate va fi o provocare:

    ![distribution](../../../../translated_images/ro/distribution.9be11df42356ca95.webp)

1. Creați un grafic scatter:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Un grafic scatter pe aceleași axe arată un model similar de convergență

    ![Facetgrid](../../../../translated_images/ro/facetgrid.9b2e65ce707eba1f.webp)

În general, pentru clustering, puteți folosi grafice scatter pentru a arăta clustere de date, așa că stăpânirea acestui tip de vizualizare este foarte utilă. În lecția următoare, vom lua aceste date filtrate și vom folosi clustering k-means pentru a descoperi grupuri în aceste date care par să se suprapună în mod interesant.

---

## 🚀Provocare

Ca pregătire pentru următoarea lecție, faceți un grafic despre diferiții algoritmi de clustering pe care îi puteți descoperi și folosi într-un mediu de producție. Ce tipuri de probleme încearcă să rezolve clusteringul?

## [Chestionar post-lectură](https://ff-quizzes.netlify.app/en/ml/)

## Recenzie & Studiu individual

Înainte de a aplica algoritmi de clustering, așa cum am învățat, este o idee bună să înțelegeți natura setului dvs. de date. Citiți mai multe pe această temă [aici](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Acest articol util](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) vă ghidează prin diferitele comportamente ale algoritmilor de clustering, în funcție de formele diferite ale datelor.

## Tema pentru acasă

[Cercetați alte vizualizări pentru clustering](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Declinare a responsabilității**:
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). În timp ce ne străduim pentru acuratețe, vă rugăm să rețineți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa nativă trebuie considerat sursa autorizată. Pentru informații critice, se recomandă traducerea profesională realizată de un om. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care decurg din utilizarea acestei traduceri.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->