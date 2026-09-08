# Įvadas į klasterizavimą

Klasterizavimas yra [nesupervizinio mokymosi](https://wikipedia.org/wiki/Unsupervised_learning) tipas, kuris daro prielaidą, kad duomenų rinkinys nėra pažymėtas arba jo įvestys nėra susietos su iš anksto apibrėžtais išvestimis. Jis naudoja įvairius algoritmus nesuprastiems duomenims rūšiuoti ir pateikti grupes pagal duomenyse pastebėtas tendencijas.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, norėdami peržiūrėti vaizdo įrašą. Mokydamiesi mašininio mokymosi su klasterizavimu, mėgaukitės Nigerijos Dance Hall dainomis – tai labai vertinamas 2014 metų PSquare kūrinys.

## [Priešpaskaitinis testas](https://ff-quizzes.netlify.app/en/ml/)

### Įvadas

[Klasterizavimas](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) labai naudingas duomenų tyrimui. Pažiūrėkime, ar jis gali padėti atrasti tendencijas ir modelius, kaip Nigerijos auditorija vartotoja muziką.

✅ Skirkite minutę pagalvoti apie klasterizavimo panaudojimą. Tikrame gyvenime klasterizavimas vyksta, kai turite krūvą skalbinių ir reikia išrūšiuoti šeimos narių drabužius 🧦👕👖🩲. Duomenų moksle klasterizavimas vyksta tada, kai analizuojamos vartotojo nuostatos arba nustatomos bet kokio nepažymėto duomenų rinkinio savybės. Klasterizavimas tam tikra prasme padeda tvarkytis su chaoso supratimu, kaip kojinių stalčius.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, norėdami peržiūrėti vaizdo įrašą: MIT John Guttag pristato klasterizavimą  

Profesionalioje aplinkoje klasterizavimas gali būti naudojamas nustatyti rinkos segmentacijai, pavyzdžiui, kokios amžiaus grupės perka tam tikrus produktus. Kita paskirtis gali būti anomalijų aptikimas, pavyzdžiui, aptikti sukčiavimą kreditinių kortelių operacijų duomenų rinkinyje. Taip pat galite naudoti klasterizavimą navikų nustatymui medicininių nuskaitymų partijoje.

✅ Pagalvokite minutę, kaip galbūt susidūrėte su klasterizavimu „gyvenime“, bankininkystės, elektroninės prekybos ar verslo kontekste.

> 🎓 Įdomu, kad klasterių analizė kilo antropologijos ir psichologijos srityse 1930-aisiais. Ar galite įsivaizduoti, kaip ji galėjo būti naudojama?

Kitaip tariant, klasterizavimą galima naudoti paieškos rezultatų grupavimui – pagal pavyzdžiui, pirkinių nuorodas, vaizdus ar atsiliepimus. Klasterizavimas naudingas, kai turite didelį duomenų rinkinį, kurį norite sumažinti ir atlikti išsamesnę analizę, taigi ši technika gali būti panaudota norint pažinti duomenis prieš kuriant kitus modelius.

✅ Kai jūsų duomenys bus suskirstyti į klasterius, priskiriate jiems klasterio Id, o ši technika naudinga išsaugoti duomenų privatumą; vietoje atskirų duomenų taškų galite nurodyti tik klasterio Id. Ar galite pagalvoti apie kitų priežasčių, dėl kurių būtų geriau naudoti klasterio Id, o ne kitus klasterio elementus identifikuoti?

Pagilinkite žinias apie klasterizavimo metodus šiame [mokymosi modulyje](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

## Pradžia su klasterizavimu

[Scikit-learn siūlo platų](https://scikit-learn.org/stable/modules/clustering.html) klasterizavimo metodų spektrą. Pasirinkimas priklausys nuo naudojimo atvejo. Pagal dokumentaciją, kiekvienas metodas turi įvairių privalumų. Štai supaprastinta lentelė su Scikit-learn palaikomais metodais ir jų tinkamais panaudojimo atvejais:

| Metodas                       | Panaudojimo atvejis                                                  |
| :--------------------------- | :------------------------------------------------------------------- |
| K-Means                      | bendras tikslas, indukcinis                                           |
| Afiniteto propagacija        | daug, nelyginiai klasteriai, indukcinis                              |
| Mean-shift                   | daug, nelyginiai klasteriai, indukcinis                              |
| Spektrinis klasterizavimas  | keli, lygūs klasteriai, transdukcinis                               |
| Ward hierarchinis klasterizavimas | daug, riboti klasteriai, transdukcinis                             |
| Agregacinis klasterizavimas | daug, riboti, ne Euklido atstumai, transdukcinis                    |
| DBSCAN                       | nelygi plokštuma, nelyginiai klasteriai, transdukcinis               |
| OPTICS                       | nelygi plokštuma, įvairaus tankio nelyginiai klasteriai, transdukcinis |
| Gauso mišiniai               | plokšti, indukciniai                                                |
| BIRCH                        | didelis duomenų rinkinys su iškritusiais, indukcinis               |

> 🎓 Kaip kuriame klasterius, labai priklauso nuo to, kaip surenkame duomenų taškus į grupes. Išskleiskime keletą terminų:
>
> 🎓 ['Transdukcinis' vs. 'indukcinis'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transdukcinė išvada gaunama iš stebimų mokymo atvejų, atitinkančių konkrečius testavimo atvejus. Indukcinė išvada gaunama iš mokymo atvejų, kurie apibrėžia bendras taisykles, kurias vėliau taiko testavimo atvejams.
> 
> Pavyzdys: Tarkime, turite duomenų rinkinį, kuris iš dalies pažymėtas. Kai kurie duomenys yra „įrašai“, kiti „CD“, o kai kurie tušti. Jūsų užduotis – pridėti žymas tuštiems. Pasirinkę indukcinį būdą, apmokytumėte modelį ieškoti „įrašų“ ir „CD“ ir priskirti tokius ženklus nepažymėtiems duomenims. Tokiu būdu sunku būtų klasifikuoti „kasetes“. Transdukcinis būdas veiksmingiau tvarkosi su nežinomu duomenų rinkiniu, nes bando grupuoti panašius elementus ir priskiria žymę grupei. Tokiu atveju klasteriai gali atspindėti „apvalius muzikinus daiktus“ ir „kampuotus muzikinus daiktus“.
> 
> 🎓 ['Nelygi plokštuma' vs. 'plokščia' geometrija](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Remiantis matematiniais terminais, „nelygi plokštuma“ ir „plokščia“ geometrija reiškia atstumų tarp taškų matavimą pasitelkiant atitinkamai „plokščią“ ([Euklido](https://wikipedia.org/wiki/Euclidean_geometry)) arba „nelygią“ (ne Euklido) geometriją.
>
> Čia „plokščia“ reiškia Euklido geometriją (jos dalis mokoma kaip „plokštumos“ geometrija), o „nelygi“ – ne Euklido geometriją. Kuo geometrija susijusi su mašininiu mokymusi? Kadangi abi sritys remiasi matematika, turi būti bendras būdas matuoti atstumus tarp taškų klasteriuose, ir tai gali būti daroma „plokščiu“ arba „nelygiu“ būdu, priklausomai nuo duomenų pobūdžio. [Euklido atstumai](https://wikipedia.org/wiki/Euclidean_distance) matuojami kaip atkarpos ilgis tarp dviejų taškų. [Ne Euklido atstumai](https://wikipedia.org/wiki/Non-Euclidean_geometry) matuojami palei kreivę. Jei jūsų duomenys, vizualizuojant, atrodo neegzistuojantys plokštumoje, gali prireikti naudoti specializuotą algoritmą.
>
>![Plokščios vs Nelygios geometrijos infografika](../../../../translated_images/lt/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infografika: [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Atstumai'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Klasteriai apibrėžiami savo atstumo matrica, t. y. atstumais tarp taškų. Šie atstumai gali būti matuojami keliais būdais. Euklido klasteriai apibrėžiami taškų reikšmių vidurkiu ir turi 'centroidą' arba centro tašką. Atstumai matuojami iki to centro taško. Ne Euklido atstumai reiškia 'klustroidus' – tašką, esančią arčiausiai kitų taškų. Klustroidai gali būti apibrėžti skirtingais būdais.
> 
> 🎓 ['Ribota'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Ribotas klasterizavimas](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) įterpia 'pusiau prižiūrimą' mokymąsi į šią nesupervizinę metodiką. Taškų ryšiai žymimi kaip 'negali būti susieti' arba 'turi būti susieti', todėl duomenims taikomi tam tikri apribojimai.
>
> Pavyzdys: Jei algoritmas darbo laisvėje ant nesutvarkytų arba pusiau pažymėtų duomenų, sugeneruoti klasteriai gali būti prastos kokybės. Aukščiau pavyzdyje klasteriai gali sugrupuoti „apvalius muzikos daiktus“, „kampuotus muzikos daiktus“, „trikampius daiktus“ ir „sausainius“. Jei suteiktume apribojimus arba taisykles („daiktas turi būti pagamintas iš plastiko“, „daiktas turi galėti groti muziką“), tai padėtų algoritmui pasirinkti geriau.
> 
> 🎓 Tankis
> 
> Duomenys, laikomi „triukšmingais“, priskiriami prie „tankaus“. Atstumai tarp taškų kiekviename klasteryje gali būti mažiau arba labiau tankūs, ar „suspausti“, todėl duomenis reikia analizuoti su tinkamu klasterizavimo metodu. [Šiame straipsnyje](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) parodytas skirtumas tarp K-Means ir HDBSCAN algoritmų, analizuojant triukšmingą duomenų rinkinį su nevienodu klasterių tankiu.

## Klasterizavimo algoritmai

Yra daugiau nei 100 klasterizavimo algoritmų, o jų naudojimas priklauso nuo turimų duomenų pobūdžio. Aptarkime kai kuriuos svarbiausius:

- **Hierarchinis klasterizavimas**. Jei objektas klasifikuojamas pagal artimumą prie kito artimo objekto, o ne prie tolimesnio, klasteriai formuojami remiantis narių atstumais iki kitų objektų. Scikit-learn aglomeracinis klasterizavimas yra hierarchinis.

   ![Hierarchinio klasterizavimo infografika](../../../../translated_images/lt/hierarchical.bf59403aa43c8c47.webp)
   > Infografika: [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroidinis klasterizavimas**. Šis populiarus algoritmas reikalauja pasirinkti „k“ – suformuojamų klasterių skaičių, po ko algoritmas nustato klasterio centro tašką ir renkasi duomenis aplink šį tašką. [K-means klasterizavimas](https://wikipedia.org/wiki/K-means_clustering) yra populiari centroidinio klasterizavimo versija. Centras nustatomas pagal artimiausią vidurkį, iš čia ir pavadinimas. Kvadratinis atstumas iki klasterio sumažinamas iki minimumo.

   ![Centroidinio klasterizavimo infografika](../../../../translated_images/lt/centroid.097fde836cf6c918.webp)
   > Infografika: [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Pasiskirstymo pagrindu klasterizavimas**. Remiasi statistiniu modeliavimu, pagrindinis dėmesys skiriamas nustatyti tikimybę, kad duomenų taškas priklauso klasteriui, ir atitinkamai priskiria jam klasterį. Priklauso Gauso mišinių metodai.

- **Tankio pagrindu klasterizavimas**. Duomenų taškai priskiriami klasteriams pagal jų tankį arba grupavimąsi vienas prie kito. Atstumai tarp tolimų taškų laikomi iškritusiais arba triukšmu. DBSCAN, Mean-shift ir OPTICS algoritmai priklauso šiai klasterizavimo rūšiai.

- **Tinklelio pagrindu klasterizavimas**. Multimatiniai duomenys paskirstomi į tinklelio langelius, kurie sukuria klasterius.

## Užduotis – klasterizuokite savo duomenis

Klasterizavimas kaip technika labai palengvinamas geru vizualizavimu, tad pradėkime nuo mūsų muzikos duomenų vizualizavimo. Ši užduotis padės nuspręsti, kurią klasterizavimo metodą efektyviausiai naudoti su šių duomenų pobūdžiu.

1. Atidarykite šį aplanke esantį [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) failą.

1. Importuokite `Seaborn` paketą, skirtą geram duomenų vaizdavimui.

    ```python
    !pip install seaborn
    ```

1. Užpildykite dainų duomenis iš [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Įkelkite duomenų rėmelį su informacija apie dainas. Pasiruoškite tirti šiuos duomenis importuodami bibliotekas ir išvedami duomenis:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Patikrinkite pirmas kelias duomenų eilutes:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Gaukite šiek tiek informacijos apie duomenų rėmelį, iškviesdami `info()`:

    ```python
    df.info()
    ```

   Išvestis atrodys maždaug taip:

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

1. Patikrinkite, ar nėra tuščių reikšmių, iškviesdami `isnull()` ir patikrindami, ar suma lygina 0:

    ```python
    df.isnull().sum()
    ```

    Atrodo gerai:

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

1. Aprašykite duomenis:

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

> 🤔 Jei dirbame su klasterizacija, nekontroliuojama metodu, kuriam nereikia žymėtų duomenų, kodėl šiuos duomenis rodome su žymomis? Duomenų tyrimo etape tai yra naudinga, bet klasterizacijos algoritmams ženklai nėra būtini. Galite tiesiog pašalinti stulpelių antraštes ir nurodyti duomenis pagal stulpelių numerį.

Pažiūrėkite į bendrą duomenų reikšmių vaizdą. Atkreipkite dėmesį, kad populiarumas gali būti '0', rodantis dainas, neturinčias reitingo. Pašalinkime jas netrukus.

1. Naudokite stulpelinę diagramą, kad sužinotumėte pačias populiariausias žanrus:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/lt/popular.9c48d84b3386705f.webp)

✅ Jei norite pamatyti daugiau didžiausių reikšmių, pakeiskite `[:5]` į didesnę reikšmę arba pašalinkite ją, kad matytumėte visas.

Atkreipkite dėmesį, kai aukščiausias žanras apibūdinamas kaip 'Missing', tai reiškia, kad Spotify jo neklasifikavo, tad pašalinkime jį.

1. Pašalinkite trūkstamus duomenis filtruodami juos

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Dabar patikrinkite žanrus dar kartą:

    ![most popular](../../../../translated_images/lt/all-genres.1d56ef06cefbfcd6.webp)

1. Tolimųjų trys žanrai dominuoja šiame duomenų rinkinyje. Susikoncentruokime į `afro dancehall`, `afropop` ir `nigerian pop`, papildomai filtruodami duomenų rinkinį, kad pašalintume viską, kur populiarumas yra 0 (tai reiškia, kad duomenų rinkinyje nebuvo priskirtas populiarumas ir tai gali būti triukšmas mums):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Atlikite greitą testą, ar duomenys rimtai koreliuoja:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/lt/correlation.a9356bb798f5eea5.webp)

    Vienintelė stipri koreliacija yra tarp `energy` ir `loudness`, kas nėra labai stebinanti, kadangi garsioji muzika dažnai yra gana energinga. Kitais atvejais koreliacijos yra gana silpnos. Įdomu bus pamatyti, ką klasterizacijos algoritmas su šiais duomenimis gali sukurti.

    > 🎓 Atkreipkite dėmesį, kad koreliacija nereiškia priežastingumo! Turime koreliacijos įrodymą, bet ne priežastingumo įrodymą. [Juokingas tinklalapis](https://tylervigen.com/spurious-correlations) turi vizualizacijų, kurios pabrėžia šį faktą.

Ar šie duomenys sutampa apie dainos suvokiamą populiarumą ir šokamumą? FacetGrid rodo, kad egzistuoja koncentrinės žiedinės linijos, nepriklausomai nuo žanro. Ar gali būti, kad Nigerijos muzikiniai skoniai sutampa tam tikru šokamumo lygiu šiam žanrui?  

✅ Išbandykite skirtingus duomenų taškus (energija, garsumas, kalbėjimo intensyvumas) ir daugiau ar skirtingų muzikos žanrų. Ką galite atrasti? Peržiūrėkite `df.describe()` lentelę, kad pamatytumėte bendrą duomenų taškų pasiskirstymą.

### Pratimai – duomenų pasiskirstymas

Ar šie trys žanrai žymiai skiriasi jų šokamumo suvokimu, remiantis jų populiarumu?

1. Išnagrinėkite mūsų trijų geriausių žanrų duomenų pasiskirstymą populiarumui ir šokamumui duotose x ir y ašyse.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Galite aptikti koncentrines žiedines linijas aplink bendrą sankirtos tašką, rodantį taškų pasiskirstymą.

    > 🎓 Atkreipkite dėmesį, kad šis pavyzdys naudoja KDE (Branduolio tankio įvertį) grafiką, kuris reprezentuoja duomenis naudojant tęstinį tikimybės tankio kreivę. Tai leidžia interpretuoti duomenis dirbant su keliais pasiskirstymais.

    Apskritai, trys žanrai silpnai sutampa pagal jų populiarumą ir šokamumą. Nustatyti klasterius šiame silpnai sutampančiame duomenyje bus iššūkis:

    ![distribution](../../../../translated_images/lt/distribution.9be11df42356ca95.webp)

1. Sukurkite išsklaidytos diagramos grafiką:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Tokia pat ašių išsklaidytos diagramos grafika rodo panašią sankirtos struktūrą

    ![Facetgrid](../../../../translated_images/lt/facetgrid.9b2e65ce707eba1f.webp)

Apskritai, klasterizacijai galite naudoti išsklaidytos diagramas, kad parodytumėte duomenų klasterius, todėl šio vizualizavimo tipo įvaldymas yra labai naudingas. Kitame pamokoje naudosime šiuos filtruotus duomenis ir atliksime k-means klasterizaciją, kad atrastume grupes duomenyse, kurios, atrodo, įdomiai sutampa.

---

## 🚀Iššūkis

Pasiruošiant kitam pamokui, sukurkite diagramą apie įvairius klasterizacijos algoritmus, kuriuos galėtumėte atrasti ir naudoti gamybos aplinkoje. Kokias problemas klasterizacija bando spręsti?

## [Po paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

## Peržiūra ir savarankiškas mokymasis

Prieš taikant klasterizacijos algoritmus, kaip mes išmokome, verta suprasti savo duomenų rinkinio pobūdį. Apie tai skaitykite daugiau [čia](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html).

[Šis naudingas straipsnis](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) paaiškina skirtingų klasterizacijos algoritmų elgesį, atsižvelgiant į skirtingas duomenų formas.

## Namų darbai

[Tirti kitas vizualizacijas klasterizacijai](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Atsakomybės apribojimas**:
Šis dokumentas buvo išverstas naudojant dirbtinio intelekto vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba laikomas autoritetingu šaltiniu. Svarbiai informacijai rekomenduojama naudoti profesionalų žmogiškąjį vertimą. Mes neatsakome už jokius nesusipratimus ar neteisingą interpretaciją, kilusią naudojantis šiuo vertimu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->