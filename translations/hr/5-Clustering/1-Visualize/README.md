# Uvod u grupiranje

Grupiranje je vrsta [nenadzirnog učenja](https://wikipedia.org/wiki/Unsupervised_learning) koja pretpostavlja da je skup podataka neoznačen ili da njegovi ulazi nisu povezani s unaprijed definiranim izlazima. Koristi različite algoritme za sortiranje neoznačenih podataka i pruža grupiranja u skladu s obrascima koje prepoznaje u podacima.

[![Ne postoji nitko poput tebe od PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "Ne postoji nitko poput tebe od PSquare")

> 🎥 Kliknite sliku iznad za video. Dok proučavate strojno učenje s grupiranjem, uživajte u nekim nigerijskim Dance Hall pjesmama - ovo je vrlo cijenjena pjesma iz 2014. godine od PSquare.

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

### Uvod

[Grupiranje](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) je vrlo korisno za istraživanje podataka. Pogledajmo može li pomoći u otkrivanju trendova i obrazaca u načinu na koji nigerijska publika konzumira glazbu.

✅ Odvojite minutu za razmišljanje o upotrebi grupiranja. U stvarnom životu, grupiranje se događa kad imate hrpu rublja i trebate razvrstati odjeću članova obitelji 🧦👕👖🩲. U data scienceu, grupiranje se događa pri pokušaju analize korisničkih preferencija ili određivanju značajki bilo kojeg neoznačenog skupa podataka. Grupiranje, na neki način, pomaže da se smisli kaos, poput ladice za čarape.

[![Uvod u ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Uvod u grupiranje")

> 🎥 Kliknite sliku iznad za video: John Guttag s MIT-a predstavlja grupiranje

U profesionalnom okruženju, grupiranje se može koristiti za određivanje stvari poput segmentacije tržišta, određivanja koje dobne skupine kupuju koje proizvode, na primjer. Druga upotreba može biti otkrivanje anomalija, možda za prepoznavanje prijevara iz skupa podataka o transakcijama kreditnih kartica. Ili biste mogli koristiti grupiranje za određivanje tumora u skupu medicinskih snimaka.

✅ Razmislite minutu o tome kako ste se možda susreli s grupiranjem 'u prirodi', u bankarstvu, e-trgovini ili poslovnom okruženju.

> 🎓 Zanimljivo, analiza skupina potječe iz područja antropologije i psihologije 1930-ih. Možete li zamisliti kako je možda bila korištena?

Alternativno, mogli biste je koristiti za grupiranje rezultata pretraživanja - po poveznicama za kupovinu, slikama ili recenzijama, na primjer. Grupiranje je korisno kad imate veliki skup podataka koji želite smanjiti i na kojem želite izvršiti detaljniju analizu, pa se tehnika može upotrijebiti za učenje o podacima prije nego što se izgrade drugi modeli.

✅ Kad su vaši podaci organizirani u skupine, dodijelite im identifikacije skupina, i ova tehnika može biti korisna pri očuvanju privatnosti skupa podataka; umjesto toga možete se pozivati na podatkovnu točku prema identifikaciji skupine, a ne po više razotkrivajućim identificirajućim podacima. Možete li smisliti druge razloge zašto biste koristili identifikaciju skupine umjesto drugih elemenata skupine za njezino identificiranje?

Produbite svoje razumijevanje tehnika grupiranja u ovom [Learn modulu](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)
## Početak rada s grupiranjem

[Scikit-learn nudi širok spektar](https://scikit-learn.org/stable/modules/clustering.html) metoda za izvođenje grupiranja. Vrsta koju odaberete ovisit će o vašem slučaju uporabe. Prema dokumentaciji, svaka metoda ima različite prednosti. Evo pojednostavljene tablice metoda podržanih od strane Scikit-learn i njihovih primjerenih slučajeva uporabe:

| Naziv metode               | Slučaj uporabe                                                      |
| :------------------------- | :----------------------------------------------------------------- |
| K-Means                    | opća namjena, induktivno                                           |
| Affinity propagation       | mnogo, nejednake skupine, induktivno                               |
| Mean-shift                 | mnogo, nejednake skupine, induktivno                               |
| Spectral clustering        | malo, jednake skupine, transduktivno                               |
| Ward hierarchical clustering | mnogo, ograničene skupine, transduktivno                         |
| Agglomerative clustering   | mnogo, ograničene, ne Euklidske udaljenosti, transduktivno         |
| DBSCAN                     | ne ravna geometrija, nejednake skupine, transduktivno             |
| OPTICS                     | ne ravna geometrija, nejednake skupine s promjenjivom gustoćom, transduktivno |
| Gaussian mixtures          | ravna geometrija, induktivno                                       |
| BIRCH                      | veliki skup podataka s odmetnicima, induktivno                     |

> 🎓 Kako stvaramo skupine jako ovisi o načinu na koji skupljamo podatkovne točke u grupe. Razjasnimo malo vokabular:
>
> 🎓 ['Transduktivno' vs. 'induktivno'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktivno zaključivanje proizlazi iz promatranih primjera za treniranje koji se preslikavaju na određene test primjere. Induktivno zaključivanje se izvodi iz primjera za treniranje koji se preslikavaju na opća pravila koja se tek tada primjenjuju na test primjere.
> 
> Primjer: Zamislite da imate skup podataka koji je samo djelomično označen. Neke stvari su 'vinili', neke 'CD-ovi', a neke su prazne. Vaš zadatak je dati oznake praznima. Ako koristite induktivni pristup, trenirali biste model koji traži 'vinile' i 'CD-ove' te te oznake primijenili na neoznačene podatke. Taj pristup će imati problema s klasificiranjem stvari koje su zapravo 'kazete'. Transduktivni pristup, s druge strane, učinkovitije rukuje nepoznatim podacima jer radi na grupiranju sličnih stvari i zatim primjenjuje oznaku na grupu. U ovom slučaju, skupine bi mogle odražavati 'okrugle glazbene stvari' i 'kockaste glazbene stvari'.
> 
> 🎓 ['Ne-ravna' vs. 'ravna' geometrija](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Izvedeno iz matematičke terminologije, ne-ravna vs. ravna geometrija odnosi se na mjerenje udaljenosti između točaka pomoću 'ravnih' ([Euklidskih](https://wikipedia.org/wiki/Euclidean_geometry)) ili 'ne-ravnih' (ne-Euklidskih) geometrijskih metoda.
>
> 'Ravno' u ovom kontekstu odnosi se na Euklidsku geometriju (dijelovi koje se uče kao 'ravninska' geometrija), a ne-ravno na ne-Euklidsku geometriju. Što geometrija ima s učenjem stroja? Kao dva područja utemeljena u matematici, mora postojati zajednički način za mjerenje udaljenosti između točaka u skupinama, a to se može učiniti 'ravnim' ili 'ne-ravnim' načinom, ovisno o prirodi podataka. [Euklidske udaljenosti](https://wikipedia.org/wiki/Euclidean_distance) mjere se kao duljina duž segmenta između dvije točke. [Ne-Euklidske udaljenosti](https://wikipedia.org/wiki/Non-Euclidean_geometry) mjere se duž krivulje. Ako vaši podaci, vizualizirani, ne postoje na ravnini, možda ćete trebati koristiti specijalizirani algoritam za njihovo rukovanje.
>
![Infografika ravna vs neravna geometrija](../../../../translated_images/hr/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infografika autora [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Udaljenosti'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Skupine se definiraju njihovom matricom udaljenosti, tj. udaljenostima između točaka. Ta se udaljenost može mjeriti na nekoliko načina. Euklidske skupine definiraju se prosjekom vrijednosti točaka i sadrže 'centroid' ili središnju točku. Udaljenosti se tako mjere do tog centroida. Ne-euklidske udaljenosti odnose se na 'klustroide', točku najbližu drugim točkama. Klustroidi se pak mogu definirati na različite načine.
> 
> 🎓 ['Ograničeno'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Ograničeno grupiranje](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) uvodi 'polunadzirano' učenje u ovu nenadziranu metodu. Odnosi između točaka označavaju se kao 'ne može se povezati' ili 'mora se povezati', pa se na skup podataka nameću neka pravila.
>
>Primjer: Ako se algoritam pusti slobodno na skup neoznačenih ili poluoznačenih podataka, skupine koje stvara mogu biti loše kvalitete. U prethodnom primjeru, skupine bi mogle objediniti 'okrugle glazbene stvari', 'kockaste glazbene stvari', 'trokutaste stvari' i 'kolačiće'. Ako se dodaju neka ograničenja ili pravila koje treba slijediti ("proizvod mora biti od plastike", "proizvod treba moći proizvoditi glazbu"), to može pomoći algoritmu da napravi bolje izbore.
> 
> 🎓 'Gustoća'
> 
> Podaci koji su 'bučni' smatraju se 'gustom' skupinom. Udaljenosti između točaka u svakoj od njihovih skupina mogu se pokazati prilikom proučavanja kao više ili manje guste, ili 'zbijene', pa te podatke treba analizirati odgovarajućom metodom grupiranja. [Ovaj članak](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) prikazuje razliku između korištenja K-Means grupiranja i HDBSCAN algoritama za istraživanje bučnog skupa podataka s nejednakom gustoćom skupina.

## Algoritmi za grupiranje

Postoji preko 100 algoritama za grupiranje, a njihova upotreba ovisi o prirodi raspoloživih podataka. Razmotrimo neke od glavnih:

- **Hijerarhijsko grupiranje**. Ako se objekt klasificira prema svojoj blizini bližem objektu, a ne onom udaljenijem, skupine se formiraju na temelju udaljenosti članova prema drugim objektima. Agglomerativno grupiranje u Scikit-learnu je hijerarhijsko.

   ![Infografika hijerarhijsko grupiranje](../../../../translated_images/hr/hierarchical.bf59403aa43c8c47.webp)
   > Infografika autora [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroidno grupiranje**. Ovaj popularni algoritam zahtijeva odabir 'k', odnosno broja skupina za formiranje, nakon čega algoritam određuje središnju točku skupine i okuplja podatke oko te točke. [K-means grupiranje](https://wikipedia.org/wiki/K-means_clustering) je popularna verzija centroidnog grupiranja. Centar se određuje prema najbližem prosjeku, odakle i naziv. Kvadratna udaljenost od skupine se minimizira.

   ![Infografika centroidno grupiranje](../../../../translated_images/hr/centroid.097fde836cf6c918.webp)
   > Infografika autora [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Grupiranje temeljeno na distribuciji**. Temeljeno na statističkom modeliranju, grupiranje temeljeno na distribuciji fokusira se na određivanje vjerojatnosti da podatkovna točka pripada skupini i odgovarajuću joj dodjelu. Metode Gaussovih mješavina pripadaju ovoj vrsti.

- **Grupiranje temeljeno na gustoći**. Podatkovnim točkama se dodjeljuju skupine prema njihovoj gustoći, odnosno oko njihove međusobne grupacije. Podatkovne točke udaljene od grupe smatraju se odmetnicima ili šumom. DBSCAN, Mean-shift i OPTICS pripadaju ovoj vrsti grupiranja.

- **Grupiranje temeljeno na mreži**. Za višedimenzionalne skupove podataka, stvara se mreža te se podaci dijele među ćelijama mreže, čime se stvaraju skupine.

## Vježba - grupirajte svoje podatke

Grupiranju kao tehnici znatno pomaže pravilna vizualizacija, pa započnimo vizualizacijom podataka o glazbi. Ova vježba pomoći će nam odlučiti koju od metoda grupiranja trebamo najefikasnije koristiti za prirodu ovih podataka.

1. Otvorite datoteku [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) u ovoj mapi.

1. Uvezite paket `Seaborn` za dobru vizualizaciju podataka.

    ```python
    !pip install seaborn
    ```

1. Dodajte podatke o pjesmama iz [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Učitajte dataframe s nekim podacima o pjesmama. Pripremite se za istraživanje ovih podataka uvozom knjižnica i ispisivanjem podataka:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Provjerite prvih nekoliko redova podataka:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Dobite neke informacije o dataframeu, pozivom `info()`:

    ```python
    df.info()
    ```

   Izlaz izgleda ovako:

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

1. Dvaput provjerite postoje li null vrijednosti, pozivanjem `isnull()` i provjerom da je suma 0:

    ```python
    df.isnull().sum()
    ```

    Izgleda dobro:

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

1. Opis podataka:

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

> 🤔 Ako radimo sa klasteriranjem, ne-nadziranim metodom koja ne zahtijeva označene podatke, zašto prikazujemo ove podatke s oznakama? U fazi istraživanja podataka koristi su korisni, ali nisu potrebni za rad klaster algoritama. Mogli biste jednako tako ukloniti zaglavlja stupaca i referirati se na podatke preko broja stupca.

Pogledajte opće vrijednosti podataka. Napomena da popularnost može biti '0', što znači pjesme koje nemaju rangiranje. Uskoro ćemo ih ukloniti.

1. Upotrijebite barplot da biste saznali najpopularnije žanrove:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/hr/popular.9c48d84b3386705f.webp)

✅ Ako želite vidjeti više vrhunskih vrijednosti, promijenite top `[:5]` u veći broj, ili ga uklonite da vidite sve.

Napomena, kada je vrhunski žanr opisan kao 'Missing', to znači da ga Spotify nije klasificirao, pa ga se riješimo.

1. Uklonite nedostajuće podatke filtriranjem

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Sada ponovno provjerite žanrove:

    ![most popular](../../../../translated_images/hr/all-genres.1d56ef06cefbfcd6.webp)

1. Daleko najdominantnija su tri žanra u ovom skupu podataka. Usredotočimo se na `afro dancehall`, `afropop` i `nigerian pop`, dodatno filtrirajte skup podataka da uklonite sve s vrijednošću popularnosti 0 (što znači da nije klasificiran s popularnošću u skupu podataka i može se smatrati bukom za naše svrhe):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Napravite brzi test da vidite postoji li jaka korelacija među podatcima:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/hr/correlation.a9356bb798f5eea5.webp)

    Jedina jaka korelacija je između `energy` i `loudness`, što nije iznenađujuće, s obzirom da je glasna glazba obično prilično energična. Inače, korelacije su relativno slabe. Bit će zanimljivo vidjeti što algoritam klasteriranja može izvući iz ovih podataka.

    > 🎓 Napomena da korelacija ne podrazumijeva uzročnost! Imamo dokaz korelacije, ali ne i dokaz uzročnosti. [Zabavna web stranica](https://tylervigen.com/spurious-correlations) ima neke vizuale koji naglašavaju ovu točku.

Postoji li konvergencija u ovom skupu podataka oko percipirane popularnosti i plesnosti pjesme? FacetGrid pokazuje da postoje koncentrični krugovi koji se poklapaju, bez obzira na žanr. Može li biti da se nigerijski ukusi konvergiraju na određenoj razini plesnosti za ovaj žanr?

✅ Isprobajte različite točke podataka (energija, glasnoća, govorljivost) i više ili različitih glazbenih žanrova. Što možete otkriti? Pogledajte tablicu `df.describe()` da vidite opći raspon podataka.

### Vježba - distribucija podataka

Jesu li ova tri žanra značajno različita u percepciji njihove plesnosti, na temelju njihove popularnosti?

1. Ispitajte distribuciju podataka naših top tri žanra za popularnost i plesnost duž zadane x i y osi.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Možete otkriti koncentrične krugove oko opće točke konvergencije, što pokazuje raspodjelu točaka.

    > 🎓 Napomena da ovaj primjer koristi KDE (Kernel Density Estimate) graf koji predstavlja podatke korištenjem kontinuirane krivulje gustoće vjerojatnosti. To nam omogućava interpretaciju podataka kada radimo s višestrukim distribucijama.

    Općenito, tri žanra se labavo poravnavaju u pogledu njihove popularnosti i plesnosti. Određivanje klastera u ovom labavo poravnanom skupu podataka bit će izazov:

    ![distribution](../../../../translated_images/hr/distribution.9be11df42356ca95.webp)

1. Napravite scatter plot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Scatterplot istih osi pokazuje sličan obrazac konvergencije

    ![Facetgrid](../../../../translated_images/hr/facetgrid.9b2e65ce707eba1f.webp)

Općenito, za klasteriranje možete koristiti scatter plotove za prikaz klastera podataka, pa je ovladavanje ovom vrstom vizualizacije vrlo korisno. U sljedećem ćemo satu uzeti ove filtrirane podatke i koristiti k-means klasteriranje da otkrijemo skupine u ovim podacima koje se na zanimljiv način preklapaju.

---

## 🚀Izazov

U pripremi za sljedeći sat, napravite grafikon o različitim algoritmima klasteriranja koje biste mogli otkriti i koristiti u proizvodnom okruženju. Koje vrste problema klasteriranje pokušava riješiti?

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Prije nego što primijenite algoritme klasteriranja, kao što smo naučili, dobra je ideja razumjeti prirodu vašeg skupa podataka. Više o ovoj temi pročitajte [ovdje](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Ovaj koristan članak](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) vodi vas kroz različite načine na koje se različiti algoritmi klasteriranja ponašaju, s obzirom na različite oblike podataka.

## Zadatak

[Istražite druge vizualizacije za klasteriranje](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Napomena**:
Ovaj dokument je preveden korištenjem AI prevoditeljskog servisa [Co-op Translator](https://github.com/Azure/co-op-translator). Iako težimo točnosti, imajte na umu da automatski prijevodi mogu sadržavati greške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za važne informacije preporuča se profesionalni ljudski prijevod. Nismo odgovorni za bilo kakva nesporazumevanja ili pogrešne interpretacije koje proizlaze iz korištenja ovog prijevoda.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->