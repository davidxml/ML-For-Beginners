# Uvod v gručenje

Gručenje je vrsta [nenadzorovanega učenja](https://wikipedia.org/wiki/Unsupervised_learning), ki predpostavlja, da je podatkovni niz neoznačen ali da njegovi vhodi niso usklajeni s predhodno določenimi izhodi. Uporablja različne algoritme za razvrščanje neoznačenih podatkov in zagotavlja skupine glede na vzorce, ki jih zazna v podatkih.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Kliknite sliko zgoraj za video. Medtem ko se učite strojno učenje z gručenjem, uživajte v afriških plesnih muzikah - to je visoko ocenjeno pesem iz leta 2014 od PSquare.

## [Kviz pred predavanjem](https://ff-quizzes.netlify.app/en/ml/)

### Uvod

[Gručenje](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) je zelo uporabno za raziskovanje podatkov. Poglejmo, ali lahko pomaga odkriti trende in vzorce v načinu, kako nigerijska publika posluša glasbo.

✅ Vzemite si minuto za razmislek o uporabi gručenja. V resničnem življenju gručenje nastane, kadar imate kup perila in morate razvrstiti oblačila družinskih članov 🧦👕👖🩲. V podatkovni znanosti nastane gručenje, ko poskušate analizirati uporabnikove preference ali določiti značilnosti katerega koli neoznačenega podatkovnega niza. Gručenje, na nek način, pomaga razumeti kaos, kot predal za nogavice.

[![Uvod v ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Uvod v gručenje")

> 🎥 Kliknite sliko zgoraj za video: John Guttag z MIT predstavlja gručenje

V profesionalnem okolju se gručenje lahko uporablja za določanje stvari, kot je segmentacija trga, določanje, katere starostne skupine kupujejo katere izdelke, na primer. Druga uporaba bi bila odkrivanje anomalij, morda za zaznavanje goljufij iz niza podatkov o transakcijah s kreditnimi karticami. Ali pa bi lahko uporabili gručenje za določanje tumorjev v nizu medicinskih posnetkov.

✅ Vzemite si minuto, da premislite, kje ste morda naleteli na gručenje 'v naravi', v bankarstvu, e-trgovini ali poslovnem okolju.

> 🎓 Zanimivo je, da je analiza gruče nastala na področjih antropologije in psihologije v tridesetih letih prejšnjega stoletja. Si lahko predstavljate, kako je bilo uporabljeno?

Alternativno bi ga lahko uporabili za združevanje rezultatov iskanja - na primer po nakupovalnih povezavah, slikah ali ocenah. Gručenje je uporabno, kadar imate velik podatkovni niz, ki ga želite zmanjšati in na katerem želite izvesti bolj granulirano analizo, zato se tehnika lahko uporablja za spoznavanje podatkov, preden so zgrajeni drugi modeli.

✅ Ko so vaši podatki organizirani v gruče, jih označite s številko gruče, in ta tehnika je lahko uporabna pri ohranjanju zasebnosti podatkov; lahko se namesto tega sklicujete na podatkovno točko po številki gruče, namesto po bolj razkrivajočih identifikacijskih podatkih. Se lahko spomnite drugih razlogov, zakaj bi se sklicevali na številko gruče namesto na druge elemente gruče za identifikacijo?

Poglobite svoje razumevanje gruče v tem [učnem modulu](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)
## Začetek z gručenjem

[Scikit-learn ponuja širok nabor](https://scikit-learn.org/stable/modules/clustering.html) metod za izvajanje gručenja. Izbran tip bo odvisen od vašega primera uporabe. Po dokumentaciji ima vsaka metoda različne prednosti. Tukaj je poenostavljena tabela metod, ki jih podpira Scikit-learn, in njihovih primernih primerov uporabe:

| Ime metode                  | Primer uporabe                                                        |
| :--------------------------- | :------------------------------------------------------------------- |
| K-Means                      | splošna uporaba, induktivno                                         |
| Affinity propagation         | veliko, neenakomerne gruče, induktivno                              |
| Mean-shift                   | veliko, neenakomerne gruče, induktivno                              |
| Spektralno gručenje          | malo, enakomerno, transduktivno                                      |
| Ward hierarhično gručenje    | veliko, omejene gruče, transduktivno                                |
| Agglomerativno gručenje      | veliko, omejeno, neevklidske razdalje, transduktivno                |
| DBSCAN                       | neploščata geometrija, neenakomerne gruče, transduktivno            |
| OPTICS                       | neploščata geometrija, neenakomerne gruče z različnimi gostotami, transduktivno |
| Gaussove mešanice            | ploščata geometrija, induktivno                                     |
| BIRCH                        | velik podatkovni niz z izstopajočimi vrednostmi, induktivno         |

> 🎓 Kako ustvarjamo gruče je močno povezano s tem, kako združujemo podatkovne točke v skupine. Poglejmo nekaj besedišča:
>
> 🎓 ['Transduktivno' proti 'induktivno'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktivno sklepanje izhaja iz opazovanih učnih primerov, ki so povezani s specifičnimi testnimi primeri. Induktivno sklepanje izhaja iz učnih primerov, ki so povezani s splošnimi pravili, ki se nato uporabijo za testne primere.
> 
> Primer: Predstavljajte si, da imate podatkovni niz, ki je le delno označen. Nekatere stvari so 'vinilke', druge 'cd-ji', nekatere pa so prazne. Vaša naloga je, da za prazne zagotovite oznake. Če se odločite za induktivni pristop, boste izučili model za 'vinilke' in 'cd-je' in te oznake uporabili na svojih neoznačenih podatkih. Ta pristop bo imel težave pri klasifikaciji stvari, ki so pravzaprav 'kasete'. Transduktivni pristop pa bolje obravnava te neznane podatke, saj jih skuša združiti v skupine in nato skupinam dodeli oznake. V tem primeru gruče lahko odražajo 'okrogle glasbene stvari' in 'kvadratne glasbene stvari'.
> 
> 🎓 ['Neploščata' proti 'ploščata' geometrija](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Izvira iz matematične terminologije; neploščata proti ploščati geometriji se nanaša na merjenje razdalj med točkami bodisi z 'ploščatimi' ([evklidskimi](https://wikipedia.org/wiki/Euclidean_geometry)) bodisi z 'neploščatimi' (neevklidskimi) geometrijskimi metodami.
>
>  'Ploščata' v tem kontekstu pomeni evklidsko geometrijo (del katere se uči kot geometrija ravnine), neploščata pa je neevklidska geometrija. Kaj ima geometrija opraviti z učenjem stroja? Ker sta obe področji zasidrani v matematiki, mora obstajati skupen način merjenja razdalj med točkami v gruči, kar je mogoče storiti na 'ploščat' ali 'neploščat' način, odvisno od narave podatkov. [Evklidske razdalje](https://wikipedia.org/wiki/Euclidean_distance) se merijo kot dolžina daljice med dvema točkama. [Nevklidske razdalje](https://wikipedia.org/wiki/Non-Euclidean_geometry) pa se merijo po krivulji. Če vaši podatki, vizualizirani, niso na ravnini, boste morda potrebovali poseben algoritem.
>
![Infografika ploščate proti neploščati geometriji](../../../../translated_images/sl/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infografika avtorja [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Razdalje'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Gruče so definirane z njihovo matriko razdalj, npr. razdaljami med točkami. Te razdalje se lahko merijo na različne načine. Evklidske gruče so definirane kot povprečje vrednosti točk in vsebujejo 'centroid' ali središčno točko. Razdalje se nato merijo do tega centroida. Neevklidske razdalje pomenijo 'clustroide', točko, ki je najbližje drugim točkam. Clustroide, v nadaljevanju, lahko definiramo na različne načine.
> 
> 🎓 ['Omejeno'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Omejeno gručenje](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) uvede 'polonadzorovano' učenje v to nenadzorovano metodo. Razmerja med točkami so označena kot 'ni mogoče povezati' ali 'mora povezati', zato so nekatera pravila prisiljena v podatkovni niz.
>
> Primer: Če je algoritem sproščen na določeni množici neoznačenih ali delno označenih podatkov, so gruče lahko nizke kakovosti. V zgornjem primeru bi gruče lahko grupirale 'okrogle glasbene stvari' in 'kvadratne glasbene stvari' ter 'trikotne stvari' in 'piškote'. Če dobite nekaj omejitev ali pravil ("izdelek mora biti iz plastike", "izdelek mora znati proizvajati glasbo"), to lahko pomaga 'omejiti' algoritem k boljšim odločitvam.
> 
> 🎓 Gostota
> 
> Podatki, ki so 'hrupni', se štejejo za 'goste'. Razdalje med točkami v vsaki gruči so lahko ob pregledu bolj ali manj goste oziroma 'gneteče', zato je treba te podatke analizirati z ustrezno metodo gručenja. [Ta članek](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) prikazuje razliko med uporabo K-Means gručenja in algoritmom HDBSCAN za raziskovanje hrupnih podatkov z neenakomerno gostoto gruče.

## Algoritmi gručenja

Obstaja več kot 100 algoritmov gručenja, njihova uporaba pa je odvisna od narave podatkov na voljo. Pogovorimo se o nekaterih glavnih:

- **Hierarhično gručenje**. Če je objekt klasificiran glede na bližino bližnjega objekta namesto na tistega dlje, nastanejo gruče na podlagi razdalj med njihovimi člani in drugimi objekti. Agglomerativno gručenje Scikit-learna je hierarhično.

   ![Infografika hierarhičnega gručenja](../../../../translated_images/sl/hierarchical.bf59403aa43c8c47.webp)
   > Infografika avtorja [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Gručenje centroidov**. Ta priljubljen algoritem zahteva izbiro 'k', števila gruče, ki jih je treba oblikovati, nato algoritem določi središčno točko gruče in zbere podatke okoli nje. [K-means gručenje](https://wikipedia.org/wiki/K-means_clustering) je priljubljena različica gručenja centroidov. Središče določi najbližje povprečje, od tod ime. Kvadratna razdalja od gruče je minimalizirana.

   ![Infografika gručenja centroidov](../../../../translated_images/sl/centroid.097fde836cf6c918.webp)
   > Infografika avtorja [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Gručenje na podlagi porazdelitve**. Temelji na statističnem modeliranju, pri katerem gručenje porazdelitve določa verjetnost, da podatkovna točka pripada gruči, in jo temu primerno dodeli. Metode gaussovih mešanic spadajo v to vrsto.

- **Gručenje na podlagi gostote**. Podatkovne točke so dodeljene gručam na podlagi njihove gostote oziroma medsebojnega združevanja. Točke, oddaljene od skupine, se štejejo za izstopajoče ali hrup. DBSCAN, Mean-shift in OPTICS so vrste tega gručenja.

- **Rastlinsko gručenje**. Za večdimenzionalne podatkovne nize se ustvari mreža in podatki se razdelijo med celice mreže, kar ustvarja gruče.

## Vaja - grupirajte svoje podatke

Gručenje kot tehnika je bistveno olajšano z ustrezno vizualizacijo, zato začnimo z vizualizacijo naših podatkov o glasbi. Ta vaja nam bo pomagala odločiti, katero metodo gručenja naj učinkoviteje uporabimo za naravo teh podatkov.

1. Odprite datoteko [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) v tej mapi.

1. Uvozite paket `Seaborn` za dobro vizualizacijo podatkov.

    ```python
    !pip install seaborn
    ```

1. Dodajte podatke o skladbah iz [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Naložite dataframe z nekaj podatki o skladbah. Pripravite se na raziskovanje teh podatkov z uvozom knjižnic in prikazom podatkov:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Preverite nekaj prvih vrstic podatkov:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Užívaj si življenje         | Lady Donli          | nigerijski pop   | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Pridobite nekaj informacij o podatkovnem okviru, tako da pokličete `info()`:

    ```python
    df.info()
    ```

   Izhod izgleda takole:

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

1. Dvakrat preverite, ali obstajajo manjkajoče vrednosti, tako da pokličete `isnull()` in preverite, ali je vsota 0:

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

1. Opis podatkov:

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

> 🤔 Če delamo s gručenjem, nenadzorovano metodo, ki ne zahteva označenih podatkov, zakaj te podatke prikazujemo z oznakami? V fazi raziskovanja podatkov pridejo prav, vendar niso potrebne za delovanje algoritmov gručenja. Stolpce lahko prav tako preprosto odstranite in se na podatke sklicujete po številki stolpca.

Poglejte splošne vrednosti podatkov. Upoštevajte, da je priljubljenost lahko '0', kar kaže na pesmi brez uvrstitve. Te bomo kmalu odstranili.

1. Uporabite stolpični grafikon, da ugotovite najbolj priljubljene zvrsti:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![najbolj priljubljene](../../../../translated_images/sl/popular.9c48d84b3386705f.webp)

✅ Če želite videti več vrhunskih vrednosti, spremenite zgornjo omejitev `[:5]` na večjo vrednost ali jo odstranite, da vidite vse.

Opomba: če je najvišja zvrst opisana kot 'Manjkajoče', to pomeni, da je Spotify ni uvrstil, zato jo odstranimo.

1. Odstranite manjkajoče podatke s filtriranjem

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Zdaj ponovno preverite zvrsti:

    ![najbolj priljubljene](../../../../translated_images/sl/all-genres.1d56ef06cefbfcd6.webp)

1. Najbolj prevladujejo tri zvrsti v tem naboru podatkov. Osredotočili se bomo na `afro dancehall`, `afropop` in `nigerian pop`, dodatno filtrirali nabor podatkov, da odstranimo vse pesmi z vrednostjo priljubljenosti 0 (kar pomeni, da ni bila ocenjena glede priljubljenosti v naboru in jo lahko za naše namene obravnavamo kot šum):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Naredite kratek test, da preverite, ali podatki korelirajo na kak poseben močan način:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![korelacije](../../../../translated_images/sl/correlation.a9356bb798f5eea5.webp)

    Edina močna korelacija je med `energetičnostjo` in `glasnostjo`, kar ni presenečenje, saj je glasna glasba ponavadi precej energična. Drugače so korelacije razmeroma šibke. Zanimivo bo videti, kaj bo gručevalni algoritem naredil s temi podatki.

    > 🎓 Upoštevajte, da korelacija ne pomeni vzročne zveze! Imamo dokaz korelacije, a ne dokaz vzročnosti. [Zabavna spletna stran](https://tylervigen.com/spurious-correlations) ponuja vizualizacije, ki to poudarjajo.

Ali obstaja kakršna koli konvergenca v tem naboru podatkov glede zaznane priljubljenosti in plesnosti pesmi? FacetGrid kaže koncentrične kroge, ki se ujemajo ne glede na zvrst. Ali bi bilo lahko, da se nigerijski okusi na določeni ravni plesnosti konvergirajo za to zvrst?

✅ Preizkusite različne podatkovne točke (energija, glasnost, govorljivost) in več ali različne glasbene zvrsti. Kaj odkrijete? Poglejte tabelo `df.describe()`, da vidite splošno razporeditev podatkovnih točk.

### Naloga - porazdelitev podatkov

Ali se te tri zvrsti bistveno razlikujejo v dojemanju plesnosti glede na njihovo priljubljenost?

1. Preglejte porazdelitev podatkov za priljubljenost in plesnost za naše tri vrhunske zvrsti na dani x in y osi.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Odkrijete lahko koncentrične kroge okoli splošne točke konvergence, ki prikazujejo razporeditev točk.

    > 🎓 Ta primer uporablja graf KDE (cenitev gostote jedra), ki podatke predstavlja z neprekinjeno krivuljo verjetnostne gostote. To omogoča interpretacijo podatkov pri delu z več distribucijami.

    Na splošno se tri zvrsti ohlapno poravnajo glede priljubljenosti in plesnosti. Določanje gruče v teh ohlapno poravnanih podatkih bo izziv:

    ![porazdelitev](../../../../translated_images/sl/distribution.9be11df42356ca95.webp)

1. Ustvarite razpršitveni grafikon:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Razpršitveni grafikon istih osi kaže podoben vzorec konvergence

    ![Facetgrid](../../../../translated_images/sl/facetgrid.9b2e65ce707eba1f.webp)

Na splošno lahko za gručenje uporabite razpršitvene grafikone za prikaz skupin podatkov, zato je obvladovanje te vrste vizualizacije zelo uporabno. V naslednjem poglavju bomo uporabili ta filtrirani nabor podatkov in uporabili gručevalni algoritem k-means, da odkrijemo skupine, ki se zdijo v podatkih zanimivo prekrivajoče.

---

## 🚀Izziv

Za pripravo na naslednje poglavje naredite grafikon različnih gručevalnih algoritmov, ki jih lahko odkrijete in uporabljate v produkcijskem okolju. Katere vrste problemov skuša gručenje rešiti?

## [Kviz po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

Preden uporabite gručevalne algoritme, kot smo se naučili, je dobro razumeti naravo vašega nabora podatkov. Preberite več o tej temi [tukaj](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Ta koristni članek](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) vas popelje skozi različne načine, kako se obnašajo različni gručevalni algoritmi glede na različne oblike podatkov.

## Domača naloga

[Raziskujte druge vizualizacije za gručenje](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Omejitev odgovornosti**:
Ta dokument je bil preveden z uporabo AI prevajalske storitve [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da avtomatizirani prevodi lahko vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za kritične informacije je priporočljiv strokovni človeški prevod. Ne odgovarjamo za morebitna nesporazume ali napačne interpretacije, ki izhajajo iz uporabe tega prevoda.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->