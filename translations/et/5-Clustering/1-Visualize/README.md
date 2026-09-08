# Sissejuhatus klasterdamisse

Klasterdamine on üheks [juhendamata õppimise](https://wikipedia.org/wiki/Unsupervised_learning) tüübiks, mis eeldab, et andmestik on märgistamata või et selle sisendid ei ole seatud eelnevalt kindlaksmääratud väljunditega. See kasutab erinevaid algoritme, et sorteerida märgistamata andmeid ja moodustada rühmi vastavalt mustritele, mida see andmetes tuvastab.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Klõpsa ülaloleval pildil video vaatamiseks. Kui õpid masinõpet klasterdamisega, naudi mõningaid Nigeeria Dance Hall lugusid – see on kõrge hinne saanud lugu aastast 2014 PSquare poolt.

## [Eel loengu test](https://ff-quizzes.netlify.app/en/ml/)

### Sissejuhatus

[Klasterdamine](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) on väga kasulik andmete uurimiseks. Vaatame, kas see aitab avastada trende ja mustreid selles, kuidas Nigeeria publik muusikat tarbib.

✅ Võta minut mõtlemiseks, milleks klasterdamist kasutada saab. Igapäevaelus juhtub klasterdamine alati, kui sul on pesukorv ja vajad oma pereliikmete riideid sorteerida 🧦👕👖🩲. Andmeteaduses toimub klasterdamine siis, kui püütakse analüüsida kasutaja eelistusi või määratleda mis tahes märgistamata andmestiku omadusi. Klasterdamine aitab omamoodi kaosest mõtestada, nagu sokisahtel.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Klõpsa ülaloleval pildil video vaatamiseks: MITi John Guttag tutvustab klasterdamist

Ametlikus keskkonnas võib klasterdamist kasutada näiteks turusegmentide määratlemiseks, näiteks et kindlaks teha, millised vanuserühmad ostavad milliseid tooteid. Teine kasutusala on anomaaliate tuvastamine, näiteks pettuste avastamine krediitkaarditehingute andmestikus. Võid ka kasutada klasterdamist vähkkasvajate määratlemiseks meditsiiniliste skaneeringute hulgas.

✅ Mõtle hetk, kus võid olla kohanud klasterdamist 'looduses', panganduse, e-kaubanduse või ärikeskkonnas.

> 🎓 Huvitaval kombel pärineb klasteranalüüs antropoloogia ja psühholoogia valdkonnast 1930. aastatel. Kas oskad ette kujutada, kuidas seda võidi kasutada?

Või võid kasutada seda otsingutulemuste grupeerimiseks – näiteks ostulinkide, piltide või arvustuste kaupa. Klasterdamine on kasulik, kui sul on suur andmestik, mida soovid kokku tõmmata ja mille peal soovid teha üksikasjalikumat analüüsi, nii et seda tehnikat saab kasutada andmete tundmaõppimiseks enne teiste mudelite ehitamist.

✅ Kui sinu andmed on organiseeritud klastritesse, määrad neile klastritele ID ja see tehnika võib olla kasulik ka andmekaitse säilitamiseks; selle asemel, et viidata andmepunktile selle kirjeldavate tundlike andmetega, võid viidata sellele ainult klastrite ID järgi. Kas suudad mõelda veel põhjuseid, miks võiksid viidata klastrite ID-le, mitte teistele klastrite elementidele, et seda identifitseerida?

Süvendage klasterdamise tehnikate mõistmist selles [õppemoodulis](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

## Klasterdamisega alustamine

[Scikit-learn pakub laia valikut](https://scikit-learn.org/stable/modules/clustering.html) meetodeid klasterdamiseks. Millist tüüpi valida, sõltub sinu kasutusjuhtumist. Dokumentatsiooni järgi on igal meetodil erinevad eelised. Siin on lihtsustatud tabel Scikit-learn poolt toetatud meetoditest ja nende sobivatest kasutusjuhtudest:

| Meetodi nimi                | Kasutusjuhtum                                                        |
| :--------------------------| :------------------------------------------------------------------- |
| K-Means                    | üldine otstarve, induktiivne                                        |
| Affinity propagation       | palju, ebaühtlased klastrid, induktiivne                            |
| Mean-shift                 | palju, ebaühtlased klastrid, induktiivne                            |
| Spectral clustering        | vähe, ühtlased klastrid, transduktsiooniline                      |
| Ward hierarhiline klasterdamine | palju, piiratud klastrid, transduktsiooniline                  |
| Agglomeratiivne klasterdamine | palju, piiratud, mitte-Eukleidilise kaugusega, transduktsiooniline |
| DBSCAN                     | mitte-lame geomeetria, ebaühtlased klastrid, transduktsiooniline   |
| OPTICS                     | mitte-lame geomeetria, ebaühtlased muutuvama tihedusega klastrid, transduktsiooniline |
| Gaussiliste segu meetodid  | lame geomeetria, induktiivne                                       |
| BIRCH                      | suur andmestik kõrvalekalletega, induktiivne                      |

> 🎓 See, kuidas me klastreid loome, sõltub palju sellest, kuidas me andmepunkte rühmadesse koondame. Vaatame mõningaid termineid:
>
> 🎓 ['Transduktsiooniline' vs 'induktiivne'](https://wikipedia.org/wiki/Transduction_(machine_learning))
>
> Transduktsiooniline järeldus tuleneb vaatlustest treeningjuhtumite kohta, mis omavad kindlaid vasteid testjuhtumitega. Induktiivne järeldus tuletatakse treeningjuhtumitest, mis loovad üldisi reegleid, mida alles seejärel rakendatakse testjuhtumitele.
>
> Näide: Kujuta ette, et sul on andmestik, mis on ainult osaliselt märgistatud. Mõned andmed on 'plaadid', mõned 'CD-d' ja mõned lüngad on tühjad. Su ülesanne on anda nimetused nendele tühjadele. Kui valid induktiivse lähenemise, treenid mudelit, mis otsib 'plaate' ja 'CDsid' ning rakendad neid nimetusi märgistamata andmetele. See lähenemine teeb raskusi, kui tuleb klassifitseerida asju, mis on tegelikult 'kassetid'. Transduktsiooniline lähenemine käsitleb seda tundmatut andmestikku tõhusamalt, sest see proovib esmalt sarnased üksused rühmitada ja alles seejärel määrab rühmale nimetuse. Selles näites võivad klastrid kajastada 'ümmargusi muusikavahendeid' ja 'ruudukujulisi muusikavahendeid'.
>
> 🎓 ['Mitte-lame' vs 'lame' geomeetria](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
>
> Võetud matemaatikaterminoloogiast, mitte-lame vs lame geomeetria tähistab punktidevaheliste kauguste mõõtmist kas 'lame' ([Eukleidiline](https://wikipedia.org/wiki/Euclidean_geometry)) või 'mitte-lame' (mitte-Eukleidiline) geomeetriliste meetodite abil.
>
> 'Lame' selles kontekstis viitab Eukleidilisele geomeetriale (mida õpetatakse ka kui 'tasapinna' geomeetriat) ning mitte-lame viitab mitte-Eukleidilisele geomeetriale. Mis pistmist on geomeetrial masinõppega? Kuna mõlemad valdkonnad on juurdunud matemaatikas, peab olema ühine viis punktidevaheliste kauguste mõõtmiseks klastrites, ja see saab olla kas 'lame' või 'mitte-lame', sõltuvalt andmete olemusest. [Eukleidilised kaugused](https://wikipedia.org/wiki/Euclidean_distance) mõõdetakse kahe punkti vahele jääva joone pikkusena. [Mitte-eukleidilised kaugused](https://wikipedia.org/wiki/Non-Euclidean_geometry) mõõdetakse kõvera pikkusena. Kui sinu andmed, kui neid visualiseerida, ei paikne tasapinnal, võib sul olla vaja kasutada spetsiaalset algoritmi selle käsitlemiseks.
>
![Lame vs mitte-lame geomeetria infograafik](../../../../translated_images/et/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infograafika autor [Dasani Madipalli](https://twitter.com/dasani_decoded)
>
> 🎓 ['Kaugused'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
>
> Klastrid määratakse nende kaugusmaatriksi põhjal, st kauguste järgi punktide vahel. Seda kaugust võib mõõta mitmel moel. Eukleidilised klastrid määratakse punktide väärtuste keskmise järgi ja sisaldavad 'tükipunkti' ehk keskpunkti. Kaugused mõõdetakse just selle keskpunkti kauguste järgi. Mitte-eukleidilised kaugused viitavad 'klastroididele', mis on punktid, mis asuvad teiste punktide suhtes kõige lähemal. Klastroidid ise võivad olla määratletud erinevalt.
>
> 🎓 ['Piiratud'](https://wikipedia.org/wiki/Constrained_clustering)
>
> [Piiratud klasterdamine](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) viib seejuures 'pooljuhendatud' õppimise sellesse juhendamata meetodisse. Punktide vahelised seosed märgistatakse kui 'ei saa ühendada' või 'peab ühendama', nii et andmetele kehtestatakse mõned reeglid.
>
> Näide: Kui algoritm lastakse 'vabalt' töötada märgistamata või poolmärgistatud andmete hulgal, võivad klastrid olla kehva kvaliteediga. Ülaltoodud näites võib klastrid moodustada 'ümmargustest muusikariistadest' ja 'ruudukujulistest muusikariistadest' ning 'kolmnurkadest' ja 'küpsistest'. Kui anda mõningaid piiranguid või reegleid ("ese peab olema plastikust", "ese peab suutma muusikat toota"), aitab see algoritmi paremaid valikuid teha.
>
> 🎓 'Tihedus'
>
> 'Mürastatud' andmeid peetakse 'tihedaks'. Punktidevahelised kaugused selles klastrites võivad olla kontrollimisel tihedamad või hõredamad ehk 'rahvarohkemad' või vähem, ja see tähendab, et andmeid tuleb analüüsida sobiva klasterdamismeetodiga. [See artikkel](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) näitab erinevust K-Meansi ja HDBSCAN algoritmide vahel mürarikka andmestiku uurimisel, millel on ebaühtlane klasterite tihedus.

## Klasterdamise algoritmid

Klasterdamise algoritme on üle 100 ja nende kasutamine sõltub olemasolevate andmete iseloomust. Arutleme mõningaid põhilisi:

- **Hierarhiline klasterdamine**. Kui objekt klassifitseeritakse selle järgi, kui lähedal see asub mõnele lähedalasuvatele objektile, mitte kaugel olevale, moodustuvad klastrid nende liikmete vahelist kaugust arvestades. Scikit-learn'i agglomeratiivne klasterdamine on hierarhiline.

   ![Hierarhilise klasterdamise infograafik](../../../../translated_images/et/hierarchical.bf59403aa43c8c47.webp)
   > Infograafika autor [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Tükipunkti klasterdamine**. See populaarne algoritm nõuab 'k' ehk klastrite arvu valikut, millele järgneb algoritmi määrata klastrite keskpunkt ning koondada andmed selle punkti ümber. [K-means klasterdamine](https://wikipedia.org/wiki/K-means_clustering) on tuntud tükipunkti klasterdamise näide. Keskpunkt määratakse lähima keskmise järgi, seega nimetus. Klastrist kauguse ruutsumma minimeeritakse.

   ![Tükipunkti klasterdamise infograafik](../../../../translated_images/et/centroid.097fde836cf6c918.webp)
   > Infograafika autor [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Jaotusepõhine klasterdamine**. Statistilisel modelleerimisel põhinev jaotusepõhine klasterdamine määrab tõenäosuse, mil määral andmepunkt kuulub mingisse klastrisse ning määrab selle vastavalt. Selle tüübi hulka kuuluvad gaussliku segu meetodid.

- **Tihedus-põhine klasterdamine**. Andmepunktid määratakse klastritesse nende tiheduse alusel ehk grupi ümber koondumise põhjal. Punktid, mis asuvad grupist kaugel, loetakse kõrvalekalleteks või müraks. DBSCAN, Mean-shift ja OPTICS kuuluvad sellesse klasterdamise tüüpi.

- **Võrgustiku-põhine klasterdamine**. Mitmemõõtmeliste andmestike jaoks luuakse võrgustik ja andmed jagatakse võrgustiku rakkudeks, moodustades nii klastrid.

## Harjutus – klasterda oma andmed

Klasterdamise tehnikat toetab suurelt hea visualiseerimine, nii et alustame muusikandmete visualiseerimisest. See harjutus aitab meil otsustada, millist klasterdusmeetodit selle andmestiku puhul kõige tõhusamalt kasutada.

1. Ava selles kaustas olev [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) fail.

1. Impordi `Seaborn` pakett hea andmete visualiseerimise jaoks.

    ```python
    !pip install seaborn
    ```

1. Lisa laulude andmed failist [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Laadi andmestikku laulude kohta. Valmista end andmete uurimiseks, impordides vajalikud teegid ja kuvades andmed:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Kontrolli andmete esimesi ridu:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Saa mõned andmed andmeraamist, kutsudes esile `info()`:

    ```python
    df.info()
    ```

   Väljund näeb välja järgmiselt:

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

1. Kinnitage nullväärtuste puudumine, kutsudes esile `isnull()` ja kontrollides, et summa on 0:

    ```python
    df.isnull().sum()
    ```

    Tundub hea:

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

1. Kirjeldage andmeid:

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

> 🤔 Kui me töötame klasterdamisega, juhendamata meetodiga, mis ei vaja märgistatud andmeid, miks me siis näitame andmeid koos siltidega? Andmete uurimise faasis on need abiks, kuid klasterdamise algoritmide jaoks pole need vajalikud. Võite ka lihtsalt veergude päised eemaldada ja viidata andmetele veeru numbri järgi.

Vaadake andmete üldisi väärtusi. Märkige, et populaarsus võib olla '0', mis näitab lugusid, millel puudub edetabelikoht. Eemaldame need peagi.

1. Kasutage tulbadiagrammi, et leida kõige populaarsemad žanrid:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![kõige populaarsemad](../../../../translated_images/et/popular.9c48d84b3386705f.webp)

✅ Kui soovite näha rohkem tipptulemusi, muutke top `[:5]` suuremaks või eemaldage see, et näha kõiki.

Pange tähele, kui tippžanr on kirjeldatud kui 'Missing', tähendab see, et Spotify ei klassifitseerinud seda, seega vabaneme sellest.

1. Eemaldage puuduvad andmed, filtreerides need välja

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Kontrollige nüüd uuesti žanre:

    ![kõige populaarsemad](../../../../translated_images/et/all-genres.1d56ef06cefbfcd6.webp)

1. Kauaoodatud kolm populaarseimat žanrit domineerivad seda andmestikku. Keskendume `afro dancehallile`, `afropopile` ja `nigerian popile`, lisaks filtreerime andmestiku, et eemaldada kõik, mille populaarsus on 0 (mis tähendab, et see polnud andmestikus populaarsusega klassifitseeritud ja võib meie jaoks olla müra):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Tehke kiire test, et näha, kas andmetel on mõni eriti tugev korrelatsioon:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![korrelatsioonid](../../../../translated_images/et/correlation.a9356bb798f5eea5.webp)

    Ainus tugev seos on `energiat` ja `valjuse` vahel, mis ei ole liiga üllatav, arvestades, et vali muusika on tavaliselt üsna energiline. Muul juhul on korrelatsioonid suhteliselt nõrgad. Huvitav on näha, mida klasterdamise algoritm selle andmestikuga teeb.

    > 🎓 Pange tähele, et korrelatsioon ei tähenda põhjuslikkust! Meil on tõestus korrelatsiooni kohta, kuid puudub põhjuslikkuse tõestus. Üks [naljakas veebisait](https://tylervigen.com/spurious-correlations) sisaldab selle punkti rõhutamiseks visuaale.

Kas selles andmestikus on mingit kokkulangevust laulu tajutud populaarsuse ja tantsitavuse vahel? FacetGrid näitab, et on kontsentrilised ringid, mis joonduvad, sõltumata žanrist. Kas on võimalik, et Nigeeria maitsed koonduvad mingil tantsitavuse tasemel selle žanri puhul?

✅ Proovige erinevaid andmepunkte (energia, valjusus, kõnetähed) ja rohkem või erinevaid muusikalisi žanre. Mida võite avastada? Vaadake üle `df.describe()` tabel, et näha andmepunktide üldist levikut.

### Harjutus - andmete jaotus

Kas need kolm žanrit erinevad tantsitavuse tajus oluliselt, lähtudes nende populaarsusest?

1. Uurige kolme tipptasemel žanri andmete jaotust populaarsuse ja tantsitavuse osas, kasutades x- ja y-telge.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Võite avastada kontsentrilisi ringe üldise kokkulangevuse punkti ümber, mis näitab punktide jaotust.

    > 🎓 Pange tähele, et see näide kasutab KDE-d (tuumatihenduse hinnangut), mis kujutab andmeid pideva tõenäosustiheduse kõverana. See võimaldab meil tõlgendada andmeid, töötades mitme jaotusega.

    Üldiselt on kolm žanrit populaarsuse ja tantsitavuse osas laialdaselt joondatud. Klasterite kindlakstegemine selles laialt joondatud andmetes on väljakutse:

    ![jaotus](../../../../translated_images/et/distribution.9be11df42356ca95.webp)

1. Looge hajuvusdiagramm:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Sama telgede hajuvusdiagramm näitab sarnast kokkulangevuse mustrit

    ![Facetgrid](../../../../translated_images/et/facetgrid.9b2e65ce707eba1f.webp)

Üldiselt võite klasterdamiseks kasutada hajuvusdiagramme, et näidata andmete klastreid, seega on selle visualiseerimise tüübi meisterdamine väga kasulik. Järgmises peatükis kasutame filtreeritud andmeid ja k-means klasterdamist, et avastada rühmi selles andmestikus, mis näivad huvitavalt kattuvat.

---

## 🚀Väljakutse

Järgmise peatüki ettevalmistamiseks koostage graafik erinevate klasterdamise algoritmide kohta, mida võite avastada ja kasutada tootmiskeskkonnas. Milliseid probleeme püüab klasterdamine lahendada?

## [Järelvaatamise viktoriin](https://ff-quizzes.netlify.app/en/ml/)

## Kordamine ja iseseisev õpe

Enne klasterdamise algoritmide rakendamist, nagu oleme õppinud, on hea mõista oma andmestiku olemust. Loe teema kohta rohkem [siit](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[See kasulik artikkel](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) juhatab teid erinevate klasterdamise algoritmide käitumise juurde, sõltuvalt andmete kujundusest.

## Kodutöö

[Uurige muid klasterdamise visualiseeringuid](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Lahtiütlus**:
See dokument on tõlgitud kasutades AI tõlketeenust [Co-op Translator](https://github.com/Azure/co-op-translator). Kuigi me püüdleme täpsuse poole, palun pange tähele, et automatiseeritud tõlgetes võib esineda vigu või ebatäpsusi. Originaaldokument selle emakeeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitatakse kasutada professionaalset inimtõlget. Me ei vastuta selle tõlkega seotud eksimustest või valesti mõistmistest.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->