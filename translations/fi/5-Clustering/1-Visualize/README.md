# Johdanto klusterointiin

Klusterointi on eräänlainen [ohjaamaton oppiminen](https://wikipedia.org/wiki/Unsupervised_learning), joka olettaa, että aineisto on merkitsemätöntä tai että sen syötteitä ei ole yhdistetty ennalta määriteltyihin tuloksiin. Se käyttää erilaisia algoritmeja järjestääkseen merkitsemätöntä dataa ja tarjotakseen ryhmittelyjä havaittujen mallien perusteella.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Klikkaa yllä olevaa kuvaa videota varten. Opiskellessasi koneoppimista klusteroinnin avulla, nauti muutamista Nigerian Dance Hall -kappaleista – tämä on erittäin suosittu kappale vuodelta 2014 PSquarelta.

## [Esiluentoharjoitus](https://ff-quizzes.netlify.app/en/ml/)

### Johdanto

[Klusterointi](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) on erittäin hyödyllinen tiedon tutkimisessa. Katsotaan, voiko se auttaa löytämään trendejä ja malleja siinä, miten Nigerian yleisöt kuluttavat musiikkia.

✅ Käytä minuutti miettiäksesi klusteroinnin käyttötarkoituksia. Todellisessa elämässä klusterointi tapahtuu aina, kun sinulla on pino pyykkiä ja sinun täytyy lajitella perheen jäsenten vaatteet 🧦👕👖🩲. Data-analytiikassa klusterointia käytetään, kun pyritään analysoimaan käyttäjän mieltymyksiä tai määrittelemään minkä tahansa merkitsemättömän aineiston ominaisuuksia. Klusterointi auttaa ikään kuin järjestämään kaaosta, kuten sukan laatikkoa.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Klikkaa yllä olevaa kuvaa videota varten: MIT:n John Guttag esittelee klusterointia

Ammatillisessa käytössä klusterointia voidaan käyttää muun muassa markkinasegmentoinnissa, esimerkiksi määrittelemään, mitkä ikäryhmät ostavat mitä tuotteita. Toinen käyttötarkoitus voi olla poikkeavuuksien havaitseminen, esimerkiksi petosten tunnistaminen luottokorttitapahtumien tiedoista. Tai klusterointia voidaan käyttää kasvaimien määrittämiseen kokoelmasta lääkinnällisiä skannauksia.

✅ Mieti hetki, kuinka olet saattanut kohdata klusterointia "luonnossa", pankki-, verkkokauppa- tai liiketoimintaympäristössä.

> 🎓 Mielenkiintoista: klusterianalyysin juuret ovat antropologiassa ja psykologiassa 1930-luvulla. Voitko kuvitella, miten sitä saatettiin käyttää?

Vaihtoehtoisesti sitä voisi käyttää hakutulosten ryhmittelyssä – esimerkiksi ostoslinkkien, kuvien tai arvostelujen mukaan. Klusterointi on hyödyllinen, kun sinulla on suuri aineisto, jota haluat vähentää ja jolle haluat tehdä tarkempaa analyysiä, joten menetelmää voidaan käyttää oppimaan aineistosta ennen muiden mallien rakentamista.

✅ Kun aineistosi on järjestetty klustereihin, sille annetaan klusteritunnus, ja tätä tekniikkaa voidaan käyttää aineiston yksityisyyden säilyttämiseen; dataan voidaan viitata sen sijaan klusteritunnuksen avulla, ennemmin kuin paljastavien tunnistettavien tietojen perusteella. Voitko keksiä muita syitä, miksi viittaisit klusteritunnukseen ennemmin kuin muihin klusterin osiin?

Syvennä klusterointitekniikoiden ymmärrystäsi tässä [Learn-moduulissa](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)
## Klusteroinnin aloittaminen

[Scikit-learn tarjoaa laajan valikoiman](https://scikit-learn.org/stable/modules/clustering.html) menetelmiä klusterointiin. Valintasi riippuu käyttötarkoituksesta. Dokumentaation mukaan jokaisella menetelmällä on erilaisia etuja. Tässä on yksinkertaistettu taulukko Scikit-learnin tukemista menetelmistä ja niiden sopivista käyttötapauksista:

| Menetelmän nimi               | Käyttötapaus                                                          |
| :--------------------------- | :-------------------------------------------------------------------- |
| K-Means                      | yleiskäyttö, induktiivinen                                            |
| Affinity propagation         | monta, epätasaisia klustereita, induktiivinen                        |
| Mean-shift                   | monta, epätasaisia klustereita, induktiivinen                        |
| Spectral clustering          | vähän, tasaisia klustereita, transduktiivinen                       |
| Ward hierarchical clustering | monta, rajattuja klustereita, transduktiivinen                      |
| Agglomerative clustering     | monta, rajattuja, epä-Euklidisia etäisyyksiä, transduktiivinen      |
| DBSCAN                       | ei-tasainen geometria, epätasaisia klustereita, transduktiivinen   |
| OPTICS                       | ei-tasainen geometria, epätasaisia vaihtelevan tiheyden klustereita, transduktiivinen |
| Gaussian mixtures            | tasainen geometria, induktiivinen                                   |
| BIRCH                        | suuri aineisto poikkeavien arvojen kanssa, induktiivinen            |

> 🎓 Miten luomme klustereita liittyy vahvasti siihen, miten keräämme datapisteet ryhmiin. Puretaan hieman sanastoa:
>
> 🎓 ['Transduktiivinen' vs. 'induktiivinen'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktiivinen päättely perustuu havainnoituihin harjoitus tapauksiin, jotka vastaavat tiettyihin testitapauksiin. Induktiivinen päättely perustuu harjoitustapauksiin, jotka yhdistetään yleisiin sääntöihin, joita sovelletaan vasta sitten testitapauksiin.
> 
> Esimerkki: Kuvittele, että sinulla on aineisto, joka on osittain merkitty. Jotkin asiat ovat 'levyjä', jotkut 'cd-levyjä', ja osa on tyhjiä. Tehtäväsi on antaa merkinnät tyhjille. Jos valitset induktiivisen lähestymistavan, koulutat mallin etsimään 'levyjä' ja 'cd-levyjä', ja sovellat noita merkintöjä merkitsemättömälle aineistolle. Tämä lähestymistapa vaikeutuu luokittelemaan asioita, jotka ovat oikeasti 'kassetteja'. Transduktiivinen lähestymistapa puolestaan käsittelee tuntematonta dataa tehokkaammin, kun se ryhmittelee samanlaisia kohteita yhteen ja sitten antaa ryhmälle merkinnän. Tässä tapauksessa klusterit voisivat kuvastaa 'pyöreitä musiikkijuttuja' ja 'neliönmuotoisia musiikkijuttuja'.
> 
> 🎓 ['Ei-tasainen' vs. 'tasainen' geometria](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Matematiikasta johdettuna, ei-tasainen vs. tasainen geometria viittaa mittaustapaan pisteiden välisille etäisyyksille joko 'tasaisen' ([Euklidisen](https://wikipedia.org/wiki/Euclidean_geometry)) tai 'ei-tasaisen' (ei-euklidisen) geometrian menetelmillä.
>
> Tässä yhteydessä 'tasainen' tarkoittaa euklidista geometriaa (jonka osia opetetaan 'tasogeometriana'), ja ei-tasainen tarkoittaa ei-euklidista geometriaa. Mitä geometrialla on tekemistä koneoppimisen kanssa? Koska molemmat alat pohjautuvat matematiikkaan, on löydettävä yleinen tapa mitata etäisyyksiä klustereiden pisteiden välillä, ja se voidaan tehdä 'tasaisesti' tai 'ei-tasaisesti', datan luonteen mukaan. [Euklidiset etäisyydet](https://wikipedia.org/wiki/Euclidean_distance) mitataan kahden pisteen välisen janan pituutena. [Ei-euklidiset etäisyydet](https://wikipedia.org/wiki/Non-Euclidean_geometry) mitataan käyrää pitkin. Jos datasi, kun sen visualisoi, ei vaikuta olevan tasolla, saatat tarvita erityisen algoritmin sen käsittelyyn.
>
![Flat vs Nonflat Geometry Infographic](../../../../translated_images/fi/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infografiikka: [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Etäisyydet'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Klusterit määritellään niiden etäisyysmatriisin perusteella, eli etäisyyksien mittausten mukaan pisteiden välillä. Tätä etäisyyttä voidaan mitata useilla tavoilla. Euklidiset klusterit määritellään pisteiden arvojen keskiarvon perusteella, ja niillä on 'keskipiste' eli sentroidi. Etäisyydet mitataan siis sentroidiin. Ei-euklidiset etäisyydet viittaavat 'klustroideihin', läheisimpään pisteeseen muiden pisteiden joukossa. Klustroidit voidaan määritellä monin tavoin.
> 
> 🎓 ['Rajoitettu'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Rajoitettu klusterointi](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) yhdistelee 'puolivalvottua' oppimista tähän ohjaamattomaan menetelmään. Pisteiden väliset suhteet merkitään 'ei voi yhdistää' tai 'täytyy yhdistää', jolloin dataan kohdistuu tiettyjä sääntöjä.
>
> Esimerkki: Jos algoritmi päästetään vapaaksi aineistoon, jossa on merkitsemätöntä tai osittain merkittyä dataa, sen tuottamat klusterit voivat olla huonolaatuisia. Yllä olevassa esimerkissä klusterit saattaisivat ryhmitellä 'pyöreät musiikkijutut', 'neliönmuotoiset musiikkijutut', 'kolmion muotoiset jutut' ja 'keksit'. Jos annetaan joukko rajoituksia tai sääntöjä ("esineen täytyy olla muovia", "esineen pitää pystyä tuottamaan musiikkia"), tämä voi auttaa rajoittamaan algoritmia tekemään parempia valintoja.
> 
> 🎓 'Tiheys'
> 
> Data, joka on 'kohinaista', katsotaan olevan 'tiheää'. Etäisyydet pisteiden välillä kussakin klusterissa voivat tarkastelun mukaan olla tiheämpiä tai harvempia, eli kyseessä voi olla 'väkijoukko' tai harvasta jakautunut joukko. Tällöin data on analysoitava sopivalla klusterointimenetelmällä. [Tässä artikkelissa](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) havainnollistetaan eroa K-Means- ja HDBSCAN-algoritmien välillä kohinaisen aineiston epätasaisen klusteritiheyden tutkimisessa.

## Klusterointialgoritmit

Klusterointialgoritmeja on yli 100, ja niiden käyttö riippuu datan luonteesta. Käydään läpi joitakin merkittävimpiä:

- **Hierarkkinen klusterointi**. Jos kohde luokitellaan sen etäisyyden perusteella lähimpään objektiin, eikä kauempana olevaan, klusterit muodostuvat läheisten jäsenten etäisyyksien perusteella muihin kohteisiin. Scikit-learnin agglomeraatio on hierarkkinen klusterointimenetelmä.

   ![Hierarchical clustering Infographic](../../../../translated_images/fi/hierarchical.bf59403aa43c8c47.webp)
   > Infografiikka: [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Sentroidi-klusterointi**. Tämä suosittu algoritmi vaatii, että valitaan 'k', eli muodostettavien klusterien lukumäärä, jonka jälkeen algoritmi määrittää klusterin keskipisteen ja kerää dataa sen ympärille. [K-means-klusterointi](https://wikipedia.org/wiki/K-means_clustering) on suosittu sentroidi-klusteroinnin versio. Keskiarvo määräytyy lähimmän keskiarvon mukaan, mistä nimi. Klusterin neliöity etäisyys minimoidaan.

   ![Centroid clustering Infographic](../../../../translated_images/fi/centroid.097fde836cf6c918.webp)
   > Infografiikka: [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Jakautumispohjainen klusterointi**. Perustuu tilastolliseen mallintamiseen, jossa arvioidaan todennäköisyys, että datapiste kuuluu klusteriin, ja osoitetaan se vastaavasti. Gaussin seosmenetelmät kuuluvat tähän tyyppiin.

- **Tiheysperusteinen klusterointi**. Datapisteet jaetaan klustereihin niiden tiheyden mukaan, eli niiden ryhmittymisen perusteella toistensa ympärille. Etäällä ryhmästä olevat pisteet katsotaan poikkeamiksi tai kohinaksi. DBSCAN, Mean-shift ja OPTICS ovat tämän tyyppisiä klusterointimenetelmiä.

- **Verkkopohjainen klusterointi**. Moniulotteisille aineistoille luodaan ruudukko, ja data jaetaan ruudukon soluihin, jolloin muodostuu klustereita.

## Harjoitus – klusteroi dataasi

Klusterointia haetaan vahvistetuksi hyvästä visualisoinnista, joten aloitetaan musiikkidatan visualisoinnilla. Tämä harjoitus auttaa meitä päättämään, mitä klusterointimenetelmiä meidän kannattaa tehokkaimmin käyttää tämän datan luonteeseen.

1. Avaa tämän kansion [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) tiedosto.

1. Tuo `Seaborn`-paketti hyvää datan visualisointia varten.

    ```python
    !pip install seaborn
    ```

1. Lisää kappaletiedot tiedostosta [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Lataa dataframe, jossa on tietoa kappaleista. Valmistaudu tutkimaan tätä dataa tuomalla kirjastot ja tulostamalla aineisto:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Tarkista muutamat ensimmäiset rivit datasta:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Hanki tietoja dataframesta kutsumalla `info()`:

    ```python
    df.info()
    ```

   Tuloste näyttää tältä:

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

1. Tarkista vielä kerran puuttuvat arvot kutsumalla `isnull()` ja varmistamalla, että summa on 0:

    ```python
    df.isnull().sum()
    ```

    Näyttää hyvälle:

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

1. Kuvaile dataa:

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

> 🤔 Jos työskentelemme klusteroinnin parissa, valvomatonta menetelmää, joka ei vaadi merkittyjä tietoja, miksi näytämme näitä tietoja tunnisteilla? Datatutkimuksen vaiheessa ne ovat hyödyllisiä, mutta klusterointialgoritmien toimintaan ne eivät ole välttämättömiä. Voisit yhtä hyvin poistaa sarakeotsikot ja viitata tietoihin sarakenumeron avulla.

Katso datan yleisiä arvoja. Huomaa, että suosiota voi olla '0', mikä tarkoittaa kappaleita, joilla ei ole sijoitusta. Poistetaan nämä pian.

1. Käytä pylväsdiagrammia selvittääksesi suosituimmat genret:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/fi/popular.9c48d84b3386705f.webp)

✅ Jos haluat nähdä enemmän ylin arvoista, vaihda ylin `[:5]` suuremmaksi arvoksi tai poista se nähdäksesi kaikki.

Huomaa, että kun ylin genre on kuvattu nimellä 'Missing', se tarkoittaa, että Spotify ei luokitellut sitä, joten poistetaan se.

1. Poista puuttuvat tiedot suodattamalla ne pois

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Tarkista nyt genret uudelleen:

    ![most popular](../../../../translated_images/fi/all-genres.1d56ef06cefbfcd6.webp)

1. Selvästi kolme parasta genreä hallitsevat tätä datasettiä. Keskitytään `afro dancehalliin`, `afropoppiin` ja `nigerian poppiin`, lisäksi suodatetaan datasetti poistamaan kaikki, joiden popularity-arvo on 0 (mikä tarkoittaa, että niitä ei ole luokiteltu suosion mukaan datasetissä ja niitä voidaan pitää kohinana tarkoituksiamme varten):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Tee nopea testi nähdäksesi, korreloiko data erityisen vahvasti:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/fi/correlation.a9356bb798f5eea5.webp)

    Ainoa vahva korrelaatio on `energian` ja `loudnessin` välillä, mikä ei ole kovin yllättävää, koska kova ääni on yleensä melko energistä. Muuten korrelaatiot ovat melko heikkoja. On mielenkiintoista nähdä, mitä klusterointialgoritmi saa tästä datasta aikaan.

    > 🎓 Huomaa, että korrelaatio ei tarkoita kausaatiota! Meillä on todiste korrelaatiosta, mutta ei kausaliteetista. [Hauska nettisivu](https://tylervigen.com/spurious-correlations) tarjoaa kuvia jotka korostavat tätä asiaa.

Onko tässä datasetissä jonkinlaista yhtenevyyttä kappaleen koetun suosion ja tanssittavuuden välillä? FacetGrid näyttää, että siellä on konsentrisia renkaita, jotka asettuvat rinnakkain genrestä riippumatta. Voisiko olla, että Nigerialaiset maut yhtenevät tietyllä tanssittavuuden tasolla tällä genrellä?

✅ Kokeile eri datapisteitä (energia, loudness, speechiness) ja useampia tai eri musiikillisia genrejä. Mitä voit löytää? Katso `df.describe()` -taulukkoa nähdäksesi datan yleistä hajontaa.

### Harjoitus - datan jakauma

Ovatko nämä kolme genreä merkittävästi erilaisia tanssittavuuden kokemuksen suhteen suosion perusteella?

1. Tarkastele kolmen parhaan genren datan jakaumaa suosion ja tanssittavuuden suhteen annetulla x- ja y-akselilla.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Voit löytää konsentrisia renkaita yleisen yhtymäkohdan ympärillä, jotka näyttävät pisteiden jakauman.

    > 🎓 Huomaa, että tämä esimerkki käyttää KDE (Kernel Density Estimate) -graafia, joka esittää dataa jatkuvana todennäköisyystiheyskäyränä. Tämä mahdollistaa datan tulkinnan työskenneltäessä useiden jakaumien kanssa.

    Yleisesti ottaen kolme genreä asettuvat löyhästi suosionsa ja tanssittavuuden suhteen. Klustereiden määrittäminen tässä löyhästi linjatussa datassa tulee olemaan haaste:

    ![distribution](../../../../translated_images/fi/distribution.9be11df42356ca95.webp)

1. Luo hajontakaavio:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Sama hajontakaavio näyttää samanlaisen yhtymän kuvion

    ![Facetgrid](../../../../translated_images/fi/facetgrid.9b2e65ce707eba1f.webp)

Yleisesti klusterointia varten hajontakaavioita voidaan käyttää näyttämään datan klustereita, joten tämän visualisoinnin hallitseminen on erittäin hyödyllistä. Seuraavassa oppitunnissa käytämme tätä suodatettua dataa ja k-means-klusterointia löytääksemme ryhmiä, jotka näyttävät limittäytyvän mielenkiintoisella tavalla.

---

## 🚀Haaste

Valmistautuessasi seuraavaan oppituntiin, tee kaavio erilaisista klusterointialgoritmeista, joita voit löytää ja käyttää tuotantoympäristössä. Minkälaisia ongelmia klusterointi yrittää ratkaista?

## [Luennon jälkeinen tietovisa](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus & Itsenäinen opiskelu

Ennen kuin sovellat klusterointialgoritmeja, kuten olemme oppineet, on hyvä ymmärtää datasetin luonne. Lue tästä aiheesta lisää [täältä](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Tämä hyödyllinen artikkeli](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) esittelee erilaisia tapoja, joilla klusterointialgoritmit toimivat erilaisissa datamuodoissa.

## Tehtävä

[Tutki muita klusteroinnin visualisointimenetelmiä](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vastuuvapauslauseke**:
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, otathan huomioon, että automaattiset käännökset saattavat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäiskielellä on virallinen lähde. Tärkeissä asioissa suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa tämän käännöksen käytöstä aiheutuvista väärinymmärryksistä tai tulkinnoista.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->