# Wprowadzenie do klasteryzacji

Klasteryzacja to rodzaj [Uczenia bez nadzoru](https://wikipedia.org/wiki/Unsupervised_learning), który zakłada, że zbiór danych jest nieoznaczony lub że jego dane wejściowe nie są dopasowane do zdefiniowanych wcześniej wyników. Wykorzystuje różne algorytmy do sortowania danych nieoznaczonych i tworzenia grup zgodnie z wzorcami, które dostrzega w danych.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Kliknij powyższy obraz, aby obejrzeć film. Podczas nauki uczenia maszynowego z klasteryzacją, ciesz się kilkoma utworami Nigerian Dance Hall - to wysoko oceniana piosenka z 2014 roku autorstwa PSquare.

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

### Wprowadzenie

[Klasteryzacja](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) jest bardzo przydatna do eksploracji danych. Zobaczmy, czy może pomóc odkryć trendy i wzorce w sposobie, w jaki nigeryjska publiczność konsumuje muzykę.

✅ Poświęć chwilę, aby przemyśleć zastosowania klasteryzacji. W rzeczywistym życiu klasteryzacja zachodzi za każdym razem, gdy masz pranie i musisz posegregować ubrania członków rodziny 🧦👕👖🩲. W nauce o danych klasteryzacja zachodzi, gdy próbuje się analizować preferencje użytkownika lub określić cechy dowolnego nieoznaczonego zbioru danych. Klasteryzacja, w pewnym sensie, pomaga zrozumieć chaos, podobnie jak szuflada na skarpetki.

[![Wprowadzenie do ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Kliknij powyższy obraz, aby obejrzeć film: John Guttag z MIT wprowadza w temat klasteryzacji

W środowisku profesjonalnym klasteryzacja może być używana do określania rzeczy takich jak segmentacja rynku, określanie, które grupy wiekowe kupują jakie przedmioty, na przykład. Innym zastosowaniem może być wykrywanie anomalii, np. w celu wykrycia oszustw na podstawie zestawu danych transakcji kartą kredytową. Możesz też użyć klasteryzacji do wykrywania guzów w partii skanów medycznych.

✅ Przemyśl przez chwilę, jak mogłeś spotkać się z klasteryzacją „w praktyce”, w bankowości, handlu elektronicznym lub biznesie.

> 🎓 Co ciekawe, analiza skupień wywodzi się z dziedzin antropologii i psychologii z lat 30. XX wieku. Czy potrafisz sobie wyobrazić, jak mogła być używana?

Alternatywnie, można ją wykorzystać do grupowania wyników wyszukiwania – na przykład według linków do sklepów, obrazów lub recenzji. Klasteryzacja jest przydatna, gdy masz duży zbiór danych, który chcesz zredukować i na którym chcesz przeprowadzić bardziej szczegółową analizę, więc technika ta może być używana do poznania danych przed budową innych modeli.

✅ Gdy twoje dane są zorganizowane w klastry, przypisujesz im identyfikator klastra, a ta technika może być użyteczna przy zachowaniu prywatności zbioru danych; możesz zamiast tego odwoływać się do punktu danych za pomocą identyfikatora klastra, a nie bardziej ujawniających danych identyfikujących. Czy potrafisz wymyślić inne powody, dla których odwoływałbyś się do identyfikatora klastra zamiast innych elementów klastra, aby go zidentyfikować?

Pogłębiaj swoją wiedzę o technikach klasteryzacji w tym [module Learn](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

## Rozpoczęcie pracy z klasteryzacją

[Scikit-learn oferuje szeroki wachlarz](https://scikit-learn.org/stable/modules/clustering.html) metod do przeprowadzania klasteryzacji. Typ, który wybierzesz, zależy od przypadku użycia. Zgodnie z dokumentacją, każda metoda ma różne zalety. Oto uproszczona tabela metod obsługiwanych przez Scikit-learn i ich odpowiednie przypadki użycia:

| Nazwa metody                | Przypadek użycia                                                      |
| :--------------------------- | :--------------------------------------------------------------------- |
| K-Means                      | ogólnego przeznaczenia, indukcyjna                                   |
| Affinity propagation         | wiele, nierównych klastrów, indukcyjna                               |
| Mean-shift                   | wiele, nierównych klastrów, indukcyjna                               |
| Spektralna klasteryzacja    | mało, równych klastrów, transdukcyjna                               |
| Ward hierarchiczna          | wiele, ograniczonych klastrów, transdukcyjna                         |
| Agglomeracyjna              | wiele, ograniczonych, odległości nieeuklidesowych, transdukcyjna     |
| DBSCAN                      | geometria niepłaska, nierówne klastry, transdukcyjna                 |
| OPTICS                      | geometria niepłaska, nierówne klastry o zmiennej gęstości, transdukcyjna |
| Mieszanki Gaussowskie       | geometria płaska, indukcyjna                                         |
| BIRCH                       | duży zbiór danych z wartościami odstającymi, indukcyjna             |

> 🎓 To, jak tworzymy klastry, w dużej mierze zależy od tego, jak grupujemy punkty danych w zbiory. Rozwińmy trochę słownictwo:
>
> 🎓 ['Transdukcyjne' a 'indukcyjne'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Wnioskowanie transdukcyjne pochodzi z obserwowanych przypadków treningowych, które są mapowane na konkretne przypadki testowe. Wnioskowanie indukcyjne pochodzi z przypadków treningowych, które odzwierciedlają ogólne reguły, które następnie stosuje się do przypadków testowych.
> 
> Przykład: Wyobraź sobie, że masz zbiór danych częściowo opisany etykietami. Niektóre rzeczy to "winyle" (records), inne "płyty CD" (cds), a niektóre są puste. Twoim zadaniem jest przypisać etykiety do pustych miejsc. Jeśli wybierzesz podejście indukcyjne, wytrenujesz model do rozpoznawania "winyli" i "płyt CD" i zastosujesz te etykiety do danych nieoznaczonych. To podejście będzie miało trudności z klasyfikacją rzeczy, które faktycznie są "kasetami". Podejście transdukcyjne natomiast skuteczniej radzi sobie z tymi nieznanymi danymi, ponieważ grupuje podobne przedmioty razem, a następnie przypisuje etykietę grupie. W tym przypadku klastry mogą odzwierciedlać "okrągłe rzeczy muzyczne" i "kwadratowe rzeczy muzyczne".
> 
> 🎓 ['Geometria niepłaska' a 'płaska'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Pochodząc z terminologii matematycznej, geometria niepłaska i płaska odnosi się do mierzenia odległości między punktami za pomocą metod geometrycznych 'płaskich' ([Euklidesowych](https://wikipedia.org/wiki/Euclidean_geometry)) lub 'niepłaskich' (nieeuklidesowych).
>
> 'Płaska' w tym kontekście odnosi się do geometrii euklidesowej (jej części nauczane jako geometria płaszczyzny), a niepłaska do geometrii nieeuklidesowej. Co geometria ma wspólnego z uczeniem maszynowym? Jako dwie dziedziny mocno osadzone w matematyce, muszą mieć wspólny sposób mierzenia odległości między punktami w klastrach, co może być wykonane na sposób 'płaski' lub 'niepłaski', w zależności od charakteru danych. [Odległości euklidesowe](https://wikipedia.org/wiki/Euclidean_distance) mierzy się jako długość odcinka między dwoma punktami. [Odległości nieeuklidesowe](https://wikipedia.org/wiki/Non-Euclidean_geometry) mierzy się wzdłuż krzywej. Jeśli twoje dane, zwizualizowane, wydają się nie istnieć na płaszczyźnie, możesz potrzebować specjalistycznego algorytmu do ich obsługi.
>
![Flat vs Nonflat Geometry Infographic](../../../../translated_images/pl/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infografika autorstwa [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Odległości'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Klastry definiuje się za pomocą macierzy odległości, np. odległości między punktami. Odległość ta może być mierzona na różne sposoby. Klastry euklidesowe definiuje się na podstawie średniej wartości punktów i zawierają 'środek' lub centroid. Odległości mierzy się zatem odległością do tego centroidu. Odległości nieeuklidesowe odnoszą się do 'klustroidów', punktów najbliższych innym punktom. Klustroidy mogą być definiowane na różne sposoby.
> 
> 🎓 ['Ograniczone'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Ograniczona klasteryzacja](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) wprowadza 'uczenie półnadzorowane' do metody bez nadzoru. Relacje między punktami są oznaczane jako 'nie mogą być połączone' lub 'muszą być połączone', dzięki czemu zestawowi danych narzucane są pewne reguły.
>
> Przykład: Jeśli algorytm zostanie puszczony luzem na partii danych nieoznaczonych lub półoznaczonych, klastry, które stworzy, mogą być niskiej jakości. W powyższym przykładzie klastry mogłyby grupować "okrągłe rzeczy muzyczne", "kwadratowe rzeczy muzyczne", "trójkątne rzeczy" i "ciastka". Jeśli nadamy pewne ograniczenia lub reguły do przestrzegania ("przedmiot musi być wykonany z plastiku", "przedmiot musi być zdolny do produkcji muzyki"), może to pomóc "ograniczyć" algorytm do lepszych wyborów.
> 
> 🎓 'Gęstość'
> 
> Dane uważane za 'szum' są traktowane jako 'gęste'. Odległości między punktami w ich klastrach mogą okazać się, po analizie, bardziej lub mniej gęste, czyli 'zatłoczone', i dlatego takie dane muszą być analizowane odpowiednią metodą klasteryzacji. [Ten artykuł](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) pokazuje różnicę między stosowaniem klasteryzacji K-Means a algorytmów HDBSCAN do eksploracji zaszumionych danych o nierównej gęstości klastrów.

## Algorytmy klasteryzacji

Istnieje ponad 100 algorytmów klasteryzacji, a ich zastosowanie zależy od charakteru danych, które mamy do dyspozycji. Omówmy niektóre z najważniejszych:

- **Hierarchiczna klasteryzacja.** Jeśli obiekt jest klasyfikowany na podstawie bliskości do obiektu w pobliżu, a nie do dalszego, klastry tworzone są na podstawie odległości między ich członkami a innymi obiektami. Agglomeratywna klasteryzacja w Scikit-learn jest hierarchiczna.

   ![Hierarchical clustering Infographic](../../../../translated_images/pl/hierarchical.bf59403aa43c8c47.webp)
   > Infografika autorstwa [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Klasteryzacja centroidów.** Ten popularny algorytm wymaga wyboru 'k', czyli liczby klastrów do utworzenia, po czym algorytm ustala punkt centralny klastra i grupuje dane wokół tego punktu. [Klasteryzacja K-means](https://wikipedia.org/wiki/K-means_clustering) jest popularną wersją klasteryzacji centroidów. Centrum jest ustalane przez najbliższą średnią, stąd nazwa. Mierzy się minimalizację kwadratu odległości od klastra.

   ![Centroid clustering Infographic](../../../../translated_images/pl/centroid.097fde836cf6c918.webp)
   > Infografika autorstwa [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Klasteryzacja oparta na rozkładzie.** Bazująca na modelowaniu statystycznym, klasteryzacja oparta na rozkładzie polega na określeniu prawdopodobieństwa przynależności punktu danych do klastra i odpowiednim jego przypisaniu. Metody mieszanki Gaussowskiej należą do tego typu.

- **Klasteryzacja oparta na gęstości.** Punkty danych są przypisywane do klastrów na podstawie ich gęstości, czyli tego, jak gęsto są ze sobą skupione. Punkty danych znacznie oddalone od grupy uznawane są za wartości odstające lub szum. Do tego typu klasteryzacji należą DBSCAN, Mean-shift i OPTICS.

- **Klasteryzacja oparta na siatce.** W przypadku wielowymiarowych zbiorów danych tworzona jest siatka, a dane dzielone są między komórki tej siatki, tworząc klastry.

## Ćwiczenie - stwórz klastry swoich danych

Technika klasteryzacji jest bardzo wspierana przez odpowiednią wizualizację, więc zacznijmy od wizualizacji naszych danych o muzyce. To ćwiczenie pomoże nam zdecydować, której z metod klasteryzacji powinniśmy najefektywniej użyć do charakteru tych danych.

1. Otwórz plik [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) w tym folderze.

1. Zaimportuj pakiet `Seaborn` do dobrej wizualizacji danych.

    ```python
    !pip install seaborn
    ```

1. Dołącz dane piosenek z [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Załaduj ramkę danych z informacjami o piosenkach. Przygotuj się do eksploracji tych danych, importując biblioteki i wyświetl je:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Sprawdź pierwsze kilka linii danych:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Uzyskaj informacje o dataframe, wywołując `info()`:

    ```python
    df.info()
    ```

   Wyjście wygląda następująco:

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

1. Podwójnie sprawdź wartości null, wywołując `isnull()` i weryfikując sumę równą 0:

    ```python
    df.isnull().sum()
    ```

    Wygląda dobrze:

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

1. Opisz dane:

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

> 🤔 Jeśli pracujemy z klasteryzacją, metodą nie nadzorowaną, która nie wymaga danych oznaczonych, dlaczego pokazujemy te dane z etykietami? W fazie eksploracji danych są one przydatne, ale nie są konieczne do działania algorytmów klasteryzacji. Można równie dobrze usunąć nagłówki kolumn i odwoływać się do danych według numeru kolumny.

Spójrz na ogólne wartości danych. Zauważ, że popularność może być '0', co oznacza utwory nie posiadające rankingów. Zaraz je usuniemy.

1. Użyj wykresu słupkowego, aby dowiedzieć się, które gatunki są najbardziej popularne:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/pl/popular.9c48d84b3386705f.webp)

✅ Jeśli chcesz zobaczyć więcej wartości na górze, zmień `[:5]` na większą wartość lub usuń ją, aby zobaczyć wszystkie.

Zwróć uwagę, że gdy najwyższy gatunek opisany jest jako 'Missing', oznacza to, że Spotify go nie sklasyfikowało, więc się go pozbądźmy.

1. Pozbądź się brakujących danych, filtrując je

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Teraz ponownie sprawdź gatunki:

    ![most popular](../../../../translated_images/pl/all-genres.1d56ef06cefbfcd6.webp)

1. Zdecydowanie trzy główne gatunki dominują w tym zbiorze danych. Skoncentrujmy się na `afro dancehall`, `afropop` i `nigerian pop`, dodatkowo filtrując zbiór, aby usunąć wszystko z wartością popularności 0 (co oznacza, że ta pozycja nie była sklasyfikowana pod względem popularności i można ją uznać za szum dla naszych celów):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Szybki test, czy dane korelują ze sobą w szczególnie silny sposób:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/pl/correlation.a9356bb798f5eea5.webp)

    Jedyną silną korelacją jest ta między `energy` i `loudness`, co nie jest zaskakujące, bo głośna muzyka zwykle jest całkiem energetyczna. Poza tym korelacje są dość słabe. Ciekawe będzie zobaczyć, co algorytm klasteryzacji wyciągnie z tych danych.

    > 🎓 Pamiętaj, że korelacja nie oznacza przyczynowości! Mamy dowód korelacji, ale nie dowód na związek przyczynowy. [Zabawna strona](https://tylervigen.com/spurious-correlations) zawiera wizualizacje, które to podkreślają.

Czy istnieje jakieś zbieżność w tym zbiorze danych wokół postrzeganej popularności piosenki i jej „danceability”? FacetGrid pokazuje koncentryczne koła, które się układają niezależnie od gatunku. Czy możliwe, że gusta nigeryjskie zbiegają się na określonym poziomie „danceability” dla tego gatunku?

✅ Wypróbuj różne punkty danych (energy, loudness, speechiness) i więcej lub inne gatunki muzyczne. Co możesz odkryć? Spójrz na tabelę `df.describe()`, aby zobaczyć ogólny rozrzut punktów danych.

### Ćwiczenie - rozkład danych

Czy te trzy gatunki różnią się znacząco pod względem postrzeganej „danceability” na podstawie ich popularności?

1. Zbadaj rozkład danych naszych trzech najpopularniejszych gatunków pod względem popularności i „danceability” na zadanych osiach x i y.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Możesz odkryć koncentryczne koła wokół ogólnego punktu zbieżności, pokazujące rozkład punktów.

    > 🎓 Zwróć uwagę, że ten przykład używa wykresu KDE (Kernel Density Estimate), który przedstawia dane za pomocą ciągłej krzywej gęstości prawdopodobieństwa. Pozwala to interpretować dane przy pracy z wieloma rozkładami.

    Ogólnie trzy gatunki są luźno zgodne pod względem popularności i „danceability”. Wyznaczenie klastrów w tych luźno dopasowanych danych będzie wyzwaniem:

    ![distribution](../../../../translated_images/pl/distribution.9be11df42356ca95.webp)

1. Utwórz wykres rozrzutu:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Wykres rozrzutu tych samych osi pokazuje podobny wzór zbieżności

    ![Facetgrid](../../../../translated_images/pl/facetgrid.9b2e65ce707eba1f.webp)

Ogólnie, do klasteryzacji można użyć wykresów rozrzutu do pokazania skupisk danych, więc opanowanie tej wizualizacji jest bardzo przydatne. W następnej lekcji weźmiemy ten przefiltrowany zestaw danych i użyjemy grupowania k-średnich, aby odkryć grupy w tych danych, które wydają się nakładać w interesujący sposób.

---

## 🚀Wyzwanie

W przygotowaniu do następnej lekcji przygotuj wykres różnych algorytmów klasteryzacji, które możesz odkryć i użyć w środowisku produkcyjnym. Jakie rodzaje problemów stara się rozwiązać klasteryzacja?

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Przegląd i samodzielna nauka

Przed zastosowaniem algorytmów klasteryzacji, jak się nauczyliśmy, dobrze jest zrozumieć charakter swojego zestawu danych. Przeczytaj więcej na ten temat [tutaj](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Ten pomocny artykuł](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) przeprowadza cię przez różne zachowania algorytmów klasteryzacji w zależności od kształtów danych.

## Zadanie

[Zbadaj inne wizualizacje dla klasteryzacji](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Zastrzeżenie**:
Niniejszy dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Choć dążymy do dokładności, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub niedokładności. Oryginalny dokument w jego języku źródłowym należy uznawać za autorytatywne źródło. W przypadku informacji krytycznych zalecane jest skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->