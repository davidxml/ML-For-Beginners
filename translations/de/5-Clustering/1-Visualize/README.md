# Einführung in Clustering

Clustering ist eine Art des [Unüberwachten Lernens](https://wikipedia.org/wiki/Unsupervised_learning), die davon ausgeht, dass ein Datensatz unbeschriftet ist oder seine Eingaben nicht mit vordefinierten Ausgaben abgeglichen werden. Es verwendet verschiedene Algorithmen, um unbeschriftete Daten zu sortieren und Gruppen gemäß den Mustern bereitzustellen, die es in den Daten erkennt.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Klicke auf das obige Bild für ein Video. Während du dich mit maschinellem Lernen und Clustering beschäftigst, genieße einige Nigerian Dance Hall Tracks – dies ist ein hoch bewertetes Lied von 2014 von PSquare.

## [Vorlesungsquiz](https://ff-quizzes.netlify.app/en/ml/)

### Einführung

[Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) ist sehr nützlich zur Datenexploration. Schauen wir, ob es helfen kann, Trends und Muster in der Art und Weise zu entdecken, wie nigerianische Zuhörer Musik konsumieren.

✅ Nimm dir eine Minute Zeit, um über die Verwendungsmöglichkeiten von Clustering nachzudenken. Im wirklichen Leben findet Clustering statt, wenn du einen Wäscheberg hast und die Kleidung deiner Familienmitglieder sortieren musst 🧦👕👖🩲. In der Datenwissenschaft findet Clustering statt, wenn versucht wird, die Vorlieben eines Nutzers zu analysieren oder die Eigenschaften eines unbeschrifteten Datensatzes zu bestimmen. Clustering hilft gewissermaßen dabei, Chaos zu verstehen, wie eine Sockenschublade.

[![Einführung in ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Einführung ins Clustering")

> 🎥 Klicke auf das obige Bild für ein Video: John Guttag vom MIT stellt Clustering vor.

In einem professionellen Umfeld kann Clustering verwendet werden, um Dinge wie Marktsegmentierung zu bestimmen, zum Beispiel welche Altersgruppen welche Artikel kaufen. Eine weitere Anwendung wäre die Anomalieerkennung, z.B. um Betrug anhand eines Datensatzes von Kreditkartentransaktionen zu erkennen. Oder du könntest Clustering verwenden, um Tumore in einer Reihe medizinischer Scans zu bestimmen.

✅ Überlege eine Minute, wie du Clusterings „in freier Wildbahn“ in einem Bank-, E-Commerce- oder Geschäftsumfeld begegnet bist.

> 🎓 Interessanterweise stammt die Clusteranalyse aus den Bereichen Anthropologie und Psychologie in den 1930ern. Kannst du dir vorstellen, wie sie damals verwendet worden sein könnte?

Alternativ kannst du es zur Gruppierung von Suchergebnissen verwenden – zum Beispiel nach Einkaufslinks, Bildern oder Rezensionen. Clustering ist nützlich, wenn du einen großen Datensatz hast, den du reduzieren und auf dem du eine detailliertere Analyse durchführen möchtest. So kann die Technik verwendet werden, um Daten kennenzulernen, bevor andere Modelle erstellt werden.

✅ Sobald deine Daten in Clustern organisiert sind, weist du ihnen eine Cluster-ID zu, und diese Technik kann nützlich sein, um die Privatsphäre eines Datensatzes zu wahren; du kannst stattdessen auf einen Datenpunkt durch seine Cluster-ID verweisen, anstatt durch offenlegende identifizierende Daten. Fallen dir weitere Gründe ein, warum du zur Identifikation eines Clusters lieber die Cluster-ID statt anderer Cluster-Elemente verwenden würdest?

Vertiefe dein Verständnis von Clustering-Techniken in diesem [Learn-Modul](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)
## Einstieg in Clustering

[Scikit-learn bietet eine große Auswahl](https://scikit-learn.org/stable/modules/clustering.html) an Methoden zur Durchführung von Clustering. Die Wahl hängt von deinem Anwendungsfall ab. Laut Dokumentation hat jede Methode verschiedene Vorteile. Hier ist eine vereinfachte Tabelle der von Scikit-learn unterstützten Methoden und ihrer geeigneten Anwendungsfälle:

| Methodenname                 | Anwendungsfall                                                       |
| :--------------------------- | :------------------------------------------------------------------ |
| K-Means                      | Allgemeiner Zweck, induktiv                                          |
| Affinity propagation         | viele, ungleichmäßige Cluster, induktiv                              |
| Mean-shift                   | viele, ungleichmäßige Cluster, induktiv                              |
| Spectral clustering          | wenige, gleichmäßige Cluster, transduktiv                           |
| Ward hierarchisches Clustering | viele, eingeschränkte Cluster, transduktiv                        |
| Agglomeratives Clustering    | viele, eingeschränkt, nicht-euklidische Distanzen, transduktiv      |
| DBSCAN                       | nicht-flache Geometrie, ungleichmäßige Cluster, transduktiv         |
| OPTICS                       | nicht-flache Geometrie, ungleichmäßige Cluster mit variabler Dichte, transduktiv |
| Gaussian mixtures            | flache Geometrie, induktiv                                          |
| BIRCH                        | großer Datensatz mit Ausreißern, induktiv                          |

> 🎓 Wie wir Cluster erstellen, hat viel damit zu tun, wie wir die Datenpunkte in Gruppen zusammenfassen. Lass uns etwas Vokabular aufschlüsseln:
>
> 🎓 ['Transduktiv' vs. 'induktiv'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktive Inferenz wird von beobachteten Trainingsfällen abgeleitet, die spezifische Testfälle abbilden. Induktive Inferenz wird von Trainingsfällen abgeleitet, die allgemeine Regeln bilden, die dann erst auf Testfälle angewendet werden.
> 
> Ein Beispiel: Stell dir vor, du hast einen Datensatz, der nur teilweise beschriftet ist. Manche Dinge sind „Schallplatten“, manche „CDs“ und manche sind leer. Deine Aufgabe ist es, die leeren Elemente zu beschriften. Wenn du einen induktiven Ansatz wählst, trainierst du ein Modell, das nach „Schallplatten“ und „CDs“ sucht, und wendest diese Beschriftungen auf deine unbeschrifteten Daten an. Dieser Ansatz wird Schwierigkeiten haben, Dinge zu klassifizieren, die tatsächlich „Kassetten“ sind. Ein transduktiver Ansatz hingegen behandelt unbekannte Daten effektiver, weil er ähnliche Elemente zusammenführt und dann einer Gruppe eine Bezeichnung zuweist. In diesem Fall könnten Cluster „runde Musiksachen“ und „quadratische Musiksachen“ reflektieren.
> 
> 🎓 ['Nicht-flache' vs. 'flache' Geometrie](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Abgeleitet aus der mathematischen Terminologie bezieht sich „nicht-flache“ vs. „flache“ Geometrie auf die Maßnahme der Abstände zwischen Punkten durch entweder „flache“ ([euklidische](https://wikipedia.org/wiki/Euclidean_geometry)) oder „nicht-flache“ (nicht-euklidische) geometrische Methoden.
>
> „Flach“ bezieht sich in diesem Kontext auf die euklidische Geometrie (Teile davon werden als „ebene“ Geometrie gelehrt) und „nicht-flach“ auf nicht-euklidische Geometrie. Was hat Geometrie mit maschinellem Lernen zu tun? Nun, da beide Bereiche in der Mathematik verankert sind, muss es eine gemeinsame Möglichkeit geben, Abstände zwischen Punkten in Clustern zu messen, und dies kann „flach“ oder „nicht-flach“ erfolgen, abhängig von der Natur der Daten. [Euklidische Abstände](https://wikipedia.org/wiki/Euclidean_distance) werden als Länge eines Liniensegments zwischen zwei Punkten gemessen. [Nicht-euklidische Abstände](https://wikipedia.org/wiki/Non-Euclidean_geometry) werden entlang einer Kurve gemessen. Wenn deine Daten visualisiert zu sein scheinen, als existierten sie nicht auf einer Ebene, brauchst du möglicherweise einen spezialisierten Algorithmus, um sie zu handhaben.
>
![Flache vs. nicht-flache Geometrie Infografik](../../../../translated_images/de/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infografik von [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Abstände'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Cluster werden durch ihre Distanzmatrix definiert, z.B. die Abstände zwischen Punkten. Dieser Abstand kann auf verschiedene Weise gemessen werden. Euklidische Cluster werden durch den Durchschnitt der Punktwerte definiert und enthalten einen „Zentrumspunkt“ oder Zentroid. Entsprechend werden Abstände durch die Distanz zu diesem Zentroid gemessen. Nicht-euklidische Abstände beziehen sich auf „Clustroide“, den Punkt, der anderen Punkten am nächsten ist. Clustroide können wiederum auf verschiedene Weise definiert werden.
> 
> 🎓 ['Eingeschränkt'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Constrained Clustering](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) führt „semi-supervised“ Lernen in diese unüberwachte Methode ein. Die Beziehungen zwischen Punkten werden als „darf nicht verbunden“ oder „muss verbunden sein“ markiert, sodass einige Regeln für den Datensatz erzwungen werden.
>
>Ein Beispiel: Wenn ein Algorithmus auf eine Menge unbeschrifteter oder halb-beschrifteter Daten losgelassen wird, können die erzeugten Cluster von minderer Qualität sein. Im obigen Beispiel könnten die Cluster „runde Musiksachen“ und „quadratische Musiksachen“ und „dreieckige Dinge“ und „Kekse“ gruppieren. Werden einige Einschränkungen oder Regeln („der Artikel muss aus Plastik sein“, „der Artikel muss in der Lage sein, Musik zu produzieren“) vorgegeben, kann dies helfen, den Algorithmus zu zwingen, bessere Entscheidungen zu treffen.
> 
> 🎓 'Dichte'
> 
> Daten, die „rauschbehaftet“ sind, gelten als „dicht“. Die Abstände zwischen Punkten in jedem ihrer Cluster können bei genauer Betrachtung mehr oder weniger dicht oder „überfüllt“ sein, weshalb diese Daten mit der geeigneten Clustering-Methode analysiert werden müssen. [Dieser Artikel](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) zeigt den Unterschied zwischen K-Means Clustering und HDBSCAN Algorithmen bei der Untersuchung eines verrauschten Datensatzes mit ungleichmäßiger Clusterdichte.

## Clustering-Algorithmen

Es gibt über 100 Clustering-Algorithmen, und ihr Einsatz hängt von der Natur der vorliegenden Daten ab. Lass uns einige der wichtigsten besprechen:

- **Hierarchisches Clustering**. Wenn ein Objekt nach seiner Nähe zu einem benachbarten Objekt klassifiziert wird, statt zu einem weiter entfernten, werden Cluster auf Basis der Entfernung ihrer Mitglieder zu und von anderen Objekten gebildet. Das Agglomerative Clustering von Scikit-learn ist hierarchisch.

   ![Hierarchisches Clustering Infografik](../../../../translated_images/de/hierarchical.bf59403aa43c8c47.webp)
   > Infografik von [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Zentroid-basiertes Clustering**. Dieser populäre Algorithmus erfordert die Wahl von „k“, also der Anzahl der zu bildenden Cluster, wonach der Algorithmus den Mittelpunkt eines Clusters bestimmt und Daten um diesen Punkt gruppiert. [K-means Clustering](https://wikipedia.org/wiki/K-means_clustering) ist eine beliebte Version des zentroid-basierten Clustering. Der Mittelpunkt wird durch den nächstgelegenen Mittelwert bestimmt, daher der Name. Die quadratische Distanz vom Cluster wird minimiert.

   ![Zentroid-basiertes Clustering Infografik](../../../../translated_images/de/centroid.097fde836cf6c918.webp)
   > Infografik von [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Verteilungsbasiertes Clustering**. Basierend auf statistischen Modellen konzentriert sich das verteilungsbasierte Clustering darauf, die Wahrscheinlichkeit zu bestimmen, dass ein Datenpunkt zu einem Cluster gehört, und ordnet ihn entsprechend zu. Gaussian Mixture-Methoden gehören zu diesem Typ.

- **Dichte-basiertes Clustering**. Datenpunkte werden Clustern basierend auf ihrer Dichte oder ihrer Gruppierung um einander zugeordnet. Datenpunkte, die weit entfernt von der Gruppe liegen, gelten als Ausreißer oder Rauschen. DBSCAN, Mean-shift und OPTICS gehören zu diesem Clustering-Typ.

- **Raster-basiertes Clustering**. Für mehrdimensionale Datensätze wird ein Raster erstellt und die Daten auf die Zellen des Rasters verteilt, wodurch Cluster entstehen.

## Übung - Clustere deine Daten

Clustering als Technik wird durch eine gute Visualisierung stark unterstützt, also lasst uns mit der Visualisierung unserer Musikdaten beginnen. Diese Übung hilft uns dabei zu entscheiden, welche der Clustering-Methoden wir für die Art dieser Daten am effektivsten einsetzen sollten.

1. Öffne die Datei [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) in diesem Ordner.

1. Importiere das `Seaborn` Paket für gute Datenvisualisierung.

    ```python
    !pip install seaborn
    ```

1. Füge die Lieddaten aus [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv) hinzu. Lade einen Dataframe mit einigen Daten über die Songs. Mache dich bereit, diese Daten zu erkunden, indem du die Bibliotheken importierst und die Daten ausgibst:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Überprüfe die ersten Zeilen der Daten:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | Indie R&B       | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | Nigerian Pop    | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | Afropop         | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Holen Sie sich einige Informationen über das DataFrame, indem Sie `info()` aufrufen:

    ```python
    df.info()
    ```

   Die Ausgabe sieht so aus:

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

1. Überprüfen Sie doppelt auf Nullwerte, indem Sie `isnull()` aufrufen und die Summe 0 bestätigen:

    ```python
    df.isnull().sum()
    ```

    Sieht gut aus:

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

1. Beschreiben Sie die Daten:

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

> 🤔 Wenn wir mit Clustering arbeiten, einer unüberwachten Methode, die keine gelabelten Daten benötigt, warum zeigen wir diese Daten mit Labels? In der Datenexplorationsphase sind sie nützlich, aber sie sind für das Funktionieren der Clustering-Algorithmen nicht erforderlich. Sie könnten auch einfach die Spaltenüberschriften entfernen und sich auf die Daten mit Spaltennummern beziehen.

Sehen Sie sich die allgemeinen Werte der Daten an. Beachten Sie, dass Popularität '0' sein kann, was Songs zeigt, die keine Rangfolge haben. Lassen Sie uns diese bald entfernen.

1. Verwenden Sie ein Balkendiagramm, um die beliebtesten Genres herauszufinden:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/de/popular.9c48d84b3386705f.webp)

✅ Wenn Sie mehr Top-Werte sehen möchten, ändern Sie `[:5]` auf einen größeren Wert oder entfernen Sie es, um alle zu sehen.

Beachten Sie, wenn das Top-Genre als 'Missing' beschrieben wird, bedeutet dies, dass Spotify es nicht klassifiziert hat, also lassen wir es weg.

1. Entfernen Sie fehlende Daten, indem Sie sie herausfiltern

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Überprüfen Sie jetzt erneut die Genres:

    ![most popular](../../../../translated_images/de/all-genres.1d56ef06cefbfcd6.webp)

1. Bei weitem dominieren die drei Top-Genres diesen Datensatz. Konzentrieren wir uns auf `afro dancehall`, `afropop` und `nigerian pop` und filtern zusätzlich den Datensatz, um alles mit einem Popularitätswert von 0 zu entfernen (was bedeutet, dass es im Datensatz keine Popularitätsklassifizierung gab und für unsere Zwecke als Rauschen betrachtet werden kann):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Führen Sie einen kurzen Test durch, um zu sehen, ob die Daten in irgendeiner besonders starken Weise korrelieren:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/de/correlation.a9356bb798f5eea5.webp)

    Die einzige starke Korrelation besteht zwischen `energy` und `loudness`, was nicht allzu überraschend ist, da laute Musik in der Regel ziemlich energetisch ist. Ansonsten sind die Korrelationen relativ schwach. Es wird interessant sein zu sehen, was ein Clustering-Algorithmus aus diesen Daten machen kann.

    > 🎓 Beachten Sie, dass Korrelation keine Kausalität impliziert! Wir haben den Beweis für eine Korrelation, aber keinen Beweis für eine Kausalität. Eine [amüsante Webseite](https://tylervigen.com/spurious-correlations) zeigt einige Visualisierungen, die diesen Punkt hervorheben.

Gibt es eine Konvergenz in diesem Datensatz bezüglich der wahrgenommenen Popularität und Tanzbarkeit eines Songs? Ein FacetGrid zeigt konzentrische Kreise, die unabhängig vom Genre übereinstimmen. Könnte es sein, dass sich der nigerianische Geschmack auf ein bestimmtes Maß an Tanzbarkeit für dieses Genre einpendelt?

✅ Probieren Sie verschiedene Datenpunkte (energy, loudness, speechiness) und mehr oder andere Musikgenres aus. Was können Sie entdecken? Schauen Sie sich die Tabelle `df.describe()` an, um die allgemeine Verteilung der Datenpunkte zu sehen.

### Übung - Datenverteilung

Unterscheiden sich diese drei Genres signifikant in der Wahrnehmung ihrer Tanzbarkeit, basierend auf ihrer Popularität?

1. Untersuchen Sie die Verteilung der Daten der drei Top-Genres hinsichtlich Popularität und Tanzbarkeit entlang einer gegebenen x- und y-Achse.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Sie können konzentrische Kreise um einen allgemeinen Konvergenzpunkt entdecken, die die Verteilung der Punkte zeigen.

    > 🎓 Beachten Sie, dass dieses Beispiel ein KDE (Kernel Density Estimate) Diagramm verwendet, das die Daten mittels einer kontinuierlichen Wahrscheinlichkeitsdichtekurve darstellt. Dies ermöglicht uns die Interpretation von Daten bei der Arbeit mit mehreren Verteilungen.

    Im Allgemeinen stimmen die drei Genres lose in Bezug auf ihre Popularität und Tanzbarkeit überein. Die Bestimmung von Clustern in diesen lose ausgerichteten Daten wird eine Herausforderung sein:

    ![distribution](../../../../translated_images/de/distribution.9be11df42356ca95.webp)

1. Erstellen Sie ein Streudiagramm:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Ein Streudiagramm mit denselben Achsen zeigt ein ähnliches Konvergenzmuster

    ![Facetgrid](../../../../translated_images/de/facetgrid.9b2e65ce707eba1f.webp)

Im Allgemeinen können Sie für Clustering Streudiagramme verwenden, um Cluster von Daten zu zeigen, daher ist das Beherrschen dieser Art der Visualisierung sehr nützlich. In der nächsten Lektion werden wir diese gefilterten Daten verwenden und k-Means-Clustering anwenden, um Gruppen in diesen Daten zu entdecken, die sich auf interessante Weise überschneiden.

---

## 🚀 Herausforderung

Bereiten Sie für die nächste Lektion ein Diagramm über die verschiedenen Clustering-Algorithmen vor, die Sie entdecken und in einer Produktionsumgebung verwenden könnten. Welche Arten von Problemen versucht das Clustering zu lösen?

## [Post-Lecture-Quiz](https://ff-quizzes.netlify.app/en/ml/)

## Rückblick & Selbststudium

Bevor Sie Clustering-Algorithmen anwenden, ist es, wie wir gelernt haben, eine gute Idee, die Natur Ihres Datensatzes zu verstehen. Lesen Sie mehr zu diesem Thema [hier](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Dieser hilfreiche Artikel](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) führt Sie durch die verschiedenen Verhaltensweisen der Clustering-Algorithmen, abhängig von unterschiedlichen Datenformen.

## Aufgabe

[Forschen Sie nach weiteren Visualisierungen für Clustering](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Haftungsausschluss**:
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner Ursprungssprache gilt als maßgebliche Quelle. Bei kritischen Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Verwendung dieser Übersetzung entstehen.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->