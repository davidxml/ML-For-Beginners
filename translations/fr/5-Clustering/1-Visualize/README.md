# Introduction au clustering

Le clustering est un type d’[Apprentissage Non Supervisé](https://wikipedia.org/wiki/Unsupervised_learning) qui présume qu’un jeu de données n’est pas étiqueté ou que ses entrées ne sont pas associées à des sorties prédéfinies. Il utilise divers algorithmes pour trier des données non étiquetées et fournir des regroupements selon les motifs qu’il distingue dans les données.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Cliquez sur l’image ci-dessus pour une vidéo. Pendant que vous étudiez l’apprentissage automatique avec le clustering, profitez de quelques morceaux de Dance Hall nigérian - c’est une chanson très appréciée de 2014 par PSquare.

## [Quiz pré-conférence](https://ff-quizzes.netlify.app/en/ml/)

### Introduction

[Le clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) est très utile pour l’exploration des données. Voyons s’il peut aider à découvrir les tendances et motifs dans la façon dont les audiences nigérianes consomment la musique.

✅ Prenez une minute pour réfléchir aux usages du clustering. Dans la vie réelle, le clustering a lieu chaque fois que vous avez un tas de linge et devez trier les vêtements des membres de votre famille 🧦👕👖🩲. En science des données, le clustering intervient lors de l’analyse des préférences d’un utilisateur, ou pour déterminer les caractéristiques de tout jeu de données non étiqueté. Le clustering, en quelque sorte, aide à donner un sens au chaos, comme un tiroir à chaussettes.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Cliquez sur l’image ci-dessus pour une vidéo : John Guttag du MIT présente le clustering

Dans un cadre professionnel, le clustering peut être utilisé pour déterminer des choses comme la segmentation de marché, déterminer quels groupes d’âge achètent quels articles, par exemple. Un autre usage serait la détection d’anomalies, peut-être pour détecter des fraudes dans un ensemble de transactions par carte de crédit. Ou vous pourriez utiliser le clustering pour détecter des tumeurs dans une série de scans médicaux.

✅ Réfléchissez une minute à la façon dont vous avez pu rencontrer le clustering 'dans la nature', dans un contexte bancaire, e-commerce, ou commercial.

> 🎓 Fait intéressant, l’analyse de clusters est née dans les domaines de l’Anthropologie et de la Psychologie dans les années 1930. Pouvez-vous imaginer comment elle a pu être utilisée ?

Alternativement, vous pourriez l’utiliser pour regrouper les résultats de recherche - par liens d’achat, images, ou avis par exemple. Le clustering est utile quand vous avez un grand ensemble de données que vous souhaitez réduire et sur lequel vous voulez effectuer une analyse plus fine, la technique peut ainsi être utilisée pour apprendre à connaître les données avant de construire d’autres modèles.

✅ Une fois vos données organisées en clusters, vous leur attribuez un identifiant de cluster, et cette technique peut être utile pour préserver la confidentialité d’un jeu de données ; vous pouvez ainsi référer un point de données par son identifiant de cluster, plutôt que par des données identifiables plus révélatrices. Pouvez-vous penser à d’autres raisons pour lesquelles vous utiliseriez un identifiant de cluster plutôt que d’autres éléments du cluster pour l’identifier ?

Approfondissez votre compréhension des techniques de clustering dans ce [module Learn](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)
## Démarrer avec le clustering

[Scikit-learn offre une large gamme](https://scikit-learn.org/stable/modules/clustering.html) de méthodes pour effectuer du clustering. Le type choisi dépendra de votre cas d’usage. Selon la documentation, chaque méthode présente divers avantages. Voici un tableau simplifié des méthodes supportées par Scikit-learn et leurs cas d’utilisation appropriés :

| Nom de la méthode           | Cas d’utilisation                                                      |
| :-------------------------- | :--------------------------------------------------------------------- |
| K-Means                    | usage général, inductif                                                |
| Affinity propagation       | nombreux, clusters inégaux, inductif                                  |
| Mean-shift                 | nombreux, clusters inégaux, inductif                                  |
| Spectral clustering        | peu nombreux, clusters équilibrés, transductif                        |
| Ward hierarchical clustering | nombreux, clusters contraints, transductif                           |
| Agglomerative clustering   | nombreux, contraintes, distances non euclidiennes, transductif       |
| DBSCAN                     | géométrie non plate, clusters inégaux, transductif                   |
| OPTICS                     | géométrie non plate, clusters inégaux avec densité variable, transductif |
| Gaussian mixtures          | géométrie plate, inductif                                            |
| BIRCH                      | grand jeu de données avec valeurs aberrantes, inductif                |

> 🎓 La façon dont nous créons les clusters dépend beaucoup de la manière dont nous regroupons les points de données. Détaillons un vocabulaire :
>
> 🎓 ['Transductif' vs. 'inductif'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> L’inférence transductive est dérivée des cas d’entraînement observés qui correspondent à des cas de test spécifiques. L’inférence inductive est dérivée des cas d'entraînement qui correspondent à des règles générales appliquées ensuite aux cas de test.
> 
> Exemple : Imaginez que vous avez un jeu de données partiellement étiqueté. Certaines choses sont des 'disques vinyle', d’autres des 'CDs', et d’autres sont vides. Votre tâche est d’étiqueter les vides. Si vous choisissez une approche inductive, vous entraînez un modèle cherchant 'disques vinyle' et 'CDs', et appliquez ces étiquettes aux données non étiquetées. Cette approche aura du mal à classer ce qui sont en réalité des 'cassettes'. Une approche transductive, en revanche, traite mieux ces données inconnues car elle regroupe les éléments similaires puis applique une étiquette au groupe. Ici, les clusters pourraient refléter les 'objets musicaux ronds' et les 'objets musicaux carrés'.
> 
> 🎓 ['Géométrie non plate' vs. 'plate'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Issu de la terminologie mathématique, la géométrie non plate versus plate réfère à la mesure des distances entre points par des méthodes géométriques 'plates' ([Euclidiennes](https://wikipedia.org/wiki/Euclidean_geometry)) ou 'non plates' (non euclidiennes).
>
> 'Plate' ici réfère à la géométrie Euclidienne (dont une partie est enseignée comme géométrie dans le plan), et non plate réfère à la géométrie non euclidienne. Quel rapport avec l’apprentissage machine ? En tant que domaines ancrés dans les mathématiques, ils nécessitent de mesurer les distances entre points dans les clusters, ce qui peut se faire de manière 'plate' ou 'non plate' selon la nature des données. Les [distances euclidiennes](https://wikipedia.org/wiki/Euclidean_distance) sont mesurées comme la longueur du segment entre deux points. Les [distances non euclidiennes](https://wikipedia.org/wiki/Non-Euclidean_geometry) sont mesurées le long d’une courbe. Si vos données, visualisées, ne semblent pas exister sur un plan, un algorithme spécialisé pourrait être nécessaire.
>
![Infographie Géométrie plate vs Non plate](../../../../translated_images/fr/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infographie par [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Distances'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Les clusters sont définis par leur matrice de distance, c’est-à-dire les distances entre points. Cette distance peut être mesurée de plusieurs façons. Les clusters euclidiens sont définis par la moyenne des valeurs des points, et contiennent un 'centreide' ou point central. Les distances sont mesurées par la distance à ce centreide. Les distances non euclidiennes réfèrent aux 'clustroïdes', le point le plus proche des autres points. Les clustroïdes peuvent être définis selon diverses méthodes.
> 
> 🎓 ['Contraint'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Le clustering contraint](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introduit l’apprentissage 'semi-supervisé' dans cette méthode non supervisée. Les relations entre points sont indiquées comme 'pas lié' ou 'doit être lié' de sorte que certaines règles sont imposées au jeu de données.
>
>Exemple : Si un algorithme est lâché sur un lot de données non étiquetées ou semi-étiquetées, les clusters produits peuvent être de mauvaise qualité. Dans l’exemple ci-dessus, les clusters pourraient regrouper 'objets musicaux ronds' et 'objets musicaux carrés' et 'objets triangulaires' et 'cookies'. Avec des contraintes ou règles (ex : "l’objet doit être en plastique", "l’objet doit pouvoir produire de la musique"), l’algorithme peut être 'contraint' à faire de meilleurs choix.
> 
> 🎓 'Densité'
> 
> Les données 'bruyantes' sont considérées comme 'denses'. Les distances entre les points de chacun de ses clusters peuvent s’avérer plus ou moins denses, autrement dit 'bondées', c’est pourquoi les données doivent être analysées avec la méthode de clustering appropriée. [Cet article](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) illustre la différence entre l’utilisation des algorithmes K-Means vs. HDBSCAN pour explorer un jeu de données bruyant avec densité de cluster inégale.

## Algorithmes de clustering

Il existe plus de 100 algorithmes de clustering, leur usage dépend de la nature des données. Discutons de certains majeurs :

- **Clustering hiérarchique**. Si un objet est classé selon sa proximité à un objet proche, plutôt qu’à un objet plus éloigné, des clusters sont formés selon la distance de leurs membres aux autres objets. L’agglomératif de Scikit-learn est hiérarchique.

   ![Infographie Clustering hiérarchique](../../../../translated_images/fr/hierarchical.bf59403aa43c8c47.webp)
   > Infographie par [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering par centroïde**. Cet algorithme populaire nécessite le choix de 'k', ou du nombre de clusters à former, après quoi l’algorithme détermine le centre d’un cluster et rassemble les données autour de ce point. Le [K-means clustering](https://wikipedia.org/wiki/K-means_clustering) est une version populaire. Le centre est déterminé par la moyenne la plus proche, d’où le nom. La distance au carré du cluster est minimisée.

   ![Infographie Clustering par centroïde](../../../../translated_images/fr/centroid.097fde836cf6c918.webp)
   > Infographie par [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering basé sur la distribution**. Fondé sur la modélisation statistique, il consiste à déterminer la probabilité qu’un point de données appartienne à un cluster, et à l’affecter en conséquence. Les méthodes de mélange gaussien appartiennent à ce type.

- **Clustering basé sur la densité**. Les points de données sont affectés à des clusters basés sur leur densité, ou leur regroupement entre eux. Les points éloignés du groupe sont considérés comme des valeurs aberrantes ou du bruit. DBSCAN, Mean-shift et OPTICS appartiennent à ce type.

- **Clustering basé sur la grille**. Pour les jeux de données multidimensionnels, une grille est créée et les données sont réparties entre les cellules de la grille, créant ainsi des clusters.

## Exercice - clusterisez vos données

Le clustering en tant que technique est grandement aidé par une visualisation appropriée, commençons par visualiser nos données musicales. Cet exercice nous aidera à choisir la méthode de clustering la mieux adaptée à la nature de ces données.

1. Ouvrez le fichier [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) dans ce dossier.

1. Importez le package `Seaborn` pour une bonne visualisation des données.

    ```python
    !pip install seaborn
    ```

1. Ajoutez les données de chansons depuis [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Chargez un dataframe avec des données sur les chansons. Préparez-vous à explorer ces données en important les bibliothèques et en affichant les données :

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Vérifiez les premières lignes des données :

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Obtenez quelques informations sur le dataframe en appelant `info()` :

    ```python
    df.info()
    ```

   Le résultat ressemble à ceci :

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

1. Vérifiez de nouveau la présence de valeurs nulles, en appelant `isnull()` et en vérifiant que la somme est 0 :

    ```python
    df.isnull().sum()
    ```

    Tout est bon :

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

1. Décrivez les données :

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

> 🤔 Si nous travaillons avec le clustering, une méthode non supervisée qui ne nécessite pas de données étiquetées, pourquoi montrons-nous ces données avec des étiquettes ? Lors de la phase d'exploration des données, elles sont utiles, mais elles ne sont pas nécessaires pour que les algorithmes de clustering fonctionnent. Vous pourriez tout aussi bien supprimer les en-têtes de colonnes et vous référer aux données par numéro de colonne.

Examinez les valeurs générales des données. Notez que la popularité peut être '0', ce qui indique des chansons sans classement. Supprimons-les rapidement.

1. Utilisez un barplot pour découvrir les genres les plus populaires :

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![les plus populaires](../../../../translated_images/fr/popular.9c48d84b3386705f.webp)

✅ Si vous souhaitez voir plus de valeurs en tête, changez le top `[:5]` en une valeur plus grande, ou supprimez-le pour voir tout.

Notez que lorsque le genre principal est décrit comme 'Missing', cela signifie que Spotify ne l'a pas classé, supprimons-le donc.

1. Éliminez les données manquantes en les filtrant

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Vérifiez de nouveau les genres :

    ![les plus populaires](../../../../translated_images/fr/all-genres.1d56ef06cefbfcd6.webp)

1. De loin, les trois genres principaux dominent ce jeu de données. Concentrons-nous sur `afro dancehall`, `afropop`, et `nigerian pop`, en filtrant en outre le jeu de données pour supprimer tout ce qui a une popularité de 0 (ce qui signifie qu'il n'a pas été classé avec une popularité dans le jeu de données et peut être considéré comme du bruit pour nos besoins) :

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Faites un test rapide pour voir si les données sont fortement corrélées d'une manière particulière :

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![corrélations](../../../../translated_images/fr/correlation.a9356bb798f5eea5.webp)

    La seule forte corrélation est entre `energy` et `loudness`, ce qui n'est pas trop surprenant, étant donné que la musique forte est généralement assez énergique. Sinon, les corrélations sont relativement faibles. Il sera intéressant de voir ce qu'un algorithme de clustering peut tirer de ces données.

    > 🎓 Notez que corrélation ne signifie pas causalité ! Nous avons la preuve d'une corrélation mais pas de causalité. Un [site web amusant](https://tylervigen.com/spurious-correlations) présente des visuels qui soulignent ce point.

Y a-t-il une convergence dans ce jeu de données autour de la popularité perçue d'une chanson et de sa danseabilité ? Un FacetGrid montre qu'il y a des cercles concentriques qui s’alignent, quel que soit le genre. Se pourrait-il que les goûts nigérians convergent à un certain niveau de danseabilité pour ce genre ?

✅ Essayez différents points de données (énergie, volume sonore, débit de parole) et plus ou différents genres musicaux. Que pouvez-vous découvrir ? Jetez un œil au tableau `df.describe()` pour voir la répartition générale des points de données.

### Exercice - répartition des données

Ces trois genres sont-ils significativement différents dans la perception de leur danseabilité, en fonction de leur popularité ?

1. Examinez la répartition des données sur la popularité et la danseabilité pour nos trois genres principaux selon un axe x et y donnés.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Vous pouvez découvrir des cercles concentriques autour d'un point général de convergence, montrant la répartition des points.

    > 🎓 Notez que cet exemple utilise un graphique KDE (estimation de la densité par noyau) qui représente les données avec une courbe continue de densité de probabilité. Cela nous permet d'interpréter les données lorsqu’on travaille avec plusieurs distributions.

    En général, les trois genres s’alignent grossièrement sur la popularité et la danseabilité. Déterminer des clusters dans ces données grossièrement alignées sera un défi :

    ![répartition](../../../../translated_images/fr/distribution.9be11df42356ca95.webp)

1. Créez un scatter plot :

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Un scatterplot des mêmes axes montre un schéma similaire de convergence

    ![Facetgrid](../../../../translated_images/fr/facetgrid.9b2e65ce707eba1f.webp)

En général, pour le clustering, vous pouvez utiliser des scatterplots pour montrer des groupes de données, donc maîtriser ce type de visualisation est très utile. Dans la prochaine leçon, nous prendrons ces données filtrées et utiliserons le clustering k-means pour découvrir des groupes dans ces données qui semblent se chevaucher de façon intéressante.

---

## 🚀Défi

En préparation de la prochaine leçon, réalisez un graphique sur les différents algorithmes de clustering que vous pourriez découvrir et utiliser en production. Quels types de problèmes le clustering cherche-t-il à résoudre ?

## [Quiz post-conférence](https://ff-quizzes.netlify.app/en/ml/)

## Revue et auto-apprentissage

Avant d'appliquer les algorithmes de clustering, comme nous l'avons appris, il est conseillé de comprendre la nature de votre jeu de données. Lisez-en davantage à ce sujet [ici](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Cet article utile](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) vous guide à travers les différents comportements des divers algorithmes de clustering, selon différentes formes de données.

## Devoir

[Recherchez d'autres visualisations pour le clustering](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Avertissement** :
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforçions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source faisant autorité. Pour les informations critiques, il est recommandé de recourir à une traduction professionnelle réalisée par un humain. Nous ne saurions être tenus responsables des malentendus ou erreurs d'interprétation découlant de l'utilisation de cette traduction.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->