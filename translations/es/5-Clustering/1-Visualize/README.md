# Introducción al clustering

El clustering es un tipo de [Aprendizaje No Supervisado](https://wikipedia.org/wiki/Unsupervised_learning) que presume que un conjunto de datos no tiene etiquetas o que sus entradas no están emparejadas con salidas predefinidas. Utiliza varios algoritmos para clasificar datos no etiquetados y proporcionar agrupaciones según los patrones que detecta en los datos. 

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Haz clic en la imagen de arriba para ver un video. Mientras estudias aprendizaje automático con clustering, disfruta de algunos temas de Dance Hall nigeriano; esta es una canción muy valorada de 2014 de PSquare.

## [Cuestionario previo a la clase](https://ff-quizzes.netlify.app/en/ml/)

### Introducción

[El clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) es muy útil para la exploración de datos. Veamos si puede ayudar a descubrir tendencias y patrones en la forma en que las audiencias nigerianas consumen música.

✅ Tómate un minuto para pensar en los usos del clustering. En la vida real, el clustering ocurre siempre que tienes un montón de ropa y necesitas ordenar la ropa de los miembros de tu familia 🧦👕👖🩲. En ciencia de datos, el clustering ocurre cuando intentas analizar las preferencias de un usuario o determinar las características de cualquier conjunto de datos sin etiquetar. El clustering, en cierto modo, ayuda a dar sentido al caos, como un cajón de calcetines.

[![Introducción a ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introducción al Clustering")

> 🎥 Haz clic en la imagen de arriba para ver un video: John Guttag del MIT presenta el clustering

En un entorno profesional, el clustering puede usarse para determinar aspectos como la segmentación de mercado, por ejemplo, qué grupos de edad compran qué artículos. Otro uso sería la detección de anomalías, quizás para detectar fraude a partir de un conjunto de datos de transacciones con tarjeta de crédito. O podrías usar clustering para detectar tumores en un lote de escáneres médicos. 

✅ Piensa un minuto en cómo podrías haberte encontrado con el clustering 'en la práctica', en un entorno bancario, de comercio electrónico o empresarial.

> 🎓 Curiosamente, el análisis de clústeres se originó en los campos de la Antropología y la Psicología en la década de 1930. ¿Puedes imaginar cómo podría haberse usado?

Alternativamente, podrías usarlo para agrupar resultados de búsqueda, por ejemplo, por enlaces de compras, imágenes o reseñas. El clustering es útil cuando tienes un conjunto de datos grande que quieres reducir y sobre el que quieres realizar un análisis más granular, por lo que la técnica puede usarse para aprender sobre los datos antes de construir otros modelos.

✅ Una vez que tus datos estén organizados en clústeres, les asignas un Id de clúster, y esta técnica puede ser útil para preservar la privacidad de un conjunto de datos; en lugar de referirte a un punto de datos por datos identificables más reveladores, puedes referirte a él por su Id de clúster. ¿Puedes pensar en otras razones por las que preferirías usar un Id de clúster en lugar de otros elementos del clúster para identificarlo?

Profundiza tu comprensión de las técnicas de clustering en este [módulo de aprendizaje](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott).

## Comenzando con el clustering

[Scikit-learn ofrece una gran variedad](https://scikit-learn.org/stable/modules/clustering.html) de métodos para realizar clustering. El tipo que elijas dependerá de tu caso de uso. Según la documentación, cada método tiene varios beneficios. Aquí tienes una tabla simplificada de los métodos admitidos por Scikit-learn y sus casos de uso adecuados:

| Nombre del método            | Caso de uso                                                            |
| :--------------------------- | :-------------------------------------------------------------------- |
| K-Means                      | propósito general, inductivo                                           |
| Propagación de afinidad      | muchos clústeres desiguales, inductivo                                |
| Mean-shift                   | muchos clústeres desiguales, inductivo                                |
| Clustering espectral         | pocos clústeres iguales, traductivo                                   |
| Clustering jerárquico de Ward| muchos clústeres restringidos, traductivo                             |
| Clustering aglomerativo      | muchos, restringidos, distancias no euclidianas, traductivo          |
| DBSCAN                       | geometría no plana, clústeres desiguales, traductivo                 |
| OPTICS                       | geometría no plana, clústeres desiguales con densidad variable, traductivo |
| Mezclas gaussianas           | geometría plana, inductivo                                            |
| BIRCH                        | gran conjunto de datos con valores atípicos, inductivo               |

> 🎓 Cómo creamos clústeres tiene mucho que ver con cómo reunimos los puntos de datos en grupos. Vamos a desglosar un poco el vocabulario:
>
> 🎓 ['Traductivo' vs. 'inductivo'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> La inferencia traductiva se deriva de casos de entrenamiento observados que se asignan a casos de prueba específicos. La inferencia inductiva se deriva de casos de entrenamiento que se asignan a reglas generales que solo entonces se aplican a casos de prueba. 
> 
> Un ejemplo: imagina que tienes un conjunto de datos que está solo parcialmente etiquetado. Algunas cosas son 'discos de vinilo', otras 'CDs' y otras están en blanco. Tu trabajo es asignar etiquetas a los vacíos. Si eliges un enfoque inductivo, entrenarías un modelo buscando 'discos de vinilo' y 'CDs', y aplicarías esas etiquetas a tus datos sin etiquetar. Este enfoque tendrá problemas para clasificar cosas que en realidad son 'cassettes'. Un enfoque traductivo, por otro lado, maneja estos datos desconocidos de manera más efectiva al trabajar para agrupar artículos similares y luego aplicar una etiqueta a un grupo. En este caso, los clústeres podrían reflejar 'cosas musicales redondas' y 'cosas musicales cuadradas'. 
> 
> 🎓 ['Geometría no plana' vs. 'plana'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Derivado de la terminología matemática, geometría no plana vs. plana se refiere a la medida de distancias entre puntos mediante métodos geométricos 'planos' ([Euclidianos](https://wikipedia.org/wiki/Euclidean_geometry)) o 'no planos' (no euclidianos). 
>
> 'Plano' en este contexto se refiere a la geometría euclidiana (partes de la cual se enseñan como geometría 'del plano'), y no plano se refiere a la geometría no euclidiana. ¿Qué tiene que ver la geometría con el aprendizaje automático? Bueno, como dos campos que están basados en las matemáticas, debe existir una forma común de medir las distancias entre puntos en los clústeres, y eso puede hacerse de manera 'plana' o 'no plana', dependiendo de la naturaleza de los datos. Las [distancias euclidianas](https://wikipedia.org/wiki/Euclidean_distance) se miden como la longitud de un segmento de línea entre dos puntos. Las [distancias no euclidianas](https://wikipedia.org/wiki/Non-Euclidean_geometry) se miden a lo largo de una curva. Si tus datos, visualizados, parecen no existir en un plano, podrías necesitar usar un algoritmo especializado para manejarlos.
>
![Infografía de geometría plana vs no plana](../../../../translated_images/es/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infografía por [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Distancias'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Los clústeres se definen por su matriz de distancias, es decir, las distancias entre puntos. Esta distancia puede medirse de varias maneras. Los clústeres euclidianos se definen por el promedio de los valores de los puntos, y contienen un 'centroide' o punto central. Las distancias se miden así por la distancia al centroide. Las distancias no euclidianas se refieren a 'clustroides', el punto más cercano a otros puntos. Los clustroides a su vez pueden definirse de varias maneras.
> 
> 🎓 ['Restringido'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [El Clustering Restringido](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introduce el aprendizaje 'semi supervisado' en este método no supervisado. Las relaciones entre puntos se marcan como 'no se puede enlazar' o 'deben enlazarse' para que algunas reglas se impongan en el conjunto de datos.
>
>Un ejemplo: Si un algoritmo se aplica libremente sobre un lote de datos sin etiqueta o semi-etiquetados, los clústeres que produce pueden ser de baja calidad. En el ejemplo anterior, los clústeres podrían agrupar 'cosas musicales redondas', 'cosas musicales cuadradas', 'cosas triangulares' y 'galletas'. Si se le dan algunas restricciones o reglas a seguir ("el artículo debe estar hecho de plástico", "el artículo debe poder producir música"), esto puede ayudar a 'restringir' el algoritmo para que tome mejores decisiones.
> 
> 🎓 'Densidad'
> 
> Los datos que son 'ruidosos' se consideran 'densos'. Las distancias entre los puntos en cada uno de sus clústeres pueden demostrar, tras examen, ser más o menos densas o 'concurridas', y por lo tanto estos datos deben analizarse con el método de clustering adecuado. [Este artículo](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) demuestra la diferencia entre usar clustering K-Means frente a algoritmos HDBSCAN para explorar un conjunto de datos ruidoso con densidad de clústeres desigual.

## Algoritmos de clustering

Existen más de 100 algoritmos de clustering, y su uso depende de la naturaleza de los datos en cuestión. Vamos a discutir algunos de los principales:

- **Clustering jerárquico**. Si un objeto se clasifica por su proximidad a un objeto cercano en lugar de a uno más lejano, se forman clústeres basados en la distancia de sus miembros hacia y desde otros objetos. El clustering aglomerativo de Scikit-learn es jerárquico.

   ![Infografía de clustering jerárquico](../../../../translated_images/es/hierarchical.bf59403aa43c8c47.webp)
   > Infografía por [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering centróide**. Este algoritmo popular requiere la elección de 'k', o el número de clústeres a formar, tras lo cual el algoritmo determina el punto central de un clúster y reúne los datos alrededor de ese punto. El [clustering K-means](https://wikipedia.org/wiki/K-means_clustering) es una versión popular del clustering centróide. El centro se determina por la media más cercana, de ahí el nombre. Se minimiza la distancia cuadrada desde el clúster.

   ![Infografía de clustering centróide](../../../../translated_images/es/centroid.097fde836cf6c918.webp)
   > Infografía por [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering basado en distribución**. Basado en modelado estadístico, el clustering basado en distribución se centra en determinar la probabilidad de que un punto de datos pertenezca a un clúster y asignarlo en consecuencia. Los métodos de mezcla gaussiana pertenecen a este tipo.

- **Clustering basado en densidad**. Los puntos de datos se asignan a clústeres según su densidad o su agrupamiento entre sí. Los puntos de datos alejados del grupo se consideran valores atípicos o ruido. DBSCAN, Mean-shift y OPTICS pertenecen a este tipo de clustering.

- **Clustering basado en rejilla**. Para conjuntos de datos multidimensionales, se crea una rejilla y los datos se dividen entre las celdas de la rejilla, creando así clústeres.

## Ejercicio - clusteriza tus datos

El clustering como técnica se ve enormemente facilitado por una visualización adecuada, así que comencemos visualizando nuestros datos musicales. Este ejercicio nos ayudará a decidir cuál de los métodos de clustering deberíamos usar de manera más efectiva para la naturaleza de estos datos.

1. Abre el archivo [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) en esta carpeta.

1. Importa el paquete `Seaborn` para una buena visualización de datos.

    ```python
    !pip install seaborn
    ```

1. Añade los datos de las canciones desde [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Carga un dataframe con algunos datos sobre las canciones. Prepárate para explorar estos datos importando las bibliotecas y mostrando los datos:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Revisa las primeras líneas de datos:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Obtén información sobre el dataframe, llamando a `info()`:

    ```python
    df.info()
    ```

   La salida se ve así:

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

1. Verifica nuevamente si hay valores nulos, llamando a `isnull()` y verificando que la suma sea 0:

    ```python
    df.isnull().sum()
    ```

    Se ve bien:

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

1. Describe los datos:

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

> 🤔 Si estamos trabajando con clustering, un método no supervisado que no requiere datos etiquetados, ¿por qué mostramos estos datos con etiquetas? En la fase de exploración de datos, resultan útiles, pero no son necesarias para que funcionen los algoritmos de clustering. Podrías perfectamente eliminar los encabezados de columna y referirte a los datos por número de columna.

Observa los valores generales de los datos. Nota que la popularidad puede ser '0', lo que muestra canciones que no tienen clasificación. Eliminaremos esos pronto.

1. Usa un diagrama de barras para encontrar los géneros más populares:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/es/popular.9c48d84b3386705f.webp)

✅ Si deseas ver más valores principales, cambia el top `[:5]` por un valor mayor, o elimínalo para ver todos.

Nota, cuando el género principal se describe como 'Missing', significa que Spotify no lo clasificó, así que deshagámonos de él.

1. Elimina los datos faltantes filtrándolos

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Ahora vuelve a revisar los géneros:

    ![most popular](../../../../translated_images/es/all-genres.1d56ef06cefbfcd6.webp)

1. Por mucho, los tres géneros principales dominan este conjunto de datos. Concentrémonos en `afro dancehall`, `afropop` y `nigerian pop`, además filtra el conjunto de datos para eliminar cualquier valor de popularidad 0 (lo que significa que no fue clasificado con una popularidad en el conjunto de datos y puede considerarse ruido para nuestros propósitos):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Realiza una prueba rápida para ver si los datos se correlacionan de manera especialmente fuerte:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/es/correlation.a9356bb798f5eea5.webp)

    La única correlación fuerte es entre `energy` y `loudness`, lo cual no es muy sorprendente, dado que la música alta suele ser bastante energética. Por lo demás, las correlaciones son relativamente débiles. Será interesante ver qué puede hacer un algoritmo de clustering con estos datos.

    > 🎓 ¡Nota que correlación no implica causalidad! Tenemos prueba de correlación pero no de causalidad. Un [sitio web divertido](https://tylervigen.com/spurious-correlations) tiene algunas visualizaciones que enfatizan este punto.

¿Existe alguna convergencia en este conjunto de datos respecto a la popularidad percibida de una canción y su bailabilidad? Un FacetGrid muestra que hay círculos concéntricos que se alinean, sin importar el género. ¿Podría ser que los gustos nigerianos converjan a cierto nivel de bailabilidad para este género?

✅ Prueba diferentes puntos de datos (energy, loudness, speechiness) y más o diferentes géneros musicales. ¿Qué puedes descubrir? Echa un vistazo a la tabla `df.describe()` para ver la distribución general de los puntos de datos.

### Ejercicio - distribución de datos

¿Son estos tres géneros significativamente diferentes en la percepción de su bailabilidad, basándonos en su popularidad?

1. Examina la distribución de datos de nuestros tres géneros principales para popularidad y bailabilidad a lo largo de un eje x y y dado.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Puedes descubrir círculos concéntricos alrededor de un punto general de convergencia, mostrando la distribución de puntos.

    > 🎓 Nota que este ejemplo usa un gráfico KDE (Estimación de Densidad de Núcleo) que representa los datos usando una curva continua de densidad de probabilidad. Esto nos permite interpretar datos cuando trabajamos con múltiples distribuciones.

    En general, los tres géneros se alinean de forma laxa en términos de su popularidad y bailabilidad. Determinar clusters en estos datos laxa y alineados será un desafío:

    ![distribution](../../../../translated_images/es/distribution.9be11df42356ca95.webp)

1. Crea un diagrama de dispersión:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Un diagrama de dispersión con los mismos ejes muestra un patrón similar de convergencia

    ![Facetgrid](../../../../translated_images/es/facetgrid.9b2e65ce707eba1f.webp)

En general, para clustering, puedes usar diagramas de dispersión para mostrar clusters de datos, por lo que dominar este tipo de visualización es muy útil. En la próxima lección, usaremos estos datos filtrados y aplicaremos k-means clustering para descubrir grupos en estos datos que parecen superponerse de maneras interesantes.

---

## 🚀Desafío

En preparación para la próxima lección, haz un gráfico sobre los distintos algoritmos de clustering que podrías descubrir y usar en un entorno productivo. ¿Qué tipos de problemas intenta abordar el clustering?

## [Cuestionario post-lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y Autoestudio

Antes de aplicar algoritmos de clustering, como hemos aprendido, es buena idea entender la naturaleza de tu conjunto de datos. Lee más sobre este tema [aquí](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Este artículo útil](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) te guía a través de las distintas formas en que se comportan varios algoritmos de clustering, dados diferentes formas de datos.

## Tarea

[Investiga otras visualizaciones para clustering](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Descargo de responsabilidad**:
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o inexactitudes. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción profesional humana. No somos responsables de cualquier malentendido o interpretación errónea que surja del uso de esta traducción.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->