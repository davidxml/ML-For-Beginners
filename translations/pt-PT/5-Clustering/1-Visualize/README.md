# Introdução ao clustering

Clustering é um tipo de [Aprendizagem Não Supervisionada](https://wikipedia.org/wiki/Unsupervised_learning) que presume que um conjunto de dados não está rotulado ou que as suas entradas não estão associadas a saídas predefinidas. Utiliza vários algoritmos para analisar dados não rotulados e fornecer agrupamentos de acordo com os padrões que identifica nos dados.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Clique na imagem acima para um vídeo. Enquanto estuda aprendizagem automática com clustering, desfrute de algumas faixas de Dance Hall nigeriano - esta é uma canção muito bem avaliada de 2014 por PSquare.

## [Teste pré-aula](https://ff-quizzes.netlify.app/en/ml/)

### Introdução

[Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) é muito útil para a exploração de dados. Vamos ver se pode ajudar a descobrir tendências e padrões na forma como os públicos nigerianos consomem música.

✅ Reserve um minuto para refletir sobre os usos do clustering. Na vida real, clustering acontece sempre que tem um monte de roupa para lavar e precisa de separar as roupas dos membros da sua família 🧦👕👖🩲. Em ciência de dados, clustering acontece quando se tenta analisar as preferências de um utilizador, ou determinar as características de qualquer conjunto de dados não rotulado. O clustering, de certa forma, ajuda a dar sentido ao caos, como numa gaveta de meias.

[![Introdução ao ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introdução ao Clustering")

> 🎥 Clique na imagem acima para ver um vídeo: John Guttag, do MIT, introduz o clustering

Num contexto profissional, o clustering pode ser usado para determinar coisas como a segmentação de mercado, determinando que grupos etários compram que itens, por exemplo. Outro uso seria a deteção de anomalias, talvez para detectar fraude num conjunto de dados de transações de cartões de crédito. Ou pode usar clustering para determinar tumores num lote de exames médicos.

✅ Pense um minuto em como poderá ter encontrado clustering 'no mundo real', num banco, comércio eletrónico ou contexto empresarial.

> 🎓 Curiosamente, a análise de clusters originou-se nos campos da Antropologia e Psicologia nos anos 1930. Consegue imaginar como pode ter sido usada?

Alternativamente, pode usá-lo para agrupar resultados de pesquisa - por links de compra, imagens ou opiniões, por exemplo. O clustering é útil quando se tem um grande conjunto de dados que se quer reduzir e sobre o qual se pretende fazer uma análise mais granular, por isso a técnica pode ser usada para aprender sobre os dados antes de outros modelos serem construídos.

✅ Uma vez que os seus dados estejam organizados em clusters, atribui-lhes um Id de cluster, e esta técnica pode ser útil para preservar a privacidade de um conjunto de dados; pode referir-se a um ponto de dados pelo seu id de cluster, em vez de por dados identificáveis mais reveladores. Consegue pensar em outras razões para referir-se a um Id de cluster em vez de outros elementos do cluster para o identificar?

Aprofunde a sua compreensão das técnicas de clustering neste [módulo Learn](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

## Começar com clustering

[Scikit-learn oferece uma grande variedade](https://scikit-learn.org/stable/modules/clustering.html) de métodos para realizar clustering. O tipo que escolher dependerá do seu caso de uso. Segundo a documentação, cada método tem vários benefícios. Aqui está uma tabela simplificada dos métodos suportados pelo Scikit-learn e os seus casos de uso apropriados:

| Nome do método              | Caso de uso                                                           |
| :-------------------------- | :------------------------------------------------------------------- |
| K-Means                    | uso geral, indutivo                                                  |
| Propagação de afinidade    | muitos, clusters irregulares, indutivo                              |
| Mean-shift                 | muitos, clusters irregulares, indutivo                              |
| Clustering espectral       | poucos, clusters uniformes, transdutivo                             |
| Clustering hierárquico Ward| muitos, clusters restritos, transdutivo                             |
| Clustering aglomerativo    | muitos, restrito, distâncias não euclidianas, transdutivo           |
| DBSCAN                    | geometria não plana, clusters irregulares, transdutivo             |
| OPTICS                    | geometria não plana, clusters irregulares com densidade variável, transdutivo |
| Misturas gaussianas       | geometria plana, indutivo                                           |
| BIRCH                     | conjunto de dados grande com outliers, indutivo                     |

> 🎓 Como criamos clusters tem muito a ver com a forma como agrupamos os pontos de dados. Vamos explorar algum vocabulário:
>
> 🎓 ['Transdutivo' vs. 'indutivo'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> A inferência transdutiva é derivada de casos de treino observados que correspondem a casos de teste específicos. A inferência indutiva é derivada de casos de treino que mapeiam para regras gerais que só depois são aplicadas aos casos de teste.
> 
> Um exemplo: Imagine que tem um conjunto de dados apenas parcialmente rotulado. Algumas coisas são 'discos', outras 'CDs' e algumas estão em branco. A sua tarefa é fornecer etiquetas para os em branco. Se optar por uma abordagem indutiva, treinaria um modelo à procura de 'discos' e 'CDs', e aplicaria essas etiquetas aos dados não rotulados. Esta abordagem terá dificuldade em classificar coisas que na verdade são 'cassetes'. Uma abordagem transdutiva, por outro lado, lida mais eficazmente com estes dados desconhecidos, pois trabalha para agrupar itens semelhantes e depois aplica uma etiqueta a um grupo. Neste caso, os clusters podem refletir 'coisas musicais redondas' e 'coisas musicais quadradas'.
> 
> 🎓 ['Geometria não plana' vs. 'plana'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Derivado da terminologia matemática, geometria não plana vs. plana refere-se à medida das distâncias entre pontos por métodos geométricos 'planos' ([Euclidianos](https://wikipedia.org/wiki/Euclidean_geometry)) ou 'não planos' (não Euclidianos).
>
> 'Plano', neste contexto, refere-se à geometria Euclidiana (partes da qual são ensinadas como geometria 'plana'), e não plano refere-se à geometria não Euclidiana. O que tem a geometria a ver com aprendizagem automática? Bem, como dois campos que têm raízes na matemática, deve haver uma forma comum de medir distâncias entre pontos em clusters, e isso pode ser feito de forma 'plana' ou 'não plana', dependendo da natureza dos dados. [Distâncias Euclidianas](https://wikipedia.org/wiki/Euclidean_distance) são medidas como o comprimento de um segmento de linha entre dois pontos. [Distâncias não Euclidianas](https://wikipedia.org/wiki/Non-Euclidean_geometry) são medidas ao longo de uma curva. Se os seus dados, visualizados, parecerem não existir num plano, poderá precisar de usar um algoritmo especializado para lidar com isso.
>
![Infográfico Geometria Plana vs Não Plana](../../../../translated_images/pt-PT/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infográfico por [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Distâncias'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Clusters são definidos pela sua matriz de distâncias, por exemplo, as distâncias entre pontos. Esta distância pode ser medida de várias formas. Clusters Euclidianos são definidos pela média dos valores dos pontos, e contêm um 'centroide' ou ponto central. As distâncias são portanto medidas pela distância para esse centroide. Distâncias não Euclidianas referem-se a 'clustroides', o ponto mais próximo dos outros pontos. Clustroides, por sua vez, podem ser definidos de várias formas.
> 
> 🎓 ['Restrito'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Clustering restrito](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introduz aprendizagem 'semi-supervisionada' neste método não supervisionado. As relações entre pontos são marcadas como 'não pode ligar' ou 'deve ligar' para que algumas regras sejam impostas ao conjunto de dados.
>
>Um exemplo: Se um algoritmo for colocado livremente num lote de dados não rotulados ou semi-rotulados, os clusters que produz podem ser de má qualidade. No exemplo acima, os clusters podem agrupar 'coisas musicais redondas', 'coisas musicais quadradas', 'coisas triangulares' e 'bolachas'. Se forem dadas algumas restrições, ou regras a seguir ("o item deve ser feito de plástico", "o item precisa ser capaz de produzir música") isto pode ajudar a 'constranger' o algoritmo a fazer escolhas melhores.
> 
> 🎓 'Densidade'
> 
> Dados que são 'ruidosos' são considerados 'densos'. As distâncias entre pontos em cada um dos seus clusters podem provar, ao exame, ser mais ou menos densas, ou 'lotadas' e assim estes dados precisam de ser analisados com o método de clustering apropriado. [Este artigo](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) demonstra a diferença entre usar clustering K-Means vs. algoritmos HDBSCAN para explorar um conjunto de dados ruidoso com densidade irregular de cluster.

## Algoritmos de clustering

Existem mais de 100 algoritmos de clustering, e o seu uso depende da natureza dos dados em questão. Vamos discutir alguns dos principais:

- **Clustering hierárquico**. Se um objeto for classificado pela sua proximidade a um objeto próximo, em vez de a um mais distante, formam-se clusters baseados na distância dos seus membros para e a partir de outros objetos. O clustering aglomerativo do Scikit-learn é hierárquico.

   ![Infográfico Clustering Hierárquico](../../../../translated_images/pt-PT/hierarchical.bf59403aa43c8c47.webp)
   > Infográfico por [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering por centróide**. Este algoritmo popular requer a escolha de 'k', ou o número de clusters a formar, após o que o algoritmo determina o ponto central de um cluster e reúne dados à volta desse ponto. [Clustering K-means](https://wikipedia.org/wiki/K-means_clustering) é uma versão popular do clustering por centróide. O centro é determinado pela média mais próxima, daí o nome. A distância quadrada ao cluster é minimizada.

   ![Infográfico Clustering por Centrôide](../../../../translated_images/pt-PT/centroid.097fde836cf6c918.webp)
   > Infográfico por [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering baseado em distribuição**. Baseado em modelagem estatística, o clustering baseado em distribuição centra-se em determinar a probabilidade de um ponto de dados pertencer a um cluster, e atribuio-lho em conformidade. Métodos de mistura Gaussiana pertencem a este tipo.

- **Clustering baseado em densidade**. Os pontos de dados são atribuídos a clusters com base na sua densidade, ou no seu agrupamento uns em torno dos outros. Pontos de dados longe do grupo são considerados outliers ou ruído. DBSCAN, Mean-shift e OPTICS pertencem a este tipo de clustering.

- **Clustering baseado em grelha**. Para conjuntos de dados multidimensionais, é criada uma grelha e os dados são divididos pelas células da grelha, criando assim clusters.

## Exercício - faça clusters dos seus dados

O clustering, enquanto técnica, é grandemente auxiliado por uma boa visualização, por isso vamos começar por visualizar os nossos dados musicais. Este exercício ajudará a decidir qual dos métodos de clustering devemos usar mais eficazmente para a natureza destes dados.

1. Abra o ficheiro [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) nesta pasta.

1. Importe o pacote `Seaborn` para boa visualização de dados.

    ```python
    !pip install seaborn
    ```

1. Anexe os dados das músicas do [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Carregue um dataframe com alguns dados sobre as músicas. Prepare-se para explorar estes dados importando as bibliotecas e despejando os dados:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Verifique as primeiras linhas dos dados:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Obtenha alguma informação sobre o dataframe, chamando `info()`:

    ```python
    df.info()
    ```

   A saída será semelhante a esta:

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

1. Verifique novamente a existência de valores nulos, chamando `isnull()` e verificando se a soma é 0:

    ```python
    df.isnull().sum()
    ```

    Está tudo bem:

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

1. Descreva os dados:

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

> 🤔 Se estamos a trabalhar com clustering, um método não supervisionado que não requer dados rotulados, por que estamos a mostrar estes dados com etiquetas? Na fase de exploração de dados, são úteis, mas não são necessárias para os algoritmos de clustering funcionarem. Poderia muito bem remover os cabeçalhos das colunas e referir-se aos dados pelo número da coluna.

Observe os valores gerais dos dados. Note que a popularidade pode ser '0', o que mostra músicas que não têm classificação. Vamos removê-las em breve.

1. Utilize um gráfico de barras para descobrir os géneros mais populares:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/pt-PT/popular.9c48d84b3386705f.webp)

✅ Se quiser ver mais valores principais, altere o topo `[:5]` para um valor maior, ou remova-o para ver todos.

Note que, quando o género principal é descrito como 'Missing', isso significa que o Spotify não o classificou, por isso vamos livrar-nos dele.

1. Elimine os dados em falta filtrando-os

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Agora verifique novamente os géneros:

    ![most popular](../../../../translated_images/pt-PT/all-genres.1d56ef06cefbfcd6.webp)

1. De longe, os três géneros principais dominam este conjunto de dados. Vamos concentrar-nos em `afro dancehall`, `afropop` e `nigerian pop`, filtrando também o conjunto de dados para remover qualquer coisa com valor de popularidade 0 (significando que não foi classificado com uma popularidade no conjunto de dados e pode ser considerado ruído para os nossos propósitos):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Faça um teste rápido para ver se os dados se correlacionam de alguma forma particularmente forte:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/pt-PT/correlation.a9356bb798f5eea5.webp)

    A única correlação forte é entre `energy` e `loudness`, o que não é surpreendente, dado que música alta normalmente é bastante energética. Caso contrário, as correlações são relativamente fracas. Será interessante ver o que um algoritmo de clustering pode fazer com estes dados.

    > 🎓 Note que correlação não implica causalidade! Temos prova de correlação, mas não de causalidade. Um [site divertido](https://tylervigen.com/spurious-correlations) tem alguns visuais que enfatizam este ponto.

Existe alguma convergência neste conjunto de dados em torno da popularidade percebida de uma música e a sua dança? Um FacetGrid mostra que há círculos concêntricos que se alinham, independentemente do género. Poderá ser que os gostos nigerianos convirjam num certo nível de dança para este género?

✅ Experimente diferentes pontos de dados (energia, volume, discurso) e mais ou diferentes géneros musicais. O que pode descobrir? Dê uma olhadela na tabela `df.describe()` para ver a dispersão geral dos pontos de dados.

### Exercício – distribuição dos dados

Estes três géneros são significativamente diferentes na perceção da sua dança, baseado na sua popularidade?

1. Examine a distribuição dos dados dos nossos três géneros principais para popularidade e dança numa dada eixo x e y.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Pode descobrir círculos concêntricos em torno de um ponto geral de convergência, mostrando a distribuição de pontos.

    > 🎓 Note que este exemplo utiliza um gráfico KDE (Estimativa de Densidade Kernel) que representa os dados usando uma curva contínua de densidade de probabilidade. Isto permite interpretar os dados quando se trabalha com múltiplas distribuições.

    Em geral, os três géneros alinham-se vagamente em termos de sua popularidade e dança. Determinar clusters neste conjunto de dados vagamente alinhado será um desafio:

    ![distribution](../../../../translated_images/pt-PT/distribution.9be11df42356ca95.webp)

1. Crie um gráfico de dispersão:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Um gráfico de dispersão dos mesmos eixos mostra um padrão semelhante de convergência

    ![Facetgrid](../../../../translated_images/pt-PT/facetgrid.9b2e65ce707eba1f.webp)

Em geral, para clustering, pode-se usar gráficos de dispersão para mostrar clusters de dados, por isso dominar este tipo de visualização é muito útil. Na próxima lição, utilizaremos estes dados filtrados e usaremos clustering k-means para descobrir grupos neste conjunto de dados que parecem sobrepor-se de maneiras interessantes.

---

## 🚀Desafio

Em preparação para a próxima lição, faça um gráfico sobre os vários algoritmos de clustering que pode descobrir e usar num ambiente de produção. Que tipo de problemas é que o clustering procura resolver?

## [Quiz pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo

Antes de aplicar algoritmos de clustering, como aprendemos, é uma boa ideia entender a natureza do seu conjunto de dados. Leia mais sobre este tópico [aqui](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Este artigo útil](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) guia-o pelos diferentes comportamentos de vários algoritmos de clustering, dados diferentes formatos de dados.

## Trabalho de casa

[Pesquise outras visualizações para clustering](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento foi traduzido utilizando o serviço de tradução automática [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes da utilização desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->