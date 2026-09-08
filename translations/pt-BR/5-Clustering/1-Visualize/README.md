# Introdução ao clustering

Clustering é um tipo de [Aprendizado Não Supervisionado](https://wikipedia.org/wiki/Unsupervised_learning) que presume que um conjunto de dados não está rotulado ou que suas entradas não estão associadas a saídas predefinidas. Ele utiliza vários algoritmos para organizar dados não rotulados e fornecer agrupamentos de acordo com os padrões que identifica nos dados.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Clique na imagem acima para um vídeo. Enquanto você estuda aprendizado de máquina com clustering, aproveite algumas faixas de Dance Hall nigeriano - esta é uma música muito bem avaliada de 2014 do PSquare.

## [Questionário pré-palestra](https://ff-quizzes.netlify.app/en/ml/)

### Introdução

[Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) é muito útil para exploração de dados. Vamos ver se ele pode ajudar a descobrir tendências e padrões na forma como o público nigeriano consome música.

✅ Reserve um minuto para pensar sobre as utilizações do clustering. Na vida real, clustering acontece sempre que você tem uma pilha de roupas para lavar e precisa separar as roupas dos membros da sua família 🧦👕👖🩲. Em ciência de dados, clustering ocorre ao tentar analisar as preferências de um usuário, ou determinar as características de qualquer conjunto de dados não rotulado. Clustering, de certa forma, ajuda a dar sentido ao caos, como uma gaveta de meias.

[![Introdução ao ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introdução ao Clustering")

> 🎥 Clique na imagem acima para um vídeo: John Guttag do MIT apresenta clustering

Em um ambiente profissional, clustering pode ser usado para determinar coisas como segmentação de mercado, determinando que faixas etárias compram quais itens, por exemplo. Outro uso seria a detecção de anomalias, talvez para identificar fraudes em um conjunto de dados de transações de cartão de crédito. Ou você pode usar clustering para identificar tumores em um conjunto de exames médicos.

✅ Pense um minuto em como você pode ter encontrado clustering 'no mundo real', em um ambiente bancário, de comércio eletrônico ou de negócios.

> 🎓 Curiosamente, a análise de clusters originou-se nos campos da Antropologia e Psicologia na década de 1930. Você consegue imaginar como ela poderia ter sido usada?

Alternativamente, você poderia usá-lo para agrupar resultados de busca - por links de compras, imagens ou avaliações, por exemplo. Clustering é útil quando você tem um grande conjunto de dados que quer reduzir e sobre o qual deseja realizar uma análise mais detalhada, assim a técnica pode ser usada para conhecer os dados antes da construção de outros modelos.

✅ Uma vez que seus dados estejam organizados em clusters, você os atribui um Id do cluster, e esta técnica pode ser útil na preservação da privacidade de um conjunto de dados; você pode, ao invés disso, referir-se a um ponto de dados pelo seu Id do cluster, em vez de por dados mais reveladores e identificáveis. Você consegue pensar em outras razões pelas quais você se referiria a um Id do cluster ao invés de outros elementos do cluster para identificá-lo?

Aprofunde seu entendimento sobre técnicas de clustering neste [módulo Learn](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)
## Começando com clustering

[Scikit-learn oferece uma grande variedade](https://scikit-learn.org/stable/modules/clustering.html) de métodos para realizar clustering. O tipo que você escolher dependerá do seu caso de uso. De acordo com a documentação, cada método possui diversos benefícios. Aqui está uma tabela simplificada dos métodos suportados pelo Scikit-learn e seus casos de uso apropriados:

| Nome do método              | Caso de uso                                                            |
| :------------------------- | :--------------------------------------------------------------------- |
| K-Means                    | uso geral, indutivo                                                    |
| Affinity propagation       | muitos clusters desiguais, indutivo                                   |
| Mean-shift                 | muitos clusters desiguais, indutivo                                   |
| Spectral clustering        | poucos clusters uniformes, transdutivo                                |
| Ward hierarchical clustering| muitos clusters restritos, transdutivo                               |
| Agglomerative clustering   | muitos, restritos, distâncias não euclidianas, transdutivo            |
| DBSCAN                     | geometria não plana, clusters desiguais, transdutivo                  |
| OPTICS                     | geometria não plana, clusters desiguais com densidade variável, transdutivo |
| Gaussian mixtures          | geometria plana, indutivo                                             |
| BIRCH                      | grande conjunto de dados com outliers, indutivo                       |

> 🎓 Como criamos clusters está muito relacionado a como agrupamos os pontos de dados em grupos. Vamos desvendar um pouco do vocabulário:
>
> 🎓 ['Transdutivo' vs. 'indutivo'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Inferência transdutiva é derivada de casos de treinamento observados que são mapeados para casos de teste específicos. Inferência indutiva é derivada de casos de treinamento que são mapeados para regras gerais que só então são aplicadas a casos de teste.
> 
> Um exemplo: Imagine que você tem um conjunto de dados parcialmente rotulado. Algumas coisas são 'discos', algumas 'cds' e algumas estão em branco. Sua tarefa é fornecer etiquetas para os campos em branco. Se você optar por uma abordagem indutiva, treinaria um modelo procurando por 'discos' e 'cds', e aplicaria essas etiquetas aos dados não rotulados. Essa abordagem terá dificuldade em classificar coisas que na verdade são 'fitas cassete'. Uma abordagem transdutiva, por outro lado, lida com esses dados desconhecidos de forma mais eficaz, pois trabalha agrupando itens similares juntos e depois aplica uma etiqueta ao grupo. Nesse caso, os clusters poderiam refletir 'coisas musicais redondas' e 'coisas musicais quadradas'.
> 
> 🎓 ['Geometria não plana' vs. 'plana'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Derivado da terminologia matemática, geometria não plana vs. plana refere-se à medida de distâncias entre pontos por métodos geométricos 'plano' ([Euclidiana](https://wikipedia.org/wiki/Euclidean_geometry)) ou 'não plano' (não euclidiana).
>
>'Plano' neste contexto refere-se à geometria Euclidiana (partes da qual são ensinadas como geometria 'plana'), e não plano refere-se à geometria não Euclidiana. O que geometria tem a ver com aprendizado de máquina? Bem, como dois campos que têm raízes na matemática, deve haver uma forma comum de medir distâncias entre pontos em clusters, e isso pode ser feito de forma 'plana' ou 'não plana', dependendo da natureza dos dados. [Distâncias Euclidianas](https://wikipedia.org/wiki/Euclidean_distance) são medidas como o comprimento de um segmento de linha entre dois pontos. [Distâncias não Euclidianas](https://wikipedia.org/wiki/Non-Euclidean_geometry) são medidas ao longo de uma curva. Se seus dados, visualizados, parecem não existir em um plano, talvez seja necessário usar um algoritmo especializado para lidar com eles.
>
![Infográfico Geometria Plana vs Não Plana](../../../../translated_images/pt-BR/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infográfico por [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Distâncias'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Clusters são definidos pela sua matriz de distâncias, ou seja, as distâncias entre pontos. Essa distância pode ser medida de algumas maneiras. Clusters Euclidianos são definidos pela média dos valores dos pontos e contêm um 'centróide' ou ponto central. As distâncias são então medidas pela distância até esse centróide. Distâncias não Euclidianas referem-se a 'clustroides', o ponto mais próximo de outros pontos. Clustroides, por sua vez, podem ser definidos de diversas maneiras.
> 
> 🎓 ['Restringido'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Clustering restrito](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introduz aprendizado 'semi-supervisionado' neste método não supervisionado. As relações entre pontos são sinalizadas como 'não podem ser vinculadas' ou 'devem ser vinculadas', de modo que algumas regras são impostas ao conjunto de dados.
>
>Um exemplo: se um algoritmo é liberado para um lote de dados não rotulados ou semi-rotulados, os clusters que produz podem ser de qualidade ruim. No exemplo acima, os clusters poderiam agrupar 'coisas musicais redondas' e 'coisas musicais quadradas', 'coisas triangulares' e 'biscoitos'. Se fornecidas algumas restrições ou regras a seguir ("o item deve ser feito de plástico", "o item precisa ser capaz de produzir música") isso pode ajudar a 'constranger' o algoritmo a fazer escolhas melhores.
> 
> 🎓 'Densidade'
> 
> Dados que são 'ruidosos' são considerados 'densos'. As distâncias entre pontos em cada um de seus clusters podem provar, ao serem examinadas, ser mais ou menos densas, ou 'aglomeradas' e, assim, esses dados precisam ser analisados com o método de clustering apropriado. [Este artigo](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) demonstra a diferença entre usar K-Means clustering versus os algoritmos HDBSCAN para explorar um conjunto de dados ruidoso com densidade desigual dos clusters.

## Algoritmos de clustering

Existem mais de 100 algoritmos de clustering, e seu uso depende da natureza dos dados em questão. Vamos discutir alguns dos principais:

- **Clustering hierárquico**. Se um objeto é classificado pela sua proximidade a outro objeto próximo, em vez de um mais distante, os clusters são formados com base na distância de seus membros para e de outros objetos. O clustering aglomerativo do Scikit-learn é hierárquico.

   ![Infográfico Clustering Hierárquico](../../../../translated_images/pt-BR/hierarchical.bf59403aa43c8c47.webp)
   > Infográfico por [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering de centróide**. Esse algoritmo popular requer a escolha de 'k', ou o número de clusters a serem formados, após o qual o algoritmo determina o ponto central de um cluster e agrupa dados em torno desse ponto. [K-means clustering](https://wikipedia.org/wiki/K-means_clustering) é uma versão popular de clustering de centróide. O centro é determinado pela média mais próxima, daí o nome. A distância quadrática do cluster é minimizada.

   ![Infográfico Clustering de Centróide](../../../../translated_images/pt-BR/centroid.097fde836cf6c918.webp)
   > Infográfico por [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering baseado em distribuição**. Baseado em modelagem estatística, clustering baseado em distribuição centra-se em determinar a probabilidade de um ponto de dados pertencer a um cluster e atribuí-lo de acordo. Métodos de mistura Gaussiana pertencem a este tipo.

- **Clustering baseado em densidade**. Pontos de dados são atribuídos a clusters com base na densidade, ou seja, seu agrupamento uns aos outros. Pontos de dados longe do grupo são considerados outliers ou ruído. DBSCAN, Mean-shift e OPTICS pertencem a esse tipo de clustering.

- **Clustering baseado em grade**. Para conjuntos de dados multidimensionais, uma grade é criada e os dados são divididos entre as células da grade, criando clusters.

## Exercício - agrupe seus dados

Clustering como técnica é muito beneficiado por uma boa visualização, então vamos começar visualizando nossos dados musicais. Este exercício nos ajudará a decidir qual dos métodos de clustering devemos usar mais efetivamente para a natureza destes dados.

1. Abra o arquivo [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) nesta pasta.

1. Importe o pacote `Seaborn` para uma boa visualização de dados.

    ```python
    !pip install seaborn
    ```

1. Anexe os dados das músicas do arquivo [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Carregue um dataframe com alguns dados sobre as músicas. Prepare-se para explorar esses dados importando as bibliotecas e exibindo os dados:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Confira as primeiras linhas dos dados:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Obtenha algumas informações sobre o dataframe, chamando `info()`:

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

1. Verifique novamente se há valores nulos, chamando `isnull()` e verificando se a soma é 0:

    ```python
    df.isnull().sum()
    ```

    Tudo parece bom:

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

> 🤔 Se estamos trabalhando com clustering, um método não supervisionado que não requer dados rotulados, por que estamos mostrando esses dados com rótulos? Na fase de exploração dos dados, eles são úteis, mas não são necessários para o funcionamento dos algoritmos de clustering. Você poderia muito bem remover os cabeçalhos das colunas e se referir aos dados pelo número da coluna.

Observe os valores gerais dos dados. Note que a popularidade pode ser '0', o que indica músicas que não têm classificação. Vamos removê-las em breve.

1. Use um gráfico de barras para descobrir os gêneros mais populares:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![mais populares](../../../../translated_images/pt-BR/popular.9c48d84b3386705f.webp)

✅ Se você quiser ver mais valores principais, mude o top `[:5]` para um valor maior, ou remova-o para ver todos.

Observe que, quando o gênero principal é descrito como 'Missing', isso significa que o Spotify não o classificou, então vamos nos livrar dele.

1. Remova os dados ausentes filtrando-os

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Agora verifique novamente os gêneros:

    ![mais populares](../../../../translated_images/pt-BR/all-genres.1d56ef06cefbfcd6.webp)

1. De longe, os três principais gêneros dominam este conjunto de dados. Vamos nos concentrar em `afro dancehall`, `afropop` e `nigerian pop`, além disso, filtre o conjunto de dados para remover qualquer coisa com valor de popularidade 0 (significando que não foi classificada com popularidade no conjunto de dados e pode ser considerada ruído para nossos propósitos):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Faça um teste rápido para ver se os dados se correlacionam de alguma maneira particularmente forte:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlações](../../../../translated_images/pt-BR/correlation.a9356bb798f5eea5.webp)

    A única correlação forte é entre `energy` e `loudness`, o que não é surpreendente, dado que música alta costuma ser bem energética. Caso contrário, as correlações são relativamente fracas. Será interessante ver o que um algoritmo de clustering pode fazer com esses dados.

    > 🎓 Note que correlação não implica causalidade! Temos prova de correlação, mas nenhuma prova de causalidade. Um [site divertido](https://tylervigen.com/spurious-correlations) tem alguns visuais que enfatizam esse ponto.

Há alguma convergência neste conjunto de dados entre a popularidade percebida de uma música e sua dançabilidade? Um FacetGrid mostra que existem círculos concêntricos que se alinham, independentemente do gênero. Pode ser que gostos nigerianos convirjam para um certo nível de dançabilidade para esse gênero?

✅ Experimente diferentes pontos de dados (energia, volume, fala) e mais ou diferentes gêneros musicais. O que você pode descobrir? Dê uma olhada na tabela `df.describe()` para ver a dispersão geral dos pontos de dados.

### Exercício - distribuição dos dados

Esses três gêneros são significativamente diferentes na percepção de sua dançabilidade, com base em sua popularidade?

1. Examine a distribuição dos dados dos nossos três gêneros principais para popularidade e dançabilidade ao longo de um eixo x e y fornecido.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Você pode descobrir círculos concêntricos ao redor de um ponto geral de convergência, mostrando a distribuição dos pontos.

    > 🎓 Note que este exemplo usa um gráfico KDE (Estimativa de Densidade de Kernel) que representa os dados usando uma curva contínua de densidade de probabilidade. Isso nos permite interpretar os dados ao trabalhar com múltiplas distribuições.

    Em geral, os três gêneros se alinham vagamente em termos de popularidade e dançabilidade. Determinar clusters nesses dados vagamente alinhados será um desafio:

    ![distribuição](../../../../translated_images/pt-BR/distribution.9be11df42356ca95.webp)

1. Crie um gráfico de dispersão:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Um gráfico de dispersão dos mesmos eixos mostra um padrão semelhante de convergência

    ![Facetgrid](../../../../translated_images/pt-BR/facetgrid.9b2e65ce707eba1f.webp)

Em geral, para clustering, você pode usar gráficos de dispersão para mostrar clusters de dados, por isso dominar esse tipo de visualização é muito útil. Na próxima lição, vamos pegar esses dados filtrados e usar o clustering k-means para descobrir grupos nesses dados que parecem se sobrepor de formas interessantes.

---

## 🚀Desafio

Em preparação para a próxima lição, faça um gráfico sobre os vários algoritmos de clustering que você possa descobrir e usar em um ambiente de produção. Que tipos de problemas o clustering tenta resolver?

## [Quiz pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo

Antes de aplicar algoritmos de clustering, como aprendemos, é uma boa ideia entender a natureza do seu conjunto de dados. Leia mais sobre esse assunto [aqui](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Este artigo útil](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) orienta você sobre as diferentes formas como vários algoritmos de clustering se comportam, dados diferentes formatos de dados.

## Tarefa

[Pesquise outras visualizações para clustering](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento foi traduzido usando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, por favor, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes do uso desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->