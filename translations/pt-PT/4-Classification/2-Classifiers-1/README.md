# Classificadores de cozinha 1

Nesta lição, irá usar o conjunto de dados que guardou na lição anterior, cheio de dados equilibrados e limpos, todos sobre cozinhas.

Usará este conjunto de dados com vários classificadores para _prever uma cozinha nacional dada com base num grupo de ingredientes_. Enquanto o faz, irá aprender mais sobre algumas das formas como os algoritmos podem ser aproveitados para tarefas de classificação.

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)
# Preparação

Assumindo que completou a [Lição 1](../1-Introduction/README.md), certifique-se de que existe um ficheiro _cleaned_cuisines.csv_ na raiz da pasta `/data` para estas quatro lições.

## Exercício - prever uma cozinha nacional

1. Trabalhando no _notebook.ipynb_ desta lição, importe esse ficheiro juntamente com a biblioteca Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Os dados parecem-se com isto:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Agora, importe várias outras bibliotecas:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Divida as coordenadas X e y em dois dataframes para treino. `cuisine` pode ser o dataframe das etiquetas:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Vai parecer isto:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Elimine a coluna `Unnamed: 0` e a coluna `cuisine`, chamando `drop()`. Guarde o resto dos dados como características para treino:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    As suas características ficam assim:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Agora está pronto para treinar o seu modelo!

## Escolher o seu classificador

Agora que os seus dados estão limpos e prontos para treino, tem que decidir qual algoritmo usar para o trabalho.

Scikit-learn agrupa classificação sob Aprendizagem Supervisionada, e nessa categoria encontrará muitas formas de classificar. [A variedade](https://scikit-learn.org/stable/supervised_learning.html) é bastante confusa à primeira vista. Os seguintes métodos incluem todos técnicas de classificação:

- Modelos Lineares
- Máquinas de Vetores de Suporte
- Descenso Estocástico do Gradiente
- Vizinhos Mais Próximos
- Processos Gaussianos
- Árvores de Decisão
- Métodos de ensemble (classificador por votação)
- Algoritmos multi-classe e multi-output (classificação multi-classe e multi-rótulo, classificação multi-classe multi-output)

> Também pode usar [redes neurais para classificar dados](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), mas isso está fora do escopo desta lição.

### Que classificador escolher?

Então, qual classificador deve escolher? Muitas vezes, experimentar vários e procurar obter um bom resultado é uma forma de testar. Scikit-learn oferece uma [comparação lado a lado](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) num conjunto de dados criado, comparando KNeighbors, SVC de duas formas, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB e QuadraticDiscrinationAnalysis, mostrando os resultados visualizados:

![comparação de classificadores](../../../../translated_images/pt-PT/comparison.edfab56193a85e7f.webp)
> Gráficos gerados na documentação da Scikit-learn

> AutoML resolve este problema de forma elegante executando essas comparações na cloud, permitindo-lhe escolher o melhor algoritmo para os seus dados. Experimente [aqui](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Uma abordagem melhor

Uma forma melhor do que adivinhar ao acaso é seguir as ideias neste [folheto de consulta rápida de ML](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) descarregável. Aqui, descobrimos que, para o nosso problema multiclass, temos algumas opções:

![folheto para problemas multiclass](../../../../translated_images/pt-PT/cheatsheet.07a475ea444d2223.webp)
> Uma secção do Folheto de Algoritmos da Microsoft, detalhando opções de classificação multiclass

✅ Descarregue este folheto, imprima-o e pendure-o na sua parede!

### Raciocínio

Vamos ver se conseguimos raciocinar diferentes abordagens dadas as restrições que temos:

- **Redes neurais são demasiado pesadas**. Dado o nosso conjunto de dados limpo, mas minimalista, e o facto de que estamos a executar o treino localmente via notebooks, as redes neurais são demasiado pesadas para esta tarefa.
- **Nenhum classificador de duas classes**. Não usamos um classificador de duas classes, pelo que isto elimina one-vs-all.
- **Árvore de decisão ou regressão logística podem funcionar**. Uma árvore de decisão pode funcionar, ou regressão logística para dados multiclass.
- **Árvores de decisão boosteadas multiclass resolvem um problema diferente**. A árvore de decisão boosteada multiclass é mais adequada para tarefas não-paramétricas, por exemplo, tarefas desenhadas para construir rankings, por isso não é útil para nós.

### Usar Scikit-learn

Vamos usar Scikit-learn para analisar os nossos dados. No entanto, há muitas formas de usar regressão logística no Scikit-learn. Veja os [parâmetros para passar](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Essencialmente há dois parâmetros importantes - `multi_class` e `solver` - que precisamos de especificar, quando pedimos ao Scikit-learn para executar uma regressão logística. O valor `multi_class` aplica um certo comportamento. O valor do solver é qual algoritmo usar. Nem todos os solvers podem ser combinados com todos os valores `multi_class`.

De acordo com a documentação, no caso multiclass, o algoritmo de treino:

- **Usa o esquema one-vs-rest (OvR)**, se a opção `multi_class` estiver definida como `ovr`
- **Usa a perda de cross-entropy**, se a opção `multi_class` estiver definida como `multinomial`. (Atualmente a opção `multinomial` só é suportada pelos solvers ‘lbfgs’, ‘sag’, ‘saga’ e ‘newton-cg’)."

> 🎓 O 'esquema' aqui pode ser 'ovr' (one-vs-rest) ou 'multinomial'. Como a regressão logística é realmente desenhada para suportar classificação binária, estes esquemas permitem-lhe lidar melhor com tarefas de classificação multiclass. [fonte](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 O 'solver' é definido como "o algoritmo a usar no problema de otimização". [fonte](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn oferece esta tabela para explicar como os solvers lidam com diferentes desafios apresentados por diferentes tipos de estruturas de dados:

![solvers](../../../../translated_images/pt-PT/solvers.5fc648618529e627.webp)

## Exercício - dividir os dados

Podemos focar na regressão logística para o nosso primeiro ensaio de treino, visto que aprendeu recentemente sobre esta numa lição anterior.  
Divida os seus dados em grupos de treino e teste chamando `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Exercício - aplicar regressão logística

Como está a usar o caso multiclass, precisa de escolher que _esquema_ usar e que _solver_ definir. Use LogisticRegression com uma configuração multiclass e o solver **liblinear** para treinar.

1. Crie uma regressão logística com multi_class definido para `ovr` e o solver definido para `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Experimente um solver diferente como `lbfgs`, que muitas vezes é o padrão

    > Nota, use a função [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) do Pandas para achatar os seus dados quando necessário.

    A precisão é boa, acima dos **80%**!

1. Pode ver este modelo em ação testando uma linha de dados (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    O resultado é impresso:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Experimente um número de linha diferente e verifique os resultados
1. Aprofundando, pode verificar a precisão desta previsão:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    O resultado é apresentado - culinária indiana é a sua melhor aposta, com boa probabilidade:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Consegue explicar por que o modelo está bastante certo de que esta é uma culinária indiana?

1. Obtenha mais detalhes imprimindo um relatório de classificação, como fez nas lições de regressão:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | chinese      | 0.73      | 0.71   | 0.72     | 229     |
    | indian       | 0.91      | 0.93   | 0.92     | 254     |
    | japanese     | 0.70      | 0.75   | 0.72     | 220     |
    | korean       | 0.86      | 0.76   | 0.81     | 242     |
    | thai         | 0.79      | 0.85   | 0.82     | 254     |
    | accuracy     |           |        | 0.80     | 1199    |
    | macro avg    | 0.80      | 0.80   | 0.80     | 1199    |
    | weighted avg | 0.80      | 0.80   | 0.80     | 1199    |

## 🚀Desafio

Nesta lição, utilizou os seus dados limpos para construir um modelo de machine learning que pode prever uma culinária nacional com base numa série de ingredientes. Tire algum tempo para ler as muitas opções que o Scikit-learn oferece para classificar dados. Aprofunde-se no conceito de 'solver' para entender o que se passa nos bastidores.

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo

Aprofunde um pouco mais na matemática por trás da regressão logística nesta [lição](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Tarefa

[Estudar os solvers](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, por favor tenha em conta que traduções automáticas podem conter erros ou imprecisões. O documento original no seu idioma nativo deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações erradas decorrentes da utilização desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->