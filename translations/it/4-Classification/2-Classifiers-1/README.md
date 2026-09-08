# Classificatori di cucina 1

In questa lezione, utilizzerai il dataset che hai salvato dalla lezione precedente pieno di dati bilanciati e puliti, tutti riguardanti cucine.

Utilizzerai questo dataset con una varietà di classificatori per _predire una cucina nazionale data un gruppo di ingredienti_. Nel farlo, imparerai di più su alcuni dei modi in cui gli algoritmi possono essere sfruttati per compiti di classificazione.

## [Quiz pre-lezione](https://ff-quizzes.netlify.app/en/ml/)
# Preparazione

Presumendo che tu abbia completato la [Lezione 1](../1-Introduction/README.md), assicurati che un file _cleaned_cuisines.csv_ esista nella cartella radice `/data` per queste quattro lezioni.

## Esercizio - predire una cucina nazionale

1. Lavorando nella cartella _notebook.ipynb_ di questa lezione, importa quel file insieme alla libreria Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    I dati appaiono così:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Ora, importa altre librerie:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Dividi le coordinate X e y in due data frame per l’addestramento. `cuisine` può essere il data frame delle etichette:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Apparirà così:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Elimina la colonna `Unnamed: 0` e la colonna `cuisine`, chiamando `drop()`. Salva il resto dei dati come caratteristiche su cui addestrare:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Le tue caratteristiche appariranno così:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Ora sei pronto per addestrare il modello!

## Scelta del classificatore

Ora che i tuoi dati sono puliti e pronti per l’addestramento, devi decidere quale algoritmo utilizzare per il lavoro.

Scikit-learn raggruppa la classificazione sotto l’apprendimento supervisionato, e in questa categoria troverai molti modi per classificare. [La varietà](https://scikit-learn.org/stable/supervised_learning.html) è piuttosto confusa a prima vista. I seguenti metodi includono tutte tecniche di classificazione:

- Modelli lineari
- Support Vector Machines
- Discesa del gradiente stocastica
- Vicini più prossimi
- Processi Gaussiani
- Alberi decisionali
- Metodi ensemble (voting Classifier)
- Algoritmi multiclass e multioutput (classificazione multiclass e multilabel, classificazione multiclass-multioutput)

> Puoi anche utilizzare [reti neurali per classificare i dati](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), ma questo è fuori dallo scopo di questa lezione.

### Quale classificatore scegliere?

Allora, quale classificatore dovresti scegliere? Spesso, provare diversi modelli e cercare un buon risultato è un modo per testare. Scikit-learn offre un [confronto affiancato](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) su un dataset creato, confrontando KNeighbors, SVC in due modi, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB e QuadraticDiscrinationAnalysis, mostrando i risultati visualizzati: 

![confronto di classificatori](../../../../translated_images/it/comparison.edfab56193a85e7f.webp)
> Grafici generati dalla documentazione di Scikit-learn

> AutoML risolve questo problema in modo ordinato eseguendo questi confronti nel cloud, permettendoti di scegliere il miglior algoritmo per i tuoi dati. Provalo [qui](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Un approccio migliore

Un modo migliore rispetto a un’improvvisazione completa, però, è seguire le idee su questa [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) scaricabile. Qui scopriamo che, per il nostro problema multiclass, abbiamo alcune scelte:

![cheatsheet per problemi multiclass](../../../../translated_images/it/cheatsheet.07a475ea444d2223.webp)
> Un estratto dell’Algorithm Cheat Sheet di Microsoft, che dettaglia le opzioni per la classificazione multiclass

✅ Scarica questa cheat sheet, stampala e appendila al muro!

### Ragionamento

Vediamo se possiamo ragionare su diversi approcci dati i vincoli che abbiamo:

- **Le reti neurali sono troppo pesanti**. Dato il nostro dataset pulito ma minimale, e il fatto che stiamo eseguendo l’addestramento localmente tramite notebook, le reti neurali sono troppo pesanti per questo compito.
- **Nessun classificatore a due classi**. Non usiamo un classificatore a due classi, quindi scartiamo one-vs-all.
- **Un albero decisionale o regressione logistica potrebbero funzionare**. Un albero decisionale potrebbe funzionare, o la regressione logistica per dati multiclass.
- **Gli alberi decisionali potenziati multiclass risolvono un problema diverso**. L’albero decisionale potenziato multiclass è più adatto per compiti non parametrici, ad esempio compiti progettati per costruire classifiche, quindi non è utile per noi.

### Uso di Scikit-learn 

Useremo Scikit-learn per analizzare i nostri dati. Tuttavia, ci sono molti modi di usare la regressione logistica in Scikit-learn. Dai un’occhiata ai [parametri da passare](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Fondamentalmente ci sono due parametri importanti - `multi_class` e `solver` - che dobbiamo specificare quando richiediamo a Scikit-learn di eseguire una regressione logistica. Il valore `multi_class` applica un certo comportamento. Il valore `solver` indica quale algoritmo usare. Non tutti i solver possono essere abbinati a tutti i valori di `multi_class`.

Secondo la documentazione, nel caso multiclass, l’algoritmo di addestramento:

- **Usa lo schema one-vs-rest (OvR)** se l’opzione `multi_class` è impostata su `ovr`
- **Usa la perdita di cross-entropy** se l’opzione `multi_class` è impostata su `multinomial`. (Attualmente l’opzione `multinomial` è supportata solo dai solver ‘lbfgs’, ‘sag’, ‘saga’ e ‘newton-cg’)."

> 🎓 Lo 'schema' qui può essere 'ovr' (one-vs-rest) o 'multinomial'. Poiché la regressione logistica è realmente progettata per supportare la classificazione binaria, questi schemi le permettono di gestire meglio compiti di classificazione multiclass. [fonte](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 Il 'solver' è definito come "l’algoritmo da usare nel problema di ottimizzazione". [fonte](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn offre questa tabella per spiegare come i solver gestiscono diverse sfide presentate da diversi tipi di strutture dati:

![solver](../../../../translated_images/it/solvers.5fc648618529e627.webp)

## Esercizio - dividere i dati

Possiamo concentrarci sulla regressione logistica per la nostra prima prova di addestramento dato che l’hai imparata recentemente in una lezione precedente.
Dividi i dati in gruppi di addestramento e test chiamando `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Esercizio - applicare regressione logistica

Dato che stai usando il caso multiclass, devi scegliere quale _schema_ usare e quale _solver_ impostare. Usa LogisticRegression con un’impostazione multiclass e il solver **liblinear** per addestrare.

1. Crea una regressione logistica con `multi_class` impostato su `ovr` e il solver impostato su `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Prova un solver diverso come `lbfgs`, che spesso è impostato come default

    > Nota, usa la funzione Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) per appiattire i tuoi dati quando necessario.

    L’accuratezza è buona, oltre l’**80%**!

1. Puoi vedere questo modello in azione testando una riga di dati (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Il risultato è stampato:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Prova un numero di riga diverso e verifica i risultati
1. Approfondendo, puoi verificare l’accuratezza di questa previsione:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Il risultato viene stampato - la cucina indiana è la sua ipotesi migliore, con buona probabilità:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Puoi spiegare perché il modello è abbastanza sicuro che questa sia una cucina indiana?

1. Ottieni più dettagli stampando un report di classificazione, come hai fatto nelle lezioni sulla regressione:

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

## 🚀Sfida

In questa lezione, hai utilizzato i tuoi dati puliti per costruire un modello di machine learning che può prevedere una cucina nazionale basata su una serie di ingredienti. Prenditi del tempo per leggere le molte opzioni che Scikit-learn fornisce per classificare i dati. Approfondisci il concetto di 'solver' per capire cosa succede dietro le quinte.

## [Quiz post-lezione](https://ff-quizzes.netlify.app/en/ml/)

## Revisione e autoapprendimento

Approfondisci un po’ di più la matematica dietro la regressione logistica in [questa lezione](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Compito

[Studia i solver](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione AI [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o inesattezze. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale umana. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->