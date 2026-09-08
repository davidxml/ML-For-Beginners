# Clasificatori de bucătărie 1

În această lecție, veți folosi setul de date pe care l-ați salvat din lecția anterioară, plin de date echilibrate și curate despre bucătării.

Veți folosi acest set de date cu o varietate de clasificatori pentru a _prezice o anumită bucătărie națională pe baza unui grup de ingrediente_. În timp ce faceți acest lucru, veți afla mai multe despre unele dintre modurile în care algoritmii pot fi utilizați pentru sarcini de clasificare.

## [Test preliminar înaintea lecției](https://ff-quizzes.netlify.app/en/ml/)
# Pregătire

Presupunând că ați terminat [Lecția 1](../1-Introduction/README.md), asigurați-vă că un fișier _cleaned_cuisines.csv_ există în folderul rădăcină `/data` pentru aceste patru lecții.

## Exercițiu - prezice o bucătărie națională

1. Lucrând în folderul _notebook.ipynb_ al acestei lecții, importați acel fișier împreună cu biblioteca Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Datele arată astfel:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Acum, importați câteva biblioteci suplimentare:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Împărțiți coordonatele X și y în două dataframes pentru antrenare. `cuisine` poate fi dataframe-ul etichetelor:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Va arăta astfel:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Eliminați coloana `Unnamed: 0` și coloana `cuisine` folosind `drop()`. Salvați restul datelor ca caracteristici de antrenament:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Caracteristicile dvs. arată astfel:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Acum sunteți gata să vă antrenați modelul!

## Alegerea clasificatorului

Acum că datele dvs. sunt curate și gata pentru antrenament, trebuie să decideți ce algoritm să folosiți pentru sarcină.

Scikit-learn grupează clasificarea sub Învățare supravegheată, iar în această categorie veți găsi multe moduri de a clasifica. [Varietatea](https://scikit-learn.org/stable/supervised_learning.html) este destul de copleșitoare la prima vedere. Următoarele metode includ toate tehnici de clasificare:

- Modele liniare
- Mașini cu vectori de susținere
- Descendentă stocastică a gradientului
- Cei mai apropiați vecini
- Procese Gaussiane
- Arbori decizionali
- Metode de ansamblu (voting Classifier)
- Algoritmi multiclasă și multioutput (clasificare multiclasă și multilabel, clasificare multiclasă-multioutput)

> Puteți folosi și [rețele neuronale pentru a clasifica date](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), dar asta este în afara scopului acestei lecții.

### Ce clasificator să alegeți?

Deci, ce clasificator ar trebui să alegeți? Deseori, rularea mai multora și căutarea unui rezultat bun este o metodă de testare. Scikit-learn oferă o [comparație alăturată](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) pe un set de date creat, comparând KNeighbors, SVC în două moduri, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB și QuadraticDiscrinationAnalysis, arătând rezultatele vizualizate:

![comparison of classifiers](../../../../translated_images/ro/comparison.edfab56193a85e7f.webp)
> Grafice generate pe documentația Scikit-learn

> AutoML rezolvă această problemă elegant rulând aceste comparații în cloud, permițându-vă să alegeți cel mai bun algoritm pentru datele dvs. Încercați-l [aici](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### O abordare mai bună

O metodă mai bună decât să ghiciți la întâmplare este să urmați ideile din acest [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) descărcabil. Aici descoperim că, pentru problema noastră multiclasă, avem câteva opțiuni:

![cheatsheet for multiclass problems](../../../../translated_images/ro/cheatsheet.07a475ea444d2223.webp)
> O secțiune a Algorithm Cheat Sheet de la Microsoft, detaliind opțiunile pentru clasificare multiclasă

✅ Descărcați această fișă de trucuri, tipăriți-o și puneți-o pe perete!

### Raționament

Să vedem dacă putem raționa prin diferite abordări date constrângerile pe care le avem:

- **Rețelele neuronale sunt prea grele**. Având în vedere setul nostru de date curat, dar minimal, și faptul că efectuăm antrenamentul local în notebook-uri, rețelele neuronale sunt prea grele pentru această sarcină.
- **Niciun clasificator cu două clase**. Nu folosim un clasificator cu două clase, deci eliminăm metoda one-vs-all.
- **Arborele decizional sau regresia logistică ar putea funcționa**. Un arbore decizional ar putea funcționa sau regresia logistică pentru date multiclasă.
- **Arborii decizionali multiclasă cu boost rezolvă o problemă diferită**. Arborii decizionali multiclasă cu boost sunt cei mai potriviți pentru sarcini neparametrice, de exemplu, sarcini de construire a unor ranking-uri, deci nu sunt utile pentru noi.

### Folosirea Scikit-learn

Vom folosi Scikit-learn pentru a analiza datele noastre. Totuși, există multe moduri de a folosi regresia logistică în Scikit-learn. Consultați [parametrii care trebuie specificați](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

În esență, există doi parametri importanți - `multi_class` și `solver` - pe care trebuie să îi specificăm când cerem Scikit-learn să execute o regresie logistică. Valoarea lui `multi_class` aplică un anumit comportament. Valoarea lui `solver` este algoritmul care va fi folosit. Nu toți solverii pot fi combinați cu toate valorile `multi_class`.

Conform documentației, în cazul multiclasă, algoritmul de antrenament:

- **Folosește schema one-vs-rest (OvR)**, dacă opțiunea `multi_class` este setată pe `ovr`
- **Folosește pierderea de tip cross-entropy**, dacă opțiunea `multi_class` este setată pe `multinomial`. (În prezent, opțiunea `multinomial` este susținută doar de solverii ‘lbfgs’, ‘sag’, ‘saga’ și ‘newton-cg’)."

> 🎓 'Schema' aici poate fi fie 'ovr' (one-vs-rest), fie 'multinomial'. Deoarece regresia logistică este concepută în principal pentru clasificare binară, aceste scheme îi permit să gestioneze mai bine sarcini de clasificare multiclasă. [sursă](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver-ul' este definit ca "algoritmul folosit în problema de optimizare". [sursă](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn oferă acest tabel pentru a explica cum solverii gestionează diferite provocări prezentate de diverse structuri de date:

![solvers](../../../../translated_images/ro/solvers.5fc648618529e627.webp)

## Exercițiu - împarte datele

Putem să ne concentrăm pe regresia logistică pentru prima noastră încercare de antrenament, deoarece tocmai ați învățat despre ea într-o lecție anterioară.

Împărțiți datele în grupuri de antrenament și testare folosind `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Exercițiu - aplică regresia logistică

Deoarece folosiți cazul multiclasă, trebuie să alegeți ce _schemă_ să folosiți și ce _solver_ să setați. Folosiți LogisticRegression cu setarea multiclasă și solverul **liblinear** pentru antrenament.

1. Creați o regresie logistică cu multi_class setat pe `ovr` și solver pe `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Încercați un solver diferit precum `lbfgs`, care este adesea setat implicit

    > Notă, folosiți funcția Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) pentru a aplatiza datele când este nevoie.

    Acuratețea este bună, peste **80%**!

1. Puteți vedea acest model în acțiune testând un rând de date (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Rezultatul este afișat:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Încercați un număr diferit de rând și verificați rezultatele
1. Explorând mai în profunzime, poți verifica acuratețea acestei predicții:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Rezultatul este afișat - bucătăria indiană este cea mai probabilă variantă, cu o probabilitate bună:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Poți explica de ce modelul este destul de sigur că aceasta este o bucătărie indiană?

1. Obține mai multe detalii prin imprimarea unui raport de clasificare, așa cum ai făcut în lecțiile despre regresie:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precizie | recall | f1-score | suport |
    | ------------ | -------- | ------ | -------- | ------- |
    | chinese      | 0.73     | 0.71   | 0.72     | 229     |
    | indian       | 0.91     | 0.93   | 0.92     | 254     |
    | japanese     | 0.70     | 0.75   | 0.72     | 220     |
    | korean       | 0.86     | 0.76   | 0.81     | 242     |
    | thai         | 0.79     | 0.85   | 0.82     | 254     |
    | accuracy     |          |        | 0.80     | 1199    |
    | macro avg    | 0.80     | 0.80   | 0.80     | 1199    |
    | weighted avg | 0.80     | 0.80   | 0.80     | 1199    |

## 🚀Provocare

În această lecție, ai folosit datele curate pentru a construi un model de învățare automată care poate prezice o bucătărie națională pe baza unui set de ingrediente. Ia-ți timp să parcurgi multiplele opțiuni pe care Scikit-learn le oferă pentru clasificarea datelor. Aprofundează conceptul de 'solver' pentru a înțelege ce se întâmplă în culise.

## [Chestionar post-lectură](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare & Studiu individual

Explorează mai mult matematica din spatele regresiei logistice în [această lecție](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Temă

[Studiază solvers](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Declinare a responsabilității**:  
Acest document a fost tradus utilizând serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim pentru acuratețe, vă rugăm să rețineți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa nativă trebuie considerat sursa autorizată. Pentru informații critice, se recomandă traducerea profesională realizată de un traducător uman. Nu ne asumăm răspunderea pentru eventualele neînțelegeri sau interpretări greșite rezultate din utilizarea acestei traduceri.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->