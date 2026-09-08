# Cuisine classifiers 1

I denne lektion vil du bruge datasættet, du gemte fra sidste lektion, fyldt med balancerede, rene data om køkkener.

Du vil bruge dette datasæt med en række klassifikatorer til at _forudsige en given national køkken baseret på en gruppe ingredienser_. Mens du gør det, vil du lære mere om nogle af de måder, algoritmer kan udnyttes til klassifikationsopgaver.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)
# Forberedelse

Forudsat at du har gennemført [Lektion 1](../1-Introduction/README.md), skal du sikre dig, at en _cleaned_cuisines.csv_-fil findes i den øverste `/data`-mappe til disse fire lektioner.

## Øvelse - forudsige et nationalt køkken

1. Arbejd i denne lektions _notebook.ipynb_ mappe, importer den fil sammen med Pandas-biblioteket:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Dataene ser sådan ud:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Importer nu flere biblioteker:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Del X og y koordinaterne i to dataframes til træning. `cuisine` kan være labels dataframe:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Det vil se sådan ud:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Drop den `Unnamed: 0` kolonne og `cuisine` kolonnen ved at kalde `drop()`. Gem resten af dataene som trænbar features:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Dine features ser sådan ud:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Nu er du klar til at træne din model!

## Valg af klassifikator

Nu hvor dine data er rene og klar til træning, skal du beslutte, hvilken algoritme der skal bruges til opgaven.

Scikit-learn grupperer klassifikation under Superviseret læring, og i den kategori finder du mange måder at klassificere på. [Variationerne](https://scikit-learn.org/stable/supervised_learning.html) kan virke overvældende ved første øjekast. Følgende metoder indeholder alle klassifikationsteknikker:

- Lineære modeller
- Support Vector Machines
- Stochastic Gradient Descent
- Nærmeste naboer
- Gaussiske processer
- Beslutningstræer
- Ensemblemetoder (votingsklassifikator)
- Multiklasse- og multioutput-algoritmer (multiklasse- og multilabel-klassifikation, multiklasse-multioutput-klassifikation)

> Du kan også bruge [neurale netværk til at klassificere data](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), men det er uden for omfanget af denne lektion.

### Hvilken klassifikator skal man vælge?

Så, hvilken klassifikator skal du vælge? Det er ofte en god idé at prøve flere og se efter et godt resultat som test. Scikit-learn tilbyder en [side-om-side sammenligning](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) på et oprettet datasæt, der sammenligner KNeighbors, SVC på to måder, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB og QuadraticDiscrinationAnalysis, med visualiserede resultater:

![comparison of classifiers](../../../../translated_images/da/comparison.edfab56193a85e7f.webp)
> Diagrammer genereret i Scikit-learns dokumentation

> AutoML løser dette problem flot ved at køre disse sammenligninger i skyen, så du kan vælge den bedste algoritme til dine data. Prøv det [her](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### En bedre tilgang

En bedre måde end at gætte vildt er at følge ideerne i dette downloadbare [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Her opdager vi, at for vores multiklasse-problem har vi nogle valg:

![cheatsheet for multiclass problems](../../../../translated_images/da/cheatsheet.07a475ea444d2223.webp)
> Et uddrag af Microsofts Algorithm Cheat Sheet, der detaljerer multiklasseklassifikationsmuligheder

✅ Download dette cheat sheet, print det ud, og hæng det op på din væg!

### Begrundelse

Lad os se, om vi kan begrunde forskellige tilgange ud fra de begrænsninger, vi har:

- **Neurale netværk er for tunge**. Givet vores rene, men minimale datasæt, og at vi kører træning lokalt via notebooks, er neurale netværk for tunge til denne opgave.
- **Ingen to-klasse klassifikator**. Vi bruger ikke en to-klasse klassifikator, så det udelukker one-vs-all.
- **Beslutningstræ eller logistisk regression kunne fungere**. Et beslutningstræ kunne fungere, eller logistisk regression til multiklasse data.
- **Multiklasse Boosted Decision Trees løser et andet problem**. Det multiklasse boosted decision tree er mest egnet til ikke-parametriske opgaver, fx opgaver designet til at bygge ranglister, så det er ikke nyttigt for os.

### Brug af Scikit-learn

Vi vil bruge Scikit-learn til at analysere vores data. Der er dog mange måder at bruge logistisk regression i Scikit-learn på. Tag et kig på [parametrene, der kan gives](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Grundlæggende er der to vigtige parametre - `multi_class` og `solver` - som vi skal specificere, når vi beder Scikit-learn om at udføre logistisk regression. `multi_class`-værdien angiver en bestemt opførsel. Værdien af solver er, hvilken algoritme der skal bruges. Ikke alle solvere kan kombineres med alle `multi_class` værdier.

Ifølge dokumentationen bruger træningsalgoritmen i multiklasse tilfælde:

- **Bruger one-vs-rest (OvR) skemaet**, hvis `multi_class`-optionen sættes til `ovr`
- **Bruger kryds-entropi-tab**, hvis `multi_class`-optionen sættes til `multinomial`. (I øjeblikket understøtter `multinomial` kun ‘lbfgs’, ‘sag’, ‘saga’ og ‘newton-cg’ solvere.)"

> 🎓 ‘Skemaet’ her kan enten være ‘ovr’ (one-vs-rest) eller ‘multinomial’. Da logistisk regression er designet til binær klassifikation, tillader disse skemaer den at håndtere multiklasse klassifikationsopgaver bedre. [kilde](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 ‘Solver’ defineres som "algoritmen til brug i optimeringsproblemet". [kilde](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn tilbyder denne tabel for at forklare, hvordan solvere håndterer forskellige udfordringer præsenteret af forskellige datatyper:

![solvers](../../../../translated_images/da/solvers.5fc648618529e627.webp)

## Øvelse - del dataene

Vi kan fokusere på logistisk regression til vores første træningsforsøg, da du for nylig lærte om denne i en tidligere lektion.
Del dine data op i trænings- og testgrupper ved at kalde `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Øvelse - anvend logistisk regression

Da du bruger multiklasse-tilfældet, skal du vælge hvilket _skema_ der skal bruges, og hvilken _solver_ der skal sættes. Brug LogisticRegression med en multiklasse-indstilling og **liblinear** solver til træning.

1. Opret en logistisk regression med multi_class sat til `ovr` og solver sat til `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Prøv en anden solver som `lbfgs`, som ofte sættes som standard

    > Bemærk, brug Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) funktion til at flade dine data ud, når det er nødvendigt.

    Præcisionen er god med over **80%**!

1. Du kan se denne model i aktion ved at teste en række data (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Resultatet printes:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Prøv et andet rækkenummer og tjek resultaterne
1. Hvis du graver dybere, kan du tjekke nøjagtigheden af denne forudsigelse:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Resultatet bliver printet - Indisk køkken er det bedste bud, med god sandsynlighed:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Kan du forklare, hvorfor modellen er ret sikker på, at dette er indisk køkken?

1. Få flere detaljer ved at printe en klassifikationsrapport, som du gjorde i regressionslektionerne:

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

## 🚀Udfordring

I denne lektion brugte du dine rensede data til at bygge en maskinlæringsmodel, der kan forudsige en national ret baseret på en række ingredienser. Tag dig tid til at læse igennem de mange muligheder, Scikit-learn tilbyder til klassificering af data. Undersøg nærmere konceptet ’solver’ for at forstå, hvad der sker bag kulisserne.

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

Grav lidt dybere i matematikken bag logistisk regression i [denne lektion](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Opgave 

[Studér solverne](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Mens vi stræber efter nøjagtighed, bedes du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det oprindelige dokument på dets modersmål bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os intet ansvar for misforståelser eller fejltolkninger som følge af brugen af denne oversættelse.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->