# Cuisine classifiers 1

In deze les ga je de dataset gebruiken die je hebt opgeslagen van de vorige les, vol met evenwichtige, schone data over keukens.

Je zult deze dataset gebruiken met verschillende classifiers om _een bepaalde nationale keuken te voorspellen op basis van een groep ingrediënten_. Terwijl je dit doet, leer je meer over enkele manieren waarop algoritmen kunnen worden ingezet voor classificatietaken.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)
# Voorbereiding

Als je [Les 1](../1-Introduction/README.md) hebt afgerond, zorg er dan voor dat er een _cleaned_cuisines.csv_ bestand bestaat in de hoofdmap `/data` voor deze vier lessen.

## Oefening - voorspel een nationale keuken

1. Werk in deze les in de _notebook.ipynb_-map en importeer dat bestand samen met de Pandas-bibliotheek:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    De data ziet er zo uit:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Importeer nu nog enkele bibliotheken:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Verdeel de X- en y-coördinaten in twee dataframes voor training. `cuisine` kan het labels-dataframe zijn:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Het zal er zo uitzien:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Verwijder die `Unnamed: 0` kolom en de `cuisine` kolom door `drop()` aan te roepen. Bewaar de rest van de data als trainbare features:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Je features zien er zo uit:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Je bent nu klaar om je model te trainen!

## Kies je classifier

Nu je data schoon en klaar is voor training, moet je beslissen welk algoritme je voor de klus gaat gebruiken.

Scikit-learn groepeert classificatie onder Supervised Learning, en in die categorie vind je vele manieren om te classificeren. [De variëteit](https://scikit-learn.org/stable/supervised_learning.html) kan in het begin nogal verwarrend zijn. De volgende methoden bevatten allemaal classificatietechnieken:

- Lineaire modellen
- Support Vector Machines
- Stochastische Gradient Descent
- Dichtstbijzijnde buren
- Gaussian Processes
- Beslissingsbomen
- Ensemble-methoden (voting Classifier)
- Multiclass en multioutput algoritmes (multiclass en multilabel classificatie, multiclass-multioutput classificatie)

> Je kunt ook [neurale netwerken gebruiken om data te classificeren](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), maar dat valt buiten de reikwijdte van deze les.

### Welke classifier te kiezen?

Dus, welke classifier moet je kiezen? Vaak is het doorlopen van meerdere en zoeken naar een goed resultaat een manier om te testen. Scikit-learn biedt een [side-by-side vergelijking](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) op een gemaakte dataset, waarbij KNeighbors, SVC op twee manieren, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB en QuadraticDiscrinationAnalysis worden vergeleken, met de resultaten visueel weergegeven:

![comparison of classifiers](../../../../translated_images/nl/comparison.edfab56193a85e7f.webp)
> Grafieken gegenereerd op de documentatie van Scikit-learn

> AutoML lost dit probleem netjes op door deze vergelijkingen in de cloud uit te voeren, waardoor je het beste algoritme voor je data kunt kiezen. Probeer het [hier](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Een betere benadering

Een betere manier dan wild raden, is echter om de ideeën op dit downloadbare [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) te volgen. Hier ontdekken we dat we bij ons multiclass-probleem enkele keuzes hebben:

![cheatsheet for multiclass problems](../../../../translated_images/nl/cheatsheet.07a475ea444d2223.webp)
> Een gedeelte van Microsoft's Algorithm Cheat Sheet, met details over multiclass classificatie-opties

✅ Download deze cheat sheet, print hem uit en hang hem op je muur!

### Redenering

Laten we eens kijken of we met redeneren tot verschillende benaderingen kunnen komen, gezien de beperkingen die we hebben:

- **Neurale netwerken zijn te zwaar**. Gezien onze schone, maar minimale dataset, en het feit dat we lokaal via notebooks trainen, zijn neurale netwerken te zwaar voor deze taak.
- **Geen tweeklassen-classifier**. We gebruiken geen tweeklassen-classifier, dus one-vs-all valt af.
- **Beslissingsboom of logistische regressie kan werken**. Een beslissingsboom kan werken, of logistische regressie voor multiclass data.
- **Multiclass Boosted Decision Trees lossen een ander probleem op**. De multiclass boosted decision tree is het meest geschikt voor niet-parametrische taken, bijvoorbeeld taken die rankings bouwen, dus is niet bruikbaar voor ons.

### Gebruik van Scikit-learn

We gaan Scikit-learn gebruiken om onze data te analyseren. Er zijn echter veel manieren om logistische regressie in Scikit-learn te gebruiken. Bekijk de [parameters die je moet meegeven](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

In wezen zijn er twee belangrijke parameters - `multi_class` en `solver` - die we moeten specificeren wanneer we Scikit-learn om logistische regressie vragen. De waarde van `multi_class` bepaalt een bepaald gedrag. De waarde van `solver` bepaalt welk algoritme wordt gebruikt. Niet alle solvers kunnen gecombineerd worden met alle `multi_class` waarden.

Volgens de docs gebruikt het trainingsalgoritme in het multiclass-geval:

- **Het one-vs-rest (OvR) schema**, als de `multi_class` optie is ingesteld op `ovr`
- **De cross-entropy verliesfunctie**, als de `multi_class` optie is ingesteld op `multinomial`. (Momenteel wordt de `multinomial` optie alleen ondersteund door de ‘lbfgs’, ‘sag’, ‘saga’ en ‘newton-cg’ solvers.)"

> 🎓 Het 'schema' hier kan 'ovr' (one-vs-rest) of 'multinomial' zijn. Omdat logistische regressie eigenlijk is ontworpen voor binaire classificatie, stellen deze schema's het beter in staat om multiclass-classificatieproblemen aan te pakken. [bron](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 De 'solver' wordt gedefinieerd als "het algoritme dat wordt gebruikt in het optimalisatieprobleem". [bron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn biedt deze tabel om uit te leggen hoe solvers verschillende uitdagingen van verschillende soorten datastructuren aanpakken:

![solvers](../../../../translated_images/nl/solvers.5fc648618529e627.webp)

## Oefening - split de data

We kunnen ons richten op logistische regressie voor onze eerste trainingsexperiment omdat je daar onlangs over hebt geleerd in een vorige les.
Splits je data in trainings- en testgroepen door `train_test_split()` aan te roepen:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Oefening - pas logistische regressie toe

Omdat je de multiclass-variant gebruikt, moet je kiezen welk _schema_ je gebruikt en welke _solver_ je instelt. Gebruik LogisticRegression met een multiclass-instelling en de **liblinear** solver om te trainen.

1. Maak een logistische regressie met multi_class ingesteld op `ovr` en de solver op `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Probeer een andere solver zoals `lbfgs`, die vaak als standaard is ingesteld

    > Let op: gebruik de Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) functie om je data af te vlakken wanneer nodig.

    De nauwkeurigheid is goed, ruim boven de **80%**!

1. Je kunt dit model in actie zien door één regel data te testen (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Het resultaat wordt afgedrukt:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Probeer een ander rij-nummer en controleer de resultaten
1. Dieper graven, je kunt de nauwkeurigheid van deze voorspelling controleren:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Het resultaat wordt afgedrukt - Indiase keuken is de beste gok, met een goede waarschijnlijkheid:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Kun je uitleggen waarom het model er vrij zeker van is dat dit een Indiase keuken is?

1. Krijg meer detail door een classificatierapport af te drukken, zoals je deed in de regressielessen:

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

## 🚀Uitdaging

In deze les heb je je opgeschoonde data gebruikt om een machine learning-model te bouwen dat een nationale keuken kan voorspellen op basis van een reeks ingrediënten. Neem de tijd om de vele opties die Scikit-learn biedt om data te classificeren door te lezen. Duik dieper in het concept van 'solver' om te begrijpen wat er achter de schermen gebeurt.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Verdiep je wat meer in de wiskunde achter logistische regressie in [deze les](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Opdracht 

[Bestudeer de solvers](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in de oorspronkelijke taal moet als de gezaghebbende bron worden beschouwd. Voor belangrijke informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->