# Cuisine classifiers 1

I den här lektionen kommer du att använda den dataset du sparade från föregående lektion, full av balanserad, ren data om olika kök.

Du kommer att använda detta dataset med en rad olika klassificerare för att _förutsäga ett givet nationellt kök baserat på en grupp ingredienser_. Samtidigt kommer du att lära dig mer om några av de sätt som algoritmer kan användas för klassificeringsuppgifter.

## [Förföreläsningsquiz](https://ff-quizzes.netlify.app/en/ml/)
# Förberedelse

Om du har genomfört [Lektion 1](../1-Introduction/README.md), se till att en fil som heter _cleaned_cuisines.csv_ finns i rotmappen `/data` för dessa fyra lektioner.

## Övning - förutsäg ett nationellt kök

1. Arbeta i den här lektionens _notebook.ipynb_-mapp, importera den filen tillsammans med Pandas-biblioteket:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Datat ser ut så här:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Importera nu flera andra bibliotek:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Dela upp X och y i två dataframes för träning. `cuisine` kan vara label-dataframen:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Det kommer att se ut så här:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Droppa kolumnen `Unnamed: 0` och kolumnen `cuisine` genom att anropa `drop()`. Spara resten av datat som träningsbara funktioner:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Dina funktioner ser ut så här:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Nu är du redo att träna din modell!

## Välj din klassificerare

Nu när ditt data är rent och redo för träning måste du bestämma dig för vilken algoritm du ska använda.

Scikit-learn grupperar klassificering under övervakad inlärning (Supervised Learning), och i den kategorin finns många sätt att klassificera. [Variationen](https://scikit-learn.org/stable/supervised_learning.html) kan vara ganska överväldigande vid första anblicken. Följande metoder inkluderar alla klassificeringstekniker:

- Linjära modeller
- Support Vector Machines
- Stokastisk gradientnedstigning
- Närmaste grannar (Nearest Neighbors)
- Gaussiska processer
- Beslutsträd
- Ensembelmetoder (votering Klassificerare)
- Multiklass- och multioutput-algoritmer (multiklass- och multilabel-klassificering, multiklass-multioutput-klassificering)

> Du kan också använda [neurala nätverk för att klassificera data](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), men det ligger utanför denna lektions omfattning.

### Vilken klassificerare ska du välja?

Så, vilken klassificerare ska du välja? Ofta är det ett sätt att testa att köra flera och leta efter ett bra resultat. Scikit-learn erbjuder en [jämförelse sida vid sida](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) på ett skapat dataset, som jämför KNeighbors, SVC på två sätt, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB och QuadraticDiscrinationAnalysis, och visar resultaten visuellt: 

![comparison of classifiers](../../../../translated_images/sv/comparison.edfab56193a85e7f.webp)
> Diagram skapade från Scikit-learns dokumentation

> AutoML löser detta problem snyggt genom att köra dessa jämförelser i molnet, så att du kan välja den bästa algoritmen för dina data. Prova det [här](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Ett bättre tillvägagångssätt

Ett bättre sätt än att gissa vilt är dock att följa idéerna i detta nedladdningsbara [ML Cheat Sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Här upptäcker vi att för vårt multiclass-problem har vi några val:

![cheatsheet for multiclass problems](../../../../translated_images/sv/cheatsheet.07a475ea444d2223.webp)
> En sektion från Microsofts Algorithm Cheat Sheet, som visar alternativ för multiklassklassificering

✅ Ladda ned detta cheatsheet, skriv ut det och häng upp det på väggen!

### Resonemang

Låt oss se om vi kan resonera oss fram till olika tillvägagångssätt givet de begränsningar vi har:

- **Neurala nätverk är för tunga**. Med vårt rena men minimala dataset, och det faktum att vi kör träningen lokalt via notebooks, är neurala nätverk för resurskrävande för denna uppgift.
- **Ingen två-klass klassificerare**. Vi använder inte en två-klass klassificerare, så det utesluter one-vs-all.
- **Beslutsträd eller logistisk regression kan fungera**. Ett beslutsträd kan fungera, eller logistisk regression för multiklassdata.
- **Multiklass Boosted Decision Trees löser ett annat problem**. Det multiklass-boostrade beslutsträdet passar bäst för icke-parametriska uppgifter, t.ex. uppgifter som är utformade för att bygga rankingar, så det är inte användbart för oss.

### Använda Scikit-learn

Vi kommer att använda Scikit-learn för att analysera våra data. Det finns dock många sätt att använda logistisk regression i Scikit-learn. Ta en titt på [parametrarna att skicka med](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

I grunden finns två viktiga parametrar - `multi_class` och `solver` - som vi behöver ange när vi ber Scikit-learn att utföra logistisk regression. Värdet på `multi_class` styr ett visst beteende. Värdet på `solver` är vilken algoritm som ska användas. Inte alla solver kan paras ihop med alla `multi_class`-värden.

Enligt dokumentationen gäller för multiklassfallet:

- **Använder one-vs-rest (OvR)-schemat**, om `multi_class`-inställningen är satt till `ovr`.
- **Använder korsentropiförlusten**, om `multi_class`-inställningen är satt till `multinomial`. (För närvarande stöds `multinomial` endast av solver-algoritmerna ‘lbfgs’, ‘sag’, ‘saga’ och ‘newton-cg’.)"

> 🎓 'schemat' här kan vara 'ovr' (one-vs-rest) eller 'multinomial'. Eftersom logistisk regression egentligen är designad för binär klassificering, tillåter dessa scheman att den bättre hanterar multiclass-klassificeringsuppgifter. [källa](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' definieras som "algoritmen som ska användas i optimeringsproblemet". [källa](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn visar denna tabell för att förklara hur solvers hanterar olika utmaningar som presenteras av olika typer av datastrukturer:

![solvers](../../../../translated_images/sv/solvers.5fc648618529e627.webp)

## Övning - dela upp datat

Vi kan fokusera på logistisk regression för vårt första träningsförsök eftersom du nyligen lärde dig om detta i en tidigare lektion.
Dela upp dina data i tränings- och testgrupper genom att anropa `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Övning - tillämpa logistisk regression

Eftersom du använder multiklassfallet behöver du välja vilket _schema_ du ska använda och vilken _solver_ du ska sätta. Använd LogisticRegression med en multiklassinställning och **liblinear** solver för träning.

1. Skapa en logistisk regression med multi_class satt till `ovr` och solver satt till `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Prova en annan solver som `lbfgs`, som ofta är standard

    > Observera att använda Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) funktion för att platta till din data vid behov.

    Noggrannheten är bra, över **80%**!

1. Du kan se denna modell i praktiken genom att testa en rad data (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Resultatet skrivs ut:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Prova ett annat radnummer och kontrollera resultaten
1. Gräv djupare, du kan kontrollera noggrannheten i denna prediktion:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Resultatet skrivs ut – Indisk matlagning är dess bästa gissning, med god sannolikhet:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Kan du förklara varför modellen är ganska säker på att detta är en indisk maträtt?

1. Få mer detalj genom att skriva ut en klassificeringsrapport, som du gjorde i regression-lektionerna:

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

## 🚀Utmaning

I denna lektion använde du dina rensade data för att bygga en maskininlärningsmodell som kan förutsäga en nationell maträtt baserat på en serie ingredienser. Ta dig tid att läsa igenom de många alternativ Scikit-learn erbjuder för att klassificera data. Gräv djupare i konceptet 'solver' för att förstå vad som händer bakom kulisserna.

## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Översikt & Självstudier

Gräv lite mer i matematiken bakom logistisk regression i [denna lektion](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Uppgift 

[Studera solvers](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfriskrivning**:
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen observera att automatiska översättningar kan innehålla fel eller unøjaktigheter. Det ursprungliga dokumentet på dess modersmål bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för några missförstånd eller feltolkningar som uppstår till följd av användningen av denna översättning.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->