# Kjøkkenklassifikatorer 1

I denne leksjonen skal du bruke datasettet du lagret fra forrige leksjon fullt av balansert, rent data om kjøkken.

Du skal bruke dette datasettet med en rekke klassifikatorer for å _forutsi en gitt nasjonal kjøkken basert på en gruppe ingredienser_. Mens du gjør dette, vil du lære mer om noen av måtene algoritmer kan utnyttes til klassifiseringsoppgaver.

## [Pre-forelesningsquiz](https://ff-quizzes.netlify.app/en/ml/)
# Forberedelse

Forutsatt at du fullførte [Leksjon 1](../1-Introduction/README.md), sørg for at en fil kalt _cleaned_cuisines.csv_ finnes i rotmappen `/data` for disse fire leksjonene.

## Øvelse - forutsi et nasjonalt kjøkken

1. Arbeid i denne leksjonens _notebook.ipynb_-mappe, importer den filen sammen med Pandas-biblioteket:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Dataene ser slik ut:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Nå, importer flere biblioteker:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Del X og y koordinatene inn i to dataframes for trening. `cuisine` kan være labels-dataframen:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Det vil se slik ut:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Dropp kolonnen `Unnamed: 0` og kolonnen `cuisine` ved å bruke `drop()`. Lagre resten av dataene som trenbare funksjoner:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Funksjonene dine ser slik ut:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Nå er du klar til å trene modellen din!

## Velge klassifikatoren din

Nå som dataene dine er rene og klare for trening, må du bestemme hvilket algoritme du vil bruke.

Scikit-learn grupperer klassifisering under Overvåket Læring, og i den kategorien finner du mange måter å klassifisere på. [Variasjonen](https://scikit-learn.org/stable/supervised_learning.html) kan virke overveldende ved første øyekast. Følgende metoder inkluderer alle klassifiseringsteknikker:

- Lineære modeller
- Støttevektormaskiner
- Stokastisk gradientnedstigning
- Nærmeste naboer
- Gaussiske prosesser
- Beslutningstrær
- Ensemble-metoder (votering Klassifikator)
- Multiklasse og multioutput-algoritmer (multiklasse og multilabel klassifisering, multiklasse-multioutput klassifisering)

> Du kan også bruke [nevrale nettverk for å klassifisere data](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), men det er utenfor omfanget av denne leksjonen.

### Hvilken klassifikator skal du velge?

Så, hvilken klassifikator bør du velge? Ofte er det en god metode å prøve flere for å finne et godt resultat. Scikit-learn tilbyr en [side-ved-side sammenligning](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) på et opprettet datasett, som sammenligner KNeighbors, SVC på to måter, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB og QuadraticDiscrinationAnalysis, og viser resultatene visualisert:

![sammenligning av klassifikatorer](../../../../translated_images/no/comparison.edfab56193a85e7f.webp)
> Plottene er generert i Scikit-learns dokumentasjon

> AutoML løser dette problemet elegant ved å kjøre disse sammenligningene i skyen, slik at du kan velge den beste algoritmen for dine data. Prøv det [her](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### En bedre tilnærming

En bedre måte enn å gjette vilt på, er å følge ideene i dette nedlastbare [ML jukseark](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Her oppdager vi at for vårt multiklasseproblem, har vi noen valg:

![jukseark for multiklasseproblemer](../../../../translated_images/no/cheatsheet.07a475ea444d2223.webp)
> En seksjon fra Microsofts Algorithm Cheat Sheet, som detaljerer alternativer for multiklasseklassifisering

✅ Last ned dette juksearket, skriv det ut og heng det opp på veggen din!

### Resonnement

La oss se om vi kan resonnere oss gjennom forskjellige tilnærminger gitt de begrensningene vi har:

- **Nevrale nettverk er for tunge**. Gitt vårt rene, men minimale datasett, og at vi kjører treningen lokalt via notebooks, er nevrale nettverk for tunge for denne oppgaven.
- **Ingen to-klasses klassifikator**. Vi bruker ikke en to-klassers klassifikator, så det utelukker one-vs-all.
- **Beslutningstre eller logistisk regresjon kan fungere**. Et beslutningstre kan fungere, eller logistisk regresjon for multiklasse-data.
- **Multiklasse Boosted Decision Trees løser et annet problem**. Multiklasse boosted beslutningstre er mest egnet for ikke-parametriske oppgaver, f.eks. oppgaver designet for å bygge rangeringer, så det er ikke nyttig for oss.

### Bruke Scikit-learn

Vi skal bruke Scikit-learn for å analysere dataene våre. Det finnes imidlertid mange måter å bruke logistisk regresjon i Scikit-learn. Ta en titt på [parametrene som kan settes](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

I hovedsak er det to viktige parametere - `multi_class` og `solver` - som vi må spesifisere når vi spør Scikit-learn om å utføre logistisk regresjon. `multi_class`-verdien bestemmer en spesiell oppførsel. Verdien for solver er hvilken algoritme som skal brukes. Ikke alle solvere kan kombineres med alle `multi_class`-verdier.

Ifølge dokumentasjonen, i multiklasse-tilfelle, bruker treningsalgoritmen:

- **En en-mot-de-andre (OvR) ordning**, hvis `multi_class`-alternativet settes til `ovr`
- **Krysstaps-tapet (cross-entropy loss)**, hvis `multi_class`-alternativet settes til `multinomial`. (Foreløpig støttes `multinomial`-alternativet kun av ‘lbfgs’, ‘sag’, ‘saga’ og ‘newton-cg’ solverne.)"

> 🎓 'Ordningen' her kan enten være 'ovr' (one-vs-rest) eller 'multinomial'. Fordi logistisk regresjon egentlig er designet for binærklassifisering, tillater disse ordningene den å håndtere multiklasseklassifikasjonsoppgaver bedre. [kilde](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solveren' defineres som "algoritmen som skal brukes i optimaliseringsproblemet". [kilde](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn tilbyr denne tabellen for å forklare hvordan solvere håndterer ulike utfordringer som forskjellige typer datastrukturer presenterer:

![solvere](../../../../translated_images/no/solvers.5fc648618529e627.webp)

## Øvelse - del opp dataene

Vi kan fokusere på logistisk regresjon for vårt første treningsforsøk siden du nylig lærte om sistnevnte i en tidligere leksjon.
Del dataene dine i trenings- og testgrupper ved å kalle `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Øvelse - bruk logistisk regresjon

Siden du bruker multiklasse-tilfelle, må du velge hvilket _oppsett_ som skal brukes og hvilken _solver_ som skal settes. Bruk LogisticRegression med en multiklasse-innstilling og **liblinear** solver til trening.

1. Opprett en logistisk regresjon med multi_class satt til `ovr` og solveren satt til `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Prøv en annen solver som `lbfgs`, som ofte settes som standard

    > Merk, bruk Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) funksjon for å flate ut dataene dine når det trengs.

    Nøyaktigheten er god, over **80%**!

1. Du kan se denne modellen i aksjon ved å teste en rad med data (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Resultatet printes:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Prøv et annet radnummer og sjekk resultatene
1. Grav dypere, du kan sjekke nøyaktigheten av denne prediksjonen:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Resultatet skrives ut - Indisk mat er det beste gjetningen, med god sannsynlighet:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Kan du forklare hvorfor modellen er ganske sikker på at dette er indisk mat?

1. Få mer detalj ved å skrive ut en klassifiseringsrapport, slik du gjorde i regresjonsleksjonene:

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

## 🚀Utfordring

I denne leksjonen brukte du dine rensede data for å bygge en maskinlæringsmodell som kan forutsi en nasjonal matrett basert på en rekke ingredienser. Ta deg tid til å lese gjennom de mange alternativene Scikit-learn tilbyr for å klassifisere data. Grav dypere inn i konseptet 'solver' for å forstå hva som skjer bak kulissene.

## [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang og Selvstudium

Grav litt mer i matematikken bak logistisk regresjon i [denne leksjonen](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Oppgave

[Studer løserne](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfraskrivelse**:
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiserte oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på dets opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feilaktige tolkninger som oppstår ved bruk av denne oversettelsen.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->