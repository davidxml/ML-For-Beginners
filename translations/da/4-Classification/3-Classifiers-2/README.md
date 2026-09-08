# Cuisine classifiers 2

I denne anden klassificeringslektion vil du udforske flere måder at klassificere numeriske data på. Du vil også lære om konsekvenserne ved at vælge den ene klassifikator frem for den anden.

## [For-forelæsning quiz](https://ff-quizzes.netlify.app/en/ml/)

### Forudsætning

Vi antager, at du har gennemført de tidligere lektioner og har et renset datasæt i din `data`-mappe kaldet _cleaned_cuisines.csv_ i roden af denne 4-lektions mappe.

### Forberedelse

Vi har indlæst din _notebook.ipynb_-fil med det rensede datasæt og delt det op i X og y dataframes, klar til modelbyggerprocessen.

## Et klassificeringskort

Tidligere lærte du om de forskellige muligheder, du har ved klassificering af data ved hjælp af Microsofts snydeark. Scikit-learn tilbyder et lignende, men mere detaljeret snydeark, der yderligere kan hjælpe med at indsnævre dine estimeringsmetoder (et andet ord for klassifikatorer):

![ML Map from Scikit-learn](../../../../translated_images/da/map.e963a6a51349425a.webp)
> Tip: [besøg dette kort online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) og klik dig igennem stien for at læse dokumentationen.

### Planen

Dette kort er meget nyttigt, når du har et klart overblik over dine data, da du kan 'gå' langs dets stier til en beslutning:

- Vi har >50 prøver
- Vi ønsker at forudsige en kategori
- Vi har mærket data
- Vi har færre end 100K prøver
- ✨ Vi kan vælge en Linear SVC
- Hvis det ikke virker, da vi har numeriske data
    - Kan vi prøve en ✨ KNeighbors Classifier 
      - Hvis det stadig ikke virker, prøv ✨ SVC og ✨ Ensemble Classifiers

Dette er en meget nyttig rute at følge.

## Øvelse - del dataene

Følgende denne sti bør vi starte med at importere nogle biblioteker til brug.

1. Importér de nødvendige biblioteker:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Del dine trænings- og testdata:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC-klassifikator

Support-Vector clustering (SVC) er en del af Support-Vector maskiner-familien af ML-teknikker (lær mere om disse nedenfor). Med denne metode kan du vælge en ‘kernel’ for at bestemme, hvordan etiketterne skal grupperes. Parameteren ‘C’ refererer til ‘regularisering’, som regulerer parametrenes indflydelse. Kernelen kan være en af [flere](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); her sætter vi den til ‘linear’ for at sikre, at vi udnytter linear SVC. Probability er som standard sat til ‘false’; her sætter vi den til ‘true’ for at indsamle sandsynlighedsskøn. Vi sætter random state til ‘0’ for at blande dataene og få sandsynligheder.

### Øvelse - anvend en linear SVC

Start med at oprette et array af klassifikatorer. Du vil gradvist tilføje til dette array, mens vi tester.

1. Start med en Linear SVC:

    ```python
    C = 10
    # Opret forskellige klassifikatorer.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Træn din model med Linear SVC og udskriv en rapport:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Resultatet er ret godt:

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## K-Neighbors klassifikator

K-Neighbors hører til i "neighbors"-familien af ML-metoder, som kan bruges til både overvåget og ikke-overvåget læring. I denne metode oprettes et foruddefineret antal punkter, og data samles omkring disse punkter, så generaliserede etiketter kan forudsiges for dataene.

### Øvelse - anvend K-Neighbors klassifikatoren

Den forrige klassifikator var god og fungerede godt med dataene, men måske kan vi få bedre nøjagtighed. Prøv en K-Neighbors klassifikator.

1. Tilføj en linje til dit klassifikator-array (tilføj et komma efter Linear SVC-objektet):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Resultatet er en smule dårligere:

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ Læs om [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vector klassifikatorer er en del af [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) familien af ML-metoder, der bruges til klassificering og regressionsopgaver. SVM'er "afbilder træningseksempler til punkter i rummet" for at maksimere afstanden mellem to kategorier. Efterfølgende data afbildes i dette rum, så deres kategori kan forudsiges.

### Øvelse - anvend en Support Vector Classifier

Lad os prøve på en lidt bedre nøjagtighed med en Support Vector Classifier.

1. Tilføj et komma efter K-Neighbors-elementet, og tilføj derefter denne linje:

    ```python
    'SVC': SVC(),
    ```

    Resultatet er ret godt!

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ Læs om [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble klassifikatorer

Lad os følge stien helt til enden, selvom den tidligere test var ganske god. Lad os prøve nogle 'Ensemble Classifiers', specifikt Random Forest og AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Resultatet er meget godt, især for Random Forest:

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ Læs om [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Denne metode inden for Maskinlæring "kombinerer forudsigelserne fra flere basismetoder" for at forbedre modellens kvalitet. I vores eksempel brugte vi Random Trees og AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), en gennemsnitsmetode, bygger en 'skov' af 'beslutningstræer' fyldt med tilfældighed for at undgå overtilpasning. Parameteren n_estimators sættes til antallet af træer.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) tilpasser en klassifikator til et datasæt og tilpasser derefter kopier af den klassifikator til det samme datasæt. Den fokuserer på vægtene af forkert klassificerede elementer og justerer tilpasningen for den næste klassifikator for at rette op.

---

## 🚀Udfordring

Hver af disse teknikker har et stort antal parametre, som du kan justere. Undersøg standardparametrene for hver enkelt, og overvej, hvad justering af disse parametre ville betyde for modellens kvalitet.

## [Efter-forelæsning quiz](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

Der er meget jargon i disse lektioner, så tag et øjeblik til at gennemgå [denne liste](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) med nyttige termer!

## Opgave

[Parameter leg](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfraskrivelse**:
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det oprindelige dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os ikke ansvar for misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->