# Kökklassificerare 2

I denna andra klassificeringslektion kommer du att utforska fler sätt att klassificera numeriska data. Du kommer också att lära dig om konsekvenserna av att välja en klassificerare över en annan.

## [Förföreläsningsquiz](https://ff-quizzes.netlify.app/en/ml/)

### Förkunskaper

Vi antar att du har slutfört de tidigare lektionerna och har en rensad dataset i din `data`-mapp som heter _cleaned_cuisines.csv_ i roten av denna 4-lektionsmapp.

### Förberedelse

Vi har laddat din _notebook.ipynb_-fil med den rensade datasetet och har delat upp den i X- och y-dataframes, redo för modellbyggnadsprocessen.

## En klassificeringskarta

Tidigare lärde du dig om de olika alternativen du har när du klassificerar data med hjälp av Microsofts fusklapp. Scikit-learn erbjuder en liknande, men mer detaljerad fusklapp som kan hjälpa dig att ytterligare begränsa dina estimatorer (ett annat ord för klassificerare):

![ML Map from Scikit-learn](../../../../translated_images/sv/map.e963a6a51349425a.webp)
> Tips: [besök denna karta online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) och klicka längs vägen för att läsa dokumentation.

### Planen

Denna karta är mycket hjälpsam när du har en klar förståelse för dina data, då du kan 'gå' längs dess vägar till ett beslut:

- Vi har >50 prover
- Vi vill förutsäga en kategori
- Vi har märkta data
- Vi har färre än 100K prover
- ✨ Vi kan välja en Linear SVC
- Om det inte fungerar, eftersom vi har numeriska data
    - Kan vi prova en ✨ KNeighbors Classifier 
      - Om det inte fungerar, prova ✨ SVC och ✨ Ensemble Classifiers

Detta är en mycket hjälpsam väg att följa.

## Övning - dela upp datan

Följande väg bör vi börja med att importera några bibliotek att använda.

1. Importera de nödvändiga biblioteken:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Dela upp din tränings- och testdata:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC-klassificerare

Support-Vector clustering (SVC) är ett barn till Support-Vector machines-familjen av ML-tekniker (läs mer om dessa nedan). I denna metod kan du välja en 'kernel' för att bestämma hur etiketterna ska grupperas. Parametern 'C' avser 'reguljärisering' som reglerar parametrarnas inflytande. Kerneln kan vara en av [flera](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); här sätter vi den till 'linear' för att säkerställa att vi utnyttjar linear SVC. Probability är standard satt till 'false'; här sätter vi den till 'true' för att samla sannolikhetsuppskattningar. Vi sätter random state till '0' för att slumpa om datan för att få sannolikheter.

### Övning - applicera en linear SVC

Börja med att skapa en array med klassificerare. Du kommer att lägga till successivt till denna array när vi testar.

1. Börja med en Linear SVC:

    ```python
    C = 10
    # Skapa olika klassificerare.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Träna din modell med Linear SVC och skriv ut en rapport:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Resultatet är ganska bra:

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

## K-Neighbors-klassificerare

K-Neighbors är en del av "neighbors"-familjen av ML-metoder, som kan användas för både övervakad och oövervakad inlärning. I denna metod skapas ett fördefinierat antal punkter och data samlas runt dessa punkter så att generaliserade etiketter kan förutsägas för datan.

### Övning - applicera K-Neighbors-klassificeraren

Den tidigare klassificeraren var bra och fungerade väl med datan, men kanske kan vi få bättre noggrannhet. Prova en K-Neighbors-klassificerare.

1. Lägg till en rad i din klassificerar-array (lägg till ett kommatecken efter Linear SVC-elementet):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Resultatet är lite sämre:

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

    ✅ Läs om [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vector-klassificerare är en del av [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine)-familjen av ML-metoder som används för klassificerings- och regressionsuppgifter. SVM "kartlägger träningsdata till punkter i rymden" för att maximera avståndet mellan två kategorier. Efterföljande data kartläggs in i denna rymd så att deras kategori kan förutsägas.

### Övning - applicera en Support Vector Classifier

Låt oss försöka få lite bättre noggrannhet med en Support Vector Classifier.

1. Lägg till ett kommatecken efter K-Neighbors-elementet, och sedan lägg till denna rad:

    ```python
    'SVC': SVC(),
    ```

    Resultatet är ganska bra!

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

    ✅ Läs om [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble-klassificerare

Låt oss följa vägen hela vägen till slutet, även om det tidigare testet var ganska bra. Låt oss prova några 'Ensemble Classifiers', specifikt Random Forest och AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Resultatet är mycket bra, särskilt för Random Forest:

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

✅ Läs om [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Denna metod inom Maskininlärning "kombinerar förutsägelser från flera basestimatorer" för att förbättra modellens kvalitet. I vårt exempel använde vi Random Trees och AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), en genomsnittlig metod, bygger en 'skog' av 'besluts-träd' infunderade med slumpmässighet för att undvika överanpassning. Parametern n_estimators är satt till antalet träd.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) anpassar en klassificerare till en dataset och anpassar sedan kopior av den klassificeraren till samma dataset. Den fokuserar på vikterna av felklassificerade objekt och justerar anpassningen för nästa klassificerare för att korrigera.

---

## 🚀Utmaning

Var och en av dessa tekniker har ett stort antal parametrar som du kan justera. Undersök standardparametrarna för var och en och fundera på vad justering av dessa parametrar skulle innebära för modellens kvalitet.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Granskning & Självstudier

Det finns mycket fackspråk i dessa lektioner, så ta en minut att gå igenom [denna lista](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) med användbar terminologi!

## Uppgift 

[Parameterlek](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfriskrivning**:
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen var medveten om att automatiska översättningar kan innehålla fel eller brister. Det ursprungliga dokumentet på dess modersmål bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för några missförstånd eller feltolkningar som uppstår från användningen av denna översättning.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->