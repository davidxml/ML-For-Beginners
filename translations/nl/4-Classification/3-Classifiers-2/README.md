# Cuisine classifiers 2

In deze tweede classificatieles verken je meer manieren om numerieke gegevens te classificeren. Je leert ook over de gevolgen van het kiezen van de ene classifier boven de andere.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Vereisten

We gaan ervan uit dat je de vorige lessen hebt afgerond en een opgeschoonde dataset hebt in je `data` map met de naam _cleaned_cuisines.csv_ in de root van deze map met 4 lessen.

### Voorbereiding

We hebben je _notebook.ipynb_ bestand geladen met de opgeschoonde dataset en deze verdeeld in X en y dataframes, klaar voor het bouwproces van het model.

## Een classificatiekaart

Eerder heb je geleerd over de verschillende opties die je hebt bij het classificeren van gegevens met behulp van Microsoft's cheat sheet. Scikit-learn biedt een soortgelijke, maar meer gedetailleerde cheat sheet die je verder kan helpen je schatters (een andere term voor classifiers) te verfijnen:

![ML Map from Scikit-learn](../../../../translated_images/nl/map.e963a6a51349425a.webp)
> Tip: [bezoek deze kaart online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) en klik door het pad om de documentatie te lezen.

### Het plan

Deze kaart is erg behulpzaam zodra je een duidelijk begrip van je gegevens hebt, want je kunt ‘wandelen’ langs de paden naar een beslissing:

- We hebben >50 voorbeelden
- We willen een categorie voorspellen
- We hebben gelabelde data
- We hebben minder dan 100K voorbeelden
- ✨ We kunnen kiezen voor een Linear SVC
- Als dat niet werkt, omdat we numerieke data hebben
    - Kunnen we een ✨ KNeighbors Classifier proberen
      - Als dat niet werkt, probeer ✨ SVC en ✨ Ensemble Classifiers

Dit is een zeer behulpzaam pad om te volgen.

## Oefening - splits de data

Volgend op dit pad, beginnen we met het importeren van enkele benodigde libraries.

1. Importeer de benodigde libraries:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Splits je trainings- en testdata:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC classifier

Support-Vector clustering (SVC) is een onderdeel van de Support-Vector machines familie van ML-technieken (lees hieronder meer over deze technieken). Bij deze methode kies je een 'kernel' om te beslissen hoe labels geclusterd worden. De parameter 'C' verwijst naar 'regularisatie' wat de invloed van parameters reguleert. De kernel kan een van [verschillende](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) zijn; hier zetten we deze op 'linear' om linear SVC te gebruiken. De standaardwaarde voor probability is 'false'; hier zetten we het op 'true' om waarschijnlijkheidsinschattingen te verzamelen. We stellen de random state in op '0' om de data te schudden en zo waarschijnlijkheden te verkrijgen.

### Oefening - pas een linear SVC toe

Begin met het maken van een array van classifiers. Je voegt hier steeds meer classifiers aan toe terwijl we testen. 

1. Begin met een Linear SVC:

    ```python
    C = 10
    # Maak verschillende classifieringsmodellen.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Train je model met de Linear SVC en print een rapport:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Het resultaat is vrij goed:

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

## K-Neighbors classifier

K-Neighbors maakt deel uit van de "neighbors" familie van ML-methoden, die kunnen worden gebruikt voor zowel supervised als unsupervised learning. Bij deze methode wordt een vooraf bepaald aantal punten gecreëerd en data worden rond deze punten verzameld zodat gegeneraliseerde labels voorspeld kunnen worden voor de data.

### Oefening - pas de K-Neighbors classifier toe

De vorige classifier was goed en werkte goed met de data, maar misschien kunnen we een betere nauwkeurigheid bereiken. Probeer een K-Neighbors classifier.

1. Voeg een regel toe aan je classifier array (voeg een komma toe na het Linear SVC item):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Het resultaat is iets slechter:

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

    ✅ Leer over [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vector classifiers zijn onderdeel van de [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) familie van ML-methoden die gebruikt worden voor classificatie- en regressietaken. SVM's "mappen trainingsvoorbeelden naar punten in de ruimte" om de afstand tussen twee categorieën te maximaliseren. Volgende data worden in deze ruimte gemapt zodat hun categorie kan worden voorspeld.

### Oefening - pas een Support Vector Classifier toe

Laten we proberen een iets betere nauwkeurigheid te behalen met een Support Vector Classifier.

1. Voeg een komma toe na het K-Neighbors item, en voeg dan deze regel toe:

    ```python
    'SVC': SVC(),
    ```

    Het resultaat is vrij goed!

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

    ✅ Leer over [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble Classifiers

Laten we het pad volgen tot het einde, ook al was de vorige test al goed. Laten we wat 'Ensemble Classifiers' proberen, specifiek Random Forest en AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Het resultaat is erg goed, vooral voor Random Forest:

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

✅ Leer over [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Deze methode van Machine Learning "combineert de voorspellingen van meerdere basis-schatters" om de kwaliteit van het model te verbeteren. In ons voorbeeld gebruikten we Random Trees en AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), een gemiddeldemethode, bouwt een 'bos' van 'beslissingsbomen' met toegevoegde willekeurigheid om overfitting te voorkomen. De parameter n_estimators staat voor het aantal bomen.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) past een classifier toe op een dataset en past vervolgens kopieën van die classifier toe op dezelfde dataset. Het richt zich op de gewichten van verkeerd geclassificeerde items en past de fit aan voor de volgende classifier om te corrigeren.

---

## 🚀Uitdaging

Elk van deze technieken heeft een groot aantal parameters die je kunt aanpassen. Onderzoek de standaardparameters van elk en denk na over wat het bijstellen van deze parameters zou betekenen voor de kwaliteit van het model.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Er zit veel jargon in deze lessen, dus neem even de tijd om [deze lijst](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) van nuttige terminologie te bekijken!

## Opdracht

[Spelen met parameters](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u er rekening mee te houden dat automatische vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het oorspronkelijke document in de oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt een professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor enige misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->