# Cuisine classifiers 2

In dieser zweiten Klassifikationslektion werden Sie weitere Möglichkeiten zur Klassifikation numerischer Daten erkunden. Außerdem erfahren Sie die Auswirkungen der Wahl eines Klassifikators gegenüber einem anderen.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Voraussetzung

Wir gehen davon aus, dass Sie die vorherigen Lektionen abgeschlossen haben und einen bereinigten Datensatz in Ihrem `data`-Ordner mit dem Namen _cleaned_cuisines.csv_ im Stammverzeichnis dieses 4-Lektionen-Ordners haben.

### Vorbereitung

Wir haben Ihre _notebook.ipynb_-Datei mit dem bereinigten Datensatz geladen und in X- und y-Datenrahmen aufgeteilt, bereit für den Modellierungsprozess.

## Eine Klassifikationskarte

Zuvor haben Sie die verschiedenen Optionen kennengelernt, die Sie bei der Klassifikation von Daten anhand des Cheat Sheets von Microsoft haben. Scikit-learn bietet ein ähnliches, aber detaillierteres Cheat Sheet, das Ihnen dabei helfen kann, Ihre Schätzer (ein anderer Begriff für Klassifikatoren) weiter einzugrenzen:

![ML Map from Scikit-learn](../../../../translated_images/de/map.e963a6a51349425a.webp)
> Tipp: [besuchen Sie diese Karte online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) und klicken Sie dem Pfad entlang, um die Dokumentation zu lesen.

### Der Plan

Diese Karte ist sehr hilfreich, sobald Sie ein klares Verständnis Ihrer Daten haben, da Sie den Pfaden zu einer Entscheidung folgen können:

- Wir haben >50 Stichproben
- Wir wollen eine Kategorie vorhersagen
- Wir haben gelabelte Daten
- Wir haben weniger als 100.000 Stichproben
- ✨ Wir können einen Linear SVC wählen
- Falls das nicht funktioniert, da wir numerische Daten haben
    - Können wir einen ✨ KNeighbors Classifier ausprobieren
      - Wenn das nicht funktioniert, versuchen Sie ✨ SVC und ✨ Ensemble Classifier

Dies ist eine sehr hilfreiche Vorgehensweise.

## Übung - Daten aufteilen

Folgen wir diesem Pfad, sollten wir zunächst einige Bibliotheken zum Verwenden importieren.

1. Importieren Sie die benötigten Bibliotheken:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Teilen Sie Ihre Trainings- und Testdaten auf:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC Klassifikator

Support-Vektor-Clustering (SVC) ist ein Teil der Familie der Support-Vektor-Maschinen (lernen Sie unten mehr über diese kennen). Bei dieser Methode können Sie einen „Kernel“ auswählen, um zu entscheiden, wie die Labels gruppiert werden. Der Parameter „C“ bezieht sich auf „Regularisierung“, also die Regulierung des Einflusses von Parametern. Der Kernel kann einer von [mehreren](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) sein; hier setzen wir ihn auf „linear“, um linear SVC zu verwenden. Die Wahrscheinlichkeit ist standardmäßig „false“; hier setzen wir sie auf „true“, um Wahrscheinlichkeitsabschätzungen zu erhalten. Wir setzen den Zufallszustand auf „0“, um die Daten zu mischen und Wahrscheinlichkeiten zu ermitteln.

### Übung - Anwenden eines Linear SVC

Beginnen Sie damit, ein Array von Klassifikatoren zu erstellen. Sie werden dieses Array nach und nach erweitern, während wir testen.

1. Beginnen Sie mit einem Linear SVC:

    ```python
    C = 10
    # Erstellen Sie verschiedene Klassifikatoren.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Trainieren Sie Ihr Modell mit dem Linear SVC und geben Sie einen Bericht aus:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Das Ergebnis ist recht gut:

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

## K-Neighbors Klassifikator

K-Neighbors gehört zur Familie der „Nachbarn“-ML-Methoden, die für überwachtes und unüberwachtes Lernen eingesetzt werden können. Bei dieser Methode wird eine vordefinierte Anzahl an Punkten erstellt, und die Daten werden um diese Punkte gruppiert, sodass verallgemeinerte Labels für die Daten vorhergesagt werden können.

### Übung - Anwenden des K-Neighbors Klassifikators

Der vorherige Klassifikator war gut und funktionierte gut mit den Daten, aber vielleicht können wir eine bessere Genauigkeit erreichen. Versuchen Sie einen K-Neighbors Klassifikator.

1. Fügen Sie Ihrem Klassifikator-Array eine Linie hinzu (setzen Sie ein Komma nach dem Linear SVC-Element):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Das Ergebnis ist etwas schlechter:

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

    ✅ Lernen Sie mehr über [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vektor-Klassifikatoren sind Teil der Familie der [Support-Vektor-Maschinen](https://wikipedia.org/wiki/Support-vector_machine), die für Klassifikations- und Regressionsaufgaben eingesetzt werden. SVMs „bilden Trainingsbeispiele als Punkte im Raum ab“, um den Abstand zwischen zwei Kategorien zu maximieren. Nachfolgende Daten werden in diesen Raum abgebildet, sodass ihre Kategorie vorhergesagt werden kann.

### Übung - Anwenden eines Support Vector Classifier

Versuchen wir eine etwas bessere Genauigkeit mit einem Support Vector Classifier.

1. Fügen Sie nach dem K-Neighbors-Element ein Komma ein und dann diese Zeile:

    ```python
    'SVC': SVC(),
    ```

    Das Ergebnis ist ziemlich gut!

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

    ✅ Lernen Sie mehr über [Support-Vektoren](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble-Klassifikatoren

Folgen wir dem Pfad bis zum Ende, obwohl der vorherige Test recht gut war. Versuchen wir einige „Ensemble-Klassifikatoren“, speziell Random Forest und AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Das Ergebnis ist sehr gut, besonders für Random Forest:

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

✅ Lernen Sie mehr über [Ensemble-Klassifikatoren](https://scikit-learn.org/stable/modules/ensemble.html)

Diese Methode des Maschinellen Lernens „kombiniert die Vorhersagen mehrerer Basis-Schätzer“, um die Qualität des Modells zu verbessern. In unserem Beispiel verwendeten wir Random Trees und AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), eine Mittelungsmethode, baut einen „Wald“ aus „Entscheidungsbäumen“, der mit Zufälligkeit versehen ist, um Overfitting zu vermeiden. Der Parameter n_estimators wird auf die Anzahl der Bäume gesetzt.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) passt einen Klassifikator an einen Datensatz an und passt dann Kopien dieses Klassifikators an denselben Datensatz an. Es fokussiert sich auf die Gewichte falsch klassifizierter Elemente und passt die Gewichtung für den nächsten Klassifikator an, um Fehler zu korrigieren.

---

## 🚀Herausforderung

Jede dieser Techniken hat eine große Anzahl von Parametern, die Sie anpassen können. Recherchieren Sie die Standardparameter jedes Verfahrens und denken Sie darüber nach, was die Anpassung dieser Parameter für die Qualität des Modells bedeuten würde.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Rückblick & Selbststudium

In diesen Lektionen gibt es viele Fachbegriffe, nehmen Sie sich also eine Minute, um [diese Liste](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) mit nützlichen Begriffen durchzugehen!

## Aufgabe

[Parameter spielen](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Haftungsausschluss**:  
Dieses Dokument wurde mithilfe des KI-Übersetzungsdienstes [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner Ursprungssprache ist als maßgebliche Quelle zu betrachten. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Nutzung dieser Übersetzung entstehen.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->