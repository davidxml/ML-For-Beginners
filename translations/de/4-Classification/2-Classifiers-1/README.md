# Küchenklassifikatoren 1

In dieser Lektion verwenden Sie den Datensatz, den Sie aus der letzten Lektion gespeichert haben, voller ausgewogener, sauberer Daten rund um Küchen.

Sie werden diesen Datensatz mit einer Vielzahl von Klassifikatoren verwenden, um _eine bestimmte nationale Küche basierend auf einer Zutatenliste vorherzusagen_. Dabei lernen Sie mehr über einige der Möglichkeiten, wie Algorithmen für Klassifizierungsaufgaben genutzt werden können.

## [Vorlesungsquiz](https://ff-quizzes.netlify.app/en/ml/)
# Vorbereitung

Vorausgesetzt, Sie haben [Lektion 1](../1-Introduction/README.md) abgeschlossen, stellen Sie sicher, dass eine Datei _cleaned_cuisines.csv_ im Stammordner `/data` für diese vier Lektionen existiert.

## Übung – eine nationale Küche vorhersagen

1. Arbeiten Sie im Ordner _notebook.ipynb_ dieser Lektion und importieren Sie diese Datei zusammen mit der Pandas-Bibliothek:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```
  
    Die Daten sehen so aus:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |

1. Importieren Sie nun mehrere weitere Bibliotheken:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```
  
1. Teilen Sie die X- und y-Koordinaten in zwei DataFrames für das Training auf. `cuisine` kann das Label-DataFrame sein:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```
  
    Es sieht so aus:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```
  
1. Löschen Sie die Spalte `Unnamed: 0` und die `cuisine`-Spalte mittels `drop()`. Speichern Sie den Rest der Daten als trainierbare Merkmale:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```
  
    Ihre Merkmale sehen so aus:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Jetzt sind Sie bereit, Ihr Modell zu trainieren!

## Auswahl des Klassifikators

Da Ihre Daten nun sauber und bereit für das Training sind, müssen Sie entscheiden, welchen Algorithmus Sie für die Aufgabe verwenden möchten. 

Scikit-learn gruppiert Klassifikation unter Überwachtem Lernen (Supervised Learning), und in dieser Kategorie finden Sie viele Klassifizierungsmethoden. [Die Vielfalt](https://scikit-learn.org/stable/supervised_learning.html) ist auf den ersten Blick ziemlich überwältigend. Die folgenden Methoden beinhalten Klassifikationstechniken:

- Lineare Modelle
- Support Vector Machines
- Stochastischer Gradientenabstieg
- Nächste Nachbarn
- Gaußsche Prozesse
- Entscheidungsbäume
- Ensemble-Methoden (Voting-Klassifikator)
- Multi-Class- und Multioutput-Algorithmen (Multiclass- und Multilabel-Klassifikation, Multiclass-Multioutput-Klassifikation)

> Sie können auch [Neuronale Netze zur Klassifikation von Daten einsetzen](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), aber das liegt außerhalb des Umfangs dieser Lektion.

### Welchen Klassifikator wählen?

Welchen Klassifikator sollten Sie wählen? Oft hilft es, mehrere auszuprobieren und nach einem guten Ergebnis zu suchen. Scikit-learn bietet einen [direkten Vergleich](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) auf einem erstellten Datensatz, in dem KNeighbors, SVC auf zwei Arten, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB und QuadraticDiscriminantAnalysis verglichen und die Ergebnisse visualisiert werden:  

![Vergleich von Klassifikatoren](../../../../translated_images/de/comparison.edfab56193a85e7f.webp)  
> Plots erstellt aus der Dokumentation von Scikit-learn

> AutoML löst dieses Problem elegant, indem es diese Vergleiche in der Cloud durchführt und Ihnen erlaubt, den besten Algorithmus für Ihre Daten auszuwählen. Probieren Sie es [hier](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Ein besserer Ansatz

Eine bessere Methode als wildes Raten ist es, den Ideen auf diesem herunterladbaren [ML-Spickzettel](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) zu folgen. Hier entdecken wir, dass wir für unser Multiclass-Problem einige Optionen haben:

![Spickzettel für Multiclass-Probleme](../../../../translated_images/de/cheatsheet.07a475ea444d2223.webp)  
> Ein Ausschnitt des Algorithmus-Spickzettels von Microsoft, der Multiclass-Klassifikationsmöglichkeiten beschreibt

✅ Laden Sie diesen Spickzettel herunter, drucken Sie ihn aus und hängen Sie ihn an Ihre Wand!

### Begründung

Versuchen wir anhand gegebener Einschränkungen die verschiedenen Ansätze zu begründen:

- **Neuronale Netze sind zu schwergewichtig.** Angesichts unseres sauberen, aber kleinen Datensatzes und der Tatsache, dass wir lokal über Notebooks trainieren, sind neuronale Netze zu schwergewichtig für diese Aufgabe.
- **Kein Zwei-Klassen-Klassifikator.** Wir nutzen keinen Zwei-Klassen-Klassifikator, daher ist One-vs-All ausgeschlossen.
- **Entscheidungsbaum oder logistische Regression sind möglich.** Ein Entscheidungsbaum könnte funktionieren oder logistische Regression für Multiclass-Daten.
- **Multiclass Boosted Decision Trees lösen ein anderes Problem.** Die multiclass boosted decision trees sind eher für nicht-parametrische Aufgaben geeignet, z.B. für Ranglisten und daher für uns nicht nützlich.

### Verwendung von Scikit-learn 

Wir werden Scikit-learn verwenden, um unsere Daten zu analysieren. Es gibt allerdings viele Möglichkeiten, logistische Regression mit Scikit-learn anzuwenden. Sehen Sie sich die [Parameter](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) an, die man übergeben kann.

Im Wesentlichen sind zwei wichtige Parameter - `multi_class` und `solver` - festzulegen, wenn wir Scikit-learn bitten, eine logistische Regression durchzuführen. Der Wert von `multi_class` bestimmt eine bestimmte Verhaltensweise. Der Wert von `solver` gibt an, welchen Algorithmus man verwenden möchte. Nicht alle Solver sind mit allen `multi_class`-Werten kompatibel.

Laut der Dokumentation verwendet der Trainingsalgorithmus im Multiclass-Fall:

- **das one-vs-rest (OvR) Schema**, wenn die Option `multi_class` auf `ovr` gesetzt ist
- **die Kreuzentropie-Verlustfunktion**, wenn die Option `multi_class` auf `multinomial` gesetzt ist. (Die Option `multinomial` wird zurzeit nur von den Solvern ‘lbfgs’, ‘sag’, ‘saga’ und ‘newton-cg’ unterstützt.)"

> 🎓 Das 'Schema' kann entweder 'ovr' (one-vs-rest) oder 'multinomial' sein. Da logistische Regression eigentlich für binäre Klassifikation ausgelegt ist, erlauben diese Schemata eine bessere Handhabung von Multiclass-Aufgaben. [Quelle](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 Der 'Solver' ist definiert als "der Algorithmus zur Lösung des Optimierungsproblems". [Quelle](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn bietet diese Tabelle, um zu zeigen, wie die Solvers unterschiedliche Herausforderungen verschiedener Datenstrukturen bewältigen:

![Solver](../../../../translated_images/de/solvers.5fc648618529e627.webp)

## Übung – Daten aufteilen

Wir konzentrieren uns bei unserem ersten Trainingsversuch auf die logistische Regression, da Sie diese Methode kürzlich in einer vorherigen Lektion gelernt haben. Teilen Sie Ihre Daten in Trainings- und Testgruppen mittels `train_test_split()` auf:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```
  
## Übung – logistische Regression anwenden

Da Sie die Multiclass-Variante verwenden, müssen Sie ein _Schema_ wählen und den _Solver_ festlegen. Verwenden Sie LogisticRegression mit einer Multiclass-Konfiguration und dem Solver **liblinear**, um zu trainieren.

1. Erstellen Sie eine logistische Regression mit `multi_class` auf `ovr` und dem Solver `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```
  
    ✅ Probieren Sie einen anderen Solver wie `lbfgs` aus, der oft als Standard eingestellt ist.

    > Hinweis: Verwenden Sie die Pandas-Funktion [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html), um Ihre Daten bei Bedarf zu glätten.

    Die Genauigkeit ist gut – über **80%**!

1. Sie können dieses Modell in Aktion sehen, indem Sie eine einzelne Datenzeile (#50) testen:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```
  
    Das Ergebnis wird ausgegeben:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```
  
   ✅ Probieren Sie eine andere Zeilennummer und überprüfen Sie die Ergebnisse.
1. Wenn Sie tiefer graben, können Sie die Genauigkeit dieser Vorhersage überprüfen:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Das Ergebnis wird ausgegeben – Indische Küche ist die beste Vermutung mit hoher Wahrscheinlichkeit:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Können Sie erklären, warum das Modell ziemlich sicher ist, dass es sich um indische Küche handelt?

1. Erhalten Sie weitere Details, indem Sie einen Klassifikationsbericht ausdrucken, wie Sie es in den Regressionslektionen getan haben:

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

## 🚀Herausforderung

In dieser Lektion haben Sie Ihre bereinigten Daten verwendet, um ein Machine-Learning-Modell zu erstellen, das basierend auf einer Reihe von Zutaten eine nationale Küche vorhersagen kann. Nehmen Sie sich Zeit, um die vielen Optionen, die Scikit-learn zur Klassifizierung von Daten bietet, zu lesen. Vertiefen Sie sich in das Konzept des 'solvers', um zu verstehen, was hinter den Kulissen passiert.

## [Post-Lecture Quiz](https://ff-quizzes.netlify.app/en/ml/)

## Rückblick & Selbststudium

Vertiefen Sie sich ein wenig mehr in die Mathematik hinter der logistischen Regression in [dieser Lektion](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)

## Aufgabe

[Studieren Sie die Solver](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir auf Genauigkeit achten, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner Ursprungssprache ist als maßgebliche Quelle zu betrachten. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Nutzung dieser Übersetzung entstehen.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->