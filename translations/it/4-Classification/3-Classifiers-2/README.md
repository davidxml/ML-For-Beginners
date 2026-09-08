# Classificatori di cucina 2

In questa seconda lezione di classificazione, esplorerai altri modi per classificare i dati numerici. Imparerai anche le ramificazioni della scelta di un classificatore rispetto a un altro.

## [Quiz pre-lezione](https://ff-quizzes.netlify.app/en/ml/)

### Prerequisiti

Presumiamo che tu abbia completato le lezioni precedenti e che tu abbia un dataset pulito nella cartella `data` chiamato _cleaned_cuisines.csv_ nella radice di questa cartella con 4 lezioni.

### Preparazione

Abbiamo caricato il tuo file _notebook.ipynb_ con il dataset pulito e lo abbiamo diviso in dataframes X e y, pronti per il processo di costruzione del modello.

## Una mappa di classificazione

In precedenza, hai imparato le varie opzioni disponibili per classificare i dati utilizzando la cheat sheet di Microsoft. Scikit-learn offre una cheat sheet simile, ma più granulare, che può aiutarti ulteriormente a restringere i tuoi stimatori (un altro termine per classificatori):

![ML Map from Scikit-learn](../../../../translated_images/it/map.e963a6a51349425a.webp)
> Suggerimento: [visita questa mappa online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) e clicca lungo il percorso per leggere la documentazione.

### Il piano

Questa mappa è molto utile una volta che hai una chiara comprensione dei tuoi dati, poiché puoi "percorrere" i suoi sentieri fino a una decisione:

- Abbiamo >50 campioni
- Vogliamo prevedere una categoria
- Abbiamo dati etichettati
- Abbiamo meno di 100K campioni
- ✨ Possiamo scegliere un Linear SVC
- Se non funziona, dato che abbiamo dati numerici
    - Possiamo provare un ✨ KNeighbors Classifier 
      - Se non funziona, prova ✨ SVC e ✨ Ensemble Classifiers

Questo è un percorso molto utile da seguire.

## Esercizio - dividi i dati

Seguendo questo percorso, dovremmo iniziare importando alcune librerie da utilizzare.

1. Importa le librerie necessarie:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

2. Dividi i dati di addestramento e di test:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Classificatore Linear SVC

Il Support-Vector Clustering (SVC) è un membro della famiglia di tecniche di ML Support-Vector Machines (scopri di più su queste sotto). In questo metodo, puoi scegliere un "kernel" per decidere come clusterizzare le etichette. Il parametro 'C' riguarda la "regolarizzazione" che regola l'influenza dei parametri. Il kernel può essere uno di [diversi](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); qui lo impostiamo su 'linear' per assicurarci di sfruttare l'SVC lineare. La probabilità ha come valore predefinito 'false'; qui lo impostiamo su 'true' per raccogliere stime di probabilità. Impostiamo lo stato casuale su '0' per mescolare i dati e ottenere probabilità.

### Esercizio - applica un Linear SVC

Inizia creando un array di classificatori. Lo arricchirai progressivamente man mano che testeremo.

1. Inizia con un Linear SVC:

    ```python
    C = 10
    # Crea diversi classificatori.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Allena il tuo modello usando il Linear SVC e stampa un report:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Il risultato è piuttosto buono:

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

## Classificatore K-Neighbors

K-Neighbors fa parte della famiglia di metodi di ML "neighbors", che possono essere usati sia per apprendimento supervisionato che non supervisionato. In questo metodo, si crea un numero predefinito di punti e i dati vengono raccolti intorno a questi punti in modo che possano essere previsti etichette generalizzate per i dati.

### Esercizio - applica il classificatore K-Neighbors

Il classificatore precedente era buono e ha funzionato bene con i dati, ma forse possiamo ottenere una migliore accuratezza. Prova un classificatore K-Neighbors.

1. Aggiungi una riga al tuo array di classificatori (aggiungi una virgola dopo l'elemento Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Il risultato è un po’ peggiore:

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

    ✅ Scopri di più su [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

I classificatori Support-Vector fanno parte della famiglia di metodi di ML [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) usati per compiti di classificazione e regressione. Gli SVM "mappano gli esempi di addestramento in punti nello spazio" per massimizzare la distanza tra due categorie. I dati successivi vengono mappati in questo spazio in modo che la loro categoria possa essere predetta.

### Esercizio - applica un Support Vector Classifier

Proviamo ad ottenere un’accuratezza un po’ migliore con un Support Vector Classifier.

1. Aggiungi una virgola dopo l’elemento K-Neighbors, quindi aggiungi questa riga:

    ```python
    'SVC': SVC(),
    ```

    Il risultato è piuttosto buono!

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

    ✅ Scopri di più su [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble Classifiers

Seguiamo il percorso fino alla fine, anche se il test precedente è stato piuttosto buono. Proviamo alcuni 'Ensemble Classifiers', nello specifico Random Forest e AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Il risultato è molto buono, specialmente per Random Forest:

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

✅ Scopri di più su [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Questo metodo di Machine Learning "combina le previsioni di diversi stimatori di base" per migliorare la qualità del modello. Nel nostro esempio, abbiamo usato Random Trees e AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), un metodo di media, costruisce una "foresta" di "alberi decisionali" infusi di casualità per evitare l’overfitting. Il parametro n_estimators è impostato sul numero di alberi.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) adatta un classificatore a un dataset e quindi adatta copie di quel classificatore allo stesso dataset. Si concentra sui pesi degli elementi classificati erroneamente e regolando l’adattamento per il classificatore successivo per correggere.

---

## 🚀Sfida

Ognuna di queste tecniche ha un gran numero di parametri che puoi modificare. Cerca i parametri predefiniti di ciascuno e pensa a cosa significherebbe modificare questi parametri per la qualità del modello.

## [Quiz post-lezione](https://ff-quizzes.netlify.app/en/ml/)

## Revisione e Autoapprendimento

Ci sono molte terminologie in queste lezioni, quindi prenditi un momento per rivedere [questa lista](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) di terminologia utile!

## Compito

[Gioca con i parametri](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica AI [Co-op Translator](https://github.com/Azure/co-op-translator). Pur impegnandoci per garantire accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua originaria deve essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda la traduzione professionale effettuata da un esperto umano. Non siamo responsabili per eventuali fraintendimenti o interpretazioni errate derivanti dall’uso di questa traduzione.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->