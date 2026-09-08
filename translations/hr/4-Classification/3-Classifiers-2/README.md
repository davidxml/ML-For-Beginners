# Klasifikatori kuhinja 2

U ovoj drugoj lekciji o klasifikaciji, istražit ćete više načina za klasifikaciju numeričkih podataka. Također ćete naučiti o posljedicama odabira jednog klasifikatora u odnosu na drugi.

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

### Pretpostavka

Pretpostavljamo da ste završili prethodne lekcije i imate očišćeni skup podataka u svojoj `data` mapi pod nazivom _cleaned_cuisines.csv_ u korijenu ove mape s 4 lekcije.

### Priprema

Učitani su vam _notebook.ipynb_ datoteka s očišćenim skupom podataka i podijelili smo ga u X i y datafrejmove, spremne za proces izgradnje modela.

## Karta klasifikacije

Prije ste naučili o različitim opcijama koje imate kod klasificiranja podataka koristeći Microsoftovu varalicu. Scikit-learn nudi sličnu, ali detaljniju varalicu koja može dodatno pomoći pri sužavanju vaših procjenitelja (drugi izraz za klasifikatore):

![ML karta iz Scikit-learn](../../../../translated_images/hr/map.e963a6a51349425a.webp)
> Savjet: [posjetite ovu kartu online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) i klikajte duž staze da pročitate dokumentaciju.

### Plan

Ova karta je vrlo korisna kad imate jasno razumijevanje svojih podataka, jer možete 'šetati' njenim stazama do odluke:

- Imamo >50 uzoraka
- Želimo predvidjeti kategoriju
- Imamo označene podatke
- Imamo manje od 100K uzoraka
- ✨ Možemo odabrati Linearni SVC
- Ako to ne uspije, budući da imamo numeričke podatke
    - Možemo pokušati sa ✨ KNeighbors klasifikatorom
      - Ako to ne uspije, pokušajte ✨ SVC i ✨ Ensemble klasifikatore

Ovo je vrlo korisna staza za praćenje.

## Vježba - podijelite podatke

Prateći ovu stazu, trebali bismo započeti uvozom nekih biblioteka za korištenje.

1. Uvezite potrebne biblioteke:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Podijelite svoje podatke za trening i test:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Linearni SVC klasifikator

Support-Vector clustering (SVC) je član obitelji Support-Vector strojeva za ML tehnike (saznajte više o njima dolje). U ovoj metodi možete odabrati 'kernel' kojim odlučujete kako grupirati oznake. Parametar 'C' odnosi se na 'regularizaciju' koja regulira utjecaj parametara. Kernel može biti jedan od [više](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); ovdje smo ga postavili na 'linearni' da osiguramo korištenje linearnog SVC-a. Vjerojatnost je prema zadanim postavkama 'false'; ovdje smo je postavili na 'true' da prikupimo procjene vjerojatnosti. Postavili smo random_state na '0' da promiješamo podatke za dobivanje vjerojatnosti.

### Vježba - primijenite linearni SVC

Započnite stvaranjem niza klasifikatora. Postupno ćete dodavati u ovaj niz dok testiramo.

1. Započnite s linearnim SVC:

    ```python
    C = 10
    # Izradite različite klasifikatore.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Naučite svoj model koristeći Linearni SVC i ispišite izvještaj:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Rezultat je prilično dobar:

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

## K-Neighbors klasifikator

K-Neighbors je dio obitelji "neighbors" ML metoda, koje se mogu koristiti za nadzirano i nenadzirano učenje. U ovoj metodi se stvara unaprijed definirani broj točaka i podaci se skupljaju oko tih točaka tako da se mogu predvidjeti generalizirane oznake za podatke.

### Vježba - primijenite K-Neighbors klasifikator

Prethodni klasifikator je bio dobar i dobro je radio s podacima, ali možda možemo postići bolju točnost. Isprobajte K-Neighbors klasifikator.

1. Dodajte liniju u svoj niz klasifikatora (dodajte zarez nakon stavke Linearni SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Rezultat je malo lošiji:

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

    ✅ Saznajte o [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vector klasifikatori su dio obitelji [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) ML metoda koje se koriste za klasifikacijske i regresijske zadatke. SVM-ovi "mapiraju primjere treninga u točke u prostoru" kako bi maksimalizirali udaljenost između dvije kategorije. Sljedeći podaci se mapiraju u taj prostor kako bi se mogla predvidjeti njihova kategorija.

### Vježba - primijenite Support Vector Classifier

Pokušajmo dobiti malo bolju točnost s Support Vector Classifierom.

1. Dodajte zarez nakon stavke K-Neighbors, pa zatim dodajte ovu liniju:

    ```python
    'SVC': SVC(),
    ```

    Rezultat je prilično dobar!

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

    ✅ Saznajte o [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble klasifikatori

Slijedimo stazu do samog kraja, iako je prethodni test bio prilično dobar. Isprobajmo neke 'Ensemble klasifikatore', konkretno Random Forest i AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Rezultat je vrlo dobar, osobito za Random Forest:

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

✅ Saznajte o [Ensemble klasifikatorima](https://scikit-learn.org/stable/modules/ensemble.html)

Ova metoda strojnog učenja "kombinira predviđanja nekoliko osnovnih procjenitelja" kako bi poboljšala kvalitetu modela. U našem smo primjeru koristili Random Trees i AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), metoda prosječavanja, gradi 'šumu' 'odlučnih stabala' obogaćenu slučajnostima kako bi se izbjeglo prekomjerno prilagođavanje. Parametar n_estimators postavljen je na broj stabala.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) trenira klasifikator na skupu podataka, a zatim trenira kopije tog klasifikatora na istom skupu podataka. Fokusira se na težine pogrešno klasificiranih elemenata i prilagođava fit sljedećem klasifikatoru da ispravi.

---

## 🚀Izazov

Svaka od ovih tehnika ima veliki broj parametara koje možete mijenjati. Istražite zadane parametre svakog i razmislite što bi mijenjanje tih parametara značilo za kvalitetu modela.

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalna studija

U ovim lekcijama ima puno žargona, pa odvojite minutu da pregledate [ovaj popis](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) korisne terminologije!

## Zadatak

[Igra s parametrima](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Napomena**:  
Ovaj je dokument preveden pomoću AI usluge prevođenja [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba se smatrati službenim i autoritativnim izvorom. Za kritične informacije preporučuje se profesionalni ljudski prijevod. Ne preuzimamo odgovornost za nesporazume ili kriva tumačenja koja proizlaze iz uporabe ovog prijevoda.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->