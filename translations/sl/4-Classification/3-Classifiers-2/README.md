# Razvrščevalci kuhinj 2

V tej drugi lekciji o razvrščanju boste raziskali več načinov razvrščanja številskih podatkov. Naučili se boste tudi posledic izbire enega razvrščevalca namesto drugega.

## [Kviz pred predavanjem](https://ff-quizzes.netlify.app/en/ml/)

### Predpogoj

Predvidevamo, da ste opravili prejšnje lekcije in imate v vaši mapi `data` očiščeno podatkovno zbirko imenovano _cleaned_cuisines.csv_ v korenu te 4-lekcijske mape.

### Priprava

Naložili smo vašo datoteko _notebook.ipynb_ z očiščeno podatkovno zbirko in jo razdelili v podatkovni okvir X in y, pripravljena za proces gradnje modela.

## Zemljevid razvrščanja

Prej ste spoznali različne možnosti, ki jih imate pri razvrščanju podatkov z uporabo Microsoftovega prevarantskega lista. Scikit-learn ponuja podoben, a bolj podroben prevarantski list, ki lahko še dodatno pomaga zožiti vaše ocenovalce (drug izraz za razvrščevalce):

![ML Map from Scikit-learn](../../../../translated_images/sl/map.e963a6a51349425a.webp)
> Namig: [obiščite ta zemljevid na spletu](https://scikit-learn.org/stable/tutorial/machine_learning_map/) in klikajte po poti, da preberete dokumentacijo.

### Načrt

Ta zemljevid je zelo koristen, ko imate jasen vpogled v svoje podatke, saj lahko ‘hodite’ po njegovih poteh do odločitve:

- Imamo >50 vzorcev
- Želimo napovedati kategorijo
- Imamo označene podatke
- Imamo manj kot 100.000 vzorcev
- ✨ Lahko izberemo Linear SVC
- Če to ne deluje, ker imamo številske podatke
    - Lahko poskusimo s ✨ KNeighbors Classifier
      - Če tudi to ne deluje, poskusimo ✨ SVC in ✨ Ensemble Classifiers

To je zelo uporabna slediti.

## Vaja - razdelite podatke

Sledi tej poti, začnemo z uvozom nekaterih knjižnic, ki jih bomo uporabili.

1. Uvoz potrebnih knjižnic:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Razdelite svoje trening in testne podatke:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Linearni SVC razvrščevalec

Support-Vector clustering (SVC) je del družine metod strojnega učenja Support-Vector machines (SVM) (o njih več spodaj). Pri tej metodi lahko izberete 'jedro' (kernel), da določite, kako zgrupirate oznake. Parameter 'C' se nanaša na 'regularizacijo', ki uravnava vpliv parametrov. Jedro je lahko eno izmed [več](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); tukaj ga nastavimo na 'linear', da zagotovimo uporabo linearnega SVC. Privzeto je Probability nastavljeno na 'false'; tukaj smo ga nastavili na 'true', da zberemo ocene verjetnosti. Za naključno stanje smo nastavili '0', da premešamo podatke za verjetnosti.

### Vaja - uporabite linearen SVC

Začnite z ustvarjanjem tabele razvrščevalcev. Postopoma boste dodajali elemente v to tabelo, ko boste testirali.

1. Začnite z Linear SVC:

    ```python
    C = 10
    # Ustvari različne klasifikatorje.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Naučite svoj model z uporabo Linear SVC in izpišite poročilo:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Rezultat je precej dober:

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

## K-najbližjih sosedov razvrščevalec

K-najbližjih sosedov spada v družino metod strojnega učenja "neighbors", ki se lahko uporabljajo tako za nadzorovano kot nenadzorovano učenje. Pri tej metodi se ustvari preddefinirano število točk in podatki se zbirajo okoli teh točk, tako da je mogoče napovedati posplošene oznake za podatke.

### Vaja - uporabite K-najbližjih sosedov

Prejšnji razvrščevalec je bil dober in je deloval dobro s podatki, vendar morda lahko dosežemo boljšo natančnost. Poskusite s K-najbližjih sosedov.

1. Dodajte vrstico v svojo tabelo razvrščevalcev (dodajte vejico za Linear SVC elementom):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Rezultat je malo slabši:

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

    ✅ Spoznajte [K-najbližjih sosedov](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector razvrščevalec

Support-Vector razvrščevalci so del družine [Support-Vector Machines](https://wikipedia.org/wiki/Support-vector_machine) metod strojnega učenja, ki se uporabljajo za razvrščanje in regresijo. SVM “preslika trening primere v točke v prostoru”, da maksimira razdaljo med dvema kategorijama. Kasnejši podatki so preslikani v ta prostor, da je mogoče napovedati njihovo kategorijo.

### Vaja - uporabite Support Vector razvrščevalec

Poskusimo doseči malo boljšo natančnost s Support Vector razvrščevalcem.

1. Dodajte vejico za K-najbližjih sosedov elementom in nato dodajte to vrstico:

    ```python
    'SVC': SVC(),
    ```

    Rezultat je precej dober!

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

    ✅ Spoznajte [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble razvrščevalci

Pojdimo do konca poti, čeprav je bil prejšnji test precej dober. Poskusimo nekaj 'Ensemble razvrščevalcev', natančneje Random Forest in AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Rezultat je zelo dober, še posebej za Random Forest:

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

✅ Spoznajte [Ensemble razvrščevalce](https://scikit-learn.org/stable/modules/ensemble.html)

Ta metoda strojnega učenja "združuje napovedi več osnovnih ocenjevalcev", da izboljša kakovost modela. V našem primeru smo uporabili Random Trees in AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), metoda povprečenja, gradi 'gosto' drevo 'odločilnih dreves' vpeto z naključnostjo, da prepreči prekomerno prileganje (overfitting). Parameter n_estimators je nastavljen na število dreves.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) prilagodi razvrščevalec na podatkovno množico in nato prilagodi kopije tega razvrščevalca na isti podatkovni množici. Osredotoča se na uteži nepravilno razvrščenih elementov in prilagaja prileganje za naslednjega razvrščevalca, da to popravi.

---

## 🚀Izziv

Vsaka od teh tehnik ima veliko parametrov, ki jih lahko spreminjate. Raziskujte privzete nastavitve vsakega in razmislite, kaj bi pomenilo prilagajanje teh parametrov za kakovost modela.

## [Kviz po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

V teh lekcijah je veliko strokovnega besedišča, zato si vzemite trenutek za pregled [tega seznama](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) uporabne terminologije!

## Domača naloga

[Igra s parametri](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Zavrnitev odgovornosti**:
Ta dokument je bil preveden z uporabo AI prevajalske storitve [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatski prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku velja za avtoritativni vir. Za pomembne informacije priporočamo strokovni prevod s strani človeka. Za morebitna nesporazume ali napačne razlage, ki izhajajo iz uporabe tega prevoda, ne prevzemamo odgovornosti.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->