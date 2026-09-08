# Konyha osztályozók 2

Ebben a második osztályozási leckében többféle módot fogsz felfedezni a numerikus adatok osztályozására. Megtanulod azt is, milyen következményekkel jár, ha az egyik osztályozót választod a másik helyett.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

### Előfeltétel

Feltételezzük, hogy elvégezted az előző leckéket, és van egy megtisztított adatállományod a `data` mappádban, amelynek neve _cleaned_cuisines.csv_, és amely ebben a 4 leckés mappában a gyökérkönyvtárban található.

### Előkészület

Betöltöttük a _notebook.ipynb_ fájlodat a megtisztított adatokkal, és szétválasztottuk őket X és y adattáblákra, készen az modellépítési folyamathoz.

## Egy osztályozási térkép

Korábban megtanultad a különféle opciókat, amikor az adatokat osztályozod, a Microsoft csalólapja alapján. A Scikit-learn hasonló, de még részletesebb csalólapot kínál, amely tovább segíthet leszűkíteni az osztályozóidat (más szóval becslőket):

![ML Map from Scikit-learn](../../../../translated_images/hu/map.e963a6a51349425a.webp)
> Tipp: [látogasd meg ezt a térképet online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) és kattints a útvonalon, hogy elolvasd a dokumentációt.

### A terv

Ez a térkép nagyon hasznos, amikor tisztán érted az adataidat, mert végig tudsz "sétálni" az útvonalain a döntéshez:

- Több mint 50 mintánk van
- Kategóriát akarunk előre jelezni
- Címkézett adataink vannak
- Kevesebb mint 100 ezer mintánk van
- ✨ Választhatunk egy Linear SVC-t
- Ha ez nem működik, mivel numerikus adataink vannak
    - Próbálkozhatunk egy ✨ KNeighbors osztályozóval
      - Ha ez sem működik, próbáljuk az ✨ SVC-t és az ✨ Ensemble osztályozókat

Ez egy nagyon hasznos útvonal, amit követni lehet.

## Gyakorlat - oszd meg az adatokat

Ezt az utat követve kezdjük azzal, hogy importálunk néhány könyvtárat használatra.

1. Importáld a szükséges könyvtárakat:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Oszd meg a tanító és teszt adataidat:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Lineáris SVC osztályozó

A Support-Vector clustering (SVC) a Support-Vector gépek családjába tartozik, amelyek gépi tanulási technikák (róluk lentebb tanulhatsz). Ebben a módszerben választhatsz "kernelt", hogy meghatározd, hogyan csoportosítod a címkéket. A 'C' paraméter a "regularizációra" utal, amely szabályozza a paraméterek hatását. A kernel lehet egyik [több](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); itt 'lineáris'-ra állítottuk, hogy lineáris SVC-t használjunk. A valószínűséget alapértelmezés szerint 'hamis'-ra állítja; itt 'igaz'-ra állítjuk, hogy valószínűségi becsléseket kapjunk. A random állapotot '0'-ra állítjuk a véletlenszerű sorbarendezéshez, hogy valószínűségeket kapjunk.

### Gyakorlat - alkalmazz lineáris SVC-t

Kezdj egy osztályozó tömb létrehozásával. Fokozatosan fogsz bővíteni ezen a tömbön, ahogy tesztelünk.

1. Kezdd egy Linear SVC-vel:

    ```python
    C = 10
    # Különböző osztályozók létrehozása.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Tanítsd meg a modelled a Lineáris SVC-vel, és nyomtass ki egy jelentést:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Az eredmény egészen jó:

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

## K-legközelebbi szomszéd osztályozó

A K-Neighbors a "szomszédok" családjába tartozik a gépi tanulási módszereknek, amelyeket felügyelt és felügyelet nélküli tanulásra is lehet használni. Ebben a módszerben előre meghatározott számú pontot hozunk létre, és az adatokat ezek köré gyűjtjük össze, hogy általánosított címkéket tudjunk előre jelezni.

### Gyakorlat - alkalmazd a K-Neighbors osztályozót

Az előző osztályozó jól működött az adatokkal, de talán jobb pontosságot érhetünk el. Próbáld ki a K-Neighbors osztályozót.

1. Adj egy sort az osztályozó tömbhöz (tegyél vesszőt a Linear SVC elem után):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Az eredmény kissé rosszabb:

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

    ✅ Ismerd meg a [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors) módszert

## Support Vector osztályozó

A Support-Vector osztályozók a [Support-Vector gép](https://wikipedia.org/wiki/Support-vector_machine) családjába tartozó gépi tanulási módszerek, amelyeket osztályozási és regressziós feladatokra használnak. Az SVM-ek "leképezik a tanító példákat térbeli pontokká", hogy maximalizálják a két kategória közötti távolságot. A későbbi adatokat ebbe a térbe képezik le, hogy előre jelezhessék a kategóriájukat.

### Gyakorlat - alkalmazz Support Vector osztályozót

Próbáljunk egy kicsit jobb pontosságot egy Support Vector osztályozóval.

1. Tegyél egy vesszőt a K-Neighbors elem után, majd add hozzá ezt a sort:

    ```python
    'SVC': SVC(),
    ```

    Az eredmény egészen jó!

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

    ✅ Ismerd meg a [Support-Vektorokat](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble osztályozók

Kövessük az utat a legvégig, még akkor is, ha az előző teszt nagyon jó volt. Próbáljunk ki néhány 'Ensemble' osztályozót, konkrétan Random Forest-et és AdaBoost-ot:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Az eredmény nagyon jó, különösen a Random Forest esetén:

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

✅ Ismerd meg az [Ensemble osztályozókat](https://scikit-learn.org/stable/modules/ensemble.html)

Ez a gépi tanulási módszer "több alapbecslő előrejelzését kombinálja", hogy javítsa a modell minőségét. A példánkban Random Trees-t és AdaBoost-ot használtunk.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), egy átlagoló módszer, amely 'döntési fákat' épít fel véletlenszerűség beépítésével az túlilleszkedés elkerülése érdekében. Az n_estimators paramétert a fák számára állítjuk.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) egy olyan osztályozót illeszt egy adatállományra, majd ugyanazt többször illeszti, fókuszálva a helytelenül osztályozott minták súlyaira, és a következő osztályozó javításához állítja az illeszkedést.

---

## 🚀Kihívás

Mindegyik technikának nagy számú paramétere van, amelyeket módosíthatsz. Kutass utána ezek alapértelmezett paramétereinek, és gondold át, milyen hatásuk lehet ezek módosításának a modell minőségére.

## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés & önálló tanulás

Sok szakkifejezés található ezekben a leckékben, így szánj egy percet arra, hogy áttekintsd [ezt a listát](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) a hasznos terminológiákról!

## Feladat

[Paraméter játék](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Nyilatkozat**:
Ez a dokumentum az [Co-op Translator](https://github.com/Azure/co-op-translator) AI fordító szolgáltatásával készült. Bár az pontosságra törekszünk, kérjük, vegye figyelembe, hogy az automatikus fordítások tartalmazhatnak hibákat vagy pontatlanságokat. Az eredeti dokumentum az anyanyelvén tekintendő hivatalos forrásnak. Kritikus információk esetén javasolt professzionális, emberi fordítást igénybe venni. Nem vállalunk felelősséget az ebből eredő félreértésekért vagy félreértelmezésekért.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->