# Virtuvės klasifikatoriai 2

Šioje antroje klasifikacijos pamokoje jūs išnagrinėsite daugiau būdų, kaip klasifikuoti skaitmeninius duomenis. Taip pat sužinosite apie pasekmes, pasirenkant vieną klasifikatorių vietoj kito.

## [Prieš paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

### Reikalavimai prieš pradedant

Tarkime, kad jūs jau baigėte ankstesnes pamokas ir turite išvalytą duomenų rinkinį savo `data` aplanke, pavadintą _cleaned_cuisines.csv_ šiame 4 pamokų aplanko šaknyje.

### Paruošimas

Mes įkėlėme jūsų _notebook.ipynb_ failą su išvalytais duomenimis ir padalinome juos į X ir y duomenų rinkinius, paruoštus modelio kūrimo procesui.

## Klasifikacijos žemėlapis

Anksčiau sužinojote apie įvairias galimybes klasifikuoti duomenis naudodami Microsoft sukurtą pagalbinį lapą. Scikit-learn siūlo panašų, bet detalesnį pagalbinį lapą, kuris gali dar labiau padėti susiaurinti jūsų estimatorius (kitas klasifikatorių pavadinimas):

![ML Map from Scikit-learn](../../../../translated_images/lt/map.e963a6a51349425a.webp)
> Patarimas: [apsilankykite šiame žemėlapyje internete](https://scikit-learn.org/stable/tutorial/machine_learning_map/) ir spustelėkite kelio taškus, kad perskaitytumėte dokumentaciją.

### Planas

Šis žemėlapis labai naudingas, kai jūs aiškiai suprantate savo duomenis, nes galite „žingsniuoti“ po jo kelius iki sprendimo:

- Turime >50 pavyzdžių
- Norime prognozuoti kategoriją
- Turime pažymėtus duomenis
- Turime mažiau nei 100 tūkst. pavyzdžių
- ✨ Galime pasirinkti Linear SVC
- Jei tai neveikia, kadangi turime skaitmeninius duomenis
    - Galime išbandyti ✨ KNeighbors klasifikatorių
      - Jei ir tai neveikia, išbandykite ✨ SVC ir ✨ Ensemble klasifikatorius

Tai labai naudingas kelias, kurio verta laikytis.

## Užduotis - padalinkite duomenis

Sekdami šiuo keliu, turėtume pradėti nuo reikalingų bibliotekų importavimo.

1. Importuokite reikiamas bibliotekas:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Padalinkite treniruočių ir testinius duomenis:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC klasifikatorius

Paramos vektorių klasterizacija (SVC) yra Paramų vektorių mašinų (Support-Vector machines) šeimos mašininio mokymosi metodas (daugiau apie juos skaitykite žemiau). Šiame metode galite pasirinkti „branduolį“ (kernel), kuris nusprendžia, kaip klasterizuoti etiketes. Parametras „C“ reiškia „reguliarizaciją“, kuri reguliuoja parametrų įtaką. Branduolys gali būti vienas iš [kelių](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); čia mes nustatome jį į „linear“, kad naudotume linijinį SVC. Tikimybė pagal nutylėjimą yra „false“; čia ją nustatome į „true“, kad gautume tikimybių įvertinimus. Atsitiktinumo būsena nustatyta į „0“, kad duomenys būtų permaišyti ir gautume tikimybes.

### Užduotis - pritaikykite linijinį SVC

Pradėkite kurdami klasifikatorių masyvą. Jį palaipsniui papildysite, kai testuosime.

1. Pradėkite nuo Linear SVC:

    ```python
    C = 10
    # Sukurkite skirtingus klasifikatorius.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Apmokykite savo modelį naudodami Linear SVC ir atspausdinkite ataskaitą:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Rezultatas gana geras:

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

## K-Neighbors klasifikatorius

K-Neighbors priklauso „kaimynų“ (neighbors) šeimai mašininio mokymosi metodų, kuriuos galima naudoti tiek prižiūrimam, tiek neprižiūrimam mokymuisi. Šiame metode sukuriamas iš anksto apibrėžtas taškų skaičius, o duomenys surenkami aplink šiuos taškus, kad būtų galima prognozuoti bendrines etiketes duomenims.

### Užduotis - pritaikykite K-Neighbors klasifikatorių

Ankstesnis klasifikatorius buvo geras ir gerai veikė su duomenimis, bet galbūt galime pasiekti geresnį tikslumą. Išbandykite K-Neighbors klasifikatorių.

1. Įtraukite eilutę į savo klasifikatorių masyvą (po Linear SVC elemento pridėkite kablelį):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Rezultatas šiek tiek prastesnis:

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

    ✅ Sužinokite apie [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector klasifikatorius

Paramų vektorių klasifikatoriai yra mašininio mokymosi metodų, skirtų klasifikacijai ir regresijai, šeimos dalis [Paramų vektorių mašinų](https://wikipedia.org/wiki/Support-vector_machine) (SVM). SVM „žemėlapiuoja treniruočių pavyzdžius į erdvės taškus“, kad maksimaliai padidintų atstumą tarp dviejų kategorijų. Tolimesni duomenys taip pat žemėlapiuojami į šią erdvę, kad būtų galima prognozuoti jų kategorijas.

### Užduotis - pritaikykite Support Vector klasifikatorių

Pabandykime gauti šiek tiek geresnį tikslumą su Support Vector klasifikatoriumi.

1. Po K-Neighbors elemento pridėkite kablelį ir tada pridėkite šią eilutę:

    ```python
    'SVC': SVC(),
    ```

    Rezultatas gana geras!

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

    ✅ Sužinokite apie [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble klasifikatoriai

Sekime kelią iki galo, nors ankstesnis testas buvo gana geras. Išbandykime kai kuriuos „Ensemble klasifikatorius“, ypatingai Random Forest ir AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Rezultatas labai geras, ypač Random Forest atveju:

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

✅ Sužinokite apie [Ensemble klasifikatorius](https://scikit-learn.org/stable/modules/ensemble.html)

Šis mašininio mokymosi metodas „jungia kelių bazinių estimatorių prognozes“, kad pagerintų modelio kokybę. Mūsų pavyzdyje naudojome Atsitiktinius Medžius (Random Trees) ir AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest) – vidurkinimo metodas, kuris kuria „mišką“ „sprendimų medžių“, įterptų su atsitiktinumu, kad būtų išvengta perdavimo. n_estimators parametras nurodo medžių skaičių.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) pritaiko klasifikatorių duomenų rinkiniui, po to pritaiko šio klasifikatoriaus kopijas tam pačiam duomenų rinkiniui. Jis sutelkia dėmesį į neteisingai klasifikuotų elementų svorius ir koreguoja tinkamumą kitam klasifikatoriui.

---

## 🚀Iššūkis

Kiekviena iš šių technikų turi daugybę parametrų, kuriuos galite koreguoti. Išnagrinėkite kiekvienos numatytuosius parametrus ir pagalvokite, ką jų pakoregavimas reikštų modelio kokybei.

## [Po paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

## Apžvalga ir savarankiškas mokymasis

Šiose pamokose yra daug terminų, todėl skirkite minutę peržiūrėti [šį terminų sąrašą](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott)!

## Užduotis 

[Parametrų žaidimas](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Atsakomybės apribojimas**:
Šis dokumentas buvo išverstas naudojant dirbtinio intelekto vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatizuoti vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turi būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojamas profesionalus žmogaus vertimas. Mes neatsakome už bet kokius nesusipratimus ar neteisingus interpretavimus, kylančius dėl šio vertimo naudojimo.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->