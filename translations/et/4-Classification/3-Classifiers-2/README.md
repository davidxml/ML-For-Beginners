# Köögikliidid 2

Selles teises klassifitseerimise õppetükis uurite rohkem viise, kuidas numbrilisi andmeid klassifitseerida. Samuti õpite, millised on tagajärjed ühe klassifikaatori valimisel teise asemel.

## [Eel-loengu viktoriin](https://ff-quizzes.netlify.app/en/ml/)

### Eeltingimus

Eeldame, et olete lõpetanud eelnevad õppetükid ja teil on puhas andmestik kaustas `data` nimega _cleaned_cuisines.csv_ selle nelja õppetüki kausta juures.

### Ettevalmistus

Oleme laadinud teie _notebook.ipynb_ faili puhta andmestikuga ning jaganud selle X ja y andmeraamistikeks, valmis mudeli loomise protsessiks.

## Klassifitseerimise kaart

Varem õppisite, millised võimalused teil on andmete klassifitseerimiseks Microsofti petutabelit kasutades. Scikit-learn pakub sarnast, kuid detailsemat petutabelit, mis aitab teil veelgi täpsemalt kitsendada oma hinnanguid (teine nimetus klassifikaatorite kohta):

![ML Map from Scikit-learn](../../../../translated_images/et/map.e963a6a51349425a.webp)
> Näpunäide: [külastage seda kaarti veebis](https://scikit-learn.org/stable/tutorial/machine_learning_map/) ja klõpsake rada, et lugeda dokumentatsiooni.

### Plaan

See kaart on väga abiks, kui teil on andmetest selge arusaam, sest saate selle radadel "liikuda" otsuse tegemiseks:

- Meil on >50 proovi
- Soovime ennustada kategooriat
- Meil on märgistatud andmed
- Meil on vähem kui 100K proovi
- ✨ Võime valida Linear SVC
- Kui see ei tööta, kuna meil on numbrilised andmed
    - Võime proovida ✨ KNeighbors Klassifikaatorit
      - Kui see ei tööta, proovime ✨ SVC ja ✨ Ensemble Klassifikaatoreid

See on väga kasulik rada järgida.

## Harjutus - andmete jagamine

Sellele rajale järgides peaksime alustama vajalike raamatukogude importimisega.

1. Importige vajalikud raamatukogud:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Jagage oma treening- ja testandmed:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Lineaarne SVC klassifikaator

Toetava vektori klasterdamine (SVC) kuulub Toetava vektori masinate perekonda ML tehnikaid (loetle allpool lisaks). Selles meetodis saate valida 'tuuma' ehk kerneli, et otsustada, kuidas siltide klastreid moodustada. Parameeter 'C' viitab 'regularisatsioonile', mis reguleerib parameetrite mõju. Kerneli valikud on [mitmed](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); siin määrame selle väärtuseks 'linear', et kasutada lineaarset SVC-d. Tõenäosus on vaikimisi 'false'; siin seame selle 'true' väärtuseks, et saada tõenäosuse hinnanguid. Määrasime juhuse seisundi '0', et andmeid segada tõenäosuste saamiseks.

### Harjutus - rakenda lineaarset SVC-d

Alusta klassifikaatorite massiivi loomisega. Selle massiivi lisad järjest, kui testime.

1. Alusta Linear SVC-d kasutades:

    ```python
    C = 10
    # Loo erinevaid klassifikaatoreid.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Treeni mudelit Linear SVC-ga ja prindi raport:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Tulemus on päris hea:

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

## K-naabrid klassifikaator

K-naabrid kuuluvad "naabrid" perekonda ML meetodites, mida saab kasutada nii juhendatud kui juhendamata õppes. Selles meetodis luuakse ette määratud arv punkte ja andmed kogutakse nende punktide ümber, et andmete jaoks üldistatud silte ennustada.

### Harjutus - rakenda K-naabrite klassifikaator

Eelmine klassifikaator oli hea ja töötas andmetega hästi, kuid võib-olla saame parema täpsuse. Proovi K-naabrite klassifikaatorit.

1. Lisa rida klassifikaatorite massiivi (pane koma Linear SVC rea järel):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Tulemus on natuke kehvem:

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

    ✅ Õpi lähemalt [K-naabritest](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Toetava vektori klassifikaator

Toetava vektori klassifikaatorid on Toetava vektori masina (Support-Vector Machine) ML meetodite perekond, mida kasutatakse klassifitseerimise ja regressiooni ülesannetes. SVM-id „määratlevad treeningnäited ruumipunktidena”, et maksimeerida kahe kategooria vahelist kaugust. Järgnevaid andmeid kaardistatakse sellesse ruumi, et ennustada nende kategooriat.

### Harjutus - rakenda Toetava vektori klassifikaator

Proovime veidi paremat täpsust Toetava vektori klassifikaatoriga.

1. Lisa koma K-naabrite rea järel, seejärel lisa see rida:

    ```python
    'SVC': SVC(),
    ```

    Tulemus on üsna hea!

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

    ✅ Õpi lähemalt [Toetavast vektorist](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ansambli klassifikaatorid

Järgneme rajale lõpuni, kuigi eelmine test oli juba päris hea. Proovime ansambli klassifikaatoreid, eriti Random Forestit ja AdaBoosti:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Tulemus on väga hea, eriti Random Foresti puhul:

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

✅ Õpi lähemalt [Ansambli klassifikaatoritest](https://scikit-learn.org/stable/modules/ensemble.html)

See Masinõppe meetod „ühendab mitme baas-hinnangu tegija ennustused”, et mudeli kvaliteeti parandada. Meie näites kasutasime juhuslikke puid ja AdaBoosti.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), keskmistamismeetod, ehitab 'puude metsa', mis on juhuslikkusega infundeeritud otsustuspuud, et vältida üleõppimist. n_estimators parameeter määrab puudede arvu.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) sobitab klassifikaatori andmestikule ja seejärel sobitab selle klassifikaatori koopiad sama andmestikuga. See keskendub valesti klassifitseeritud esemete kaaludele ja kohandab järgmise klassifikaatori sobivust, et vea parandada.

---

## 🚀Väljakutse

Iga selle meetodi puhul on palju parameetreid, mida saate häälestada. Uurige iga meetodi vaikeparameetreid ja mõelge, mida nende parameetrite häälestamine mudeli kvaliteedi jaoks tähendaks.

## [Pärast loengu viktoriin](https://ff-quizzes.netlify.app/en/ml/)

## Ülevaade ja iseseisev õpe

Nendes õppetükkides on palju erialatermineid, seega võtke hetk ja vaadake üle [see nimekiri](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) kasulikust terminoloogiast!

## Kodune ülesanne

[Parameetrite mäng](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vastutusest loobumine**:
See dokument on tõlgitud kasutades tehisintellektil põhinevat tõlke teenust [Co-op Translator](https://github.com/Azure/co-op-translator). Kuigi püüame täpsust, tuleb arvestada, et automaatsed tõlked võivad sisaldada vigu või ebatäpsusi. Originaaldokument selle emakeeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitatakse kasutada professionaalset inimtõlget. Me ei vastuta ühegi arusaamatuse ega tõlgenduse eest, mis võivad selle tõlke kasutamisest tekkida.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->