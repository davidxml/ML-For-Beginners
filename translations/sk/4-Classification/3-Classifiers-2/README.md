# Classifikátory kuchýň 2

V tejto druhej lekcii klasifikácie preskúmate ďalšie spôsoby klasifikácie numerických dát. Tiež sa naučíte o dôsledkoch výberu jedného klasifikátora namiesto druhého.

## [Prednáškový kvíz](https://ff-quizzes.netlify.app/en/ml/)

### Predpoklad

Predpokladáme, že ste dokončili predchádzajúce lekcie a máte v zložke `data` vyčistenú dátovú sadu s názvom _cleaned_cuisines.csv_ v koreňovom adresári tejto štvorlekciovej zložky.

### Príprava

Do vášho súboru _notebook.ipynb_ sme načítali vyčistenú dátovú sadu a rozdelili ju na dáta X a y, pripravené na proces budovania modelu.

## Mapa klasifikácie

Predtým ste sa dozvedeli o rôznych možnostiach klasifikácie dát podľa Microsoftovej pomôcky. Scikit-learn ponúka podobnú, no detailnejšiu pomôcku, ktorá vám môže ešte viac zúžiť výber odhadcov (ďalší pojem pre klasifikátory):

![ML Map from Scikit-learn](../../../../translated_images/sk/map.e963a6a51349425a.webp)
> Tip: [navštívte túto mapu online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) a klikajte postupne na ceste, aby ste si prečítali dokumentáciu.

### Plán

Táto mapa je veľmi užitočná, keď máte jasnú predstavu o svojich dátach, pretože môžete „prejsť“ jej cestami k rozhodnutiu:

- Máme >50 vzoriek
- Chceme predpovedať kategóriu
- Máme označené dáta
- Máme menej ako 100 tisíc vzoriek
- ✨ Môžeme zvoliť Lineárny SVC
- Ak to nefunguje, keďže máme numerické dáta
    - Môžeme skúsiť ✨ KNeighbors Classifier
      - Ak to nefunguje, skúste ✨ SVC a ✨ Ensemble Classifiers

Toto je veľmi užitočná cesta, ktorú sa oplatí sledovať.

## Cvičenie - rozdelenie dát

Podľa tejto cesty by sme mali začať importovaním niektorých knižníc.

1. Importujte potrebné knižnice:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Rozdeľte tréningové a testovacie dáta:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Lineárny SVC klasifikátor

Support-Vector clustering (SVC) je podmnožina rodiny Support-Vector strojov (SVM) v strojovom učení (dozviete sa o nich nižšie). V tejto metóde môžete zvoliť 'kernel' (jadro), ktoré rozhoduje o tom, ako sa labely zhluku. Parameter 'C' označuje 'regularizáciu', ktorá reguluje vplyv parametrov. Kernel môže byť jeden z [niekoľkých](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); tu ho nastavujeme na 'linear', aby sme využili lineárny SVC. Pravdepodobnosť je štandardne 'false'; tu ju nastavujeme na 'true', aby sme získali odhady pravdepodobnosti. Náhodný stav nastavujeme na '0', aby sa dáta zamiešali a získali pravdepodobnosti.

### Cvičenie - aplikujte lineárny SVC

Začnite vytvorením poľa klasifikátorov. Budete do neho postupne pridávať podľa testovania.

1. Začnite s Lineárnym SVC:

    ```python
    C = 10
    # Vytvorte rôzne klasifikátory.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Natrénujte model pomocou Lineárneho SVC a vytlačte správu:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Výsledok je celkom dobrý:

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

## K-Neighbors klasifikátor

K-Neighbors patrí do rodiny metód „neighbors“, ktoré možno použiť na riadené aj neriadené učenie. V tejto metóde sa vytvorí preddefinovaný počet bodov a okolo nich sa zhromažďujú dáta, aby sa mohli predpovedať všeobecné labely pre dáta.

### Cvičenie - aplikujte K-Neighbors klasifikátor

Predchádzajúci klasifikátor bol dobrý a dobre fungoval s dátami, ale možno môžeme dosiahnuť lepšiu presnosť. Skúste K-Neighbors klasifikátor.

1. Pridajte riadok do poľa klasifikátorov (pridajte čiarku za položku Lineárny SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Výsledok je mierne horší:

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

    ✅ Naučte sa o [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vector klasifikátory sú súčasťou rodiny [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine), ktoré sa používajú pre klasifikačné a regresné úlohy. SVM „mapuje tréningové príklady do bodov v priestore“, aby maximalizoval vzdialenosť medzi dvoma kategóriami. Následné dáta sa mapujú do tohto priestoru, aby sa mohla predpovedať ich kategória.

### Cvičenie - aplikujte Support Vector Classifier

Skúsme dosiahnuť trocha lepšiu presnosť pomocou Support Vector Classifier.

1. Pridajte čiarku za položku K-Neighbors a potom pridajte tento riadok:

    ```python
    'SVC': SVC(),
    ```

    Výsledok je celkom dobrý!

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

    ✅ Naučte sa o [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble Classifiers

Poďme pokračovať až do konca cesty, aj keď bol predchádzajúci test dosť dobrý. Skúsme „Ensemble Classifiers“, konkrétne Random Forest a AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Výsledok je veľmi dobrý, najmä pre Random Forest:

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

✅ Naučte sa o [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Táto metóda strojového učenia "kombinuje predpovede niekoľkých základných odhadcov" pre zlepšenie kvality modelu. V našom príklade sme použili náhodné stromy a AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), metóda priemerovania, vytvára „les“ rozhodovacích stromov obohatených náhodnosťou, aby sa predišlo preučeniu. Parameter n_estimators je nastavený na počet stromov.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) prispôsobí klasifikátor dátovej sade a potom prispôsobí kópie tohto klasifikátora danej sade. Zameriava sa na váhy nesprávne klasifikovaných položiek a upravuje prispôsobenie pre ďalší klasifikátor, aby to opravil.

---

## 🚀Výzva

Každá z týchto techník má veľké množstvo parametrov, ktoré môžete upravovať. Preskúmajte predvolené parametre každého z nich a zamyslite sa, čo by znamenalo ich upravovanie pre kvalitu modelu.

## [Po prednáške kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Recenzia a samostatné štúdium

Týchto lekcií je veľa odborných výrazov, preto si dajte chvíľu na preštudovanie [tohto zoznamu](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) užitočnej terminológie!

## Zadanie

[Hra s parametrami](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Upozornenie**:
Tento dokument bol preložený pomocou AI prekladateľskej služby [Co-op Translator](https://github.com/Azure/co-op-translator). Aj keď sa snažíme o presnosť, prosím, majte na pamäti, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Originálny dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre dôležité informácie sa odporúča profesionálny ľudský preklad. Nezodpovedáme za akékoľvek nepochopenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->