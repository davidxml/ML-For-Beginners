# Classifikátory kuchyní 2

V této druhé lekci klasifikace prozkoumáte více způsobů, jak klasifikovat číselná data. Také se dozvíte o důsledcích výběru jednoho klasifikátoru před druhým.

## [Přednáškový kvíz](https://ff-quizzes.netlify.app/en/ml/)

### Předpoklady

Předpokládáme, že jste dokončili předchozí lekce a máte vyčištěný dataset ve složce `data` pojmenovaný _cleaned_cuisines.csv_ v kořenové složce této sady čtyř lekcí.

### Příprava

Do vašeho souboru _notebook.ipynb_ jsme vložili vyčištěný dataset a rozdělili jej na datové rámce X a y, připravené pro tvorbu modelu.

## Mapa klasifikace

Dříve jste se seznámili s různými možnostmi při klasifikaci dat podle Microsoftova přehledu. Scikit-learn nabízí podobný, ale detailnější přehled, který vám může dále pomoci zúžit výběr odhadovačů (jiný termín pro klasifikátory):

![ML Map from Scikit-learn](../../../../translated_images/cs/map.e963a6a51349425a.webp)
> Tip: [navštivte tuto mapu online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) a klikáním postupujte po cestě k dokumentaci.

### Plán

Tato mapa je velmi užitečná, pokud máte jasný přehled o svých datech, protože můžete „jít“ po jejích cestách k rozhodnutí:

- Máme více než 50 vzorků
- Chceme předpovědět kategorii
- Máme označená data
- Máme méně než 100 tisíc vzorků
- ✨ Můžeme zvolit Linear SVC
- Pokud to nefunguje, protože máme číselná data
    - Můžeme zkusit ✨ KNeighbors Classifier
      - Pokud to nepomůže, zkusit ✨ SVC a ✨ Ensemble Classifiers

Toto je velmi užitečná cesta, kterou lze následovat.

## Cvičení – rozdělení dat

Podle této cesty bychom měli začít importem některých knihoven, které použijeme.

1. Naimportujte potřebné knihovny:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Rozdělte vaše trénovací a testovací data:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Klasifikátor Linear SVC

Support-Vector clustering (SVC) je metodou z rodiny Support-Vector machine ML technik (o těchto se dozvíte dále). V této metodě můžete zvolit „kernel“, který rozhoduje, jak seskupit štítky. Parametr „C“ odkazuje na „regularizaci“, která reguluje vliv parametrů. Kernel může být jeden z [několika](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); zde jej nastavujeme na ‚linear‘, abychom využili lineární SVC. Pravděpodobnost (probability) je standardně „false“; zde nastavujeme na „true“, aby bylo možné získat odhady pravděpodobností. Náhodný stav je nastaven na „0“ pro promíchání dat a získání pravděpodobností.

### Cvičení – aplikujte lineární SVC

Začněte vytvořením pole klasifikátorů. Postupně do něj budete přidávat další při testování.

1. Začněte s Linear SVC:

    ```python
    C = 10
    # Vytvořte různé klasifikátory.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Natrénujte svůj model pomocí Linear SVC a zobrazte report:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Výsledek je poměrně dobrý:

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

## Klasifikátor K-Neighbors

K-Neighbors je součástí rodiny ML metod „neighbors“, které lze použít pro jak učení s učitelem, tak bez učitele. V této metodě je vytvořen předem definovaný počet bodů a data jsou shromažďována kolem těchto bodů, aby bylo možné pro data předpovídat obecné štítky.

### Cvičení – aplikujte klasifikátor K-Neighbors

Předchozí klasifikátor byl dobrý a dobře fungoval s daty, ale možná můžeme dosáhnout lepší přesnosti. Vyzkoušejte klasifikátor K-Neighbors.

1. Přidejte řádek do pole klasifikátorů (přidejte čárku za položku Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Výsledek je trochu horší:

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

    ✅ Naučte se o [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vector klasifikátory jsou součástí rodiny [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) ML metod, které se používají pro klasifikaci a regresní úlohy. SVM mapují tréninkové příklady do bodů ve vesmíru za účelem maximalizace vzdálenosti mezi dvěma kategoriemi. Následná data jsou pak do tohoto prostoru namapována tak, aby bylo možné předpovědět jejich kategorii.

### Cvičení – aplikujte Support Vector Classifier

Zkuste dosáhnout trochu lepší přesnosti pomocí Support Vector Classifier.

1. Přidejte čárku za položku K-Neighbors a potom přidejte tento řádek:

    ```python
    'SVC': SVC(),
    ```

    Výsledek je docela dobrý!

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

    ✅ Naučte se o [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble Classifiers

Pojďme jít cestou až do konce, i když předchozí test byl poměrně dobrý. Vyzkoušejme některé 'Ensemble Classifiers', konkrétně Random Forest a AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Výsledek je velmi dobrý, zvláště u Random Forest:

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

✅ Naučte se o [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Tato metoda strojového učení „kombinuje předpovědi několika základních odhadovačů“ ke zlepšení kvality modelu. V našem příkladu jsme použili Random Trees a AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), metoda průměrování, vytváří „les“ rozhodovacích stromů obohacených náhodností, aby se předešlo přetrénování. Parametr n_estimators je nastaven na počet stromů.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) učí klasifikátor na datové sadě a potom učí jeho kopie na téže datové sadě. Zaměřuje se na váhy nesprávně klasifikovaných položek a upravuje přizpůsobení pro další klasifikátor, aby chyby korigoval.

---

## 🚀 Výzva

Každá z těchto technik má mnoho parametrů, které můžete ladit. Prozkoumejte výchozí parametry každé a zamyslete se, co by změna těchto parametrů znamenala pro kvalitu modelu.

## [Po přednáškový kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Přehled a samostudium

V těchto lekcích je hodně odborných výrazů, takže si udělejte chvíli na zopakování [tohoto seznamu](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) užitečné terminologie!

## Zadání

[Hra s parametry](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Upozornění**:  
Tento dokument byl přeložen pomocí služby automatického překladu [Co-op Translator](https://github.com/Azure/co-op-translator). Přestože usilujeme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Originální dokument v původním jazyce by měl být považován za závazný zdroj. Pro zásadní informace se doporučuje profesionální lidský překlad. Nejsme odpovědni za jakékoliv nedorozumění nebo mylné výklady vzniklé používáním tohoto překladu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->