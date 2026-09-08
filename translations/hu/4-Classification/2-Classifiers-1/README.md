# Konyhák osztályozói 1

Ebben a leckében a legutóbbi leckéből mentett, kiegyensúlyozott, tiszta adatokat tartalmazó, konyhákról szóló adathalmazt fogod használni.

Ezt az adathalmazt különféle osztályozókkal fogod használni, hogy _megjósolj egy adott nemzeti konyhát egy alapanyagcsoport alapján_. Miközben ezt teszed, többet megtudsz az algoritmusok osztályozási feladatokra való felhasználásának néhány módjáról.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)
# Előkészület

Feltételezve, hogy befejezted az [1. leckét](../1-Introduction/README.md), győződj meg róla, hogy a _cleaned_cuisines.csv_ fájl létezik a négy leckét lefedő gyökér `/data` mappában.

## Gyakorlat - egy nemzeti konyha előrejelzése

1. Az ebben a leckében található _notebook.ipynb_ mappában importáld be a fájlt a Pandas könyvtárral együtt:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Az adatok így néznek ki:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Most importálj be még több könyvtárat:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Oszd szét az X és y koordinátákat két adattáblára tanításhoz. A `cuisine` lehet a címke adattábla:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Így fog kinézni:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Dobd el az `Unnamed: 0` oszlopot és a `cuisine` oszlopot `drop()` meghívásával. A maradék adatot mentsd el tanításra alkalmas jellemzőkként:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    A jellemzőid így néznek ki:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Most készen állsz a modell betanítására!

## Az osztályozó kiválasztása

Most, hogy az adatok tiszták és készek a tanításra, el kell döntened, melyik algoritmust használod a feladathoz.

A Scikit-learn az osztályozást a Felügyelt tanulás (Supervised Learning) alá sorolja, és ezen a területen sokféle módszert találsz. [A választék](https://scikit-learn.org/stable/supervised_learning.html) először meglehetősen zavaró lehet. Az alábbi módszerek mind tartalmaznak osztályozási technikákat:

- Lineáris modellek
- Támogató vektor gépek (Support Vector Machines)
- Stokasztikus gradiens csökkenés (Stochastic Gradient Descent)
- Legközelebbi szomszédok (Nearest Neighbors)
- Gaussi folyamatok
- Döntési fák
- Együttes módszerek (voting Classifier)
- Többosztályos és többkimenetű algoritmusok (többosztályos és többcímkés osztályozás, többosztályos-többkimenetű osztályozás)

> [Neurális hálókat](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification) is használhatsz az adatok osztályozására, de ez nem része ennek a leckének.

### Melyik osztályozót válasszam?

Tehát, melyik osztályozót válaszd? Gyakran jó módszer több osztályozó tesztelése, hogy melyik ad jó eredményt. A Scikit-learn kínál egy [oldal-by-oldal összehasonlítást](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) egy létrehozott adathalmazon, összehasonlítva KNeighbors, az SVC két módját, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB és QuadraticDiscrinationAnalysis eredményeit, vizualizálva:

![osztályozók összehasonlítása](../../../../translated_images/hu/comparison.edfab56193a85e7f.webp)
> Grafikonok a Scikit-learn dokumentációjából

> Az AutoML ezt a problémát szépen megoldja azáltal, hogy ezeket az összehasonlításokat a felhőben futtatja, lehetővé téve a legjobb algoritmus kiválasztását az adataidhoz. Próbáld ki [itt](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Egy jobb megközelítés

Egy jobb, mint találomra próbálgatni, ha követed az ötleteket ezen a letölthető [ML Cheat sheet-en](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Itt megtudjuk, hogy szakítós problémánk esetén néhány lehetőségünk van:

![csalólap többosztályos problémákhoz](../../../../translated_images/hu/cheatsheet.07a475ea444d2223.webp)
> A Microsoft algoritmus csalólapjának egy része, a többosztályos osztályozási opciók részletezése

✅ Töltsd le ezt a csalólapot, nyomtasd ki, és tedd ki a faladra!

### Érvelés

Nézzük meg, hogy hogyan gondolkodhatunk különböző megközelítésekről az adott feltételek mellett:

- **A neurális hálók túl nehezek**. Adataink tiszták, de minimálisak, és mivel helyileg, notebookokban futtatjuk a tanítást, a neurális hálók túl nehézsúlyúak ehhez a feladathoz.
- **Nincs kétosztályos osztályozó**. Nem használunk kétosztályos osztályozót, ezért kizárjuk az egy-az-összes elleni módszert.
- **A döntési fa vagy a logisztikus regresszió működhet**. Egy döntési fa működhet, illetve a logisztikus regresszió többosztályos adatokra.
- **A többosztályos Boosted Decision Tree más problémát old meg**. Ez a módszer inkább nem-paraméteres feladatokra alkalmas, például rangsorok készítésére, így nem hasznos nekünk.

### Scikit-learn használata

A Scikit-learn-t használjuk adatelemzésre. Viszont számos módon lehet logisztikus regressziót alkalmazni benne. Nézd meg a [paramétereket](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Alapvetően két fontos paraméter van – a `multi_class` és a `solver` –, amelyet meg kell adni a Scikit-learn-nek logisztikus regresszió kéréséhez. A `multi_class` egy bizonyos viselkedést határoz meg. A `solver` az alkalmazott algoritmust jelöli. Nem minden megoldó párosítható minden `multi_class` értékkel.

A dokumentáció szerint többosztályos esetben az algoritmus:

- **Az egy-az-összes (OvR) sémát használja**, ha a `multi_class` opció `ovr` értékre van állítva
- **A keresztentrópia veszteséget használja**, ha a `multi_class` beállítás `multinomial`. (Jelenleg a `multinomial` opciót csak az ‘lbfgs’, ‘sag’, ‘saga’ és ‘newton-cg’ megoldók támogatják.)"

> 🎓 Az itt lévő 'séma' lehet 'ovr' (egy-az-összes) vagy 'multinomial'. Mivel a logisztikus regresszió eredetileg bináris osztályozásra készült, ezek a sémák lehetővé teszik számára a többosztályos feladatok hatékonyabb kezelését. [forrás](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 A 'solver' az "optimalizációs probléma megoldására használt algoritmust" jelenti. [forrás](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

A Scikit-learn az alábbi táblázattal mutatja be, miként kezelik a különböző megoldók az eltérő adatszerkezeteket:

![megoldók](../../../../translated_images/hu/solvers.5fc648618529e627.webp)

## Gyakorlat - oszd fel az adatot

Az első tanítási próbánkhoz a logisztikus regresszióra koncentrálhatunk, hiszen ezt az előző leckében tanultad.
Oszd fel az adataidat tanító és tesztelő csoportokra a `train_test_split()` meghívásával:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Gyakorlat - alkalmazz logisztikus regressziót

Mivel többosztályos esetet használsz, ki kell választanod, hogy milyen _sémát_ és milyen _megoldót_ használj. Használd a LogisticRegression-t többosztályos beállítással és **liblinear** megoldóval a tanításhoz.

1. Hozz létre egy logisztikus regressziót, ahol `multi_class` értéke `ovr`, a `solver` pedig `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Próbálj ki egy másik megoldót, például az alapértelmezett gyakran használt `lbfgs`-t.

    > Megjegyzés: használd a Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) függvényét, ha szükséges az adatok kiterítése.

    Az pontosság jó, több mint **80%**!

1. Megnézheted ezt a modellt működés közben, ha megvizsgálsz egy adat sort (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Az eredmény kiírásra kerül:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Próbálj ki egy másik sorszámot és nézd meg az eredményeket
1. Mélyebbre ásva ellenőrizheted ennek az előrejelzésnek a pontosságát:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Az eredmény kiírásra kerül - az indiai konyha a legvalószínűbb találat:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ El tudod magyarázni, miért biztos a modell abban, hogy ez egy indiai konyha?

1. Részletesebb információért nyomtasd ki a besorolási jelentést, ahogy a regressziós leckékben is tettél:

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

## 🚀Kihívás

Ebben a leckében a megtisztított adatoddal építettél egy gépi tanulási modellt, amely egy sor összetevő alapján meg tudja jósolni a nemzeti konyhát. Szánj egy kis időt arra, hogy áttekintsd a Scikit-learn számos lehetőségét az adatok osztályozására. Mélyedj el a 'solver' fogalmában, hogy megértsd, mi zajlik a háttérben.

## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés & Önálló tanulás

Merülj el egy kicsit mélyebben a logisztikus regresszió mögötti matematikában ebben a [leckében](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)

## Házi feladat

[A solver-ek tanulmányozása](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Nyilatkozat**:
Ezt a dokumentumot az AI fordító szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével fordítottuk. Bár a pontosságra törekszünk, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum a saját nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt a szakmai emberi fordítás igénybevétele. Nem vállalunk felelősséget a fordítás használatából eredő félreértésekért vagy félreértelmezésekért.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->