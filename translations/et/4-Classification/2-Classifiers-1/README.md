# Köökide klassifikaatorid 1

Selles õppetükis kasutad andmekogumit, mille salvestasid eelmisest õppetükist, täis tasakaalustatud, puhast infot kõikide köökide kohta.

Sa kasutad seda andmekogumit mitmesuguste klassifikaatoritega, et _ennustada kindlat rahvuskööki koostisosade grupi põhjal_. Selle käigus õpid rohkem meetodite kohta, kuidas algoritme saab kasutada klassifitseerimisülesannetes.

## [Loengu eelne viktoriin](https://ff-quizzes.netlify.app/en/ml/)
# Ettevalmistus

Eeldades, et oled lõpetanud [Õppetüki 1](../1-Introduction/README.md), veendu, et puhastatud fail _cleaned_cuisines.csv_ on olemas juurkaustas `/data` selleks neljaks õppetükiks.

## Harjutus - rahvusköögi ennustamine

1. Töötades selle õppetüki kaustas _notebook.ipynb_, impordi see fail koos Pandas teegiga:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Andmed näevad välja sellised:

|     | Unnamed: 0 | köök   | mandel | angelika | aniis | aniisi_seeme | õun   | õunabrandy   | aprikoos | armagnac | ... | viski | valge_leib | valge_vein | täistera_nisu_jahu | vein | puit | bataat | pärm  | jogurt | suvikõrvits |
| --- | ---------- | ------ | ------ | -------- | ----- | ------------ | ----- | ------------ | -------- | -------- | --- | ------ | ---------- | ---------- | ------------------ | ---- | ---- | ------ | ----- | ------ | ----------- |
| 0   | 0          | india   | 0      | 0        | 0     | 0            | 0     | 0            | 0        | 0        | ... | 0      | 0          | 0          | 0                  | 0    | 0    | 0      | 0     | 0      | 0           |
| 1   | 1          | india   | 1      | 0        | 0     | 0            | 0     | 0            | 0        | 0        | ... | 0      | 0          | 0          | 0                  | 0    | 0    | 0      | 0     | 0      | 0           |
| 2   | 2          | india   | 0      | 0        | 0     | 0            | 0     | 0            | 0        | 0        | ... | 0      | 0          | 0          | 0                  | 0    | 0    | 0      | 0     | 0      | 0           |
| 3   | 3          | india   | 0      | 0        | 0     | 0            | 0     | 0            | 0        | 0        | ... | 0      | 0          | 0          | 0                  | 0    | 0    | 0      | 0     | 0      | 0           |
| 4   | 4          | india   | 0      | 0        | 0     | 0            | 0     | 0            | 0        | 0        | ... | 0      | 0          | 0          | 0                  | 0    | 0    | 0      | 0     | 1      | 0           |

1. Nüüd impordi veel mõningaid teeke:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Jaga X ja y koordinaadid kaheks DataFrame’iks treenimiseks. `küök` võib olla sildi DataFrame:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    See näeb välja selline:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Eemalda sarjad `Unnamed: 0` ja `küök` `drop()` abil. Salvestage ülejäänud andmed treenimiseks sobivate tunnustena:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Sinu tunnused näevad välja sellised:

|      | mandel | angelika | aniis | aniisi_seeme | õun | õunabrandy | aprikoos | armagnac | artemisia | artišokk |  ... | viski | valge_leib | valge_vein | täistera_nisu_jahu | vein | puit | bataat | pärm | jogurt | suvikõrvits |
| ---: | ------: | -------: | -----: | -----------: | ---: | ----------: | --------: | -------: | ---------: | ---------: | ---: | -----: | ---------: | ---------: | -----------------: | ----: | ----: | ------: | ----: | ------: | ----------: |
|    0 |       0 |        0 |      0 |            0 |   0 |           0 |         0 |        0 |          0 |          0 |  ... |      0 |          0 |          0 |                  0 |    0 |    0 |       0 |     0 |       0 |           0 | 0 |
|    1 |       1 |        0 |      0 |            0 |   0 |           0 |         0 |        0 |          0 |          0 |  ... |      0 |          0 |          0 |                  0 |    0 |    0 |       0 |     0 |       0 |           0 | 0 |
|    2 |       0 |        0 |      0 |            0 |   0 |           0 |         0 |        0 |          0 |          0 |  ... |      0 |          0 |          0 |                  0 |    0 |    0 |       0 |     0 |       0 |           0 | 0 |
|    3 |       0 |        0 |      0 |            0 |   0 |           0 |         0 |        0 |          0 |          0 |  ... |      0 |          0 |          0 |                  0 |    0 |    0 |       0 |     0 |       0 |           0 | 0 |
|    4 |       0 |        0 |      0 |            0 |   0 |           0 |         0 |        0 |          0 |          0 |  ... |      0 |          0 |          0 |                  0 |    0 |    0 |       0 |     0 |       1 |           0 | 0 |

Nüüd oled valmis oma mudelit treenima!

## Klassifikaatori valimine

Nüüd, kui sinu andmed on puhtad ja treenimiseks valmis, pead otsustama, millist algoritmi kasutada.

Scikit-learn ühendab klassifikatsioonid Juhendatud Õppimise alla ning selles kategoorias leiad palju võimalusi klassifikatsiooniks. [Valik](https://scikit-learn.org/stable/supervised_learning.html) on esmapilgul üsna segadusttekitav. Järgmised meetodid sisaldavad kõik klassifikatsioonitehnikaid:

- Lineaarsed mudelid
- Tugivektormasinad
- Stohhastiline gradientlangus
- Lähimad naabrid
- Gaussi protsessid
- Otsustuspuud
- Ansamblimeetodid (hääletuskassifikaator)
- Multiklasse ja multiväljundiga algoritmid (multiklasse ja multilabel klassifikatsioon, multiklasse-multiväljundiga klassifikatsioon)

> Võid kasutada ka [neurvõrke andmete klassifitseerimiseks](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), kuid see jääb selle õppetüki teemast väljapoole.

### Millise klassifikaatori valida?

Millise klassifikaatori valida? Sageli testitakse ennast läbi erinevate variantide ja otsitakse head tulemust. Scikit-learn pakub [kõrvuti võrdlust](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) loodud andmestikul, võrdeldes KNeighbors, kahte SVC võimalust, GaussianProcessClassifierit, DecisionTreeClassifierit, RandomForestClassifierit, MLPClassifierit, AdaBoostClassifierit, GaussianNB ja QuadraticDiscrinationAnalysist ning kuvades tulemused visuaalselt: 

![klassifikaatorite võrdlus](../../../../translated_images/et/comparison.edfab56193a85e7f.webp)
> Graafikud on genereeritud Scikit-learn dokumentatsioonis

> AutoML lahendab selle probleemi kenasti, käivitades need võrdlused pilves, võimaldades valida parima algoritmi oma andmete jaoks. Proovi seda [siin](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Parem lähenemine

Parem kui lihtsalt meelevaldselt arvata, on järgida mõtteid selle allalaaditava [ML petulehe](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) põhjal. Siin saame teada, et meie mitmeklassilise probleemi puhul on mitmeid valikuid:

![mitmeklassilise probleemi petuleht](../../../../translated_images/et/cheatsheet.07a475ea444d2223.webp)
> Microsofti algoritmipetulehe osa, mis kirjeldab mitmeklassilise klassifikatsiooni võimalusi

✅ Laadi see petuleht alla, prindi välja ja kleebi seinale!

### Loogika

Vaatame, kas suudame loogiliselt läbi mõelda erinevad lähenemised antud piirangute valguses:

- **Neurvõrgud on liiga mahukad**. Arvestades meie puhast, kuid minimaalseid andmeid ning asjaolu, et me treenime lokaalselt notebookide kaudu, on neurvõrgud selleks ülesandeks liiga rasked.
- **Ühe- või kahekordse klassifikaatori puudumine**. Me ei kasuta kaheklassi klassifikaatorit, seega välistatakse one-vs-all.
- **Otsustuspuu või logistiline regressioon võiks töötada**. Otsustuspuu võiks töötada või logistiline regressioon mitmeklassiliste andmete jaoks.
- **Mitmeklassilised tugevdussõelad lahendavad teistsuguse probleemi**. Mitmeklassiline tugevdusotsustuspuu sobib rohkem mitteparametrilisteks ülesanneteks, näiteks järjestuse loomiseks, seega see ei ole meile kasulik.

### Scikit-learn kasutamine

Analüüsime andmeid Scikit-learniga. Siiski on logistilise regressiooni kasutamiseks Scikit-learnis palju viise. Vaata [parameetrite dokumentatsiooni](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Põhimõtteliselt on kaks olulist parameetrit - `multi_class` ja `solver` -, mida pead määrama, kui palud Scikit-learnil logistilist regressiooni kasutada. `multi_class` määrab teatud käitumise, `solver` määrab algoritmi. Kõik lahendajad ei sobi kõigi `multi_class` väärtustega.

Dokumentatsiooni järgi, mitmeklassilises juhul kasutab treening:

- **one-vs-rest (OvR) skeemi**, kui `multi_class` on `ovr`
- **ristentropia kaotust**, kui `multi_class` on `multinomial`. (Praegu toetavad `multinomial` võimalust ainult ‘lbfgs’, ‘sag’, ‘saga’ ja ‘newton-cg’ lahendajad.)"

> 🎓 "Skeem" võib olla 'ovr' (üks-võrdne-kõigiga) või 'multinomial'. Kuna logistiline regressioon on mõeldud peamiselt binaarklassifikatsiooniks, võimaldavad need skeemid paremini käsitleda mitmeklassilisi klassifikatsioone. [allikas](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' on määratletud kui "optimeerimisprobleemis kasutatav algoritm". [allikas](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn pakub selle tabeli, mis selgitab, kuidas erinevad lahendajad käitlevad erinevat tüüpi andmete omadusi:

![lahendajad](../../../../translated_images/et/solvers.5fc648618529e627.webp)

## Harjutus - andmete jagamine

Võime keskenduda logistilisele regressioonile oma esimeses treeningukatsetuses, sest sa just õppisid seda eelnevas õppetükis.
Jaga andmed treening- ja testkomplektideks, kasutades `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Harjutus - logistilise regressiooni rakendamine

Kuna kasutad mitmeklassilist juhtumit, pead valima, millist _skeemi_ kasutada ja millise _lahendaja_ seada. Kasuta LogisticRegressioni mitmeklassilise seade ja lahendajaga **liblinear** treenimiseks.

1. Loo logistiline regressioon seadistusega multi_class = `ovr` ja lahendajaga `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Proovi ka teist lahendajat nagu `lbfgs`, mis on sageli vaikimisi seatud

    > Märkus, kasuta Pandas `ravel` funktsiooni andmete tasandamiseks, kui vaja.

    Täpsus on hea, üle **80%**!

1. Sa saad seda mudelit toimimas näha, testides ühe rea (#50) andmeid:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Tulemus trükitakse välja:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Proovi teist rea numbrit ja vaata tulemusi
1. Süvenedes võite kontrollida selle prognoosi täpsust:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Tulemused prinditakse välja - India köök on selle parim aimdus, hea tõenäosusega:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Kas saate seletada, miks mudel on üsna kindel, et tegemist on India köögiga?

1. Saage täpsemat teavet, trükkides välja klassifikatsiooniraporti, nagu tegite regressioonitundides:

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

## 🚀Väljakutse

Selles õppetükis kasutasite oma puhastatud andmeid masinaõppemudeli loomiseks, mis suudab prognoosida riiklikku kööki koostisosade põhjal. Võtke aega, et lugeda läbi Scikit-learni pakutavad arvukad võimalused andmete klassifitseerimiseks. Süvenege rohkem 'lahendaja' (solver) mõistetesse, et mõista, mis toimub kaadri taga.

## [Loengu järel võistlus](https://ff-quizzes.netlify.app/en/ml/)

## Kordamine & Iseteadmine

Süvenege natuke rohkem logistilise regressiooni matemaatikasse [selles õppetükis](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Kodutöö

[Õppige lahendajaid](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vastutusest loobumine**:  
See dokument on tõlgitud kasutades tehisintellekti tõlkimisteenust [Co-op Translator](https://github.com/Azure/co-op-translator). Kuigi püüame tagada täpsust, palun pidage meeles, et automaatsed tõlked võivad sisaldada vigu või ebatäpsusi. Originaaldokument oma algkeeles tuleks pidada autoriteetseks allikaks. Kriitilise teabe puhul soovitatakse kasutada professionaalset inimtõlget. Me ei võta vastutust selle tõlke kasutamisest tingitud arusaamatuste ega valesti mõistmiste eest.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->