# Virtuvių klasifikatoriai 1

Šiame pamokoje naudosite duomenų rinkinį, kurį išsaugojote iš paskutinės pamokos, pilną subalansuotų, švarių duomenų apie virtuves.

Naudosite šį duomenų rinkinį su įvairiais klasifikatoriais, kad _numatytumėte tam tikrą nacionalinę virtuvę pagal ingredientų grupę_. Tai darydami sužinosite daugiau apie kai kuriuos būdus, kaip algoritmai gali būti panaudoti klasifikavimo užduotims.

## [Išankstinis paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)
# Paruošimas

Jei baigėte [1-ą pamoką](../1-Introduction/README.md), įsitikinkite, kad šių keturių pamokų metu turite _cleaned_cuisines.csv_ failą pagrindiniame `/data` kataloge.

## Užduotis - numatyti nacionalinę virtuvę

1. Šios pamokos _notebook.ipynb_ aplanke importuokite tą failą kartu su Pandas biblioteka:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Duomenys atrodo taip:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Dabar importuokite dar kelias bibliotekas:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Padalinkite X ir y koordinates į du duomenų kadrus treniravimui. `cuisine` gali būti žymių duomenų kadras:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Atrodys taip:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Išmeskite `Unnamed: 0` ir `cuisine` stulpelius, naudodami `drop()`. Likusius duomenis išsaugokite kaip treniruojamus požymius:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Jūsų požymiai atrodo taip:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Dabar jūs pasiruošę treniruoti modelį!

## Klasifikatoriaus pasirinkimas

Kadangi jūsų duomenys švarūs ir pasiruošę mokymuisi, turite nuspręsti, kurį algoritmą naudoti šiai užduočiai.

Scikit-learn klasifikaciją priskiria prie Prižiūrimo mokymosi (Supervised Learning), o šioje kategorijoje rasite daug skirtingų būdų klasifikuoti. [Įvairovė](https://scikit-learn.org/stable/supervised_learning.html) iš pirmo žvilgsnio yra ganėtinai didelė. Šie metodai visi apima klasifikavimo technikas:

- Linijiniai modeliai
- Atraminiai vektorių mašinos
- Stochastinis nuolydžio nusileidimas
- Artimiausių kaimynų metodas
- Gauso procesai
- Sprendimų medžiai
- Ansamblio metodai (voting Classifier)
- Keliaklasiai ir kelių išėjimų algoritmai (multiclass ir multilabel klasifikacija, multiclass-multioutput klasifikacija)

> Taip pat galite naudoti [dirbtinius neuroninius tinklus duomenims klasifikuoti](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), tačiau tai ne šios pamokos tema.

### Kuri klasifikatorių pasirinkti?

Taigi, kurį klasifikatorių rinktis? Dažnai naudinga išbandyti kelis ir ieškoti geriausio rezultato. Scikit-learn siūlo [šalia šalia palyginimą](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) sukurtame duomenų rinkinyje, palyginant KNeighbors, SVC dviem būdais, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB ir QuadraticDiscrinationAnalysis, rodydami rezultatus vizualiai:

![klasifikatorių palyginimas](../../../../translated_images/lt/comparison.edfab56193a85e7f.webp)
> Grafikai pagal Scikit-learn dokumentaciją

> AutoML šią problemą sprendžia efektyviai vykdydamas šiuos palyginimus debesyje ir leydamas išsirinkti geriausią algoritmą jūsų duomenims. Išbandykite [čia](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Geresnis požiūris

Tačiau geresnis būdas nei laužtu būdu spėlioti – sekti mintimis šioje atsisiunčiamoje [ML „Cheat sheet“](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Čia mes atrandame, kad mūsų keliaklasiui problemos sprendimui turime šiuos pasirinkimus:

![keliaklasių problemų cheat sheet](../../../../translated_images/lt/cheatsheet.07a475ea444d2223.webp)
> Microsoft algoritmų „Cheat sheet“ ištrauka, aprašanti keliaklasių klasifikavimo galimybes

✅ Atsisiųskite šį lapelį, išspausdinkite ir prikabinkite prie sienos!

### Argumentacija

Pažiūrėkime, ar galime pamąstyti apie skirtingus požiūrius atsižvelgiant į turimas sąlygas:

- **Neuroniniai tinklai yra per sunkūs**. Atsižvelgiant į švarius, bet minimaliai didelius duomenis ir faktą, kad mokymą vykdome vietoje su užrašų knygelėmis, neuroniniai tinklai yra per sunkūs šiai užduočiai.
- **Nenaudojamas dviejų klasių klasifikatorius**. Mes nenaudojame dviejų klasių klasifikatoriaus, taigi „one-vs-all“ metodas netinka.
- **Sprendimų medis arba logistinė regresija gali veikti**. Galėtų tikti sprendimų medis arba logistinė regresija keliaklasiams duomenims.
- **Keliaklasiai stiprinami sprendimų medžiai sprendžia kitokį uždavinį**. Keliakliai stiprinami sprendimų medžiai labiau tinka neparametrinėms užduotims, pvz., reitingų kūrimui, tad mums nėra naudingi.

### Naudojant Scikit-learn

Naudosime Scikit-learn duomenų analizei. Tačiau logistinę regresiją Scikit-learn galima naudoti įvairiais būdais. Pažvelkite į [parametrus, kuriuos reikia perduoti](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Iš esmės yra du svarbūs parametrai – `multi_class` ir `solver` – kuriuos reikia nurodyti prašant Scikit-learn atlikti logistinę regresiją. `multi_class` nustato tam tikrą elgesį. `solver` nurodo, kurį algoritmą naudoti. Ne visi sprendikliai tinka visiems `multi_class` parinkčių deriniams.

Pagal dokumentaciją, keliaklasiu atveju, treniravimo algoritmas:

- **Naudoja one-vs-rest (OvR) schemą**, jei `multi_class` nustatyta į `ovr`
- **Naudoja kryžminio entropijos nuostolio funkciją**, jei `multi_class` nustatyta į `multinomial`. (Šiuo metu `multinomial` parinktis palaikoma tik su `lbfgs`, `sag`, `saga` ir `newton-cg` sprendikliais.)"

> 🎓 Čia „schema“ gali būti arba `ovr` (one-vs-rest), arba „multinomial“. Kadangi logistinė regresija iš esmės skirta dvejetainiaiems klasifikavimo uždaviniams, šios schemos leidžia geriau tvarkyti keliaklasius uždavinius. [šaltinis](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 „Solver“ apibrėžiamas kaip „algoritmas, naudojamas optimizacijos problemai spręsti“. [šaltinis](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn pateikia lentelę, kurioje paaiškinama, kaip sprendikliai elgiasi su įvairiais duomenų struktūrų iššūkiais:

![sprendikliai](../../../../translated_images/lt/solvers.5fc648618529e627.webp)

## Užduotis - padalyti duomenis

Galime sutelkti dėmesį į logistinės regresijos naudojimą kaip pirmąjį treniravimo bandymą, kadangi tai ką tik peržiūrėjote ankstesnėje pamokoje.
Padalinkite duomenis į mokymo ir testavimo grupes, naudodami `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Užduotis - pritaikyti logistinę regresiją

Kadangi naudojate keliaklasių atvejį, turite pasirinkti, kokią _schemą_ naudoti ir kokį _sprendiklį_ nustatyti. Naudokite LogisticRegression su keliaklasių nustatymu ir **liblinear** sprendikliu mokymui.

1. Sukurkite logistinę regresiją su multi_class nustatytu į `ovr` ir sprendikliu į `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Išbandykite kitą sprendiklį, pvz. `lbfgs`, kuris dažnai yra numatytasis

    > Atkreipkite dėmesį, kad kai reikia, naudokite Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) funkciją duomenims suplokšti.

    Tikslumas yra geras – virš **80%**!

1. Galite pamatyti, kaip veikia šis modelis, ištestavę vieną duomenų eilutę (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Rezultatas išspausdinamas:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Išbandykite kitą eilutės numerį ir patikrinkite rezultatus
1. Giliau nagrinėjant, galite patikrinti šio spėjimo tikslumą:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Rezultatas atspausdintas - geriausias spėjimas yra Indijos virtuvė, su gera tikimybe:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Ar galite paaiškinti, kodėl modelis yra gana tikras, kad tai yra Indijos virtuvė?

1. Gaukite daugiau detalių atspausdinę klasifikavimo ataskaitą, kaip darėte regresijos pamokose:

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

## 🚀Iššūkis

Šioje pamokoje naudojote savo išvalytus duomenis, kad sukurtumėte mašininio mokymosi modelį, galintį prognozuoti nacionalinę virtuvę pagal ingredientų rinkinį. Skirkite laiko perskaityti daugybę variantų, kuriuos Scikit-learn siūlo duomenų klasifikavimui. Gilinkitės į 'solver' sąvoką, kad suprastumėte, kas vyksta užkulisiuose.

## [Po paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

## Peržiūra ir savarankiškas mokymasis

Giliau nagrinėkite logistinės regresijos matematiką [šioje pamokoje](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Namų darbai

[Išstudijuokite solverius](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Atsakomybės atsisakymas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors stengiamės užtikrinti tikslumą, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritiniais atvejais rekomenduojamas profesionalus žmogaus vertimas. Mes neatsakome už jokius nesusipratimus ar neteisingus interpretavimus, kylant iš šio vertimo naudojimo.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->