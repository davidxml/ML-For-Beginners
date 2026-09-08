# Klasifikatori kuhinja 1

U ovoj lekciji koristit ćete skup podataka koji ste spremili iz prošle lekcije, pun izbalansiranih, čistih podataka o kuhinjama.

Koristit ćete ovaj skup podataka s različitim klasifikatorima kako biste _predvidjeli određenu nacionalnu kuhinju na temelju grupe sastojaka_. Istovremeno ćete naučiti više o nekim načinima na koje se algoritmi mogu koristiti za zadatke klasifikacije.

## [Kvizz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)
# Priprema

Pod pretpostavkom da ste dovršili [Lekciju 1](../1-Introduction/README.md), provjerite da li u korijenskoj mapi `/data` za ove četiri lekcije postoji datoteka _cleaned_cuisines.csv_.

## Vježba - predvidi nacionalnu kuhinju

1. Uvježbajte u mapi _notebook.ipynb_ ove lekcije te uvezite tu datoteku zajedno s Pandas knjižnicom:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Podaci izgledaju ovako:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Sada uvezite još nekoliko knjižnica:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Podijelite koordinate X i y u dva podatkovna okvira za treniranje. `cuisine` može biti podatkovni okvir oznaka:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Izgledat će ovako:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Izbrišite stupce `Unnamed: 0` i `cuisine` pozivom na `drop()`. Ostatak podataka spremite kao značajke za treniranje:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Vaše značajke izgledaju ovako:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Sada ste spremni za treniranje vašeg modela!

## Odabir klasifikatora

Sada kada su vam podaci čisti i spremni za treniranje, morate odlučiti koji algoritam koristiti za taj posao.

Scikit-learn grupira klasifikaciju pod Nadzorovano učenje, i u toj kategoriji pronaći ćete mnogo načina za klasifikaciju. [Raznolikost](https://scikit-learn.org/stable/supervised_learning.html) može u početku biti zbunjujuća. Sljedeće metode uključuju tehnike klasifikacije:

- Linearni modeli
- Strojevi potpornih vektora
- Stohastički gradijentni spust
- Najbliži susjedi
- Gaussovi procesi
- Odluke stabala
- Metode ansambla (voting Classifier)
- Multiklasni i multiizlazni algoritmi (multiklasna i multilabel klasifikacija, multiklasno-multiizlazna klasifikacija)

> Također možete koristiti [neuronske mreže za klasifikaciju podataka](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), ali to je izvan opsega ove lekcije.

### Koji klasifikator odabrati?

Dakle, koji klasifikator odabrati? Često je isprobavanje nekoliko njih i traženje dobrog rezultata dobar način testiranja. Scikit-learn nudi [usporedni prikaz jedan pored drugoga](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) na kreiranom skupu podataka, uspoređujući KNeighbors, SVC na dva načina, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB i QuadraticDiscrinationAnalysis, prikazujući rezultate vizualizirano:

![usporedba klasifikatora](../../../../translated_images/hr/comparison.edfab56193a85e7f.webp)
> Grafikon generiran u dokumentaciji Scikit-learn

> AutoML rješava ovaj problem uredno izvođenjem ovih usporedbi u oblaku, omogućujući vam izbor najboljeg algoritma za vaše podatke. Isprobajte [ovdje](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Bolji pristup

Bolji način nego nasumično pogađati je pratiti ideje na ovom preuzimljivom [ML Cheat sheetu](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Ovdje otkrivamo da za naš problem s više klasa imamo neke izbore:

![cheatsheet za probleme s više klasa](../../../../translated_images/hr/cheatsheet.07a475ea444d2223.webp)
> Dio Microsoftovog Algorithm Cheat Sheeta, s detaljima opcija za višeklasnu klasifikaciju

✅ Preuzmite ovaj cheat sheet, isprintajte ga i zalijepite na zid!

### Razmišljanje

Pogledajmo možemo li racionalno pristupiti različitim pristupima s obzirom na uvjete koje imamo:

- **Neuronske mreže su preteške**. S obzirom na naš čisti, ali minimalni skup podataka i činjenicu da treniramo lokalno preko bilježnica, neuronske mreže su preteške za ovaj zadatak.
- **Nema klasifikatora za dvije klase**. Ne koristimo klasifikator za dvije klase, pa je opcija jedan-protiv-svih isključena.
- **Stablo odluke ili logistička regresija mogu funkcionirati**. Stablo odluke bi moglo funkcionirati, ili logistička regresija za višeklasne podatke.
- **Višeklasno pojačano stablo odluke rješava drugačiji problem**. Višeklasno pojačano stablo odluke najprikladnije je za neparametarske zadatke, npr. zadatke dizajnirane za izradu rangiranja, pa nije korisno za nas.

### Korištenje Scikit-learn

Koristit ćemo Scikit-learn za analizu naših podataka. Međutim, postoji mnogo načina za korištenje logističke regresije u Scikit-learnu. Pogledajte [parametre koje treba proslijediti](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

U suštini postoje dva važna parametra - `multi_class` i `solver` - koje trebamo specificirati kad tražimo od Scikit-learna da izvrši logističku regresiju. Vrijednost `multi_class` specificira određeno ponašanje. Vrijednost `solver` označava koji algoritam se koristi. Nisu svi rješavači (solvers) kompatibilni sa svim vrijednostima `multi_class`.

Prema dokumentaciji, kod višeklasne varijante, algoritam treniranja:

- **Koristi shemu jedan-protiv-ostalih (OvR)** ako je opcija `multi_class` postavljena na `ovr`
- **Koristi cross-entropy loss** ako je opcija `multi_class` postavljena na `multinomial`. (Trenutno opcija `multinomial` podržana je samo kod rješavača ‘lbfgs’, ‘sag’, ‘saga’ i ‘newton-cg’)."

> 🎓 'shema' ovdje može biti 'ovr' (jedan-protiv-ostalih) ili 'multinomial'. Budući da je logistička regresija izvorno dizajnirana za binarnu klasifikaciju, ove sheme joj omogućuju da bolje rukuje višeklasnim zadacima. [izvor](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' je definiran kao "algoritam koji se koristi u problemu optimizacije". [izvor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn nudi ovu tablicu za objašnjenje kako rješavači rukuju različitim izazovima koje predstavljaju različite strukture podataka:

![rješavači](../../../../translated_images/hr/solvers.5fc648618529e627.webp)

## Vježba - razdvojite podatke

Možemo se usredotočiti na logističku regresiju za naš prvi pokušaj treniranja budući da ste o njoj nedavno učili u prethodnoj lekciji.
Podijelite svoje podatke u grupe za treniranje i testiranje pozivom `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Vježba - primijenite logističku regresiju

Budući da koristite višeklasni slučaj, morate odabrati koju _shemu_ ćete koristiti i koji _rješavač_ postaviti. Koristite LogisticRegression s višeklasnim postavkama i rješavačem **liblinear** za treniranje.

1. Kreirajte logističku regresiju s `multi_class` postavljenim na `ovr` i rješavačem postavljenim na `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Isprobajte drugi rješavač poput `lbfgs`, koji je često zadani

    > Napomena, koristite Pandas funkciju [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) za spljoštenje podataka kad je potrebno.

    Točnost je dobra, preko **80%**!

1. Možete vidjeti ovaj model u akciji testiranjem jednog reda podataka (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Rezultat je ispisan:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Isprobajte drugi broj retka i provjerite rezultate
1. Kopajući dublje, možete provjeriti točnost ove predikcije:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Rezultat je ispisan - indijska kuhinja je najbolja procjena, s dobrom vjerojatnošću:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Možete li objasniti zašto je model prilično siguran da je ovo indijska kuhinja?

1. Dobijte više detalja ispisivanjem izvještaja o klasifikaciji, kao što ste to činili u lekcijama o regresiji:

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

## 🚀Izazov

U ovoj lekciji ste koristili očišćene podatke za izgradnju modela strojnog učenja koji može predvidjeti nacionalnu kuhinju na temelju niza sastojaka. Odvojite malo vremena da proučite mnoge opcije koje Scikit-learn pruža za klasifikaciju podataka. Istražite dublje koncept 'solver' kako biste razumjeli što se događa iza scene.

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Istražite malo više matematiku iza logističke regresije u [ovoj lekciji](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Zadatak

[Proučite solvere](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Odricanje od odgovornosti**:
Ovaj je dokument preveden korištenjem AI usluge za prijevod [Co-op Translator](https://github.com/Azure/co-op-translator). Iako težimo točnosti, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na njegovom izvornom jeziku treba smatrati autoritativnim izvorom. Za kritične informacije preporučuje se profesionalni ljudski prijevod. Ne snosimo odgovornost za bilo kakve nesporazume ili pogrešna tumačenja koja proizlaze iz korištenja ovog prijevoda.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->