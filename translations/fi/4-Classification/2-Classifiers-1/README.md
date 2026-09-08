# Ruokakulttuurien luokittelijat 1

Tässä oppitunnissa käytät viimeiseltä oppitunnilta tallentamaasi tasapainoista, siistiä ruokakulttuureja käsittelevää aineistoa.

Käytät tätä aineistoa erilaisten luokittelijoiden kanssa _ennustaaksesi tietyn kansallisen ruokakulttuurin perustuen joukkoon ainesosia_. Samalla opit lisää erilaisista tavoista, joilla algoritmeja voidaan hyödyntää luokittelutehtävissä.

## [Luennon ennakkotentti](https://ff-quizzes.netlify.app/en/ml/)
# Valmistautuminen

Oletetaan, että olet suorittanut [Oppitunti 1](../1-Introduction/README.md), varmista että _cleaned_cuisines.csv_ -tiedosto on olemassa `/data`-kansion juuressa näitä neljää oppituntia varten.

## Harjoitus - ennusta kansallinen ruokakulttuuri

1. Työskentele tässä oppitunnin _notebook.ipynb_-kansiossa ja tuo tiedosto yhdessä Pandas-kirjaston kanssa:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Data näyttää tältä:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Tuo nyt useita muita kirjastoja:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Jaa X- ja y-koordinaatit kahdeksi dataframeksi koulutusta varten. `cuisine` voi olla labelien dataframe:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Se näyttää tältä:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Pudota `Unnamed: 0`-sarake ja `cuisine`-sarake kutsumalla `drop()`. Tallenna loput datasta koulutettaviksi ominaisuuksiksi:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Ominaisuutesi näyttävät tältä:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Nyt olet valmis kouluttamaan mallisi!

## Luokittelijan valinta

Nyt kun data on puhdas ja valmis koulutukseen, sinun täytyy päättää, mitä algoritmia käyttää.

Scikit-learn ryhmittelee luokittelun ohjattuun oppimiseen, ja tässä kategoriassa on monia tapoja luokitella. [Monipuolisuus](https://scikit-learn.org/stable/supervised_learning.html) voi ensi näkemältä olla hämmentävää. Seuraavat menetelmät sisältävät kaikki luokittelutekniikoita:

- Lineaariset mallit
- Tukivektorikoneet
- Stokastinen gradienttilaskenta
- Lähimmät naapurit
- Gaussiset prosessit
- Päätöspuut
- Kokonaismenetelmät (äänestävä luokittelija)
- Moniluokka- ja usean ulostulon algoritmit (moniluokkainen ja monimerkintäinen luokittelu, moniluokkainen usean ulostulon luokittelu)

> Voit myös käyttää [neuroverkkoja datan luokitteluun](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), mutta se on tämän oppitunnin ulkopuolella.

### Mikä luokittelija valitaan?

Joten, minkä luokittelijan valitset? Usein on järkevää kokeilla useampaa ja etsiä hyvää tulosta. Scikit-learn tarjoaa [rinnakkaisvertailun](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) luodulla datalla vertaillen KNeighborsia, kahta SVC-versiota, GaussianProcessClassifieria, DecisionTreeClassifieria, RandomForestClassifieria, MLPClassifieria, AdaBoostClassifieria, GaussianNB:tä ja QuadraticDiscrinationAnalysistä tuloksineen visuaalisesti:

![luokittelijoiden vertailu](../../../../translated_images/fi/comparison.edfab56193a85e7f.webp)
> Kaaviot luotu Scikit-learnin dokumentaatiossa

> AutoML ratkaisee tämän ongelman siististi ajamalla nämä vertailut pilvessä, jolloin voit valita parhaan algoritmin datallesi. Kokeile sitä [tässä](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Parempi lähestymistapa

Parempi tapa kuin arvailla villeinä on seurata tämän ladattavan [ML Cheat sheetin](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) vinkkejä. Sieltä löydämme, että moniluokkaiseen ongelmaamme on muutama vaihtoehto:

![moniluokkaisten ongelmien vinkkilista](../../../../translated_images/fi/cheatsheet.07a475ea444d2223.webp)
> Ote Microsoftin algoritmivinkkilistasta, joka kuvaa moniluokkalun luokitteluvaihtoehtoja

✅ Lataa tämä vinkkilista, tulosta se ja laita seinällesi roikkumaan!

### Päättely

Katsotaan, voimmeko perustella eri lähestymistavat annetuin rajoituksin:

- **Neuroverkot ovat liian raskaita**. Ottaen huomioon puhtaan, mutta pienen datasetimme ja että koulutus tapahtuu paikallisesti muistikirjojen kautta, neuroverkot ovat liian raskaita tähän tehtävään.
- **Ei kaksiluokkaista luokittelijaa**. Emme käytä kaksiluokkaista luokittelijaa, joten one-vs-all suljetaan pois.
- **Päätöspuu tai logistinen regressio voisi toimia**. Päätöspuu voisi toimia, samoin logistinen regressio moniluokkaiseen dataan.
- **Moniluokkainen Boosted Decision Trees ratkaisee eri ongelman**. Moniluokkainen vahvistettu päätöspuu soveltuu paremmin parametrisiin tehtäviin, kuten rankingien rakentamiseen, joten se ei ole meille käyttökelpoinen.

### Scikit-learnin käyttö

Käytämme Scikit-learnia datan analysointiin. Kuitenkin logistiikkaregressiota on monia erilaisia tapoja käyttää Scikit-learnissa. Tutustu [parametreihin](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Käytännössä tärkeimmät parametrit ovat `multi_class` ja `solver` - joita tarvitsee määrittää pyytäessämme Scikit-learnia suorittamaan logistisen regression. `multi_class` määrittää tietyn käyttäytymisen. `solver` on käytettävä algoritmi. Kaikki solverit eivät sovi yhteen kaikkien `multi_class` -arvojen kanssa.

Dokumenttien mukaan moniluokkaisessa tapauksessa koulutusalgoritmi:

- **Käyttää one-vs-rest (OvR) -metodia**, jos `multi_class` on asetettu arvoksi `ovr`
- **Käyttää ristientropiahäviötä**, jos `multi_class` on asetettu arvoksi `multinomial`. (Tällä hetkellä `multinomial` on tuettu vain `lbfgs`, `sag`, `saga` ja `newton-cg` -solverien kanssa.)

> 🎓 'Skenaario' voi olla 'ovr' (one-vs-rest) tai 'multinomial'. Koska logistinen regressio on suunniteltu binääriluokitteluun, nämä skenaariot mahdollistavat paremman tuen moniluokkaluille luokittelutehtäville. [lähde](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solveri' on määritelty algoritmiksi, jota käytetään optimointiongelmassa. [lähde](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn tarjoaa tämän taulukon selittämään, miten solverit käsittelevät erilaisia datarakenteiden haasteita:

![solverit](../../../../translated_images/fi/solvers.5fc648618529e627.webp)

## Harjoitus - jaa data

Voimme keskittyä logistiseen regressioon ensimmäisen koulutusyrityksenä, kun opit siitä hiljattain edellisellä tunnilla.  
Jaa data koulutus- ja testiryhmiin kutsumalla `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Harjoitus - käytä logistista regressiota

Koska käytät moniluokkaista tapausta, sinun täytyy valita, mitä _skenaariota_ ja _solveria_ käytetään. Käytä LogisticRegressionia moniluokkaisella asetuksella ja **liblinear**-solveria kouluttaaksesi.

1. Luo logistinen regressio, jossa `multi_class` on asetettu `ovr`:ksi ja `solver` `liblinear`iksi:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Kokeile eri solveria, esimerkiksi `lbfgs`, joka usein on oletus

    > Huomaa, käytä Pandasin [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) -funktiota, kun haluat tasoittaa dataasi tarvittaessa.

    Tarkkuus on hyvä, yli **80%**!

1. Voit nähdä tämän mallin toiminnassa testaamalla yhden datarivin (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Tuloste on:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Kokeile eri rivinumeroa ja tarkista tulokset
1. Digging deeper, you can check for the accuracy of this prediction:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    The result is printed - Indian cuisine is its best guess, with good probability:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Voitko selittää, miksi malli on melko varma, että kyseessä on intialainen keittiö?

1. Get more detail by printing a classification report, as you did in the regression lessons:

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

## 🚀Challenge

In this lesson, you used your cleaned data to build a machine learning model that can predict a national cuisine based on a series of ingredients. Take some time to read through the many options Scikit-learn provides to classify data. Dig deeper into the concept of 'solver' to understand what goes on behind the scenes.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Dig a little more into the math behind logistic regression in [this lesson](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Assignment 

[Study the solvers](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, ole hyvä ja huomioi, että automatisoiduissa käännöksissä saattaa esiintyä virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäiskielellä tulee pitää auktoriteettisena lähteenä. Tärkeiden tietojen kohdalla suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa tämän käännöksen käytöstä johtuvista väärinkäsityksistä tai virheellisistä tulkinnoista.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->