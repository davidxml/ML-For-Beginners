# Razvrščevalci kuhinj 1

V tej lekciji boste uporabili podatkovni niz, ki ste ga shranili v zadnji lekciji, poln uravnoteženih, čistih podatkov o kuhinjah.

Ta podatkovni niz boste uporabili z različnimi razvrščevalci, da _napovejo nacionalno kuhinjo glede na skupino sestavin_. Med tem se boste naučili več o nekaterih načinih, kako je mogoče algoritme uporabiti za naloge razvrščanja.

## [Predpredavanja kviz](https://ff-quizzes.netlify.app/en/ml/)
# Priprava

Če ste zaključili [Lekcijo 1](../1-Introduction/README.md), se prepričajte, da datoteka _cleaned_cuisines.csv_ obstaja v korenski mapi `/data` za te štiri lekcije.

## Naloga - napovedati nacionalno kuhinjo

1. V delovni mapi _notebook.ipynb_ te lekcije uvozite to datoteko skupaj s knjižnico Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Podatki izgledajo tako:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Zdaj uvozite še nekaj drugih knjižnic:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Razdelite spremenljivki X in y na dva podatkovna okvira za učenje. `cuisine` je lahko podatkovni okvir z oznakami:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Izgledalo bo takole:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Odstranite stolpca `Unnamed: 0` in `cuisine` z uporabo `drop()`. Preostale podatke shranite kot značilke za učenje:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Vaše značilke izgledajo takole:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Zdaj ste pripravljeni za učenje modela!

## Izbira razvrščevalca

Zdaj, ko so vaši podatki čisti in pripravljeni za učenje, morate izbrati algoritem za to nalogo.

Scikit-learn združuje razvrščanje pod Nadzorovanim učenjem, in v tej kategoriji boste našli veliko načinov za razvrščanje. [Raznolikost](https://scikit-learn.org/stable/supervised_learning.html) je na prvi pogled precej zmedena. Naslednje metode vključujejo tehnike razvrščanja:

- Linearni modeli
- Podporni vektorski stroji
- Stohastični gradientni spust
- Najbližji sosedje
- Gausovi procesi
- Odločitvena drevesa
- Metode združevanja (voting Classifier)
- Multiklasni in multi-output algoritmi (multiklasno in multilabel razvrščanje, multiklasno multioutput razvrščanje)

> Uporabiti lahko tudi [nevronske mreže za razvrščanje podatkov](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), vendar to ni del te lekcije.

### Kateri razvrščevalec izbrati?

Torej, katerega razvrščevalca izbrati? Pogosto je dobra praksa preizkusiti več razvrščevalcev in iskati lep rezultat. Scikit-learn ponuja [primerjavo ob strani](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) na ustvarjenem podatkovnem nizu, ki primerja KNeighbors, SVC na dva načina, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB in QuadraticDiscrinationAnalysis, ter prikazuje rezultate vizualno:

![primerjava razvrščevalcev](../../../../translated_images/sl/comparison.edfab56193a85e7f.webp)
> Grafi iz dokumentacije Scikit-learn

> AutoML to težavo elegantno reši tako, da te primerjave izvaja v oblaku, kar vam omogoča izbiro najboljšega algoritma za vaše podatke. Poskusite [tukaj](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Boljši pristop

Boljši način kot naključno ugibanje je, da sledite idejam na tem prenosljivem [ML Cheat sheetu](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Tukaj odkrijemo, da imamo za naš multiklasni problem nekaj možnosti:

![cheatsheet za multiklasne probleme](../../../../translated_images/sl/cheatsheet.07a475ea444d2223.webp)
> Del Microsoftovega Algorithm Cheat Sheeta, ki podrobno opisuje možnosti multiklasnega razvrščanja

✅ Prenesite ta cheat sheet, ga natisnite in obesite na steno!

### Razlogi

Poglejmo, če lahko presodimo različne pristope glede na dane omejitve:

- **Nevronske mreže so pretežke**. Glede na naš čist, a minimalen podatkovni niz in dejstvo, da učenje izvajamo lokalno prek zvezkov, so nevronske mreže za to nalogo preveč težke.
- **Ni razvrščevalca za dve razredi**. Nimamo razvrščevalca za dve razredi, zato izključimo one-vs-all metodi.
- **Odločitveno drevo ali logistična regresija bi delovala**. Morda bi delovalo odločitveno drevo ali pa logistična regresija za multiklasne podatke.
- **Multiklasno boostano odločitveno drevo rešuje drugačno težavo**. Multiklasno boostano odločitveno drevo je najbolj primerno za neparametrične naloge, npr. naloge za izdelavo rangovanj, zato za nas ni uporabno.

### Uporaba Scikit-learn

Za analizo podatkov bomo uporabljali Scikit-learn. Vendar obstaja veliko načinov uporabe logistične regresije v Scikit-learn. Poglejte si [parametre, ki jih je mogoče nastaviti](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression). 

Ključno sta dva parametra - `multi_class` in `solver` - ki ju morate nastaviti, ko zahtevate od Scikit-learn, da izvede logistično regresijo. Vrednost `multi_class` določa določeno vedenje. Vrednost `solver` pa določa, kateri algoritem se bo uporabljal. Ne vsi reševalci (solverji) so združljivi z vsemi vrednostmi `multi_class`.

Po dokumentaciji v primeru multiklase učenja:

- **Uporablja shemo one-vs-rest (OvR)**, če je `multi_class` nastavljen na `ovr`
- **Uporablja izgubo križne entropije**, če je `multi_class` nastavljen na `multinomial`. (Trenutno so `multinomial` možnost podprte samo pri reševalcih ‘lbfgs’, ‘sag’, ‘saga’ in ‘newton-cg’)."

> 🎓 "Shema" je lahko 'ovr' (one-vs-rest) ali 'multinomial'. Ker je logistična regresija prvotno zasnovana za binarno razvrščanje, te sheme omogočajo boljše upravljanje nalog multiklasnega razvrščanja. [vir](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 "Solver" je definiran kot "algoritem za reševanje optimizacijskega problema". [vir](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn ponuja to tabelo, ki pojasnjuje, kako reševalci obravnavajo različne izzive, povezane z različnimi vrstami podatkovnih struktur:

![reševalci](../../../../translated_images/sl/solvers.5fc648618529e627.webp)

## Naloga - razdeliti podatke

Za našo prvo preizkusno učenje se lahko osredotočimo na logistično regresijo, saj ste o njej nedavno že izvedeli v prejšnji lekciji.
Podatke razdelite na učne in testne skupine z uporabo `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Naloga - uporabite logistično regresijo

Ker uporabljate multiklasni primer, morate izbrati, katero _shemo_ uporabiti in kateri _solver_ nastaviti. Uporabite LogisticRegression z nastavitvijo za multiklasno in solverjem **liblinear** za učenje.

1. Ustvarite logistično regresijo z multi_class nastavljeno na `ovr` in solver nastavljen na `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Preizkusite drugačnega solverja, na primer `lbfgs`, ki je pogosto privzeti

    > Opomba: uporabite Pandas funkcijo [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) za sploščitev podatkov, kadar je to potrebno.

    Natančnost je dobra in presega **80 %**!

1. Model lahko preizkusite tudi tako, da testirate enega od vrstic podatkov (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Rezultat se izpiše:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Preizkusite drugo številko vrstice in preverite rezultate
1. Če želite podrobneje raziskati, lahko preverite natančnost te napovedi:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Rezultat je izpisan - indijska kuhinja je najbolj verjetna napoved, z dobro verjetnostjo:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Ali lahko pojasnite, zakaj je model dokaj prepričan, da gre za indijsko kuhinjo?

1. Za več podrobnosti izpišite poročilo o klasifikaciji, kot ste to počeli v lekcijah o regresiji:

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

## 🚀Izziv

V tej lekciji ste uporabili očiščene podatke za izdelavo modela strojnega učenja, ki lahko napove nacionalno kuhinjo na osnovi vrste sestavin. Vzemite si čas in preberite različne možnosti, ki jih Scikit-learn ponuja za klasifikacijo podatkov. Podrobneje raziščite koncept 'solver', da razumete, kaj se dogaja v ozadju.

## [Kviz po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

Podrobneje raziščite matematiko za logistično regresijo v [tej lekciji](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Naloga 

[Preučite solverje](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Omejitev odgovornosti**:
Ta dokument je bil preveden z uporabo storitve za samodejni prevod [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, upoštevajte, da lahko samodejni prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem maternem jeziku velja za avtoritativni vir. Za ključne informacije priporočamo strokoven človeški prevod. Za morebitne nesporazume ali napačne interpretacije, ki nastanejo zaradi uporabe tega prevoda, ne odgovarjamo.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->