# Klasifikátory kuchyne 1

V tejto lekcii použijete dataset, ktorý ste si uložili z predchádzajúcej lekcie, plný vyvážených, čistých údajov o kuchyniach.

Tento dataset použijete s rôznymi klasifikátormi na _predpoveď danej národnej kuchyne na základe skupiny ingrediencií_. Počas toho sa dozviete viac o niektorých spôsoboch, ako možno algoritmy využiť na klasifikačné úlohy.

## [Prednáškový kvíz](https://ff-quizzes.netlify.app/en/ml/)
# Príprava

Ak ste dokončili [Lekciu 1](../1-Introduction/README.md), uistite sa, že súbor _cleaned_cuisines.csv_ existuje v koreňovej priečinok `/data` pre tieto štyri lekcie.

## Cvičenie - predpovedať národnú kuchyňu

1. V priečinku _notebook.ipynb_ tejto lekcie importujte tento súbor spolu s knižnicou Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Údaje vyzerajú takto:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Teraz importujte niekoľko ďalších knižníc:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Rozdeľte súradnice X a y do dvoch dataframeov na trénovanie. `cuisine` môže byť dataframe s návestiami:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Bude to vyzerať takto:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Odstráňte stĺpec `Unnamed: 0` a stĺpec `cuisine` pomocou funkcie `drop()`. Zvyšok dát uložte ako trénovateľné vlastnosti:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Vaše vlastnosti vyzerajú takto:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Teraz ste pripravení trénovať svoj model!

## Výber klasifikátora

Keď už sú vaše dáta čisté a pripravené na trénovanie, musíte sa rozhodnúť, ktorý algoritmus použiť.

Scikit-learn zaraďuje klasifikáciu do kategórie Supervised Learning a v tejto kategórii nájdete mnoho spôsobov klasifikácie. [Rozmanitosť](https://scikit-learn.org/stable/supervised_learning.html) je na prvý pohľad dosť ohromujúca. Nasledujúce metódy zahŕňajú klasifikačné techniky:

- Lineárne modely
- Support Vector Machines
- Stochastic Gradient Descent
- Najbližší susedia
- Gaussovské procesy
- Rozhodovacie stromy
- Metódy ansámblov (voting Classifier)
- Multitriedne a multivýstupové algoritmy (multitriedna a multilabel klasifikácia, multitriedna multivýstupová klasifikácia)

> Môžete tiež použiť [neurónové siete na klasifikáciu dát](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), ale to presahuje rozsah tejto lekcie.

### Ktorý klasifikátor zvoliť?

Ktorý klasifikátor by ste si teda mali vybrať? Často je spôsob, že skúsite niekoľko a hľadáte dobrý výsledok. Scikit-learn ponúka [bezprostredné porovnanie](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) na vytvorenom datasete, ktoré porovnáva KNeighbors, SVC dvoma spôsobmi, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB a QuadraticDiscrinationAnalysis a vizualizuje výsledky:

![porovnanie klasifikátorov](../../../../translated_images/sk/comparison.edfab56193a85e7f.webp)
> Grafy vytvorené v dokumentácii Scikit-learn

> AutoML túto úlohu elegantne rieši tým, že vykonáva tieto porovnania v cloude, čo vám umožní vybrať najlepší algoritmus pre vaše dáta. Vyskúšajte to [tu](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Lepší prístup

Lepší spôsob ako hádať naslepo, je postupovať podľa odporúčaní v tomto stiahnuteľnom [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Tu objavujeme, že pri našom multitriednom probléme máme niekoľko možností:

![cheatsheet pre multitriedne problémy](../../../../translated_images/sk/cheatsheet.07a475ea444d2223.webp)
> Časť Microsoft Algorithm Cheat Sheet, rozoberajúca možnosti multitriednej klasifikácie

✅ Stiahnite si tento cheat sheet, vytlačte si ho a zavesíte na stenu!

### Odôvodnenie

Pozrime sa, či dokážeme rozumovo prejsť rôznymi prístupmi vzhľadom na dané obmedzenia:

- **Neurónové siete sú príliš náročné**. Vzhľadom na náš čistý, ale minimálny dataset a fakt, že trénovanie beží lokálne cez notebooky, sú neurónové siete príliš náročné na túto úlohu.
- **Žiadny klasifikátor pre dve triedy**. Nepoužívame klasifikátor pre dve triedy, takže možnosť "one-vs-all" vylučujeme.
- **Decision tree alebo logická regresia môže fungovať**. Môže fungovať rozhodovací strom alebo logistická regresia pre multitriedne dáta.
- **Multitriedne boosted rozhodovacie stromy riešia inú úlohu**. Multitriedny boosted rozhodovací strom je vhodnejší pre neparametrické úlohy, napríklad konštruovanie rebríčkov, takže pre nás nie je užitočný.

### Použitie Scikit-learn

Použijeme Scikit-learn na analýzu našich dát. Existuje však mnoho spôsobov, ako použiť logistickú regresiu v Scikit-learn. Pozrite si [parametre](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression), ktoré je možné odovzdať.

Podstatné sú dva parametre - `multi_class` a `solver` - ktoré musíme určiť, keď žiadame Scikit-learn o vykonanie logistickej regresie. Hodnota `multi_class` aplikuje určitý spôsob správania. Hodnota solveru je algoritmus, ktorý sa použije. Nie všetky solvery sú kompatibilné so všetkými hodnotami `multi_class`.

Podľa dokumentácie, pri multitriednom prípade, tréningový algoritmus:

- **Používa schému one-vs-rest (OvR),** ak je `multi_class` nastavený na `ovr`
- **Používa cross-entropy loss**, ak je `multi_class` nastavený na `multinomial`. (Momentálne je možnosť `multinomial` podporovaná iba solvermi ‘lbfgs’, ‘sag’, ‘saga’ a ‘newton-cg’)."

> 🎓 'schéma' tu môže byť 'ovr' (one-vs-rest) alebo 'multinomial'. Keďže logistická regresia je navrhnutá na binárnu klasifikáciu, tieto schémy jej umožňujú lepšie zvládnuť multitriedne klasifikačné úlohy. [zdroj](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' je definovaný ako "algoritmus, ktorý sa použije v optimalizačnej úlohe". [zdroj](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn ponúka túto tabuľku, ktorá vysvetľuje, ako solvery spracovávajú rôzne výzvy, ktoré predstavujú rôzne štruktúry dát:

![solvery](../../../../translated_images/sk/solvers.5fc648618529e627.webp)

## Cvičenie - rozdeliť dáta

Môžeme sa zamerať na logistickú regresiu pre náš prvý pokus o trénovanie, keďže ste o nej nedávno čítali v predchádzajúcej lekcii.
Rozdeľte dáta na trénovacie a testovacie skupiny pomocou volania `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Cvičenie - aplikovať logistickú regresiu

Keďže používate multitriedny prípad, musíte si vybrať, akú _schému_ použiť a aký _solver_ nastaviť. Použite LogisticRegression s multitriednym nastavením a **liblinear** solverom na trénovanie.

1. Vytvorte logistickú regresiu s `multi_class` nastaveným na `ovr` a solverom `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Vyskúšajte iný solver ako `lbfgs`, ktorý je často nastavený ako predvolený

    > Poznámka: Použite funkciu Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) na zploštenie dát, keď je to potrebné.

    Presnosť je dobrá, vyše **80%**!

1. Model si môžete vyskúšať tak, že otestujete jeden riadok dát (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Výsledok sa vytlačí:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Vyskúšajte iné číslo riadku a skontrolujte výsledky
1. Hlbšie preskúmajte a overte presnosť tohto predpovede:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Výsledok je vytlačený – najpravdepodobnejšia je indická kuchyňa, s dobrou pravdepodobnosťou:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Dokážete vysvetliť, prečo si model dosť iste myslí, že ide o indickú kuchyňu?

1. Získajte viac detailov vytlačením klasifikačnej správy, ako ste to robili v lekciách o regresii:

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

## 🚀Výzva

V tejto lekcii ste použili vyčistené dáta na vybudovanie modelu strojového učenia, ktorý dokáže predpovedať národnú kuchyňu na základe série ingrediencií. Venujte čas preštudovaniu mnohých možností, ktoré Scikit-learn poskytuje na klasifikáciu dát. Hlbšie sa oboznámte s konceptom „solver“ a pochopte, čo sa deje v pozadí.

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium

Málo sa ponorte do matematiky za logistickou regresiou v [tejto lekcii](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)

## Úloha

[Študujte solver-y](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Zrieknutie sa zodpovednosti**:
Tento dokument bol preložený pomocou služby prekladov AI [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, majte prosím na pamäti, že automatické preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho rodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne výklady vyplývajúce z použitia tohto prekladu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->