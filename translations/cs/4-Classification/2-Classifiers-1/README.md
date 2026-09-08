# Klasifikátory kuchyní 1

V této lekci použijete dataset, který jste si uložili z minulé lekce, plný vyvážených, čistých dat o kuchyních.

Tento dataset použijete s různými klasifikátory k _predikci dané národní kuchyně na základě skupiny surovin_. Při tom se dozvíte více o některých způsobech, jak lze algoritmy využít pro klasifikační úlohy.

## [Přednáškový kvíz](https://ff-quizzes.netlify.app/en/ml/)
# Příprava

Pokud jste dokončili [Lekci 1](../1-Introduction/README.md), ujistěte se, že soubor _cleaned_cuisines.csv_ existuje v kořenové složce `/data` pro tyto čtyři lekce.

## Cvičení - predikce národní kuchyně

1. Pracujte ve složce _notebook.ipynb_ v této lekci, importujte tento soubor spolu s knihovnou Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Data vypadají takto:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Nyní importujte několik dalších knihoven:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Rozdělte X a y do dvou dataframe pro trénování. `cuisine` může být dataframe s popisky:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Bude to vypadat takto:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Odstraňte sloupec `Unnamed: 0` a sloupec `cuisine` pomocí `drop()`. Zbytek dat uložte jako trénovací vlastnosti:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Vaše vlastnosti vypadají takto:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Nyní jste připraveni na trénování modelu!

## Volba klasifikátoru

Nyní, když jsou data čistá a připravená k tréninku, musíte se rozhodnout, který algoritmus použít.

Scikit-learn řadí klasifikaci pod Supervised Learning a v této kategorii najdete mnoho způsobů, jak klasifikovat. [Pestrá nabídka](https://scikit-learn.org/stable/supervised_learning.html) může být zpočátku ohromující. Následující metody zahrnují klasifikační techniky:

- Lineární modely
- Support Vector Machines
- Stochastic Gradient Descent
- Nejbližší sousedé
- Gaussian procesy
- Rozhodovací stromy
- Ensemble metody (voting Classifier)
- Multitřídní a vícevýstupové algoritmy (multitřídní a multilabel klasifikace, multitřídní vícevýstupová klasifikace)

> Můžete také použít [neurální sítě k klasifikaci dat](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), ale to je mimo rozsah této lekce.

### Jaký klasifikátor zvolit?

Takže, jaký klasifikátor zvolit? Často je dobré zkusit několik a hledat dobrý výsledek. Scikit-learn nabízí [porovnání vedle sebe](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) na vytvořeném datasetu, kde porovnává KNeighbors, SVC dvěma způsoby, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB a QuadraticDiscrinationAnalysis a výsledky vizualizuje:

![comparison of classifiers](../../../../translated_images/cs/comparison.edfab56193a85e7f.webp)
> Grafy vytvořené v dokumentaci Scikit-learn

> AutoML tento problém elegantně řeší tím, že tyto porovnání provádí v cloudu, což vám umožňuje vybrat nejlepší algoritmus pro vaše data. Vyzkoušejte to [zde](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Lepší přístup

Lepší než hádat naslepo je sledovat nápady z tohoto stahovatelného [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Zde zjistíte, že pro náš multitřídní problém máme několik možností:

![cheatsheet for multiclass problems](../../../../translated_images/cs/cheatsheet.07a475ea444d2223.webp)
> Úsek z Microsoft Algorithm Cheat Sheet, zobrazující možnosti multitřídní klasifikace

✅ Stáhněte si tento cheat sheet, vytiskněte si ho a pověste na zeď!

### Uvažování

Podívejme se, zda můžeme logicky projít různými přístupy vzhledem k daným omezením:

- **Neurální sítě jsou příliš náročné**. Vzhledem k našemu čistému, ale minimálnímu datasetu a tomu, že trénink probíhá lokálně přes notebooky, jsou neurální sítě příliš těžkopádné na tento úkol.
- **Žádný klasifikátor pro dvě třídy**. Nepoužíváme klasifikátor pro dvě třídy, takže vylučujeme one-vs-all.
- **Rozhodovací strom nebo logistická regrese by mohly fungovat**. Rozhodovací strom může fungovat, nebo logistická regrese pro multitřídní data.
- **Multitřídní Boosted Decision Trees řeší jiný problém**. Multitřídní boosted decision tree je nejvhodnější pro neparametrické úlohy, např. úlohy navržené pro budování žebříčků, takže pro nás není užitečný.

### Použití Scikit-learn

Budeme používat Scikit-learn k analýze našich dat. Existuje však mnoho způsobů, jak v Scikit-learn použít logistickou regresi. Podívejte se na [parametry, které lze předat](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

V zásadě jsou dva důležité parametry - `multi_class` a `solver` - které musíme specifikovat, když požadujeme, aby Scikit-learn provedl logistickou regresi. Hodnota `multi_class` určuje chování. Hodnota `solver` určuje, jaký algoritmus se použije. Ne všechny solvery lze použít se všemi hodnotami `multi_class`.

Podle dokumentace v případě multitřídy:

- **Používá schéma one-vs-rest (OvR)**, pokud je volba `multi_class` nastavena na `ovr`
- **Používá funkci ztráty křížové entropie**, pokud je volba `multi_class` nastavena na `multinomial`. (Momentálně jsou volby `multinomial` podporovány pouze solvery ‘lbfgs’, ‘sag’, ‘saga’ a ‘newton-cg’)."

> 🎓 ‘Schéma’ zde může být buď 'ovr' (one-vs-rest) nebo 'multinomial'. Jelikož je logistická regrese primárně navržena pro binární klasifikaci, tato schémata jí umožňují lépe zvládat multitřídní klasifikační úkoly. [zdroj](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 ‘Solver’ je definován jako „algoritmus použitý v optimalizačním problému“. [zdroj](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn nabízí tuto tabulku, která vysvětluje, jak jednotlivé solvery zvládají různé výzvy datových struktur:

![solvers](../../../../translated_images/cs/solvers.5fc648618529e627.webp)

## Cvičení - rozdělení dat

Můžeme se zaměřit na logistickou regresi pro náš první tréninkový pokus, protože jste se o ní nedávno učili v předchozí lekci.
Rozdělte svá data do tréninkových a testovacích skupin voláním funkce `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Cvičení - aplikace logistické regrese

Protože používáte multitřídní případ, musíte zvolit, jaké _schéma_ použít a jaký _solver_ nastavit. Použijte LogisticRegression s multitřídním nastavením a **liblinear** solverem k tréninku.

1. Vytvořte logistickou regresi s multi_class nastaveným na `ovr` a solverem `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Vyzkoušejte jiný solver, například `lbfgs`, který je často nastaven jako výchozí

    > Poznámka: použijte funkci Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) k zploštění dat v případě potřeby.

    Přesnost je dobrá, přes **80 %**!

1. Model si můžete vyzkoušet na jednom řádku dat (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Výsledek je vytisknut:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Vyzkoušejte jiný řádek a ověřte výsledky
1. Pokud se do toho ponoříte hlouběji, můžete zkontrolovat přesnost tohoto předpovědního výsledku:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Výsledek je vytištěn - indická kuchyně je jeho nejlepší odhad s dobrou pravděpodobností:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Můžete vysvětlit, proč je model docela jistý, že se jedná o indickou kuchyni?

1. Získejte více podrobností vytištěním klasifikační zprávy, jak jste to dělali v lekcích o regresi:

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

## 🚀 Výzva

V této lekci jste použili vyčištěná data k vytvoření modelu strojového učení, který dokáže předpovědět národní kuchyni na základě řady ingrediencí. Věnujte čas přečtení si mnoha možností, které Scikit-learn poskytuje pro klasifikaci dat. Ponořte se hlouběji do konceptu „solver“ (řešiče), abyste pochopili, co se děje za scénou.

## [Kvíz po přednášce](https://ff-quizzes.netlify.app/en/ml/)

## Přehled a samostudium

Ponořte se více do matematiky za logistickou regresí v [této lekci](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf).

## Zadání

[Studujte řešiče](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Upozornění**:  
Tento dokument byl přeložen pomocí AI překladatelské služby [Co-op Translator](https://github.com/Azure/co-op-translator). Přestože usilujeme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho rodném jazyce by měl být považován za autoritativní zdroj. Pro kritické informace se doporučuje profesionální lidský překlad. Nejsme odpovědní za jakékoli nedorozumění nebo nesprávné výklady vyplývající z použití tohoto překladu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->