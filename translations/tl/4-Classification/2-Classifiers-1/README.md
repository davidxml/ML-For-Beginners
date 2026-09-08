# Cuisine classifiers 1

Sa leksiyong ito, gagamitin mo ang dataset na iyong in-save mula sa nakaraang leksiyon na puno ng balanseng, malinis na datos tungkol sa mga lutuin.

Gagamitin mo ang dataset na ito sa iba't ibang classifiers upang _hulaan ang isang pambansang lutuin batay sa isang grupo ng mga sangkap_. Habang ginagawa ito, matututo ka pa tungkol sa ilang paraan kung paano magagamit ang mga algorithm para sa mga classification na gawain.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)
# Preparation

Kung natapos mo na ang [Lesson 1](../1-Introduction/README.md), siguraduhing may _cleaned_cuisines.csv_ na file sa root `/data` folder para sa apat na leksiyon na ito.

## Exercise - hulaan ang isang pambansang lutuin

1. Sa paglulunsad ng leksiyon na ito sa _notebook.ipynb_ folder, i-import ang file kasama ng Pandas library:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Ganito ang hitsura ng datos:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Ngayon, mag-import ng ilan pang mga library:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Hatiin ang X at y coordinates sa dalawang dataframes para sa training. Ang `cuisine` ay pwedeng gawing labels dataframe:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Ganito ang magiging hitsura nito:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. I-drop ang `Unnamed: 0` na column at ang `cuisine` na column gamit ang `drop()`. I-save ang natitirang data bilang trainable features:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Ganito ang hitsura ng iyong mga features:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Handa ka nang i-train ang iyong modelo!

## Pagpili ng iyong classifier

Ngayon na malinis at handa na ang iyong datos para sa training, kailangan mong pumili kung anong algorithm ang gagamitin para sa gawain. 

Pinasasaklaw ng Scikit-learn ang classification sa ilalim ng Supervised Learning, at sa kategoryang iyon makikita mo ang maraming paraan upang magklasipika. [Ang iba't ibang uri](https://scikit-learn.org/stable/supervised_learning.html) ay nakakahilo sa unang tingin. Kasama sa mga sumusunod na paraan ang iba’t ibang teknik para sa classification:

- Linear Models
- Support Vector Machines
- Stochastic Gradient Descent
- Nearest Neighbors
- Gaussian Processes
- Decision Trees
- Ensemble methods (voting Classifier)
- Multiclass at multioutput algorithms (multiclass at multilabel classification, multiclass-multioutput classification)

> Maaari mo ring gamitin ang [neural networks para magklasipika ng datos](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), ngunit ito ay lampas sa saklaw ng leksiyong ito.

### Alin ang classifier na pipiliin?

Kaya, alin nga ba ang classifier na pipiliin mo? Madalas, ang pagtakbo ng ilan at paghahanap ng magandang resulta ay isang paraan ng pagsubok. Nagbibigay ang Scikit-learn ng [side-by-side comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) sa isang ginawa na dataset, na ikinukumpara ang KNeighbors, SVC sa dalawang paraan, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB at QuadraticDiscrinationAnalysis, na ipinapakita ang mga resulta na na-visualize:

![comparison of classifiers](../../../../translated_images/tl/comparison.edfab56193a85e7f.webp)
> Mga plot na ginawa mula sa dokumentasyon ng Scikit-learn

> Nilulutas ng AutoML ang problemang ito nang maayos sa pamamagitan ng pagpapatakbo ng mga paghahambing na ito sa cloud, na nagbibigay-daan sa iyo upang piliin ang pinakamahusay na algorithm para sa iyong datos. Subukan ito [dito](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Isang mas mabuting pamamaraan

Isang mas mabuting paraan, kaysa sa bulag na paghuhula, ay ang sundin ang mga ideya sa nai-download na [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Dito, natuklasan natin na, para sa ating multiclass na problema, may ilang pagpipilian tayo:

![cheatsheet for multiclass problems](../../../../translated_images/tl/cheatsheet.07a475ea444d2223.webp)
> Isang bahagi ng Microsoft's Algorithm Cheat Sheet, na nagpapaliwanag ng mga pagpipilian para sa multiclass classification

✅ I-download ang cheat sheet na ito, i-print at idikit sa iyong dingding!

### Pangangatwiran

Subukan nating hasain ang ating pag-iisip sa iba't ibang pamamaraan sa ibinigay na mga limitasyon natin:

- **Masyadong mabigat ang neural networks**. Dahil sa malinis ngunit minimal na dataset, at sapagkat nagpapatakbo tayo ng training nang lokal sa pamamagitan ng mga notebook, masyadong mabigat para sa gawain ito ang neural networks.
- **Walang two-class classifier**. Hindi natin ginagamit ang two-class classifier, kaya hindi nito kasama ang one-vs-all.
- **Maaaring gumana ang decision tree o logistic regression**. Maaaring gumana ang decision tree, o logistic regression para sa multiclass na datos.
- **Iba ang nilulutas ng Multiclass Boosted Decision Trees**. Ang multiclass boosted decision tree ay mas angkop para sa mga nonparametric na gawain, hal. gawain na nilalayong bumuo ng mga ranggo, kaya hindi ito kapaki-pakinabang para sa atin.

### Paggamit ng Scikit-learn

Gagamitin natin ang Scikit-learn para suriin ang ating datos. Gayunpaman, maraming paraan upang gamitin ang logistic regression sa Scikit-learn. Tingnan ang [mga parameter na ipapasa](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Sa pangkalahatan may dalawang mahahalagang parameter - `multi_class` at `solver` - na kailangang tukuyin kapag hilingin natin sa Scikit-learn na magsagawa ng logistic regression. Ang halaga ng `multi_class` ay nag-aaplay ng isang tiyak na pag-uugali. Ang halaga ng solver ay kung anong algorithm ang gagamitin. Hindi lahat ng solvers ay maaaring gamitin sa lahat ng `multi_class` na halaga.

Ayon sa dokumentasyon, sa multiclass na kaso, ang training algorithm:

- **Gumagamit ng one-vs-rest (OvR) scheme**, kung ang `multi_class` ay naka-set sa `ovr`
- **Gumagamit ng cross-entropy loss**, kung ang `multi_class` ay naka-set sa `multinomial`. (Sa kasalukuyan, ang `multinomial` ay sinusuportahan lamang ng ‘lbfgs’, ‘sag’, ‘saga’ at ‘newton-cg’ na solvers.)"

> 🎓 Ang 'scheme' dito ay maaaring 'ovr' (one-vs-rest) o 'multinomial'. Dahil ang logistic regression ay dinisenyo talaga para sa binary classification, pinahihintulutan ng mga scheme na ito na mas mahusay itong makayanan ang multiclass classification tasks. [source](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 Ang 'solver' ay tinukoy bilang "ang algorithm na gagamitin sa optimization problem". [source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Nag-aalok ang Scikit-learn ng talahanayan upang ipaliwanag kung paano hinahandle ng mga solvers ang iba't ibang hamon na dala ng iba't ibang uri ng istruktura ng datos:

![solvers](../../../../translated_images/tl/solvers.5fc648618529e627.webp)

## Exercise - hatiin ang datos

Maaari tayong magpokus sa logistic regression para sa ating unang trial ng training dahil kamakailan mo lang itong natutunan sa isang naunang leksiyon.
Hatiin ang iyong datos sa training at testing na mga grupo gamit ang `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Exercise - ilapat ang logistic regression

Dahil ginagamit mo ang multiclass na kaso, kailangan mong pumili kung anong _scheme_ ang gagamitin at anong _solver_ ang ise-set. Gamitin ang LogisticRegression na may multiclass setting at ang **liblinear** solver para mag-train.

1. Gumawa ng logistic regression na naka-set ang multi_class sa `ovr` at ang solver sa `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Subukan ang ibang solver tulad ng `lbfgs`, na madalas na default na ginagamit

    > Tandaan, gamitin ang Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) function para i-flatten ang iyong datos kapag kailangan.

    Maganda ang accuracy, higit sa **80%**!

1. Maaari mong makita ang modelong ito na gumagana sa pamamagitan ng pag-test ng isang row ng datos (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Ang resulta ay ipi-print:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Subukan ang ibang bilang ng row at suriin ang mga resulta
1. Masusing suriin, maaari mong tingnan ang katumpakan ng prediksyon na ito:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Ipinapakita ang resulta - ang Indian na lutuing pagkain ang pinakamalapit na hula, na may magandang probabilidad:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Maaari mo bang ipaliwanag kung bakit medyo sigurado ang modelo na ito ay isang Indian na lutuing pagkain?

1. Kunin ang mas detalyadong impormasyon sa pamamagitan ng pagprint ng classification report, tulad ng ginawa mo sa mga aralin sa regression:

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

## 🚀Hamunin

Sa araling ito, ginamit mo ang malinis mong data upang bumuo ng machine learning model na maaaring hulaan ang isang pambansang lutuing pagkain base sa serye ng mga sangkap. Maglaan ng oras upang basahin ang maraming opsyon na inaalok ng Scikit-learn para mag-classify ng data. Masusing pag-aralan ang konsepto ng 'solver' upang maunawaan kung ano ang nangyayari sa likod ng mga tagpo.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Pag-aaral sa Sarili

Suriin pa nang mas malalim ang matematika sa likod ng logistic regression sa [araling ito](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Takdang Aralin 

[Pag-aralan ang mga solvers](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Pagsusuri**:
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Habang nagsusumikap kami para sa katumpakan, pakitandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o di pagkakatugma. Ang orihinal na dokumento sa kanyang sariling wika ang dapat ituring na opisyal na sanggunian. Para sa mahahalagang impormasyon, inirerekomenda ang propesyonal na salin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na nagmula sa paggamit ng pagsasaling ito.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->