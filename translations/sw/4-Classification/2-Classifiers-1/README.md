# Wataalam wa vyakula 1

Katika somo hili, uta tumia seti ya data uliyoihifadhi kutoka somo la mwisho iliyojaa data safi na yenye usawa kuhusu vyakula vya kitaifa.

Utatumia seti hii ya data pamoja na wataalam mbalimbali ili _kutabiri aina ya chakula cha kitaifa kulingana na kundi la viungo_. Wakati wa kufanya hivyo, utajifunza zaidi kuhusu baadhi ya njia ambazo algorithimu zinaweza kutumika kwa kazi za utambuzi.

## [Jaribio kabla ya somo](https://ff-quizzes.netlify.app/en/ml/)
# Maandalizi

Kama umekamilisha [Somo la 1](../1-Introduction/README.md), hakikisha kwamba faili la _cleaned_cuisines.csv_ lipo katika folda ya mizizi `/data` kwa masomo haya manne.

## Zojo - tabiri aina ya chakula cha kitaifa

1. Ukiwa katika folda ya _notebook.ipynb_ ya somo hili, ingiza faili hilo pamoja na maktaba ya Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Data inaonekana kama ifuatavyo:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Sasa, ingiza maktaba zaidi:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Gawanya viwango vya X na y katika dataframes mbili kwa mafunzo. `cuisine` itakuwa dataframe ya lebo:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Itaonekana hivi:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Futa safu ya `Unnamed: 0` na safu ya `cuisine` kwa kutumia `drop()`. Hifadhi data inayobaki kama sifa za mafunzo:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Sifa zako zinaonekana kama ifuatavyo:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Sasa uko tayari kufundisha mfano wako!

## Kuchagua mtambuzi wako

Sasa data yako iko safi na tayari kwa mafunzo, lazima uamue algorithimu gani utumie kwa kazi hii.

Scikit-learn inaweka uainishaji chini ya Mafunzo Yaliyoongozwa (Supervised Learning), na katika kundi hilo utapata njia nyingi za kuainisha. [Aina tofauti](https://scikit-learn.org/stable/supervised_learning.html) zinaweza kukufanya machungu mwanzoni. Njia zifuatazo zote zinajumuisha mbinu za uainishaji:

- Mifano ya Mstari
- Mashine za Msaada wa Vektor (Support Vector Machines)
- Kushuka kwa Hatua za Kistokastiki (Stochastic Gradient Descent)
- Majirani Karibu (Nearest Neighbors)
- Mchakato wa Gaussian
- Miti ya Maamuzi (Decision Trees)
- Njia za Kikundi (voting Classifier)
- Algorithimu za Mifumo Mingi na Matokeo Mengi (multiclass and multilabel classification, multiclass-multioutput classification)

> Unaweza pia kutumia [mitandao ya neva kuainisha data](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), lakini hiyo iko nje ya wigo wa somo hili.

### Mtambuzi gani uchague?

Kwa hiyo, mtambuzi gani unapaswa kuchagua? Mara nyingi, kujaribu kadhaa na kutafuta matokeo mazuri ni njia ya kupima. Scikit-learn inatoa [kulinganisha kando kwa kando](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) kwa dataset iliyoanzishwa, ikilinganisha KNeighbors, SVC njia mbili, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB na QuadraticDiscrinationAnalysis, ikionyesha matokeo kwa picha: 

![comparison of classifiers](../../../../translated_images/sw/comparison.edfab56193a85e7f.webp)
> Michoro iliyotengenezwa katika nyaraka za Scikit-learn

> AutoML inalitatua tatizo hili kwa urahisi kwa kuendesha kulinganisha haya kwenye wingu, ikikupa uwezo wa kuchagua algorithimu bora kwa data yako. Jaribu [hapa](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Njia bora zaidi

Njia bora zaidi badala ya kubahatisha ni kufuata mawazo yaliyomo katika jarida la [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) ambalo unaweza kupakua. Hapa, tunagundua kwamba kwa tatizo letu la wunaji wengi, tuna chaguzi kadhaa:

![cheatsheet for multiclass problems](../../../../translated_images/sw/cheatsheet.07a475ea444d2223.webp)
> Sehemu ya Jarida la Microsoft’s Algorithm Cheat Sheet, likielezea chaguzi za uainishaji wa wunaji wengi

✅ Pakua jarida hili, lipige picha na ulike kwenye ukuta wako!

### Sababu

Tuchunguze kama tunaweza kupata muktadha wa njia tofauti ukizingatia vikwazo tulivyonavyo:

- **Mitandao ya neva ni mizito sana**. Kutokana na seti yetu safi lakini ndogo, na ukweli kwamba tunaendesha mafunzo kwa kutumia notebooks kwa mtaa, mitandao ya neva ni mizito sana kwa kazi hii.
- **Hakuna mtambuzi wa aina mbili**. Hatutumii mtambuzi wa aina mbili, kwa hiyo hatuwezi tumia one-vs-all.
- **Mti wa maamuzi au usambazaji wa logistic unaweza kufanya kazi**. Mti wa maamuzi unaweza kufanya kazi, au usambazaji wa logistic kwa data ya wunaji wengi.
- **Miti ya Maamuzi yenye Kuimarishwa kwa Wunaji Wengi ni tofauti**. Miti ya maamuzi yenye kuimarishwa kwa wunaji wengi ni bora kwa kazi zisizo na kanuni maalum, mfano kazi za kuunda vikundi, kwa hiyo haitatufaa sisi.

### Kutumia Scikit-learn

Tutakuwa tunatumia Scikit-learn kuchambua data yetu. Hata hivyo, kuna njia nyingi za kutumia usambazaji wa logistic katika Scikit-learn. Angalia [vigezo vya kuingiza](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Kwa ujumla kuna vigezo viwili muhimu - `multi_class` na `solver` - ambavyo tunahitaji kubainisha tunapoomba Scikit-learn ifanye usambazaji wa logistic. Thamani ya `multi_class` hutumia tabia fulani. Thamani ya solver ni algorithimu gani itumike. Sio solvers zote zinaweza kuambatana na thamani zote za `multi_class`.

Kulingana na nyaraka, katika kesi ya multiclass, algorithimu ya mafunzo:

- **Inatumia mpango wa one-vs-rest (OvR)**, ikiwa chaguo la `multi_class` limewekwa kuwa `ovr`
- **Inatumia hasara ya msalaba wa entropi**, ikiwa chaguo la `multi_class` limewekwa kuwa `multinomial`. (Kwa sasa chaguo la `multinomial` linaungwa mkono tu na solvers ‘lbfgs’, ‘sag’, ‘saga’ na ‘newton-cg’.)"

> 🎓 'Mpango' hapa unaweza kuwa 'ovr' (one-vs-rest) au 'multinomial'. Kwa kuwa usambazaji wa logistic umeundwa hasa kusaidia uainishaji wa binary, mipango hii huruhusu kushughulikia vizuri kazi za uainishaji wa wunaji wengi. [chanzo](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' yamefafanuliwa kama "algorithimu inayotumika kwenye tatizo la uboreshaji". [chanzo](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn inatoa jedwali hili kuelezea jinsi solvers zinavyoshughulikia changamoto tofauti zinazotokea kwa aina tofauti za muundo wa data:

![solvers](../../../../translated_images/sw/solvers.5fc648618529e627.webp)

## Zojo - gawanya data

Tunaweza kuzingatia usambazaji wa logistic kwa jaribio letu la kwanza la mafunzo kwa sababu umepata maelezo yake katika somo lililopita.
Gawanya data yako katika makundi ya mafunzo na majaribio kwa kutumia `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Zojo - tumia usambazaji wa logistic

Kwa kuwa unatumia kesi ya multiclass, unahitaji kuchagua _mpango_ utakao tumia na _solver_ utakao weka. Tumia LogisticRegression yenye mpangilio wa multiclass na solver ya **liblinear** kufundisha.

1. Tengeneza usambazaji wa logistic na multi_class ukiwa `ovr` na solver ukiwa `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Jaribu solver tofauti kama `lbfgs`, ambayo mara nyingi hutumika kwa default

    > Kumbuka, tumia Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) ili kurahisisha data yako unapotakiwa.

    Usahihi ni mzuri, zaidi ya **80%**!

1. Unaweza kuona mfano huu ukiwa unafanya jaribio kwa mstari mmoja wa data (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Matokeo yamechapishwa:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Jaribu nambari nyingine ya mstari na angalia matokeo yake
1. Kuchunguza zaidi, unaweza kukagua usahihi wa utabiri huu:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Matokeo yamechapishwa - vyakula vya India ndicho kinachokadiriwa vyema zaidi, kwa uwezekano mzuri:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Je, unaweza kuelezea kwanini mfano una uhakika mkubwa hii ni chakula cha India?

1. Pata maelezo zaidi kwa kuchapisha ripoti ya uainishaji, kama ulivyofanya katika masomo ya usawa:

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

## 🚀Changamoto

Katika somo hili, ulitumia data yako safi kujenga mfano wa kujifunza mashine ambao unaweza kutabiri chakula cha kitaifa kulingana na mfululizo wa viungo. Chukua muda kusoma kwa undani chaguzi nyingi Scikit-learn inazotoa kwa ajili ya kuainisha data. Kuchunguza zaidi dhana ya 'solver' ili kuelewa kinachotokea nyuma ya pazia.

## [Mtihani baada ya mihadhara](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio & Kujisomea

Chunguza kidogo zaidi hesabu nyuma ya usawa wa logistic katika [somo hili](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Kazi ya nyumbani 

[Jifunze kuhusu solvers](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Kandiyo**:
Hati hii imetafsiriwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kwa usahihi, tafadhali fahamu kuwa tafsiri za otomatiki zinaweza kuwa na makosa au upungufu wa usahihi. Hati ya asili katika lugha yake ya mama inapaswa kuchukuliwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatuna dhamana kwa maelewano au tafsiri potofu zinazotokana na matumizi ya tafsiri hii.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->