# వంటకాల శ్రేణీకరణకారులు 1

ఈ పాఠంలో, మీరు గత పాఠం నుండి సేవ్ చేసిన సమతుల్యమైన, శుభ్రమైన వంటకాల గురించి డేటా ఉన్న dataset ను ఉపయోగిస్తారు.

మీరు ఈ డేటాసెట్‌ను వివిధ శ్రేణీకరణకారులతో ఉపయోగించి _ఉన్న పదార్థాల సమూహం ఆధారంగా ఇచ్చిన జాతీయ వంటకాన్ని ఊహించడానికి_ ప్రయత్నిస్తారు. ఈ ప్రక్రియలో, శ్రేణీకరణ పనులకు ఆల్గోరిథమ్లు ఎలా ఉపయోగించబడుతాయో మీరు మరింత తెలుసుకుంటారు.

## [పూర్వ- విభాగ పరీక్ష](https://ff-quizzes.netlify.app/en/ml/)
# సరిపోల్చు

మీరు [పాఠం 1](../1-Introduction/README.md) పూర్తి చేశారని ఊహిస్తే, ఈ నాలుగు పాఠాల కోసం _cleaned_cuisines.csv_ ఫైల్ మూల /data ఫోల్డరులో ఉండడం నిర్థారించండి.

## వ్యాయామం - జాతీయ వంటకాన్ని ఊహించండి

1. ఈ పాఠం యొక్క _notebook.ipynb_ ఫోల్డర్లో, ఆ ఫైల్ ను మరియు Pandas లైబ్రరీని దిగుమతి చేసుకోండి:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    డేటా ఇలా ఉంటుందిః

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. ఇప్పుడు, మరికొన్ని లైబ్రరీలను దిగుమతి చేసుకోండి:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. X మరియు y కోఆర్డినేట్‌లను రెండు డేటాఫ్రేమ్లుగా విభజించి శిక్షణ కోసం తయారుచేయండి. `cuisine` ను లేబుల్స్ డేటాఫ్రేమ్ గా పరిగణించవచ్చు:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    ఇది ఇలా కనిపిస్తుంది:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. `Unnamed: 0` కాలమ్ మరియు `cuisine` కాలమ్ ను `drop()` ఫంక్షన్ తో తీసివేయండి. మిగిలిన డేటాను శిక్షణకు అనువుగా ఫీచర్లు గా సేవ్ చేయండి:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    మీ ఫీచర్లు ఈ విధంగా ఉంటాయి:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

ఇప్పుడు మీరు మీ మోడల్ శిక్షణకు సిద్ధంగా ఉన్నారు!

## మీ శ్రేణీకరణకారుడు ఎంచుకోవడం

మీ డేటా శుభ్రం మరియు శిక్షణకు సిద్ధంగా ఉన్నప్పటి నుంచి, మీరు ఏ ఆల్గోరిథమ్ ను ఉపయోగించాలో నిర్ణయించుకోవాలి.

Scikit-learn శ్రేణీకరణను సూపర్వైజ్డ్ లర్నింగ్ క్రింద గ్రూప్ చేస్తుంది, ఆ విభాగంలో మీరు ఎన్నో శ్రేణీకరణ పద్ధతులను కనుగొంటారు. [ఈ వైవిధ్యం](https://scikit-learn.org/stable/supervised_learning.html) మొదట దృష్టికి కొంచెం గందరగోళంగా ఉంటుంది. క్రింది పద్ధతులు శ్రేణీకరణ సాంకేతికతలను కలిగి ఉంటాయి:

- రేఖీయ నమూనాలు (Linear Models)
- సపోర్ట్ వెక్టర్ మెషీన్లు (Support Vector Machines)
- స్టోచాస్టిక్ గ్రేడియెంట్ డిసెంట్
- సమీప పొరుగువార్లు (Nearest Neighbors)
- గౌసియన్ ప్రాసెస్‌లు
- నిర్ణయం చెట్లు (Decision Trees)
- అనేక పద్ధతులు (Ensemble methods, ఉదాహరణకు వోటింగ్ శ్రేణీకరణకారులు)
- బహుళ తరగతులు మరియు బహుళ అవుట్‌పుట్ ఆల్పోరిథములు (multiclass మరియు multilabel classification, multiclass-multioutput classification)

> మీరు కూడా [న్యూరల్ నెట్‌వర్క్లు ఉపయోగించి డేటాను వర్గీకరించవచ్చు](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), కానీ ఇది ఈ పాఠం పరిధికి సంబంధించదు.

### ఏ శ్రేణీకరణకారుడ్ని ఎంచుకోవాలి?

అంటే, ఏ శ్రేణీకరణకారుడ్ని ఎంచుకోవాలి? తరచుగా, పలు శ్రేణీకరణకారులను ప్రయత్నించి, మంచి ఫలితాన్ని చూసి నిర్ణయం తీసుకోవడం ఉత్తమం. Scikit-learn ఒక [సంకలనం](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)ను అందిస్తుంది, ఇందులో క్రింది శ్రేణీకరణకారులను ఒక సృష్టించబడిన dataset మీద పక్కిన పక్క ధరించి కొలుస్తుంది: KNeighbors, రెండు రకాల SVC, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB మరియు QuadraticDiscrinationAnalysis, ఫలితాలను దృశ్యమానం చేస్తూ:

![classification కాంపారిజన్](../../../../translated_images/te/comparison.edfab56193a85e7f.webp)
> ప్లాట్లు Scikit-learn డాక్యుమెంటేషన్ నుండి

> AutoML ఈ సమస్యను క్లౌడ్‌లో ఈ కాంపారిజన్లను నడుపుతూ సులభంగా పరిష్కరించగలదు, మీ డేటాకు ఉత్తమ ఆల్గోరిథమ్ ఎంచుకోవడానికి వీలు కల్పిస్తుంది. దీన్ని [ఇక్కడ](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott) ట్రై చేయండి

### మరింత ఉత్తమ విధానం

అనుకున్నదానికన్నా మంచి విధానం ఈ డౌన్లోడ్ చేసుకునే [ML చీట్‌షీట్](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) లోని ఆలోచనలను అనుసరించడం. ఇక్కడ మేము మా బహుళ తరగతి సమస్యకు కొన్ని ఎంపికలు కనిపిస్తాయి:

![బహుళ తరగతి సమస్యలకు చీట్‌షీట్](../../../../translated_images/te/cheatsheet.07a475ea444d2223.webp)
> మైక్రోసాఫ్ట్ యొక్క Algorithm Cheat Sheet నుండి ఒక భాగం, బహుళ తరగతి శ్రేణీకరణ ఎంపికలను వివరించే విధానం

✅ ఈ చీట్‌షీట్‌ను డౌన్లోడ్ చేసి, ముద్రించి, మీ గోడపై చిందించండి!

### తర్కం

మాకు ఉన్న పరిమితులు ఆధారంగా వివిధ పద్ధతులను మనం తర్కం చేయవచ్చా చూద్దాం:

- **న్యూరల్ నెట్‌వర్క్లు చాలా భారమైనవి**. మా శుభ్రపరిచిన మరియు తక్కువ డేటాను తీసుకొని, నోట్‌బుక్స్ లో స్థానికంగా శిక్షణ ఇస్తుండగా, న్యూరల్ నెట్‌వర్క్లు ఈ పని కోసం భారమైనవి అవుతాయి.
- **రెండు తరగతుల శ్రేణీకరణకారి లేదు**. మేము రెండు తరగతుల శ్రేణీకరణకారిని ఉపయోగించవద్దు, కాబట్టి ఒకటి వర్సస్ అన్ని విధానం ఉపయోగించరు.
- **నిర్ణయ పెద్ద చెట్టు లేదా లాజిస్టిక్ రిగ్రెషన్ పనిచేయచ్చు**. నిర్ణయ పెద్ద చెట్టు పనిచేయవచ్చు, లేదా బహుళ తరగతి డేటాకు లాజిస్టిక్ రిగ్రెషన్ ఉపయోగించవచ్చు.
- **బహుళ తరగతి బూస్ట్ చేసిన నిర్ణయ చెట్లు వేరు సమస్యను పరిష్కరిస్తాయి**. బహుళ తరగతి బూస్ట్ చేసిన నిర్ణయ చెట్టు ఎక్కువగా ర్యాంకింగ్‌లను తయారు చేయడానికి ఉపయోగపడుతుంది, కాబట్టి ఇది మన పని కోసం ఉపయోగకరం కాదు.

### Scikit-learn వాడకం

మేము మా డేటా విశ్లేషించడానికి Scikit-learn ఉపయోగించబోతున్నాం. అయితే, Scikit-learn లో లాజిస్టిక్ రిగ్రెషన్ ఉపయోగించే అనేక మార్గాలు ఉన్నాయి. [పారామీటర్లను](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) ఒకసారి చూడండి.  

ప్రధానంగా రెండు ముఖ్యమైన పారామీటర్లు - `multi_class` మరియు `solver` - ని గుర్తించాలి, మీరు Scikit-learn దగ్గర లాజిస్టిక్ రిగ్రెషన్ చేయమని అడిగేటప్పుడు. `multi_class` విలువు ఒక నిర్దిష్ట ప్రవర్తనను అమలు చేస్తుంది. solver విలువు అంటే ఏ ఆల్గోరిథమ్ ఉపయోగించాలో నిర్ణయం. అన్ని solver లు అన్ని `multi_class` విలువలతో జత చేయకపోవచ్చు.

డాక్యుమెంటేషన్ ప్రకారం, బహుళ తరగతి సందర్భంలో శిక్షణ ఆల్గోరిథమ్:

- **`multi_class` ఎంపిక `ovr` గా ఉంటే, ఒకటి వర్సెస్ ఇతరుల (OvR) స్కీమ్ ఉపయోగిస్తుంది**
- **`multi_class` ఎంపిక `multinomial` గా ఉంటే, క్రాస్ ఎంట్రోపీ నష్టం (cross-entropy loss) ఉపయోగిస్తుంది**. (ప్రస్తుతానికి `multinomial` ఎంపిక ‘lbfgs’, ‘sag’, ‘saga’ మరియు ‘newton-cg’ solvers తో మాత్రమే మద్దతు ఉంది.)"

> 🎓 ఇక్కడి 'scheme' అనగా 'ovr' (ఒకటి వర్సెస్ రెస్ట్) లేదా 'multinomial'. లాజిస్టిక్ రిగ్రెషన్ సాధారణంగా ద్విమూల శ్రేణీకరణకు తొమ్మిదిటే అయితే, ఈ స్కీములు దానిని బహుళ తరగతుల శ్రేణీకరణ సమస్యలకు ఉత్తమంగా నిర్వహించడానికి సహాయపడతాయి. [మూలం](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' అనగా "ఆప్టిమైజేషన్ సమస్యలో ఉపయోగించే ఆల్గోరిథమ్". [మూలం](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn ఈ కింది పట్టిక ద్వారా solvers ఎలా వివిధ డేటా నిర్మాణాలను నిర్వహిస్తాయో వివరిస్తుంది:

![solvers](../../../../translated_images/te/solvers.5fc648618529e627.webp)

## వ్యాయామం - డేటా విభజించడం

మీరు గత పాఠంలో తెలుసుకున్న లాజిస్టిక్ రిగ్రెషన్ కోసం ప్రథమ శిక్షణ ప్రయత్నం focus చేసుకోవచ్చు.
`train_test_split()` ను పిలిచి మీ డేటాను శిక్షణ మరియు పరీక్షా సమూహాలుగా విభజించండి:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## వ్యాయామం - లాజిస్టిక్ రిగ్రెషన్ వర్తింపజేయండి

మీరు బహుళ తరగతి కేటగిరీలో ఉన్నందున, మీరు ఏ _scheme_ ఉపయోగించబడాలో మరియు ఏ _solver_ ను సెట్ చేయాలో ఎంచుకోవాలి. **liblinear** solverతో multi_class ని `ovr` గా సెట్ చేసి LogisticRegression ఉపయోగించి శిక్షణ ఇవ్వండి.

1. multi_class ను `ovr`, solver ను `liblinear` గా సెట్ చేసి లాజిస్టిక్ రిగ్రెషన్ రూపొందించండి:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ `lbfgs` వంటి వేరొక solverను ప్రయత్నించండి, ఇది తరచుగా డిఫాల్ట్ గా సెట్ ఉంటుంది

    > గమనిక: అవసరమైనప్పుడు మీ డేటాను సరళీకరించడానికి Pandas యొక్క [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) ఫంక్షన్ ఉపయోగించండి.

    ఖచ్చితత్వం **80%కి పైగా** బాగుంది!

1. ఒకటి వర్ధించండి, డేటాలో (#50) ఓ డేటా వరుసను పరీక్షించి మీరు ఈ మోడల్ పనిచేస్తున్నట్లు చూడవచ్చు:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    ఫలితం ముద్రించబడుతుంది:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

    ✅ వేరొక వరుస సంఖ్యను ప్రయత్నించి ఫలితాలను పరిశీలించండి
1. మరింత లోతుగా వెతకడం ద్వారా, ఈ ఊహించడంయ‌క‌ర‌త‌ను మీరు సరి చూసుకోవచ్చు:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    ఫలితం ప్రింట్ అయింది - భారతీయ వంటకం దీని ఉత్తమ అంచనా, మంచి సంభావ్యతతో:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ ఈ భయంకరమైన మోడల్ ఈ వంటకాన్ని భారతీయ వంటకం అని ఎందుకు భావిస్తున్నదో మీరు వివరిస్తారా?

1. మీరు రిగ్రెషన్ పాఠాలలో చేసినట్లు, క్లాసిఫికేషన్ నివేదికను ముద్రించి మరింత వివరణ పొందండి:

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

## 🚀సవాలు

ఈ పాఠంలో, మీరు శుభ్రపరిచిన డేటాను ఉపయోగించి పదార్థాల శ్రేణి ఆధారంగా జాతీయ వంటకం ఊహించే మెషిన్ లెర్నింగ్ మోడల్‌ను నిర్మించారు. డేటాను వర్గీకరించడానికి స్కైకిట్-లెర్న్ అందించే అనేక ఎంపికలను చదవడానికి కొన్ని సమయంగా వెచ్చించండి. సాల్వర్ (solver) అనే భావనను లోతుగా అర్థం చేసుకోవడం ద్వారా పిచ్చి వెనుక జరుగుతున్నది ఏమో తెలుసుకోండి.

## [పోస్ట్-లెక్చర్ క్విజ్](https://ff-quizzes.netlify.app/en/ml/)

## సమీక్ష & స్వయంఅధ్యయనం

లోజిస్టిక్ రిగ్రెషన్ వెనుక గణితాన్ని మరింత లోతుగా తెలుసుకోవడానికి [ఈ పాఠం](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf) చదవండి
## అసైన్‌మెంట్

[సాల్వర్లను అధ్యయనం చేయండి](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**అస్పష్టత**:  
ఈ డాక్యుమెంట్ [Co-op Translator](https://github.com/Azure/co-op-translator) అనే AI అనువాద సేవ ద్వారా అనువదించబడింది. మనం ఖచ్చితత్వానికి ప్రయత్నించినప్పటికీ, ఆటోమేటెడ్ అనువాదాల్లోలో తప్పులు లేదా అపరిష్కృతాంశాలు ఉండకపోవచ్చు. మౌలిక భాషలో ఉన్న ఒరిజినల్ డాక్యుమెంట్ ని అధికారిక మూలంగా పరిగణించాలి. అత్యవసర సమాచారానికి, వృత్తిపరమైన మానవ అనువాదం తీసుకోవడం సలహాగా ఇవ్వబడింది. ఈ అనువాదం వలన కలిగే ఏ మార్మిక అవగాహన లోపాలకు మేము బాధ్యత వహించము.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->