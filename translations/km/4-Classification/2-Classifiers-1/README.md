# ក្រុមចំរៀងម្ហូប ១

នៅក្នុងមេរៀននេះ អ្នកនឹងប្រើទិន្នន័យដែលបានរក្សាទុកពីមេរៀនមុន ដែលស្អាត និង មានតុល្យភាព ស្តីពីម្ហូបជាតិ។

អ្នកនឹងប្រើទិន្នន័យនេះជាមួយនឹងក្រុមចំរៀងដើម្បី _ទស្សន៍ទាយអំពីម្ហូបជាតិតាមគោលលទ្ធផលអំពីគ្រឿងផ្សំណាមួយ។_ ពេលធ្វើបែបនេះ អ្នកនឹងរៀនបន្ថែមអំពីវិធីផ្សេងៗដែលអាល់ហ្គូរីធម៍អាចប្រើសម្រាប់ភារកិច្ចចំរៀង។

## [តេស្តមុនវគ្គ](https://ff-quizzes.netlify.app/en/ml/)
# ការត្រៀមខ្លួន

សន្មត់ថាអ្នកបានបញ្ចប់ [មេរៀន ១](../1-Introduction/README.md) រួច សូមប្រាកដថា​មាន​ឯកសារ _cleaned_cuisines.csv_ នៅក្នុងថតមួយ​ដើម `/data` សម្រាប់មេរៀនទាំងបួននេះ។

## ធ្វើហាត់ - ទស្សន៍ទាយម្ហូបជាតិ

1. នៅក្នុងថត _notebook.ipynb_ នៃមេរៀននេះ នាំចូលឯកសារនោះ ព្រមជាមួយបណ្ណាល័យ Pandas៖

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    ទិន្នន័យបង្ហាញដូចជា៖

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. ឥឡូវនេះ នាំចូលបណ្ណាល័យបន្ថែមទៀត៖

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. ចែក X និង y ជាទម្រង់ជា dataframe ពីរ សម្រាប់ការបណ្តុះបណ្តាល។ `cuisine` អាចជាទម្រង់ស្លាក៖

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    វានឹងហាក់ដូចជា៖

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. លុបកូឡុំណ `Unnamed: 0` និង `cuisine` ដោយប្រើ `drop()` និងរក្សាទុកសំណុំទិន្នន័យសម្រាប់បណ្តុះបណ្តាលជាលក្ខណៈ features៖

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    លក្ខណៈ features របស់អ្នកដូចជា៖

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

ឥឡូវនេះអ្នកសម្រាកណាស់ក្នុងការបណ្តុះបណ្តាលម៉ូដែលរបស់អ្នក!

## ជ្រើសរើសក្រុមចំរៀងរបស់អ្នក

ឥឡូវនេះទិន្នន័យរបស់អ្នកបានស្អាត និងរួចរាល់សម្រាប់បណ្តុះបណ្តាល អ្នកត្រូវសំរេចថាតើអាល់ហ្គូរីធម៍ណាមួយត្រូវបានប្រើសម្រាប់ការងារ។

Scikit-learn ចែកចាយចំរៀងក្រោម ការរៀនដោយគ្រប់គ្រង (Supervised Learning) ហើយក្នុងប្រភេទនេះ អ្នកនឹងឃើញវិធីច្រើនសម្រាប់ចំរៀង។ [ភាពចម្រូងចម្រាស](https://scikit-learn.org/stable/supervised_learning.html) គឺគ្រាន់តែស្មុគស្មាញនៅលើភ្នែកដំបូង។ វិធីដូចខាងក្រោមទាំងអស់រួមបញ្ចូលបច្ចេកទេសចំរៀង៖

- ម៉ូដែលរាងបន្ទាត់ (Linear Models)
- ម៉ាស៊ីនពេញលេញគាំទ្រ (Support Vector Machines)
- Stochastic Gradient Descent
- អ្នកជិតខាងជិតបំផុត (Nearest Neighbors)
- Gaussian Processes
- សេចក្តីសម្រេចចិត្តដើមឈើ (Decision Trees)
- វិធីសាស្រ្តក្រុមប្រឹក្សា (ensemble methods - voting Classifier)
- អាល់ហ្គូរីធម៍មួយថ្នាក់ច្រើន និងចេញច្រើន (multiclass និង multilabel classification, multiclass-multioutput classification)

> អ្នកក៏អាចប្រើ [បណ្ដាញប្រសាទ (neural networks) សម្រាប់ចំរៀងទិន្នន័យ](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification) ប៉ុន្តែវាខាងក្រៅវិសាលភាពមេរៀននេះ។

### តើនឹងប្រើក្រុមចំរៀងណា?

ហេតុអ្វីបានជា អ្នកគួរជ្រើសរើសក្រុមចំរៀង? ជាញឹកញាប់ ការរត់តាមរយៈអ្នកជាច្រើន ហើយដើម្បីរកលទ្ធផលល្អគឺជាវិធីសាកល្បងមួយ។ Scikit-learn ផ្ដល់ [ការប្រៀបធៀបមុខមាត់](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) នៅលើទិន្នន័យដែលបានបង្កើត ប្រើប្រាស់ KNeighbors, SVC ២ ប្រភេទ, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB និង QuadraticDiscrinationAnalysis នៅក្នុងពិពណ៌នារាងតាងគ្នា៖ 

![ការប្រៀបធៀបនៃក្រុមចំរៀង](../../../../translated_images/km/comparison.edfab56193a85e7f.webp)
> គំនូសតាងបង្កើតឡើងក្នុងឯកសាររបៀបប្រើរបស់ Scikit-learn

> AutoML ដោះស្រាយបញ្ហានេះដោយរត់ការប្រៀបធៀបទាំងនេះនៅក្នុងពពក ជួយឱ្យអ្នកជ្រើសរើសអាល់ហ្គូរីធម៍ល្អបំផុតសម្រាប់ទិន្នន័យរបស់អ្នក។ សាកល្បងបាន [ទីនេះ](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### មុខងារល្អជាងនេះ

វិធីល្អជាងការព្យាយាមមូល ដោយគ្មានផ្ទៃ អាចជាមើលទៅតាមគំនិតនៅលើ [សន្លឹក​ជំនួយ ML Cheat sheet នេះ](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) ដែលអ្នកអាចទាញយកបាន។ នៅទីនេះ អ្នករកឃើញថា សម្រាប់បញ្ហាប្រភេទ multiclass មានជម្រើសខ្លះៗ៖

![សន្លឹកជំនួយសម្រាប់បញ្ហា multiclass](../../../../translated_images/km/cheatsheet.07a475ea444d2223.webp)
> ផ្នែកមួយនៃសន្លឹកបច្ចេកទេសម៉ាស៊ីនរៀនរបស់ Microsoft ពិពណ៌នាជម្រើសចំរៀង multiclass

✅ ទាញយកសន្លឹកនេះ ម្រី.Print និងតាមដានវាលើជញ្ជាំងរបស់អ្នក!

### ហេតុផល

មកមើលថាតើយើងអាចប្រើហេតុផលលើវិធីផ្សេងៗដោយផ្អែកលើកម្រិតដូចខាងក្រោម៖

- **បណ្ដាញប្រសាទធ្ងន់ពេក**។ ពីព្រោះទិន្នន័យរបស់យើងស្អាត បូមតិច ហើយការបណ្តុះបណ្តាលធ្វើនៅកន្លែងនីមួយៗ តាមរយៈកំណត់ត្រា notebook បណ្ដាញប្រសាទធ្ងន់ពេកសម្រាប់ភារកិច្ចនេះ។
- **មិនប្រើក្រុមចំរៀងពីរថ្នាក់**។ យើងមិនប្រើក្រុមចំរៀងពីរថ្នាក់ ដូច្នេះមិនអាចប្រើ one-vs-all ទេ។
- **ដើមឈើសម្រេចចិត្ត ឬ logistic regression អាចដំណើរការ**។ ដើមឈើសម្រេចចិត្តអាចដំណើរការ ឬ logistic regression សម្រាប់ទិន្នន័យ multiclass។
- **Multiclass Boosted Decision Trees ដំណោះស្រាយបញ្ហាផ្សេង**។ Multiclass boosted decision tree សាកសមសម្រាប់ភារកិច្ចគ្មានប៉ារ៉ាម៉ែត្រ ឧ. ភារកិច្ចបង្កើតតារាងជាដើម ដូច្នេះវាមិនប្រើបានសម្រាប់យើងទេ។

### ការប្រើប្រាស់ Scikit-learn

យើងនឹងប្រើ Scikit-learn សម្រាប់វិភាគទិន្នន័យ។ ទោះយ៉ាងណា មានវិធីច្រើនក្នុងការប្រើ logistic regression ក្នុង Scikit-learn។ សូមមើល [ប៉ារ៉ាម៉ែត្រ](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) ដែលត្រូវផ្ដល់។

សំខាន់គឺមានប៉ារ៉ាម៉ែត្រ ២ - `multi_class` និង `solver` - ដែលយើងត្រូវកំណត់នៅពេលសុំឱ្យ Scikit-learn ប្រតិបត្តិ logistic regression។ តម្លៃ `multi_class` បញ្ជាក់អំពីតម្រូវការមួយ។ តម្លៃ `solver` ជាអាល់ហ្គូរីធម៍ដែលត្រូវប្រើ។ មិនមែនគ្រប់ solver អាចប្រើជាមួយតម្លៃ `multi_class` ទាំងអស់បានទេ។

យោងតាមឯកសារ ក្នុងករណី multiclass អាល់ហ្គូរីធម៍បណ្តុះបណ្តាល៖

- **ប្រើវិធី one-vs-rest (OvR)** ប្រសិនបើ `multi_class` ត្រូវបានកំណត់ជាសោភាគ `ovr`
- **ប្រើបាត់បង់ cross-entropy** ប្រសិនបើ `multi_class` ត្រូវបានកំណត់ជាសោភាគ `multinomial`។ (បច្ចុប្បន្ន, ជម្រើស `multinomial` គាំទ្រដោយ solvers `lbfgs`, `sag`, `saga` និង `newton-cg` តែម្តង។)"

> 🎓 វិធីសាស្រ្ត 'scheme' នៅទីនេះអាចជា 'ovr' (one-vs-rest) ឬ 'multinomial'។ ពីព្រោះ logistic regression គិតគឺរចនាឡើងសម្រាប់ចំរៀងពីរជំរើស ប៉ុន្តែលំហែ ផែនការនេះអាចជួយដល់ការចំរៀង multiclass។ [ប្រភព](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' ត្រូវបានកំណត់ថា "អាល់ហ្គូរីធម៍ដែលប្រើសម្រាប់បញ្ហាកំណត់អត្តសញ្ញាណ"។ [ប្រភព](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn ផ្ដល់តារាងនេះដើម្បីពន្យល់ពីរបៀបដែល solvers ដោះស្រាយបញ្ហាតាមរយៈរចនាសម្ព័ន្ធទិន្នន័យផ្សេងៗ៖

![solvers](../../../../translated_images/km/solvers.5fc648618529e627.webp)

## ធ្វើហាត់ - ផែក្នុងទិន្នន័យ

យើងអាចផ្នែកទិន្នន័យទៅក្រុមបណ្តុះបណ្តាល និងក្រុមសាកល្បង ដោយហៅ `train_test_split()`៖

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## ធ្វើហាត់ - ប្រើ logistic regression

ដោយសារតែអ្នកប្រើกรณี multiclass អ្នកត្រូវជ្រើសរើស _scheme_ មួយ និង _solver_ មួយ។ ប្រើ LogisticRegression ជាមួយការ​កំណត់ multiclass និង solver ជា **liblinear** ដើម្បីបណ្តុះបណ្តាល។

1. បង្កើត logistic regression ជាមួយ `multi_class` កំណត់ជា `ovr` និង `solver` កំណត់ជា `liblinear`៖

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ សាកល្បង solver ផ្សេងទៀតដូចជា `lbfgs` ដែលជាការកំណត់លំនាំដើមជាញឹកញាប់។

    > កំណត់សម្គាល់ ប្រើមុខងារ Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) ដើម្បីបង្រួមទិន្នន័យរបស់អ្នកពេលត្រូវការ។

    ភាពត្រឹមត្រូវល្អជាង **៨០%**!

1. អ្នកអាចមើលម៉ូដែលនេះដំណើរការដោយសាកល្បងមួយជំរើស (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    លទ្ធផលបង្ហាញ៖

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ សាកល្បងជួរលេខផ្សេងទៀត ហើយពិនិត្យលទ្ធផល។
1. ការចាក់ចូលជ្រៅ, អ្នកអាចពិនិត្យមើលភាពត្រឹមត្រូវនៃការព្យាករណ៍នេះ៖

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    លទ្ធផលត្រូវបានបោះពុម្ព - មុខម្ហូបឥណ្ឌា គឺការព្យាករណ៍ល្អបំផុតរបស់វា មានលទ្ធភាពល្អ៖

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ តើអ្នកអាចពន្យល់ថា ហេតុអ្វីបានជា ម៉ូដែលប្រាកដជាមានចិត្តថាផលិតផលនេះគឺម្ហូបឥណ្ឌា?

1. ទទួលបានព័ត៌មានលម្អិតបន្ថែមដោយបោះពុម្ពរបាយការណ៍ចំណាត់ថ្នាក់ ដូចដែលអ្នកបានធ្វើនៅមេរៀនស្តីពីការ​បង្រួម​សមាមាត្រ៖

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

## 🚀ការប្រកួតប្រជែង

នៅក្នុងមេរៀននេះ អ្នកបានប្រើទិន្នន័យដែលបានសម្អាតរួច ដើម្បីបង្កើតម៉ូដែលរៀនដោយម៉ាស៊ីនមួយ ដែលអាចព្យាករណ៍មុខម្ហូបជាតិតាមបច្ចេកទេសជាជួរដោយផ្អែកលើគ្រឿងផ្សំជាច្រើន។ ចំណាយពេលអានតាមជម្រើសជាច្រើនដែល Scikit-learn ផ្ដល់ជូនសម្រាប់ចំណាត់ថ្នាក់ទិន្នន័យ។ ចាក់បញ្ចូលជ្រៅក្នុងគំនិត 'solver' ដើម្បីយល់ពីអ្វីដែលកើតឡើងនៅខាងក្រោយឆាក។

## [សំនួរបន្ទាប់មេរៀន](https://ff-quizzes.netlify.app/en/ml/)

## សរុប & សិក្សាឯក្យ

ចាក់បញ្ចូលជ្រៅជាងនេះទៀតចំពោះគណិតវិទ្យាខាងក្រោយនៃការបង្រួមសមាមាត្រ នៅក្នុង [មេរៀននេះ](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## ភារកិច្ច 

[សិក្សាអំពី solvers](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ការបដិសេធ**៖  
ឯកសារនេះត្រូវបានបកប្រែដោយប្រើសេវាបកប្រែ AI [Co-op Translator](https://github.com/Azure/co-op-translator)។ ខណៈពេលយើងព្យាយាមឲ្យបានច្បាស់លាស់ សូមយល់ថាការបកប្រែដោយស្វ័យប្រវត្តិនេះអាចមានកំហុស ឬការមិនត្រឹមត្រូវ។ ឯកសារដើមក្នុងភាសាដើមគេគួរត្រូវបានចាត់ទុកថាជាធនធានមានអំណាច។ សម្រាប់ព័ត៌មានសំខាន់ៗ ការបកប្រែដោយមនុស្សជំនាញត្រូវបានណែនាំ។ យើងមិនទទួលខុសត្រូវចំពោះការយល់ច្រឡំនូវអ្វីៗណាមួយ ឬការបកប្រែខុសដែលកើតមានពីការប្រើប្រាស់ការបកប្រែនេះឡើយ។
<!-- CO-OP TRANSLATOR DISCLAIMER END -->