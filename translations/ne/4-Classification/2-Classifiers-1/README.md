# पकवान वर्गीकरणकर्ता 1

यस पाठमा, तपाईंले अघिल्लो पाठबाट बचत गर्नुभएको सामग्रीहरू भरिएको सन्तुलित, सफा डाटासेट प्रयोग गर्नुहुनेछ जुन सबै पकवानहरूका बारेमा छ।

तपाईं यस डाटासेटलाई विभिन्न वर्गीकरणकर्ताहरूका साथ प्रयोग गरेर _दिएको सामग्री समूहको आधारमा राष्ट्रिय पकवान पूर्वानुमान गर्न_ प्रयोग गर्नुहुनेछ। यस क्रममा, तपाईंले वर्गीकरण कार्यहरूको लागि एल्गोरिदमहरूलाई कसरी उपयोग गर्न सकिन्छ भन्ने केहि तरिकाहरूको बारेमा थप जान्नु हुनेछ।

## [पूर्व-व्याख्यान प्रश्नोत्तरी](https://ff-quizzes.netlify.app/en/ml/)
# तयारी

मानौं तपाईंले [पाठ 1](../1-Introduction/README.md) पूरा गर्नुभयो भने, पक्का गर्नुहोस् कि _cleaned_cuisines.csv_ फाइल मूल `/data` फोल्डरमा यी चार पाठका लागि अवस्थित छ।

## अभ्यास - राष्ट्रिय पकवान अनुमान गर्न

1. यस पाठको _notebook.ipynb_ फोल्डर भित्र, त्यो फाइल र Pandas लाइब्रेरी आयात गर्नुहोस्:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    डाटा यस प्रकार देखिन्छ:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. अब, धेरै लाइब्रेरीहरू आयात गर्नुहोस्:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. X र y कोष्ठकलाई दुई डाटाफ्रेमहरूमा विभाजन गर्नुहोस् प्रशिक्षणका लागि। `cuisine` लेबलहरू भएको डाटाफ्रेम हुन सक्छ:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    यसले यसरी देखिन्छ:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. `Unnamed: 0` स्तम्भ र `cuisine` स्तम्भलाई `drop()` कल गरेर हटाउनुहोस्। बाँकी डाटालाई प्रशिक्षण योग्य विशेषताहरूको रूपमा सेभ गर्नुहोस्:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    तपाईँका विशेषताहरू यस प्रकार छन्:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

अब तपाईं आफ्नो मोडल तालिम दिन तयार हुनुहुन्छ!

## तपाईँको वर्गीकरणकर्ता चयन

अब जब तपाईंको डाटा सफा र तालिमका लागि तयार छ, तपाईंले कुन एल्गोरिदम कामका लागि प्रयोग गर्ने निर्णय गर्नुपर्छ।

Scikit-learn ले वर्गीकरणलाई Supervised Learning अन्तर्गत राख्दछ, र त्यस वर्गमा तपाईंलाई वर्गीकरण गर्ने धेरै तरिकाहरू पाइनुहुनेछ। [विविधता](https://scikit-learn.org/stable/supervised_learning.html) पहिलो दृष्टिमा अलि भ्रमपूर्ण हुन सक्छ। निम्न तरिकाहरू सबैमा वर्गीकरण प्रविधिहरू समावेश छन्:

- रैखिक मोडेलहरू (Linear Models)
- समर्थन भेक्टर मेशिनहरू (Support Vector Machines)
- स्टोकेस्टिक ग्रेडियन्ट डिजेन्ट (Stochastic Gradient Descent)
- नजिकैका छिमेकीहरू (Nearest Neighbors)
- गौसियन प्रोसेसहरू (Gaussian Processes)
- निर्णय वृक्षहरू (Decision Trees)
- समूह विधिहरू (voting Classifier)
- बहुवर्गीय र बहु-आउटपुट एल्गोरिदमहरू (multiclass and multilabel classification, multiclass-multioutput classification)

> तपाईं [न्यूरल नेटवर्कहरू](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification) पनि डाटालाई वर्गीकरण गर्न प्रयोग गर्न सक्नुहुन्छ, तर त्यो यस पाठको दायरामा पर्दैन।

### कुन वर्गीकरणकर्ता छनौट गर्ने?

त्यसैले, कुन वर्गीकरणकर्ता चयन गर्ने? प्रायः, थुप्रै विकल्पहरूको प्रयोग गरेर राम्रो नतिजा खोज्न एउटा तरिका हो परीक्षण गर्न। Scikit-learn ले [पारस्परिक तुलना](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) प्रदान गर्दछ जुन सिर्जना गरिएको डाटासेटमा KNeighbors, SVC दुई तरिकाले, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB र QuadraticDiscrinationAnalysis लाई तुलना गरी नतिजा दृश्यहरु देखाउँछ:

![classification को तुलना](../../../../translated_images/ne/comparison.edfab56193a85e7f.webp)
> Scikit-learn को डकुमेन्टेसनमा बनेका प्लटहरू

> AutoML ले यो समस्या कुशलतापूर्वक समाधान गर्दछ जसले यी तुलना बादलमा चलाएर तपाईंको डाटाका लागि सबैभन्दा उपयुक्त एल्गोरिदम छनौट गर्न अनुमति दिन्छ। यसलाई [यहाँ](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott) प्रयास गर्नुहोस्।

### राम्रो तरिका

तरिकाहरूको अनुमान गर्नुभन्दा राम्रो तरिका, यो डाउनलोड गर्न मिल्ने [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) मा भएका सुझावहरू पालना गर्नु हो। यहाँ हामी पाउँछौं कि हाम्रो बहुवर्गीय समस्याको लागि हामीसँग केहि विकल्पहरू छन्:

![बहुवर्ग समस्याहरूको लागि cheatsheet](../../../../translated_images/ne/cheatsheet.07a475ea444d2223.webp)
> Microsoft को Algorithm Cheat Sheet को एक भाग, बहुवर्ग वर्गीकरण विकल्पहरू विवरण गर्दै

✅ यो cheat sheet डाउनलोड गरेर प्रिन्ट गर्नुहोस् र आफ्नो पर्खालमा झुण्ड्याउनुहोस्!

### तार्किक सोच

हेरौं कि हामीले विभिन्न दृष्टिकोणहरूलाई हाम्रा सिमितताहरूलाई मध्यनजर गर्दै कसरी सोच्न सक्छौं:

- **न्यूरल नेटवर्कहरू धेरै भारी हुन्छन्।** हाम्रो सफा तर न्यूनतम डाटासेट र स्थानीय नोटबुक मार्फत तालिम चलाउने तथ्यलाई ध्यानमा राख्दा, न्यूरल नेटवर्कहरू यस कार्यका लागि धेरै भारी हुन्छन्।
- **दुई-वर्गीय वर्गीकरणकर्ता छैन।** हामी दुई-वर्गीय वर्गीकरणकर्ता प्रयोग गर्दैनौं, यसले one-vs-all विकल्पलाई बाहिर गर्छ।
- **निर्णय वृक्ष वा Logistic Regression काम हुन सक्छ।** निर्णय वृक्ष काम गर्न सक्छ, वा बहुवर्गीय डाटाको लागि Logistic Regression।
- **बहुवर्ग बूस्टेड निर्णय वृक्ष फरक समस्या समाधान गर्छ।** बहुवर्ग बूस्टेड निर्णय वृक्ष गैर-प्यारामेट्रिक कार्यहरूमा उपयुक्त हुन्छ, जस्तै र्‍याङ्किङ निर्माण गर्न डिजाईन गरिएको कार्यहरूमा, त्यसैले हाम्रो लागि उपयोगी हुँदैन।

### Scikit-learn को प्रयोग

हामी हाम्रो डाटा विश्लेषण गर्न Scikit-learn प्रयोग गर्नेछौं। यद्यपि, Scikit-learn मा Logistic Regression प्रयोग गर्ने धेरै तरिकाहरू छन्। [पास गर्नुपर्ने प्यारामिटरहरू](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) हेर्नुस्।

मूल रूपमा दुई महत्वपूर्ण प्यारामिटरहरू छन् - `multi_class` र `solver` - जुन हामीले Logistic Regression प्रदर्शन गर्न Scikit-learn लाई सोध्दा निर्दिष्ट गर्नुपर्छ। `multi_class` मानले एउटा व्यवहार लागू गर्छ। solver को मान कुन एल्गोरिदम प्रयोग गर्ने हो भनेर जनाउँछ। सबै solver हरू सबै प्रकारका `multi_class` मानहरूसँग जोडी हुँदैनन्।

डकुमेन्टेसन अनुसार, बहुवर्ग अवस्थामा, तालिम एल्गोरिदम:

- **one-vs-rest (OvR) योजना प्रयोग गर्छ**, जब `multi_class` विकल्प `ovr` मा सेट गरिएको छ
- **cross-entropy हानि प्रयोग गर्छ**, जब `multi_class` विकल्प `multinomial` मा सेट गरिएको छ। (हाल `multinomial` विकल्पले ‘lbfgs’, ‘sag’, ‘saga’ र ‘newton-cg’ solver हरूलाई मात्र समर्थन गर्दछ।)"

> 🎓 यहाँ 'योजना' हुन सक्छ 'ovr' (one-vs-rest) वा 'multinomial'। Logistic regression वास्तवमा द्विक वर्गीकरणलाई समर्थन गर्न बनाइएकोले यी योजनाहरूले बहुवर्ग वर्गीकरण कार्यहरूलाई राम्रोसँग सम्हाल्न अनुमति दिन्छ। [स्रोत](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' लाई "अप्टिमाइजेशन समस्यामा प्रयोग गरिने एल्गोरिदम" भनिएको छ। [स्रोत](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)

Scikit-learn ले यस तालिकाले विभिन्न डाटा संरचनाहरूले प्रस्तुत गरेको विभिन्न चुनौतीहरूलाई कसरी solver हरूले सम्हाल्छन् भन्ने बुझाउँछ:

![solvers](../../../../translated_images/ne/solvers.5fc648618529e627.webp)

## अभ्यास - डाटा विभाजन गर्ने

हामी पहिलो प्रशिक्षण प्रयासको लागि Logistic Regression मा केन्द्रित हुन सक्छौं किनभने तपाईंले यो अघिल्लो पाठमा सिक्नुभएको छ।
तपाईंको डाटालाई प्रशिक्षण र परीक्षण समूहमा `train_test_split()` कल गरेर विभाजन गर्नुहोस्:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## अभ्यास - Logistic Regression लागू गर्ने

तपाईंले बहुवर्ग अवस्था प्रयोग गर्दै हुनुहुन्छ, त्यसैले तपाईंले कुन _योजना_ प्रयोग गर्ने र कुन _solver_ सेट गर्ने छनौट गर्नुपर्छ। LogisticRegression लाई बहुवर्ग सेटिङ र **liblinear** solver सहित तालिम दिन प्रयोग गर्नुहोस्।

1. `multi_class`लाई `ovr` र solver लाई `liblinear` मा सेट गर्दै logistic regression बनाउनुहोस्:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ प्रायः डिफल्टका रूपमा सेट गरिने `lbfgs` जस्ता अर्को solver प्रयास गर्नुहोस्।

    > ध्यान दिनुहोस्, जब आवश्यक पर्छ तपाईँले डाटालाई फ्ल्याटन गर्न Pandas को [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) फंक्सन प्रयोग गर्न सक्नुहुन्छ।

    शुद्धता ८०% भन्दा माथि राम्रो छ!

1. तपाईं एक डाटाको एउटा पंक्ति (#50) परीक्षण गरेर यो मोडल क्रियाशील देख्न सक्नुहुन्छ:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    नतिजा छापिन्छ:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ अर्को पंक्ति संख्या प्रयास गर्नुहोस् र नतिजा जाँच गर्नुहोस्।
1. अझ गहिराईमा हेर्दा, तपाईंले यस भविष्यवाणीको सटीकता जाँच गर्न सक्नुहुन्छ:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    परिणाम मुद्रित हुन्छ - भारतीय भोजन यसको उत्तम अनुमान हो, राम्रो सम्भावनासहित:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ के तपाईं व्याख्या गर्न सक्नुहुन्छ किन मोडेललाई यो भारतीय भोजन भएकोमा धेरै पक्का छ?

1. तपाईंले प्रतिबिम्ब पाठहरूमा गरेको जस्तै वर्गीकरण प्रतिवेदन मुद्रण गरेर थप विवरण प्राप्त गर्नुहोस्:

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

यस पाठमा, तपाईंले क्रमबद्ध सामग्रीहरूको श्रृंखलाको आधारमा राष्ट्रिय भोजन पूर्वानुमान गर्न सक्ने मेसिन लर्निङ मोडेल निर्माण गर्न आफ्नो सफा गरेको डाटा प्रयोग गर्नुभयो। स्कikit-learn ले डेटा वर्गीकृत गर्न प्रदान गर्ने धेरै विकल्पहरूलाई पढ्न केही समय निकाल्नुहोस्। 'solvers' को अवधारणा मा अझ गहिराईमा जानुहोस् ताकि तपाईंलाई पछाडि भइरहेको बुझ्न सकियोस्।

## [पोस्ट-लेक्चर क्विज](https://ff-quizzes.netlify.app/en/ml/)

## समीक्षा र आत्म-अध्ययन

[यस पाठमा](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf) logistic regression को गणितमा अझ गहिरो रूपमा पुग्नुहोस्।
## असाइनमेन्ट

[solvers अध्ययन गर्नुहोस्](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**अस्वीकरण**:
यो दस्तावेज AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) प्रयोग गरेर अनुवाद गरिएको हो। हामी शुद्धताको लागि प्रयासरत छौं, कृपया ध्यान दिनुहोस् कि स्वचालित अनुवादमा त्रुटिहरु वा अशुद्धता हुन सक्छ। मूल दस्तावेज यसको स्वदेशी भाषामा आधिकारिक स्रोत मानिनुपर्छ। महत्वपूर्ण जानकारीको लागि व्यावसायिक मानव अनुवाद सिफारिस गरिन्छ। यो अनुवादको प्रयोगबाट उत्पन्न कुनै पनि गलतफहमी वा गलत व्याख्याका लागि हामी जिम्मेवार छैनौं।
<!-- CO-OP TRANSLATOR DISCLAIMER END -->