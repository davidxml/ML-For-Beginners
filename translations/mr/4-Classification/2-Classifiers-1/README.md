# पाककृती वर्गीकरण 1

या धड्यात, तुम्ही मागील धड्यापासून जतन केलेला डेटा संच वापराल ज्यामध्ये समतोल, स्वच्छ डेटा आहे जो पूर्णपणे काही विशिष्ट भोजनपरंपरांच्या बद्दल आहे.

तुम्ही या डेटा संचासह विविध वर्गीकरण करणाऱ्या अल्गोरिदम वापराल जे _एक गट घटकांवर आधारित राष्ट्रीय भोजनपरंपरा भाकीत_ करतात. तसे करताना, तुम्हाला वर्गीकरणाच्या कामांसाठी अल्गोरिदम कसे वापरले जाऊ शकतात याबद्दल अधिक माहिती मिळेल.

## [धडा आधीचा प्रश्नमंजुषा](https://ff-quizzes.netlify.app/en/ml/)
# तयारी

गृहीत धरा की तुम्ही [धडा 1](../1-Introduction/README.md) पूर्ण केला आहे, तर या चार धड्यांसाठी रूट `/data` फोल्डरमध्ये _cleaned_cuisines.csv_ नावाचा फाईल अवश्य आहे.

## सराव - राष्ट्रीय भोजनपरंपरा भाकीत करा

1. या धड्याच्या _notebook.ipynb_ फोल्डरमध्ये काम करताना, तो फाईल आणि Pandas लायब्ररी इंपोर्ट करा:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    डेटा असे दिसतो:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. आता, अजून काही लायब्ररी इंपोर्ट करा:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. X आणि y कोऑर्डिनेट्सना दोन वेगळ्या डेटा फ्रेममध्ये विभाजित करा प्रशिक्षणासाठी. `cuisine` ही लेबल्सची डेटा फ्रेम असू शकते:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    ते असे दिसेल:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. `Unnamed: 0` आणि `cuisine` स्तंभ काढून टाका `drop()` वापरून. उर्वरित डेटा ट्रेनिंग फीचर्स म्हणून जतन करा:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    तुमच्या फीचर्सचा हा प्रकार दिसेल:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

आता तुम्ही तुमचा मॉडेल प्रशिक्षणासाठी तयार आहात!

## तुमचा वर्गीकरण करणारा अल्गोरिदम निवडा

तुमचा डेटा स्वच्छ आणि प्रशिक्षणासाठी तयार झाल्यानंतर, तुम्हाला या कामासाठी कोणता अल्गोरिदम वापरायचा हे ठरवावे लागेल.

Scikit-learn वर्गीकरणाला Supervised Learning अंतर्गत समूहित करते आणि त्या श्रेणीत तुम्हाला वर्गीकरणाचे अनेक मार्ग सापडतील. [विविधता](https://scikit-learn.org/stable/supervised_learning.html) पहिल्या दृष्टीक्षेपात खूपच गुंतागुंतीची वाटू शकते. पुढील पद्धती सर्व वर्गीकरण तंत्रज्ञानाची उदाहरणे आहेत:

- रेषीय मॉडेल्स (Linear Models)
- सहाय्यक व्हेक्टर मशीन (Support Vector Machines)
- स्टोकास्टिक ग्रेडियंट डिसेंट (Stochastic Gradient Descent)
- सर्वात जवळची शेजारी (Nearest Neighbors)
- गॉसियन प्रोसेसेस (Gaussian Processes)
- निर्णय वृक्ष (Decision Trees)
- संयुक्‍त पद्धती (Ensemble methods, उदा. voting Classifier)
- मल्टिक्लास आणि मल्टिआउटपुट अल्गोरिदम (मल्टिक्लास आणि मल्टीलेबल वर्गीकरण, मल्टिक्लास-मल्टिआउटपुट वर्गीकरण)

> तुम्ही [न्यूरल नेटवर्क्स वापरून डेटा वर्गीकरण देखील करू शकता](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), पण तो या धड्याच्या परीघाबाहेर आहे.

### कोणता वर्गीकरण करणारा वापरायचा?

तर, तुम्ही कोणता वर्गीकरण करणारा निवडावा? अनेक वेळा, अनेक वापरून आणि चांगले निकाल पाहून चाचणी करणे उपयुक्त असते. Scikit-learn तयार केलेल्या डेटासेटवर KNeighbors, SVC (दोन प्रकारे), GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB आणि QuadraticDiscriminantAnalysis यांची [पक्ष-द्वेष तुलना](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) देते आणि निकाल दृष्य स्वरूपात उत्पन्न करते:

![comparison of classifiers](../../../../translated_images/mr/comparison.edfab56193a85e7f.webp)
> प्लॉट्स Scikit-learn च्या दस्तऐवजात तयार केलेले आहेत

> AutoML हा प्रश्न क्लाउडमध्ये हे तुलना चालवून तुमच्या डेटासाठी सर्वोत्तम अल्गोरिदम निवडण्यास मदत करतो. ते [इथे](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott) वापरून पहा

### चांगला दृष्टिकोन

अंधाधुंद अंदाज लावण्याऐवजी, डाउनलोड करता येणाऱ्या [ML Cheat Sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) मधील कल्पना अनुसरण करणे चांगले आहे. येथे आपल्याला आपल्या मल्टिक्लास समस्येसाठी काही निवडी आढळतात:

![cheatsheet for multiclass problems](../../../../translated_images/mr/cheatsheet.07a475ea444d2223.webp)
> Microsoft च्या Algorithm Cheat Sheet चा एक भाग, जे मल्टिक्लास वर्गीकरण पर्याय तपशीलवार दाखवते

✅ हा Cheat Sheet डाउनलोड करा, प्रिंट करा आणि तुमच्याजवळ लावा!

### विचारसरणी

चला पाहूया आपल्याकडे असलेल्या अटींनुसार विविध दृष्टिकोनांबद्दल जर विवेचन करता येईल का:

- **न्यूरल नेटवर्क्स खूप जड आहेत**. आमचा डेटा स्वच्छ आहे पण कमी, तसेच प्रशिक्षण स्थानिक नोटबुकवर चालवले जात आहे, त्यामुळे न्यूरल नेटवर्क्स या कामासाठी जड आहेत.
- **दोन वर्ग करत नाही**. आपण दोन-श्रेणी वर्गीकरण वापरत नाही, म्हणून one-vs-all युक्ती वापरत नाही.
- **निर्णय वृक्ष किंवा लॉजिस्टिक रिग्रेशन काम करू शकते**. निर्णय वृक्ष किंवा मल्टिक्लास डेटा साठी लॉजिस्टिक रिग्रेशन वापरता येऊ शकतो.
- **मल्टिक्लास बूस्टेड निर्णय वृक्ष वेगळ्या समस्येसाठी आहे**. मल्टिक्लास बूस्टेड निर्णय वृक्ष मुख्यतः नॉन-पॅरामीट्रिक कामांसाठी उपयुक्त आहे, जसे की रँकिंग तयार करण्यासाठी, त्यामुळे तो आपल्यासाठी उपयुक्त नाही.

### Scikit-learn वापरणे

आम्ही Scikit-learn वापरून डेटा विश्लेषण करू. मात्र, Scikit-learn मध्ये लॉजिस्टिक रिग्रेशन वापरण्याचे अनेक मार्ग आहेत. [पॅरामीटर्स](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) एकदा बघा.

मूळतः दोन महत्त्वाचे पर्याय आहेत - `multi_class` आणि `solver` - जे आपण लॉजिस्टिक रिग्रेशन करताना सांगावे लागतात. `multi_class` मूल्य विशिष्ट वर्तन ठरवते. `solver` म्हणजे कोणता अल्गोरिदम वापरायचा आहे ते सूचित करते. सर्व solvers सर्व multi_class पर्यायांसोबत जुळत नाहीत.

दस्तऐवजांनुसार, मल्टिक्लास केस मध्ये, प्रशिक्षण अल्गोरिदम:

- **one-vs-rest (OvR) पद्धत वापरतो**, जर `multi_class` पर्याय `ovr` असेल तर
- **cross-entropy loss वापरतो**, जर `multi_class` पर्याय `multinomial` असेल तर. (सध्या `multinomial` पर्याय 'lbfgs’, ‘sag’, ‘saga’ आणि ‘newton-cg’ solvers ना समर्थन देतो.)"

> 🎓 'scheme' म्हणजे 'ovr' (one-vs-rest) किंवा 'multinomial' हे असू शकते. लॉजिस्टिक रिग्रेशन मुख्यतः द्विश्रेणी वर्गीकरणासाठी तयार असल्यामुळे, या युक्त्या मल्टिक्लास वर्गीकरण अधिक चांगल्या प्रकारे हाताळतात. [source](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 ‘solver’ म्हणजे "ऑप्टिमायझेशन समस्येत वापरला जाणारा अल्गोरिदम". [source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn खालील टेबल देते ज्यामध्ये वेगवेगळ्या डेटा प्रकारांसाठी कसे solvers काम करतात हे दाखवले आहे:

![solvers](../../../../translated_images/mr/solvers.5fc648618529e627.webp)

## सराव - डेटा विभाजन करा

आपण आपल्या पहिल्या प्रशिक्षणासाठी लॉजिस्टिक रिग्रेशनवर लक्ष केंद्रित करू शकतो कारण तुम्हाला मागील धड्यात याबद्दल नुकताच शिकायला मिळाले आहे. 

`train_test_split()` वापरून तुमचा डेटा प्रशिक्षण आणि चाचणी समूहांत विभाजित करा:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## सराव - लॉजिस्टिक रिग्रेशन लागू करा

तुम्ही मल्टिक्लास केस वापरत असल्याने कोणती _scheme_ आणि कोणता _solver_ वापरायचा हे ठरवा. मल्टिक्लास सेटिंगसाठी LogisticRegression वापरा आणि **liblinear** solver वापरून प्रशिक्षण द्या.

1. multi_class साठी `ovr` आणि solver साठी `liblinear` असलेली लॉजिस्टिक रिग्रेशन तयार करा:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ इतर solver `lbfgs` देखील वापरून पाहा, जे अनेकदा डिफॉल्ट असतो

    > लक्षात घ्या, आवश्यक असताना तुमचा डेटा flatten करण्यासाठी Pandas ची [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) फंक्शन वापरा.

    अचूकता ८०% पेक्षा जास्त चांगली आहे!

1. तुम्ही या मॉडेलची एक ओळ (#50) चाचणी देऊन ते कसे कार्य करते ते पाहू शकता:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    निकाल हा मुद्रित होतो:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ दुसरी ओळ क्रमांक वापरून निकाल तपासा
1. अधिक खोलवर तपासण्यासाठी, तुम्ही या भविष्यवाणीसाठी अचूकता तपासू शकता:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    निकाल छापला आहे - भारतीय जेवण हे त्याचे सर्वोत्तम अंदाज आहे, चांगल्या संभाव्यतेसह:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ तुम्ही समजावून सांगू शकता का की मॉडेल का इतके निश्चित आहे की हे भारतीय जेवण आहे?

1. रिग्रेशन धडे जसे केलं तसे वर्गीकरण अहवाल छापून अधिक तपशील मिळवा:

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

## 🚀आव्हान

या धड्यात, तुम्ही स्वच्छ केलेल्या डेटाचा वापर करून असा मशीन लर्निंग मॉडेल तयार केला जो घटकांच्या मालिकेवरून राष्ट्रीय जेवणाचा अंदाज लावू शकतो. डेटा वर्गीकरणासाठी Scikit-learn अनेक पर्याय पुरवते त्यातील विविध पर्याय वाचण्यासाठी काही वेळ द्या. 'सॉल्वर' या संकल्पनेत अधिक खोलवर जाणून घ्या ज्यामुळे समजेल की मागच्या प्रक्रियेत काय होते.

## [पोस्ट-लेक्चर क्विझ](https://ff-quizzes.netlify.app/en/ml/)

## पुनरावलोकन व स्वअभ्यास

[या धड्यात](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf) लॉजिस्टिक रिग्रेशनच्या गणितामागील तत्त्वे अधिक खोलात बघा.
## असायन्मेंट

[सॉल्वर अभ्यास करा](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**सूचना**:  
हा दस्तऐवज [Co-op Translator](https://github.com/Azure/co-op-translator) या AI भाषांतर सेवेचा वापर करून भाषांतरित केला आहे. आम्ही अचूकतेसाठी प्रयत्नशील असतो, तरीही कृपया ध्यानात ठेवा की ऑटोमेटेड भाषांतरांमध्ये चुका किंवा अचूकतेची कमतरता असू शकते. मूळ दस्तऐवज त्याच्या स्थानिक भाषेत अधिकृत स्रोत मानला पाहिजे. महत्वाची माहिती असल्यास व्यावसायिक मानवी भाषांतर करण्याची शिफारस केली जाते. या भाषांतराच्या वापरातून उद्भवणार्‍या कुठल्याही गैरसमजुतीसाठी किंवा चुकीच्या अर्थ लावण्याबद्दल आम्ही जबाबदार नाही.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->