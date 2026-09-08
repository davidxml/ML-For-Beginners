# व्यंजन वर्गीकर्ता 1

इस पाठ में, आप पिछले पाठ से सहेजे गए डेटासेट का उपयोग करेंगे, जिसमें संतुलित, साफ-सुथरा डेटा व्यंजनों के बारे में है।

आप इस डेटासेट का उपयोग विभिन्न वर्गीकर्ताओं के साथ _एक समूह के घटकों के आधार पर दिए गए राष्ट्रीय व्यंजन की भविष्यवाणी करने के लिए_ करेंगे। ऐसा करते समय, आप सीखेंगे कि कैसे कुछ एल्गोरिदमों का उपयोग वर्गीकरण कार्यों के लिए किया जा सकता है।

## [प्रारंभिक व्याख्यान क्विज़](https://ff-quizzes.netlify.app/en/ml/)
# तैयारी

मान लेते हैं कि आपने [पाठ 1](../1-Introduction/README.md) पूरा कर लिया है, तो सुनिश्चित करें कि एक _cleaned_cuisines.csv_ फ़ाइल रूट `/data` फ़ोल्डर में इन चार पाठों के लिए मौजूद है।

## अभ्यास - एक राष्ट्रीय व्यंजन का अनुमान लगाएं

1. इस पाठ के _notebook.ipynb_ फ़ोल्डर में काम करते हुए, उस फ़ाइल को Pandas लाइब्रेरी के साथ आयात करें:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    डेटा इस प्रकार दिखता है:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |

1. अब, कई और लाइब्रेरी आयात करें:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. X और y निर्देशांकों को दो डेटा फ्रेमों में विभाजित करें प्रशिक्षण के लिए। `cuisine` लेबल वाला डेटा फ्रेम हो सकता है:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    यह इस प्रकार दिखेगा:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. उस `Unnamed: 0` कॉलम और `cuisine` कॉलम को ड्रॉप करें, `drop()` कॉल करते हुए। बाकी डेटा को प्रशिक्षण योग्य फ़ीचर्स के रूप में सहेजें:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    आपके फीचर्स इस प्रकार दिखेंगे:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

अब आप अपने मॉडल को प्रशिक्षण देने के लिए तैयार हैं!

## अपना वर्गीकर्ता चुनना

अब जब आपका डेटा साफ़ और प्रशिक्षण के लिए तैयार है, आपको यह निर्णय लेना होगा कि किस एल्गोरिदम का उपयोग करना है।

Scikit-learn वर्गीकरण को Supervised Learning के अंतर्गत रखता है, और इस श्रेणी में आप कई वर्गीकरण के तरीके पाएंगे। [विविधता](https://scikit-learn.org/stable/supervised_learning.html) पहली नज़र में काफी भ्रमित कर सकती है। निम्नलिखित विधियाँ सभी वर्गीकरण तकनीकों को शामिल करती हैं:

- रैखिक मॉडल
- सपोर्ट वेक्टर मशीन
- स्टोकेस्टिक ग्रेडिएंट डिसेंट
- निकटतम पड़ोसी
- गौसियन प्रोसेस
- निर्णय वृक्ष
- एन्सेम्बल विधियाँ (वोटिंग क्लासिफायर)
- मल्टीक्लास और मल्टीआउटपुट एल्गोरिदम (मल्टीक्लास और मल्टीलेबल वर्गीकरण, मल्टीक्लास-मल्टीआउटपुट वर्गीकरण)

> आप [तंत्रिका नेटवर्क](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification) का उपयोग करके भी डेटा वर्गीकृत कर सकते हैं, लेकिन वह इस पाठ का हिस्सा नहीं है।

### किस वर्गीकर्ता को चुनें?

तो, आपको कौन सा वर्गीकर्ता चुनना चाहिए? अक्सर, कई को आजमाकर और अच्छे परिणाम की तलाश करके परीक्षण किया जाता है। Scikit-learn एक [साइड-बाय-साइड तुलना](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) प्रदान करता है, जहां KNeighbors, SVC दो तरीकों से, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB और QuadraticDiscrinationAnalysis की तुलना एक बनाए गए डेटासेट पर की जाती है, जो परिणामों को विज़ुअलाइज्ड दिखाती है:

![क्लासिफायर की तुलना](../../../../translated_images/hi/comparison.edfab56193a85e7f.webp)
> प्लॉट Scikit-learn के दस्तावेज़ में बनाए गए

> AutoML इस समस्या को क्लाउड में इन तुलनाओं को चलाकर अच्छी तरह से हल करता है, जिससे आप अपने डेटा के लिए सबसे अच्छा एल्गोरिदम चुन सकते हैं। इसे [यहाँ](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott) आज़माएं।

### एक बेहतर दृष्टिकोण

एक बेहतर तरीका होता है अंधाधुंध अनुमान लगाने से, इस डाउनलोड योग्य [एमएल चीट शीट](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) के विचारों का पालन करना। यहाँ, हम पाते हैं कि हमारे मल्टीक्लास समस्या के लिए कुछ विकल्प हैं:

![मल्टीक्लास समस्याओं के लिए चीट शीट](../../../../translated_images/hi/cheatsheet.07a475ea444d2223.webp)
> माइक्रोसॉफ्ट के Algorithm Cheat Sheet का एक खंड, जो मल्टीक्लास वर्गीकरण विकल्पों को विस्तार से बताता है

✅ इस चीट शीट को डाउनलोड करें, प्रिंट करें, और इसे अपनी दीवार पर लगाएं!

### तर्क

आइए देखें कि हमारे पास मौजूद प्रतिबंधों के आधार पर विभिन्न दृष्टिकोणों के बारे में तर्क कैसे किया जा सकता है:

- **तंत्रिका नेटवर्क बहुत भारी हैं।** हमारे साफ़, लेकिन न्यूनतम डेटासेट और स्थानीय नोटबुक के माध्यम से प्रशिक्षण चलाने के कारण, तंत्रिका नेटवर्क इस कार्य के लिए बहुत भारी हैं।
- **कोई दो-क्लास वर्गीकर्ता नहीं।** हम दो-क्लास वर्गीकर्ता का उपयोग नहीं करते, इसलिए वन-वर्सेस-ऑल rule लागू नहीं होता।
- **निर्णय वृक्ष या लॉजिस्टिक रिग्रेशन काम कर सकते हैं।** निर्णय वृक्ष काम कर सकता है, या मल्टीक्लास डेटा के लिए लॉजिस्टिक रिग्रेशन सही हो सकता है।
- **मल्टीक्लास बूस्टेड निर्णय वृक्ष एक अलग समस्या हल करता है।** मल्टीक्लास बूस्टेड निर्णय वृक्ष गैर-पैरामीट्रिक कार्यों के लिए उपयुक्त होता है, जैसे रैंकिंग बनाने के कार्य, इसलिए यह हमारे लिए उपयोगी नहीं है।

### Scikit-learn का उपयोग करना

हम अपने डेटा का विश्लेषण करने के लिए Scikit-learn का उपयोग करेंगे। हालांकि, Scikit-learn में लॉजिस्टिक रिग्रेशन के उपयोग के कई तरीके हैं। [पैरा‍मीटर देखें](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)।

मूल रूप से दो महत्वपूर्ण पैरामीटर हैं - `multi_class` और `solver` - जिन्हें हमें निर्दिष्ट करना होगा, जब हम Scikit-learn से लॉजिस्टिक रिग्रेशन करवाते हैं। `multi_class` का मान कुछ व्यवहार लागू करता है। `solver` का मान एल्गोरिथ्म बताता है। सभी solvers सभी `multi_class` मानों के साथ संगत नहीं होते।

दस्तावेज़ के अनुसार, मल्टीक्लास मामले में, प्रशिक्षण एल्गोरिद्म:

- **वन-वर्सेस-रेस्ट (OvR) स्कीम का उपयोग करता है**, अगर `multi_class` विकल्प `ovr` पर सेट हो
- **क्रॉस-एंट्रॉपी लॉस का उपयोग करता है**, अगर `multi_class` विकल्प `multinomial` पर सेट हो। (वर्तमान में `multinomial` विकल्प केवल ‘lbfgs’, ‘sag’, ‘saga’ और ‘newton-cg’ solvers के द्वारा समर्थित है।)

> 🎓 यहाँ 'स्कीम' हो सकती है 'ovr' (वन-वर्सेस-रेस्ट) या 'multinomial'। क्योंकि लॉजिस्टिक रिग्रेशन वास्तव में बाइनरी वर्गीकरण का समर्थन करता है, ये स्कीमें इसे मल्टीक्लास वर्गीकरण कार्यों को बेहतर हैंडल करने देती हैं। [स्रोत](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' को "ऑप्टिमाइजेशन समस्या में उपयोग किए जाने वाले एल्गोरिद्म" के रूप में परिभाषित किया गया है। [स्रोत](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)

Scikit-learn इस तालिका के माध्यम से समझाता है कि विभिन्न solvers विभिन्न डेटा संरचनाओं की चुनौतियों को कैसे संभालते हैं:

![सॉल्वर](../../../../translated_images/hi/solvers.5fc648618529e627.webp)

## अभ्यास - डेटा को विभाजित करें

हम अपने पहले प्रशिक्षण प्रयास के लिए लॉजिस्टिक रिग्रेशन पर ध्यान केंद्रित कर सकते हैं, क्योंकि आपने हाल ही में पिछले पाठ में इसके बारे में सीखा है।  
अपने डेटा को प्रशिक्षण और परीक्षण समूहों में विभाजित करें `train_test_split()` कॉल करके:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## अभ्यास - लॉजिस्टिक रिग्रेशन लागू करें

चूंकि आप मल्टीक्लास केस का उपयोग कर रहे हैं, इसलिए आपको यह चुनना होगा कि कौन-सी _स्कीम_ इस्तेमाल करनी है और किस _solver_ को सेट करना है। लॉजिस्टिक रिग्रेशन का उपयोग करें मल्टीक्लास सेटिंग के साथ और प्रशिक्षण के लिए **liblinear** solver का चयन करें।

1. `multi_class` को `ovr` पर सेट करें और `solver` को `liblinear` पर सेट करते हुए लॉजिस्टिक रिग्रेशन बनाएं:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ एक अलग solver जैसे `lbfgs` भी आजमाएं, जो अक्सर डिफ़ॉल्ट होता है

    > ध्यान दें, जरूरत पड़ने पर अपने डेटा को फ्लैट करने के लिए Pandas का [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) फ़ंक्शन उपयोग करें।

    सटीकता **80%** से ऊपर अच्छी है!

1. आप इस मॉडल को एक पंक्ति (#50) के डेटा का परीक्षण करके सक्रिय रूप में देख सकते हैं:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    परिणाम प्रिंट होता है:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ किसी अलग पंक्ति संख्या की कोशिश करें और परिणाम देखें।
1. गहराई से जांच करने के लिए, आप इस भविष्यवाणी की सटीकता की जांच कर सकते हैं:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    परिणाम प्रिंट किया गया है - भारतीय व्यंजन इसका सबसे अच्छा अनुमान है, अच्छी संभावना के साथ:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ क्या आप समझा सकते हैं कि मॉडल क्यों काफी सुनिश्चित है कि यह भारतीय व्यंजन है?

1. रिग्रेशन पाठों की तरह, वर्गीकरण रिपोर्ट प्रिंट करके और अधिक विवरण प्राप्त करें:

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

## 🚀चुनौती

इस पाठ में, आपने अपनी साफ़ की गई डेटा का उपयोग करके एक मशीन लर्निंग मॉडल बनाया जो सामग्री की एक श्रृंखला के आधार पर राष्ट्रीय व्यंजन की भविष्यवाणी कर सकता है। डेटा को वर्गीकृत करने के लिए Scikit-learn द्वारा प्रदान किए गए कई विकल्पों को पढ़ने के लिए कुछ समय निकालें। 'सॉल्वर' की अवधारणा में गहराई से जाएं ताकि आप समझ सकें कि पर्दे के पीछे क्या होता है।

## [पाठ के बाद का क्विज़](https://ff-quizzes.netlify.app/en/ml/)

## समीक्षा और स्व-अध्ययन

[इस पाठ](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf) में लॉजिस्टिक रिग्रेशन के पीछे की गणित पर थोड़ा और गहराई से जानें।  
## असाइनमेंट

[सॉल्वर का अध्ययन करें](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**अस्वीकरण**:  
इस दस्तावेज़ का अनुवाद AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके किया गया है। जबकि हम सटीकता के लिए प्रयासरत हैं, कृपया ध्यान रखें कि स्वचालित अनुवाद में त्रुटियाँ या असंगतियाँ हो सकती हैं। मूल दस्तावेज़ अपनी मूल भाषा में प्रामाणिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सलाह दी जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।
<!-- CO-OP TRANSLATOR DISCLAIMER END -->