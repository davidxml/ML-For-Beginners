# রান্নার শ্রেণীবিভাজক ১

এই পাঠে, আপনি গত পাঠ থেকে সংরক্ষণকৃত একটি পরিপূর্ণ, পরিষ্কার ডেটাসেট ব্যবহার করবেন যা রান্নার ধরন সম্পর্কে।

আপনি এই ডেটাসেট ব্যবহার করে বিভিন্ন শ্রেণীবিভাজকের মাধ্যমে _একটি নির্দিষ্ট জাতীয় রান্নার ধরন পূর্বানুমান করবেন একটি উপাদানের গ্রুপের ভিত্তিতে_। এই প্রক্রিয়ায়, আপনি শিখবেন কীভাবে অ্যালগরিদমগুলিকে শ্রেণীবিভাগের কাজের জন্য ব্যবহার করা যায়।

## [পঠনের পূর্বে কুইজ](https://ff-quizzes.netlify.app/en/ml/)
# প্রস্তুতি

ধরা যাক আপনি [পাঠ ১](../1-Introduction/README.md) সমাপ্ত করেছেন, নিশ্চিত করুন যে _cleaned_cuisines.csv_ ফাইলটি মূল `/data` ফোল্ডারে রয়েছে এই চারটি পঠনের জন্য।

## অনুশীলন - একটি জাতীয় রান্না পূর্বানুমান করুন

1. এই পাঠের _notebook.ipynb_ ফোল্ডারে কাজ করে, সেই ফাইলটি এবং Pandas লাইব্রেরি আমদানি করুন:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    ডেটা এরকম দেখায়:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. এখন, আরও কিছু লাইব্রেরি আমদানি করুন:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. প্রশিক্ষণের জন্য X ও y কোঅর্ডিনেটকে দুটি ডেটাফ্রেমে ভাগ করুন। `cuisine` হতে পারে লেবেল ডেটাফ্রেম:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    এটি এরকম দেখাবে:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. `Unnamed: 0` কলাম এবং `cuisine` কলামটি বাদ দিন, `drop()` কল ব্যবহার করে। বাকী ডেটা সংরক্ষণ করুন প্রশিক্ষণযোগ্য বৈশিষ্ট্য হিসেবে:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    আপনার বৈশিষ্ট্যগুলো এরকম দেখায়:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

এখন আপনার মডেল প্রশিক্ষণের জন্য প্রস্তুত!

## শ্রেণীবিভাজক নির্বাচন

আপনার ডেটা পরিষ্কার এবং প্রশিক্ষণের জন্য প্রস্তুত হওয়ায়, আপনাকে সিদ্ধান্ত নিতে হবে কোন অ্যালগরিদমটি কাজের জন্য ব্যবহার করবেন।

Scikit-learn শ্রেণীবিভাগকে সুপারভাইজড লার্নিংয়ের অন্তর্ভুক্ত করেছে, এবং সেই বিভাগের মধ্যে আপনি অনেক ধরনের শ্রেণীবিভাজক পাবেন। [বিভিন্ন প্রকার](https://scikit-learn.org/stable/supervised_learning.html) প্রথম দৃষ্টিতে বিভ্রান্তিকর হতে পারে। নিম্নলিখিত পদ্ধতিগুলোতে শ্রেণীবিভাজন কৌশল রয়েছে:

- লিনিয়ার মডেল
- সাপোর্ট ভেক্টর মেশিন
- স্টোকাস্টিক গ্রেডিয়েন্ট ডিসেন্ট
- নিকটতম প্রতিবেশী
- গাউসিয়ান প্রসেস
- ডিসিশন ট্রি
- এনসম্বল পদ্ধতি (ভোটিং শ্রেণীবিভাজক)
- মাল্টিক্লাস ও মাল্টি আউটপুট অ্যালগরিদম (মাল্টিক্লাস ও মাল্টিলেবেল শ্রেণীবিভাগ, মাল্টিক্লাস-মাল্টি আউটপুট শ্রেণীবিভাগ)

> আপনি [নিউরাল নেটওয়ার্ক দিয়ে ডেটা শ্রেণীবিভাগ](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification) করতেও পারেন, তবে সেটি এই পাঠের সীমার বাইরে।

### কোন শ্রেণীবিভাজক বেছে নেবেন?

তাহলে, কোন শ্রেণীবিভাজক বেছে নেবেন? প্রায়ই, একাধিক চালিয়ে দেখভাল করা এবং ভাল ফলাফল খোঁজা বেছে নেওয়ার একটি উপায়। Scikit-learn একটি [সাইড বাই সাইড তুলনা](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) দেখায় একটি তৈরি ডেটাসেটের উপর, যেখানে KNeighbors, SVC দুই ধরনের, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB ও QuadraticDiscrinationAnalysis তুলনা করা হয়েছে; ফলাফল চিত্রায়িত হয়েছে:

![শ্রেণীবিভাজকদের তুলনা](../../../../translated_images/bn/comparison.edfab56193a85e7f.webp)
> প্লটগুলি Scikit-learn এর ডকুমেন্টেশনে তৈরি

> AutoML এই সমস্যা মসৃণভাবে সমাধান করে ক্লাউডে এসব তুলনা চালিয়ে, আপনাকে আপনার ডেটারের জন্য সেরা অ্যালগরিদম বেছে নিতে দেয়। এটাকে চেষ্টা করুন [এখানে](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### আরও ভাল উপায়

তবে, অনুমান ছাড়াই আরও ভাল উপায় হলো এই ডাউনলোডযোগ্য [এমএল চিট শীট](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) অনুসরণ করা। এখানে, আমরা আবিষ্কার করি, আমাদের মাল্টিক্লাস সমস্যার জন্য আমাদের কিছু বিকল্প আছে:

![মাল্টিক্লাস সমস্যার জন্য চিটশীট](../../../../translated_images/bn/cheatsheet.07a475ea444d2223.webp)
> মাইক্রোসফটের অ্যালগরিদম চিট শীট এর একটি অংশ, মাল্টিক্লাস শ্রেণীবিভাগের বিকল্পগুলি বর্ণনা করে

✅ এই চিট শীটটি ডাউনলোড করুন, প্রিন্ট করে আপনার দেয়ালে খুঁজে লেগিয়ে রাখুন!

### যুক্তি

চলুন দেখি আমরা বিভিন্ন উপায়ের যুক্তি দিতে পারি কি না আমাদের সীমাবদ্ধতাগুলো বিবেচনা করে:

- **নিউরাল নেটওয়ার্ক খুব ভারী**। আমাদের পরিষ্কার তবে সীমিত ডেটাসেট এবং আমরা নোটবুকের মাধ্যমে স্থানীয়ভাবে প্রশিক্ষণ চালাচ্ছি, সেজন্য নিউরাল নেটওয়ার্ক এই কাজের জন্য অতিরিক্ত ভারী।
- **কোনো দুই-ক্লাস শ্রেণীবিভাজক নেই**। আমরা দুই-ক্লাস শ্রেণীবিভাজক ব্যবহার করব না, সেজন্য one-vs-all বাদ।
- **ডিসিশন ট্রি বা লগিস্টিক রিগ্রেশন কাজ করতে পারে**। ডিসিশন ট্রি হতে পারে, বা মাল্টিক্লাস ডেটার জন্য লগিস্টিক রিগ্রেশন।
- **মাল্টিক্লাস বুস্টেড ডিসিশন ট্রি অন্য ধরণের কাজের জন্য**। মাল্টিক্লাস বুস্টেড ডিসিশন ট্রি অ-পরিমিত কাজের জন্য উপযোগী, যেমন র্যাঙ্কিং নির্মাণের কাজ, তাই এটি আমাদের জন্য উপযোগী নয়।

### Scikit-learn ব্যবহার

আমরা আমাদের ডেটা বিশ্লেষণের জন্য Scikit-learn ব্যবহার করব। তবে, Scikit-learn এ লগিস্টিক রিগ্রেশন চালানোর অনেক উপায় আছে। দেখে নিন [পারামিটারসমূহ](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) কীভাবে দিতে হয়।

মোটামুটি দুটি গুরুত্বপূর্ণ পারামিটার আছে - `multi_class` এবং `solver` - যেগুলো Scikit-learn কে লগিস্টিক রিগ্রেশন চালানোর সময় বসাতে হয়। `multi_class` একটি নির্দিষ্ট আচরণ প্রয়োগ করে। solver এর মান নির্ধারণ করে কোন অ্যালগরিদম ব্যবহার হবে। সব solver সব `multi_class` মানের সঙ্গে ব্যবহার হবে না।

ডকুমেন্ট অনুযায়ী, মাল্টিক্লাস ক্ষেত্রে, ট্রেনিং অ্যালগরিদম:

- **one-vs-rest (OvR) পদ্ধতি ব্যবহার করে**, যদি `multi_class` অপশন `ovr` এ থাকে
- **ক্রস-এন্ট্রপি লস ব্যবহার করে**, যদি `multi_class` অপশন `multinomial` এ থাকে। (বর্তমানে `multinomial` অপশন শুধুমাত্র ‘lbfgs’, ‘sag’, ‘saga’ এবং ‘newton-cg’ সলভার দ্বারা সমর্থিত)"

> 🎓 এখানে 'স্কিম' মানে হয় 'ovr' (one-vs-rest) অথবা 'multinomial'। যেহেতু লগিস্টিক রিগ্রেশন মূলত বাইনারি শ্রেণীবিভাগের জন্য ডিজাইন করা, এই পদ্ধতিগুলো মাল্টিক্লাস শ্রেণীবিভাগ ভালোভাবে পরিচালনা করতে দেয়। [তথ্যসূত্র](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'সলভার' এর অর্থ হলো "অপ্টিমাইজেশন সমস্যায় ব্যবহৃত অ্যালগরিদম"। [তথ্যসূত্র](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)

Scikit-learn নিম্নলিখিত টেবিল দিয়ে দেখায় কীভাবে সলভার বিভিন্ন ধরণের ডেটা কাঠামোর চ্যালেঞ্জ মোকাবেলা করে:

![সলভার](../../../../translated_images/bn/solvers.5fc648618529e627.webp)

## অনুশীলন - ডেটা ভাগ করুন

আমরা প্রথম প্রশিক্ষণে লগিস্টিক রিগ্রেশন এ ফোকাস করতে পারি কারণ আপনি পূর্বের একটি পাঠে এই বিষয়ে শিখেছেন।
আপনার ডেটা প্রশিক্ষণ ও পরীক্ষা গ্রুপে ভাগ করুন `train_test_split()` কল করে:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## অনুশীলন - লগিস্টিক রিগ্রেশন প্রয়োগ করুন

আপনি মাল্টিক্লাস কেস ব্যবহার করছেন, সেজন্য আপনাকে সিদ্ধান্ত নিতে হবে কোন _স্কিম_ এবং কোন _সলভার_ বসাবেন। মাল্টিক্লাস সেটিং সহ এবং **liblinear** সলভার দিয়ে লগিস্টিক রিগ্রেশন তৈরি করুন।

1. `multi_class` সেট করুন `ovr` এবং solver সেট করুন `liblinear` দিয়ে একটি লগিস্টিক রিগ্রেশন তৈরি করুন:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ অন্য কোনো সলভার যেমন `lbfgs` চেষ্টা করুন, যা প্রায়ই ডিফল্ট থাকে

    > লক্ষ্য করুন, দরকার হলে Pandas এর [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) ফাংশন ব্যবহার করুন ডেটা ফ্ল্যাট করতে।

    নির্ভুলতা ৮০% এর বেশি ভালো!

1. এই মডেলটি কার্যকর দেখতে পারেন একটি ডেটার সারি (#50) পরীক্ষা করে:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    ফলাফল মুদ্রিত হয়:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ অন্য কোনো সারি নম্বর চেষ্টা করে ফলাফল পরীক্ষা করুন
1. আরও গভীরে খুঁজে দেখলে, আপনি এই পূর্বাভাসের সঠিকতা পরীক্ষা করতে পারেন:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    ফলাফল মুদ্রিত হয়েছে - ভারতীয় রান্না এই মডেলের সেরা অনুমান, ভাল সম্ভাবনার সাথে:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ আপনি কি ব্যাখ্যা করতে পারেন কেন মডেলটি বেশ নিশ্চিত যে এটি একটি ভারতীয় রান্না?

1. আপনার রিগ্রেশন পাঠগুলোর মতো, একটি শ্রেণীবিভাগ প্রতিবেদনের মাধ্যমে আরও বিস্তারিত তথ্য পান:

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

## 🚀চ্যালেঞ্জ

এই পাঠে, আপনি আপনার পরিস্কারকৃত ডেটা ব্যবহার করে একটি মেশিন লার্নিং মডেল তৈরি করেছেন যা উপাদানের সিরিজের ভিত্তিতে একটি জাতীয় রান্না পূর্বাভাস দিতে পারে। Scikit-learn যে অসংখ্য অপশন প্রদান করে তা মনোযোগ দিয়ে পড়ুন। 'solver' ধারণাটি গভীরভাবে জানুন যেন পিছনের কাজগুলো বোঝা যায়।

## [পোস্ট-লেকচার কুইজ](https://ff-quizzes.netlify.app/en/ml/)

## পর্যালোচনা ও স্ব-অধ্যয়ন

[এই পাঠে](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf) লগিস্টিক রিগ্রেশনের গণিতের আরও গভীর অধ্যয়ন করুন।
## অ্যাসাইনমেন্ট

[সোলভারগুলি অধ্যয়ন করুন](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**অস্বীকৃতি**:
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনূদিত হয়েছে। আমরা যথাসাধ্য সঠিকতার জন্য চেষ্টা করি, তবে অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল নথিটি তার স্থানীয় ভাষায় একটি কর্তৃত্বপূর্ণ উৎস হিসেবে বিবেচনা করা উচিত। গুরুত্বপূর্ণ তথ্যের জন্য, পেশাদার মানব অনুবাদের পরামর্শ দেওয়া হয়। এই অনুবাদ ব্যবহারের কারণে সৃষ্ট কোনো ভুল বোঝাবুঝি বা ভুল ব্যাখ্যার জন্য আমরা দায়ী নই।
<!-- CO-OP TRANSLATOR DISCLAIMER END -->