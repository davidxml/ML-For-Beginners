# باورچی خانے کی قسم بندی کرنے والے 1

اس سبق میں، آپ پچھلے سبق سے محفوظ کردہ متوازن، صاف ڈیٹا پر مشتمل غذائی اقسام کے بارے میں ڈیٹا سیٹ استعمال کریں گے۔

آپ اس ڈیٹا سیٹ کو مختلف قسم بندی کرنے والے الگورتھمز کے ساتھ استعمال کریں گے تاکہ _اجزاء کے ایک گروپ کی بنیاد پر دی گئی قومی غذا کی پیش گوئی کی جائے_. ایسا کرتے ہوئے، آپ الگورتھمز کے ذریعے قسم بندی کے کاموں کے لیے مختلف طریقوں کے بارے میں مزید جانیں گے۔

## [سبق سے پہلے کوئز](https://ff-quizzes.netlify.app/en/ml/)
# تیاری

فرض کریں کہ آپ نے [سبق 1](../1-Introduction/README.md) مکمل کر لیا ہے، تو اس بات کو یقینی بنائیں کہ ایک _cleaned_cuisines.csv_ فائل جڑ `/data` فولڈر میں ان چار اسباق کے لیے موجود ہو۔

## مشق - قومی کھانے کی پیش گوئی کریں

1. اس سبق کے _notebook.ipynb_ فولڈر میں کام کرتے ہوئے، اس فائل کو Pandas لائبریری کے ساتھ امپورٹ کریں:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    ڈیٹا کچھ یوں نظر آتا ہے:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. اب، مزید چند لائبریریاں امپورٹ کریں:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. ٹریننگ کے لیے X اور y کو دو ڈیٹا فریمز میں تقسیم کریں۔ `cuisine` لیبلز کا ڈیٹا فریم ہو سکتا ہے:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    یہ کچھ یوں دکھائی دے گا:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. وہ `Unnamed: 0` کالم اور `cuisine` کالم کو `drop()` کال کر کے حذف کریں۔ باقی ڈیٹا کو ٹریننگ خصوصیات کے طور پر محفوظ کریں:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    آپ کی خصوصیات ایسی نظر آتی ہیں:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

اب آپ اپنا ماڈل ٹرین کرنے کے لیے تیار ہیں!

## اپنا قسم بندی کرنے والا منتخب کرنا

اب جب کہ آپ کا ڈیٹا صاف اور تربیت کے لیے تیار ہے، آپ کو یہ فیصلہ کرنا ہوگا کہ کون سا الگورتھم استعمال کرنا ہے۔

Scikit-learn قسم بندی کو سپروائزڈ لرننگ کے تحت رکھتا ہے، اور اس زمرے میں آپ کو قسم بندی کرنے کے کئی طریقے ملیں گے۔ [مختلف طریقے](https://scikit-learn.org/stable/supervised_learning.html) دیکھنے میں پہلے پہل کافی پیچیدہ لگ سکتے ہیں۔ درج ذیل طریقے سب قسم بندی کی تکنیکیں شامل کرتے ہیں:

- لینیئر ماڈلز
- سپورٹ ویکٹر مشینز
- اسٹوکاسٹک گریڈینٹ ڈی سینٹ
- قریب ترین پڑوسی
- گوسیئن پراسیسز
- فیصلہ ساز درخت
- ایسینبل طریقے (ووٹنگ کلاسفیئر)
- ملٹی کلاس اور ملٹی آؤٹ پٹ الگورتھمز (ملٹی کلاس اور ملٹی لیبل قسم بندی، ملٹی کلاس-ملٹی آؤٹ پٹ قسم بندی)

> آپ [نیورل نیٹ ورکس بھی استعمال کر سکتے ہیں](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification) لیکن وہ اس سبق کے دائرہ کار سے باہر ہیں۔

### کون سا کلاسفیئر منتخب کیا جائے؟

تو، آپ کون سا کلاسفیئر منتخب کریں؟ اکثر، کئی کلاسفیئرز کو آزمانا اور اچھا نتیجہ تلاش کرنا آزمائشی طریقہ ہوتا ہے۔ Scikit-learn ایک [موازنہ](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) پیش کرتا ہے جو مختلف الگورتھمز جیسے KNeighbors، SVC کے دو طریقے، GaussianProcessClassifier، DecisionTreeClassifier، RandomForestClassifier، MLPClassifier، AdaBoostClassifier، GaussianNB اور QuadraticDiscrinationAnalysis کا موازنہ کرتا ہے اور نتائج کو بصری طور پر دکھاتا ہے:

![comparison of classifiers](../../../../translated_images/ur/comparison.edfab56193a85e7f.webp)
> یہ گراف Scikit-learn کی دستاویزات میں بنائے گئے ہیں

> AutoML اس مسئلے کو آسانی سے حل کرتا ہے کلاؤڈ میں یہ کمپیریزنز چلا کر، تاکہ آپ اپنے ڈیٹا کے لیے سب سے بہتر الگورتھم منتخب کر سکیں۔ اسے [یہاں آزمائیں](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### ایک بہتر طریقہ

اندازے لگانے کے بجائے بہتر ہے کہ اس [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) کو ڈاؤنلوڈ کر کے اس میں بتائی گئی تجاویز پر عمل کریں۔ یہاں، ہم دیکھتے ہیں کہ ہمارے ملٹی کلاس مسئلے کے لیے ہمارے پاس کچھ آپشنز ہیں:

![cheatsheet for multiclass problems](../../../../translated_images/ur/cheatsheet.07a475ea444d2223.webp)
> مائیکروسافٹ کے الگورتھم چیٹ شیٹ کا ایک حصہ، جو ملٹی کلاس قسم بندی کے آپشنز کی تفصیل بتاتا ہے

✅ اس چیٹ شیٹ کو ڈاؤنلوڈ کریں، پرنٹ نکالیں، اور اپنی دیوار پر لگائیں!

### منطق

آئیے دیکھیں کہ موجودہ پابندیوں کے پیش نظر مختلف طریقوں پر کیسے غور کیا جا سکتا ہے:

- **نیورل نیٹ ورکس بہت بھاری ہیں**۔ ہمارے صاف لیکن کم حجم ڈیٹا سیٹ اور لوکل نوٹ بکس میں ٹریننگ چلانے کی وجہ سے، نیورل نیٹ ورکس اس کام کے لیے زیادہ بھاری ہیں۔
- **کوئی دو-کلاس کلاسفیئر نہیں**۔ ہم دو-کلاس کلاسفیئر استعمال نہیں کرتے، اس لیے ون-ورسس-آل آؤٹ ہو جاتا ہے۔
- **فیصلہ ساز درخت یا لاجسٹک ریگریشن کام کر سکتا ہے**۔ فیصلہ ساز درخت یا ملٹی کلاس ڈیٹا کے لیے لاجسٹک ریگریشن مفید ہو سکتا ہے۔
- **ملٹی کلاس بوسٹڈ فیصلہ ساز درخت مختلف مسئلے کے لیے ہیں**۔ ملٹی کلاس بوسٹڈ فیصلہ ساز درخت غیر پیرا میٹرک کاموں کے لیے بہتر ہیں، جیسے رینکنگ بنانا، اس لیے ہمارے لیے یہ مفید نہیں۔

### Scikit-learn کا استعمال

ہم اپنے ڈیٹا کا تجزیہ کرنے کے لیے Scikit-learn استعمال کریں گے۔ تاہم، Scikit-learn میں لاجسٹک ریگریشن کے کئی استعمالات ہیں۔ اس [دستاویز](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) کو دیکھیں کہ کون سے پیرامیٹرز پاس کیے جا سکتے ہیں۔

بنیادی طور پر دو اہم پیرامیٹرز ہیں - `multi_class` اور `solver` - جو آپ کو لاجسٹک ریگریشن انجام دیتے ہوئے بتانے ہوں گے۔ `multi_class` ویلیو ایک خاص رویہ اپناتی ہے۔ `solver` ویلیو وہ الگورتھم بتاتی ہے جو استعمال ہوگا۔ ہر solver تمام `multi_class` ویلیوز کے ساتھ مطابقت نہیں رکھتا۔

دستاویزات کے مطابق، ملٹی کلاس صورت میں، تربیتی الگورتھم:

- اگر `multi_class` آپشن `ovr` پر سیٹ ہو تو **ون-ورسس-ریسٹ (OvR) اسکیم** استعمال کرتا ہے
- اگر `multi_class` آپشن `multinomial` ہو تو **کراس انٹروپی لاس** استعمال کرتا ہے (اس وقت یہ آپشن صرف ‘lbfgs’, ‘sag’, ‘saga’ اور ‘newton-cg’ solvers کے ساتھ سپورٹڈ ہے)"

> 🎓 'اسکیم' یہاں 'ovr' (ون-ورسس-ریسٹ) یا 'multinomial' ہو سکتی ہے۔ چونکہ لاجسٹک ریگریشن بنیادی طور پر بائنری قسم بندی کے لیے ہے، یہ اسکیمز اسے بہتر طریقے سے ملٹی کلاس کاموں میں مدد دیتی ہیں۔ [ذرائع](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' کا مطلب ہے "آپٹیمائزیشن مسئلہ میں استعمال ہونے والا الگورتھم"۔ [ذرائع](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)

Scikit-learn یہ جدول فراہم کرتا ہے کہ کیسے solvers مختلف قسم کے ڈیٹا ڈھانچوں کے چیلنجز کو ہینڈل کرتے ہیں:

![solvers](../../../../translated_images/ur/solvers.5fc648618529e627.webp)

## مشق - ڈیٹا کو تقسیم کریں

چونکہ آپ نے حال ہی میں لاجسٹک ریگریشن کے بارے میں سیکھا ہے، تو اپنے پہلے ٹریننگ ٹرائل کے لیے ہم لاجسٹک ریگریشن پر توجہ دیں گے۔ اپنے ڈیٹا کو `train_test_split()` کال کر کے تربیتی اور ٹیسٹنگ گروپس میں تقسیم کریں:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## مشق - لاجسٹک ریگریشن لگائیں

چونکہ آپ ملٹی کلاس کیس استعمال کر رہے ہیں، آپ کو فیصلہ کرنا ہوگا کہ کون سا _اسکیم_ اور کون سا _solver_ استعمال کرنا ہے۔ لاجسٹک ریگریشن بنائیں جس میں multi_class `ovr` پر اور solver **liblinear** پر مقرر ہو۔

1. ایک لاجسٹک ریگریشن بنائیں جس میں multi_class `ovr` ہو اور solver `liblinear` ہو:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ مختلف solver جیسے `lbfgs` بھی آزمائیں، جو اکثر ڈیفالٹ ہوتا ہے

    > نوٹ کریں، جب ضرورت ہو تو آپ اپنے ڈیٹا کو فلٹین کرنے کے لیے Pandas کا [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) فنکشن استعمال کریں۔

    درستگی 80٪ سے زیادہ اچھی ہے!

1. آپ اس ماڈل کو فعال دیکھنے کے لیے کسی بھی ایک قطار (#50) کا ڈیٹا ٹیسٹ کر کے نتائج دیکھ سکتے ہیں:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    نتیجہ پرنٹ کیا جائے گا:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ مختلف قطار نمبر آزمائیں اور نتائج چیک کریں
1. مزید گہرائی میں جا کر، آپ اس پیشگوئی کی درستگی چیک کر سکتے ہیں:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    نتیجہ پرنٹ کیا گیا ہے - انڈین کھانا اس کی بہترین پیش گوئی ہے، اچھی امکانیت کے ساتھ:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ کیا آپ وضاحت کر سکتے ہیں کہ ماڈل کیوں کافی حد تک یقینی ہے کہ یہ انڈین کھانا ہے؟

1. مزید تفصیل حاصل کرنے کے لیے ایک درجہ بندی رپورٹ پرنٹ کریں، جیسا کہ آپ نے رجریشن کے اسباق میں کیا تھا:

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

## 🚀چیلنج

اس سبق میں، آپ نے اپنی صاف شدہ ڈیٹا کا استعمال کرتے ہوئے ایک مشین لرننگ ماڈل بنایا ہے جو اجزاء کی ایک سیریز کی بنیاد پر ایک قومی کھانے کی پیشگوئی کر سکتا ہے۔ مختلف اختیارات کو پڑھنے کے لیے کچھ وقت نکالیں جو Scikit-learn فراہم کرتا ہے تاکہ ڈیٹا کو درجہ بند کیا جا سکے۔ 'solver' کے تصور میں مزید گہرائی میں جائیں تاکہ سمجھ سکیں کہ پس منظر میں کیا ہو رہا ہے۔

## [لیکچر کے بعد کوئز](https://ff-quizzes.netlify.app/en/ml/)

## جائزہ اور خود مطالعہ

لاجسٹک رجریشن کے پیچھے ریاضی میں تھوڑا مزید غوطہ لگائیں [اس سبق](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf) میں
## اسائنمنٹ

[solvers کا مطالعہ کریں](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ڈس کلیمر**:  
یہ دستاویز AI ترجمہ سروس [Co-op Translator](https://github.com/Azure/co-op-translator) کا استعمال کرتے ہوئے ترجمہ کی گئی ہے۔ اگرچہ ہم درستگی کے لیے کوشاں ہیں، براہ کرم اس بات سے آگاہ رہیں کہ خودکار تراجم میں غلطیاں یا نا درستیاں ہو سکتی ہیں۔ اصل دستاویز اپنی مادری زبان میں ہی معتبر ذریعہ سمجھی جانی چاہیے۔ اہم معلومات کے لیے پیشہ ور انسانی ترجمہ کی سفارش کی جاتی ہے۔ اس ترجمے کے استعمال سے پیدا ہونے والی کسی بھی غلط فہمی یا غلط تشریح کی ذمہ داری ہم پر عائد نہیں ہوتی۔
<!-- CO-OP TRANSLATOR DISCLAIMER END -->