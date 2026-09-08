# طبقه‌بندهای آشپزی ۱

در این درس، شما از مجموعه داده‌ای که در درس قبلی ذخیره کردید استفاده خواهید کرد، مجموعه داده‌ای متعادل و تمیز که تماماً درباره آشپزی‌هاست.

شما از این مجموعه داده با انواع مختلف طبقه‌بندها برای _پیش‌بینی یک آشپزی ملی مشخص بر اساس گروهی از مواد اولیه_ استفاده خواهید کرد. در حین انجام این کار، بیشتر با برخی از روش‌های استفاده از الگوریتم‌ها برای وظایف طبقه‌بندی آشنا خواهید شد.

## [آزمون پیش‌از‌درس](https://ff-quizzes.netlify.app/en/ml/)
# آماده‌سازی

فرض کنید درس [درس ۱](../1-Introduction/README.md) را کامل کرده‌اید، اطمینان حاصل کنید که یک فایل _cleaned_cuisines.csv_ در پوشه ریشه `/data` برای این چهار درس وجود دارد.

## تمرین - پیش‌بینی یک آشپزی ملی

۱. در پوشه این درس با نام _notebook.ipynb_، آن فایل را همراه با کتابخانه Pandas وارد کنید:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    داده‌ها شبیه به این هستند:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

۱. حال، چند کتابخانه دیگر وارد کنید:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

۱. مختصات X و y را به دو دیتا‌فریم برای آموزش تقسیم کنید. `cuisine` می‌تواند دیتا‌فریم برچسب‌ها باشد:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    به این شکل خواهد بود:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

۱. آن ستون `Unnamed: 0` و ستون `cuisine` را با فراخوانی `drop()` حذف کنید. بقیه داده‌ها را به‌عنوان ویژگی‌های قابل آموزش ذخیره کنید:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    ویژگی‌های شما شبیه به این خواهند بود:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

حالا آماده‌اید تا مدل خود را آموزش دهید!

## انتخاب طبقه‌بند شما

اکنون که داده‌های شما تمیز و آماده آموزش هستند، باید تصمیم بگیرید کدام الگوریتم را برای این کار استفاده کنید.

Scikit-learn طبقه‌بندی را در زیر مجموعه یادگیری نظارت‌شده دسته‌بندی می‌کند، و در آن دسته چندین روش برای طبقه‌بندی خواهید یافت. [تنوع](https://scikit-learn.org/stable/supervised_learning.html) در نگاه اول بسیار گیج‌کننده است. روش‌های زیر همه شامل تکنیک‌های طبقه‌بندی هستند:

- مدل‌های خطی
- ماشین‌های بردار پشتیبان
- نزول گرادیان تصادفی
- نزدیک‌ترین همسایه‌ها
- فرآیندهای گاوسی
- درخت‌های تصمیم
- روش‌های مجموعه‌ای (voting Classifier)
- الگوریتم‌های چندکلاسه و چندخروجی (طبقه‌بندی چندکلاسه و چندبرچسبه، طبقه‌بندی چندکلاسه-چندخروجی)

> همچنین می‌توانید از [شبکه‌های عصبی برای طبقه‌بندی داده‌ها استفاده کنید](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification)، اما این خارج از محدوده این درس است.

### کدام طبقه‌بند را انتخاب کنیم؟

پس، کدام طبقه‌بند را باید انتخاب کنید؟ اغلب، اجرای چند تا و جستجو برای نتیجه خوب روش خوبی برای آزمایش است. Scikit-learn یک [مقایسه کنار هم](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) روی یک مجموعه داده ایجاد شده ارائه می‌دهد که KNeighbors، دو روش SVC، GaussianProcessClassifier، DecisionTreeClassifier، RandomForestClassifier، MLPClassifier، AdaBoostClassifier، GaussianNB و QuadraticDiscrinationAnalysis را مقایسه کرده و نتایج را به‌صورت تصویری نشان می‌دهد:

![مقایسه طبقه‌بندها](../../../../translated_images/fa/comparison.edfab56193a85e7f.webp)
> نمودارهای تولید شده در مستندات Scikit-learn

> AutoML این مشکل را به خوبی با اجرای این مقایسه‌ها در فضای ابری حل می‌کند، که به شما امکان می‌دهد بهترین الگوریتم را برای داده‌های خود انتخاب کنید. آن را [اینجا](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott) امتحان کنید.

### رویکردی بهتر

یک روش بهتر از حدس زدن کورکورانه، پیروی از ایده‌های این [برگه تقلب یادگیری ماشین](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) قابل دانلود است. در اینجا، کشف می‌کنیم که برای مسئله چندکلاسه خود، چند انتخاب داریم:

![برگه تقلب برای مسائل چندکلاسه](../../../../translated_images/fa/cheatsheet.07a475ea444d2223.webp)
> بخشی از برگه تقلب الگوریتم مایکروسافت، جزئیات گزینه‌های طبقه‌بندی چندکلاسه

✅ این برگه تقلب را دانلود، چاپ کنید و روی دیوار خود آویزان کنید!

### استدلال

بیایید ببینیم آیا می‌توانیم رویکردهای مختلف را با توجه به محدودیت‌هایی که داریم استدلال کنیم:

- **شبکه‌های عصبی خیلی سنگین هستند**. با توجه به مجموعه داده تمیز اما کوچک ما و اینکه آموزش به صورت محلی از طریق نوت‌بوک‌ها انجام می‌شود، شبکه‌های عصبی برای این کار سنگین است.
- **طبقه‌بند دوکلاسه نداریم**. ما از طبقه‌بند دوکلاسه استفاده نمی‌کنیم، پس روش یک‌درمیان را کنار می‌گذاریم.
- **درخت تصمیم یا رگرسیون لجستیک می‌تواند کار کند**. درخت تصمیم ممکن است موثر باشد، یا رگرسیون لجستیک برای داده‌های چندکلاسه.
- **درخت تصمیم تقویت‌شده چندکلاسه مسئله متفاوتی را حل می‌کند**. درخت تصمیم تقویت‌شده چندکلاسه بیشتر برای کارهای غیرپارامتریک مناسب است، مثلاً وظایف طراحی‌شده برای ساخت رتبه‌بندی، پس برای ما مفید نیست.

### استفاده از Scikit-learn

ما از Scikit-learn برای تحلیل داده‌های خود استفاده خواهیم کرد. اما راه‌های زیادی برای استفاده از رگرسیون لجستیک در Scikit-learn وجود دارد. نگاهی به [پارامترهایی که باید ارسال کنید](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) بیندازید.

اساساً دو پارامتر مهم - `multi_class` و `solver` - وجود دارد که باید مشخص کنیم وقتی از Scikit-learn می‌خواهیم رگرسیون لجستیک انجام دهد. مقدار `multi_class` نوع خاصی از رفتار را اعمال می‌کند. مقدار solver الگوریتم مورد استفاده است. همه حل‌کننده‌ها نمی‌توانند با همه `multi_class`ها جفت شوند.

طبق مستندات، در حالت چندکلاسه، الگوریتم آموزش:

- **از طرح یک‌درمیان (OvR) استفاده می‌کند**، اگر گزینه `multi_class` روی `ovr` تنظیم شود
- **از تابع هزینه آنتروپی متقاطع استفاده می‌کند**، اگر گزینه `multi_class` روی `multinomial` تنظیم شود. (فعلاً گزینه `multinomial` فقط توسط حل‌کننده‌های ‘lbfgs’, ‘sag’, ‘saga’ و ‘newton-cg’ پشتیبانی می‌شود.)"

> 🎓 این 'طرح' اینجا می‌تواند 'ovr' (یک‌درمیان) یا 'multinomial' باشد. از آنجا که رگرسیون لجستیک در اصل برای طبقه‌بندی دودویی طراحی شده است، این طرح‌ها به آن کمک می‌کنند تا بهتر وظایف طبقه‌بندی چندکلاسه را مدیریت کند. [منبع](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'حل‌کننده' به عنوان "الگوریتمی که در مسئله بهینه‌سازی استفاده می‌شود" تعریف شده است. [منبع](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)

Scikit-learn این جدول را برای توضیح چگونگی برخورد حل‌کننده‌ها با چالش‌های مختلف ناشی از ساختارهای مختلف داده ارائه می‌دهد:

![حل‌کننده‌ها](../../../../translated_images/fa/solvers.5fc648618529e627.webp)

## تمرین - تقسیم داده‌ها

ما می‌توانیم برای اولین آزمایش آموزش خود روی رگرسیون لجستیک تمرکز کنیم، چون اخیراً درباره آن در درس قبلی یاد گرفتید.
داده‌های خود را به گروه‌های آموزش و تست تقسیم کنید با فراخوانی `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## تمرین - اعمال رگرسیون لجستیک

از آنجا که شما از حالت چندکلاسه استفاده می‌کنید، باید انتخاب کنید چه _طرحی_ استفاده شود و چه _حل‌کننده‌ای_ تنظیم شود. از LogisticRegression با تنظیمات چندکلاسه و حل‌کننده **liblinear** برای آموزش استفاده کنید.

۱. یک رگرسیون لجستیک با `multi_class` برابر `ovr` و حل‌کننده برابر `liblinear` بسازید:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ یک حل‌کننده دیگر مثل `lbfgs` که معمولا به صورت پیش‌فرض تنظیم می‌شود را امتحان کنید

    > توجه کنید، از تابع [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) پانداها برای صاف کردن داده‌ها در صورت نیاز استفاده کنید.

    دقت بالای **۸۰٪** خوبی دارد!

۱. می‌توانید این مدل را با تست یک سطر داده (#50) در عمل ببینید:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    نتیجه چاپ می‌شود:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ شماره ردیف دیگری را امتحان کنید و نتایج را بررسی کنید
1. کاوش عمیق‌تر، می‌توانید دقت این پیش‌بینی را بررسی کنید:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    نتیجه چاپ می‌شود - بهترین حدس آن غذاهای هندی است، با احتمال خوب:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ می‌توانید توضیح دهید چرا مدل تقریباً مطمئن است که این غذاهای هندی است؟

1. با چاپ گزارش طبقه‌بندی، همانطور که در درس‌های رگرسیون انجام دادید، جزئیات بیشتری بگیرید:

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

## 🚀چالش

در این درس، با استفاده از داده‌های تمیز شده خود، مدلی برای یادگیری ماشین ساختید که می‌تواند غذاهای ملی را بر اساس مجموعه‌ای از مواد تشکیل دهنده پیش‌بینی کند. زمانی را صرف خواندن گزینه‌های بسیاری کنید که Scikit-learn برای طبقه‌بندی داده‌ها ارائه می‌دهد. عمیق‌تر به مفهوم «حل‌کننده» (solver) بپردازید تا بفهمید پشت صحنه چه می‌گذرد.

## [آزمون پس از درس](https://ff-quizzes.netlify.app/en/ml/)

## بازبینی و مطالعه خودآموز

کمی بیشتر در ریاضیات پشت رگرسیون لجستیک در [این درس](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf) عمیق شوید.

## تکلیف

[مطالعه حل‌کننده‌ها](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**سلب مسئولیت**:  
این سند با استفاده از سرویس ترجمه ماشینی [Co-op Translator](https://github.com/Azure/co-op-translator) ترجمه شده است. در حالی که ما در تلاش برای دقت هستیم، لطفاً توجه داشته باشید که ترجمه‌های خودکار ممکن است شامل خطاها یا نواقصی باشند. سند اصلی به زبان بومی آن باید به عنوان منبع معتبر در نظر گرفته شود. برای اطلاعات مهم، ترجمه حرفه‌ای انسانی توصیه می‌شود. ما مسئول هیچ گونه سوءتفاهم یا برداشت نادرستی که از استفاده از این ترجمه به وجود آید، نیستیم.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->