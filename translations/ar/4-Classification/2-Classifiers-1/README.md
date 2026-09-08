# مصنفات المطبخ 1

في هذا الدرس، ستستخدم مجموعة البيانات التي حفظتها من الدرس السابق والتي تحتوي على بيانات متوازنة ونظيفة حول المطابخ.

ستستخدم هذه المجموعة مع مجموعة متنوعة من المصنفات لـ _التنبؤ بمطبخ وطني معين بناءً على مجموعة من المكونات_. أثناء القيام بذلك، ستتعلم المزيد عن بعض الطرق التي يمكن استخدام الخوارزميات من خلالها لمهام التصنيف.

## [اختبار ما قبل المحاضرة](https://ff-quizzes.netlify.app/en/ml/)
# التحضير

بافتراض أنك أكملت [الدرس 1](../1-Introduction/README.md)، تأكد من وجود ملف _cleaned_cuisines.csv_ في المجلد الجذر `/data` لهذه الدروس الأربعة.

## التمرين - التنبؤ بمطبخ وطني

1. بالعمل في مجلد _notebook.ipynb_ لهذا الدرس، قم باستيراد ذلك الملف بالإضافة إلى مكتبة Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    تبدو البيانات بهذا الشكل:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. الآن، استورد المزيد من المكتبات:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. قسم إحداثيات X و y إلى إطارين بيانات للتدريب. يمكن أن يكون `cuisine` إطار بيانات للتسميات:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    سيبدو هكذا:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. اسقط عمود `Unnamed: 0` وعمود `cuisine` باستخدام `drop()`. احفظ بقية البيانات كميزات قابلة للتدريب:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    تبدو ميزاتك هكذا:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

الآن أنت جاهز لتدريب نموذجك!

## اختيار المصنف الخاص بك

الآن بعد أن أصبحت بياناتك نظيفة وجاهزة للتدريب، عليك أن تقرر أي خوارزمية تستخدم للمهمة.

يصنف Scikit-learn التصنيف تحت تعلم مراقب، وفي تلك الفئة ستجد العديد من الطرق لتصنيف البيانات. [التنوع](https://scikit-learn.org/stable/supervised_learning.html) مربك في الوهلة الأولى. تشمل الطرق التالية جميع تقنيات التصنيف:

- النماذج الخطية
- آلات الدعم الناقل
- الهبوط العشوائي المتدرج
- الجيران الأقرب
- العمليات الغاوسية
- أشجار القرار
- طرق التجميع (مصنف التصويت)
- خوارزميات متعددة الفئات ومتعددة النتائج (تصنيف متعدد الفئات ومتعدد التصنيفات، وتصنيف متعدد الفئات متعدد النتائج)

> يمكنك أيضًا استخدام [الشبكات العصبية لتصنيف البيانات](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification)، لكن ذلك خارج نطاق هذا الدرس.

### أي مصنف تختار؟

فأي مصنف يجب اختيار؟ غالبًا، التجربة عبر عدة مصنفات والبحث عن نتيجة جيدة هو طريقة للاختبار. يقدم Scikit-learn [مقارنة جنبًا إلى جنب](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) على مجموعة بيانات مصطنعة، يقارن فيها KNeighbors، وSVC بطريقتين، وGaussianProcessClassifier، وDecisionTreeClassifier، وRandomForestClassifier، و MLPClassifier، وAdaBoostClassifier، وGaussianNB وQuadraticDiscrinationAnalysis، ويعرض النتائج بشكل مرئي:

![مقارنة بين المصنفات](../../../../translated_images/ar/comparison.edfab56193a85e7f.webp)
> الرسوم المولدة في توثيق Scikit-learn

> يحل AutoML هذه المشكلة بشكل أنيق من خلال إجراء هذه المقارنات في السحابة، مما يتيح لك اختيار أفضل خوارزمية لبياناتك. جربه [هنا](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### نهج أفضل

طريقة أفضل من التخمين العشوائي، هي اتباع الأفكار في [ورقة الغش](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) القابلة للتنزيل هذه. هنا، نكتشف أنه لمشكلتنا متعددة الفئات، لدينا بعض الخيارات:

![ورقة الغش لمشاكل متعددة الفئات](../../../../translated_images/ar/cheatsheet.07a475ea444d2223.webp)
> قسم من ورقة الغش لخوارزميات مايكروسوفت، يوضح خيارات التصنيف متعدد الفئات

✅ قم بتحميل ورقة الغش هذه، واطبعها وعلقها على جدارك!

### السبب

دعنا نحاول التفكير في الطرق المختلفة استنادًا إلى القيود التي نمتلكها:

- **الشبكات العصبية ثقيلة جدًا**. نظرًا لأن بياناتنا نظيفة وقليلة، وحقيقة أننا ندرب محلياً عبر دفاتر الملاحظات، الشبكات العصبية ثقيلة جدًا لهذه المهمة.
- **لا نستخدم مصنف ثنائي الفئة**. لا نستخدم مصنفًا ثنائي الفئة، لذا فإن طريقة one-vs-all مستبعدة.
- **شجرة القرار أو الانحدار اللوجستي قد تعمل**. شجرة القرار قد تعمل، أو الانحدار اللوجستي للبيانات متعددة الفئات.
- **أشجار القرار المعززة متعددة الفئات تحل مشكلة مختلفة**. شجرة القرار المعززة متعددة الفئات مناسبة أكثر للمهام غير المعلمية، مثل المهام المصممة لبناء التصنيفات، لذا فهي ليست مفيدة لنا.

### استخدام Scikit-learn

سنستخدم Scikit-learn لتحليل بياناتنا. ومع ذلك، هناك العديد من الطرق لاستخدام الانحدار اللوجستي في Scikit-learn. ألق نظرة على [المعلمات التي يمكن تمريرها](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

أساسًا، هناك معلمان مهمان - `multi_class` و `solver` - يجب تحديدهما عند طلب Scikit-learn إجراء انحدار لوجستي. تطبق قيمة `multi_class` سلوكًا معينًا. قيمة `solver` هي الخوارزمية المستخدمة. ليست كل الخوارزميات متوافقة مع كل قيم `multi_class`.

وفقًا للتوثيق، في حالة متعدد الفئات، خوارزمية التدريب:

- **تستخدم نظام one-vs-rest (OvR)** إذا تم تعيين خيار `multi_class` إلى `ovr`
- **تستخدم خسارة الإنتروبيا المتقاطعة** إذا تم تعيين خيار `multi_class` إلى `multinomial`. (حاليًا خيار `multinomial` مدعوم فقط من المحللات ‘lbfgs’, ‘sag’, ‘saga’ و ‘newton-cg’)." 

> 🎓 "النظام" هنا يمكن أن يكون 'ovr' (واحد ضد الباقي) أو 'multinomial'. بما أن الانحدار اللوجستي مصمم بشكل أساسي لدعم التصنيف الثنائي، فإن هذه الأنظمة تسمح له بمعالجة مهام التصنيف متعددة الفئات بشكل أفضل. [المصدر](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 "المحلل" (solver) معرف بأنه "الخوارزمية المستخدمة في مشكلة التحسين". [المصدر](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

يقدم Scikit-learn هذا الجدول ليشرح كيف تتعامل المحللات مع تحديات مختلفة مقدمة من أنواع هياكل البيانات المختلفة:

![المحللات](../../../../translated_images/ar/solvers.5fc648618529e627.webp)

## تمرين - تقسيم البيانات

يمكننا التركيز على الانحدار اللوجستي كتجربة تدريب أولى لأنك تعلمت عنه مؤخرًا في درس سابق.
اقسم بياناتك إلى مجموعات تدريب واختبار عن طريق استدعاء `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## تمرين - تطبيق الانحدار اللوجستي

بما أنك تستخدم حالة متعدد الفئات، تحتاج إلى اختيار أي _نظام_ تستخدم وأي _محلل_ تختار. استخدم LogisticRegression مع إعداد متعدد الفئات والمحّل **liblinear** للتدريب.

1. أنشئ نموذج انحدار لوجستي مع تعيين multi_class إلى `ovr` والمحلل إلى `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ جرب محللًا مختلفًا مثل `lbfgs`، والذي غالبًا ما يكون الافتراضي

    > ملاحظة، استخدم دالة Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) لتسطيح بياناتك عند الحاجة.

    الدقة جيدة بأكثر من **80%**!

1. يمكنك مشاهدة هذا النموذج أثناء العمل عن طريق اختبار صف واحد من البيانات (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    يتم طباعة النتيجة:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ جرب رقم صف مختلف وتحقق من النتائج
1. بالتعمق أكثر، يمكنك التحقق من دقة هذا التنبؤ:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    يتم طباعة النتيجة - المطبخ الهندي هو أفضل تخمين، مع احتمال جيد:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ هل يمكنك شرح لماذا النموذج متأكد إلى حد كبير أن هذا هو المطبخ الهندي؟

1. احصل على مزيد من التفاصيل عن طريق طباعة تقرير التصنيف، كما فعلت في دروس الانحدار:

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

## 🚀التحدي

في هذا الدرس، استخدمت بياناتك المنظفة لبناء نموذج تعلم آلي يمكنه التنبؤ بنوع مطبخ وطني بناءً على سلسلة من المكونات. خذ بعض الوقت لقراءة الخيارات العديدة التي يوفرها Scikit-learn لتصنيف البيانات. تعمق أكثر في مفهوم 'المُحلل' لفهم ما يحدث وراء الكواليس.

## [اختبار ما بعد المحاضرة](https://ff-quizzes.netlify.app/en/ml/)

## مراجعة ودراسة ذاتية

اغص أكثر في الرياضيات وراء الانحدار اللوجستي في [هذا الدرس](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## المهمة

[ادرس المحللات](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**إخلاء المسؤولية**:  
تمت ترجمة هذا المستند باستخدام خدمة الترجمة الآلية [Co-op Translator](https://github.com/Azure/co-op-translator). بينما نسعى لتحقيق الدقة، يرجى العلم بأن الترجمات الآلية قد تحتوي على أخطاء أو عدم دقة. يجب اعتبار المستند الأصلي بلغته الأصلية المصدر المعتمد. للحصول على معلومات هامة، يُنصح بالاعتماد على ترجمة بشرية محترفة. نحن غير مسؤولين عن أي سوء فهم أو تفسير خاطئ ناتج عن استخدام هذه الترجمة.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->