# အစားအသောက် အမျိုးအစား ခွဲခြားသူများ ၁

ဒီသင်ခန်းစာမှာ သင်ဟာ ယင်းအစားအသောက်အမျိုးအစားတွေနဲ့ပတ်သက်ပြီး ချိန်ညှိပြီး သန့်ရှင်းသော ဒေတာတစ်ခုကို မပြီးဆုံးခင်သင်ခန်းစာကနေ သိမ်းဆည်းထားတဲ့ ဒေတာ set ကိုအသုံးပြုပါမည်။

အဲဒီ dataset ကို အမျိုးအစားခွဲခြားသူမျိုးစုံနဲ့ သုံးပြီး _ပေးထားတဲ့ အစားအစာပစ္စည်း အုပ်စုတစ်ခုအပေါ်အခြေခံပြီး အမျိုးအစားခွဲခြားဖော်ထုတ်ခြင်း_ ပြုလုပ်မှာဖြစ်ပါတယ်။ ဒီအလုပ်လုပ်နေစဉ်အတွင်း အပိုင်းခွဲခြားလုပ်ငန်းများအတွက် algorithm တွေကို ဘယ်လိုအသုံးချနိုင်ကြောင်း အပိုများစွာ လေ့လာနိုင်မှာဖြစ်ပါတယ်။

## [သင်ခန်းစာမတိုင်မီ စမ်းမေးပွဲ](https://ff-quizzes.netlify.app/en/ml/)
# ပြင်ဆင်မှုပြုလုပ်ခြင်း

[သင်ခန်းစာ ၁](../1-Introduction/README.md) ပြီးဆုံးပြီးသားဖြစ်ကြောင်း သတ်မှတ်ပြီးသားဆိုထားပြီး၊ ဤသင်ခန်းစာလေးခုအတွက် root `/data` ဖိုလ်ဒါအတွင်း _cleaned_cuisines.csv_ ဖိုင်ရှိနေပြီးဖြစ်ကြောင်း သေချာစေပါ။

## လေ့ကျင့်ခန်း - အမျိုးအစားခွဲခြားရန် ခန့်မှန်းခြင်း

1. ဒီသင်ခန်းစာ၏ _notebook.ipynb_ ဖိုလ်ဒါအတွင်းမှာ အဲဒီဖိုင်နဲ့ Pandas စာကြည့်တိုက်ကို တင်သွင်းပါ။

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    ဒေတာဟာ ဒီလိုမျိုးပုံစံရှိပါတယ် -

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. ယခု အခြားစာကြည့်တိုက်များကို တင်သွင်းပါ။

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. X နှင့် y ကို သင်ကြားရေးအတွက် နှစ်ခုသော dataframe များအဖြစ် ခွဲထုတ်ပါ။ `cuisine` ကို label အဖြစ်သတ်မှတ်နိုင်သည်။

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    ဒီလိုမျိုးရှိပါလိမ့်မယ် -

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. `Unnamed: 0`နှင့် `cuisine` ကော်လံများကို `drop()` ဖြင့် ဖယ်ရှားပြီး training features အဖြစ် အခြားဒေတာများအား သိမ်းဆည်းပါ။

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    သင့် features ဟာ ဒီလိုပုံစံရှိတော့မယ် -

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

ယခု သင့်မှာ သင်တန်းပေးဖို့ အဆင်သင့်ဖြစ်ပါပြီ။

## သင့်အတွက် အမျိုးအစားခွဲခြားသူ ရွေးချယ်ခြင်း

ဒေတာ သန့်ရှင်းပြီး သင်ကြားရန် ပြင်ဆင်ပြီး ဖြစ်တဲ့အခါ၊ အလုပ်အတွက် ဘယ် algorithm ကို သုံးမလဲ ဆိုတာဆုံးဖြတ်ရပါမယ်။

Scikit-learn ဟာ Classification ကို Supervised Learning အတွင်း ထည့်သွင်းပြီး၊ အဲဒီအုပ်စုမှာ သင်တွေ့နိုင်တဲ့ အမျိုးအစားလာလားဖြစ်စေ classification နည်းလမ်းမျိုးစုံကို ရှာတွေ့နိုင်ပါတယ်။ [အမျိုးမျိုး](https://scikit-learn.org/stable/supervised_learning.html) မှာ ရှိတာက ပထမကြည့်တဲ့အခါ စိတ်မချမ်းသာ ဖြစ်စေတတ်ပါတယ်။ အောက်ပါနည်းလမ်းများတွင် classification နည်းပညာများ ပါဝင်သည် -

- လိုင်းနီးယား မော်ဒယ်များ
- Support Vector Machines
- Stochastic Gradient Descent
- အနီးဆုံးအိမ်နီးများ
- Gaussian Processes
- ဆုံးဖြတ်ရိုက်ပုံများ
- Ensemble နည်းလမ်းများ (voting Classifier)
- Multiclass နှင့် multioutput algorithm များ (multiclass နှင့် multilabel classification, multiclass-multioutput classification)

> သင်ဟာ [နယူးရာလ်နက်ဝက်များဖြင့် ဒေတာပြုလုပ်ခြင်း](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification)ကိုပါ အသုံးပြုနိုင်သော်လည်း၊ ဒီသင်ခန်းစာ၏ ပတ်ဝန်းကျင်ဟာ အဲဒါနဲ့ မဟုတ်ပါ။

### ဘယ်အမျိုးအစားခွဲခြားသူကို ရွေးမလဲ?

ဒါဆို အမျိုးအစားခွဲခြားသူ ဘယ်အမျိုးကိုရွေးသင့်သလဲ? ပုံမှန်အားဖြင့် အမျိုးမျိုးကို တွန်းလှန် စမ်းသပ်ပြီး ရလဒ်ကောင်းလာလားကြည့်ရင် စမ်းသပ်နိုင်ပါတယ်။ Scikit-learn ဟာ [တပြိုင်နက်မှာ နှိုင်းယှဉ်ခြင်း](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) ကို dataset တစ်ခုကိုဖန်တီးပြီး နည်းလမ်းအချို့ (KNeighbors, SVC နှစ်မျိုး, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB နှင့် QuadraticDiscrinationAnalysis) နဲ့ အဆင်ပြေမှုကို မြင်ယောင် ပြထားပါတယ် -

![comparison of classifiers](../../../../translated_images/my/comparison.edfab56193a85e7f.webp)
> Scikit-learn မှ အထောက်အထားစာတမ်းအတွင်း ဖန်တီးထားသော ပြဇာတ်များ ဖြစ်သည်။

> AutoML က ဒီပြဿနာကို သွားရောက်ပြီး ဆန်းစစ်ကာ သင့်ဒေတာအတွက် အကောင်းဆုံး algorithm ကို ရွေးနိုင်စေပါတယ်။ [ဒီမှာ ကြိုးစားကြည့်ပါ](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### ပိုမိုကောင်းမွန်သောနည်းလမ်း

လွဲမှားတွေးတာထက် ပိုကောင်းသောနည်းလမ်းကတော့ ဒီ [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) ကို ဒင်းလိုက်ဒေါင်းလုတ် ယူပြီး အကြံဉာဏ်တွေကို လိုက်နာခြင်းဖြစ်ပါတယ်။ ဒီမှာ ကျွန်တော်တို့ multiclass ပြဿနာအတွက် ရွေးချယ်စရာ အတော်များများ ရှိကြောင်း တွေ့ရှိရပါသည်။

![cheatsheet for multiclass problems](../../../../translated_images/my/cheatsheet.07a475ea444d2223.webp)
> Microsoft ၏ Algorithm Cheat Sheet မှ အစိတ်အပိုင်းတစ်ခုဖြစ်ပြီး multiclass classification အတွက် ရွေးချယ်နည်းများ ဖော်ပြထားသည်။

✅ ဒီ cheat sheet ကို ဒေါင်းလုဒ် ယူ၍ ပုံနှိပ်ထားပြီး နံရံပေါ်တွင် ပေါ်လာအောင် တပ်ထားပါ။

### အကြောင်းပြချက်များ

ကြည့်ကြပါစို့ မတူညီသော လမ်းစဉ်များကို ငြင်းဆိုရာတွင် ကျွန်တော်တို့ ရှိတဲ့ကန့်သတ်ချက်များအရ -

- **နယူးရာလ်နက်ဝက်များသည် များစွာကြီးလွန်းသည်**။ သန့်ရှင်းပြီး သေးငယ်သည့် dataset ဖြင့်၊ သင့်က နောက်ခံနက်ဘုတ်ဖြင့် ထည့်သင်ကြားခြင်း များ ပြုလုပ်နေသောကြောင့် Neural network များသည် အလေးမတင်နိုင်သည်။
- **နှစ်သိမ့်အမျိုးအစား ခွဲခြားသူ မရှိပါ**။ နှစ်သိမ့် classifier မသုံးတော့သောကြောင့် one-vs-all ကို မသုံးပါ။
- **Decision tree သို့မဟုတ် logistic regression ကောင်းကောင်းလည်ပတ်နိုင်သည်**။ Decision tree သို့မဟုတ် multiclass အချက်အလက်များအတွက် logistic regression အသုံးပြုနိုင်ပါတယ်။
- **Multiclass Boosted Decision Trees သည် အခြားပြဿနာများဖြေရှင်းသည်**။ Multiclass boosted decision tree သည် ranking ကို ဖန်တီးရန် ဒီဇိုင်းပြုလုပ်သော nonparametric task များမှသာ အသုံးပြုသင့်သည်၊ သို့င်း ကျွန်ုပ်တို့အတွက် မသင့်တော်ပါ။

### Scikit-learn ကို အသုံးပြုခြင်း

ကျွန်တော်တို့ Scikit-learn ကို ဒေတာအား စစ်တမ်းဖော်မြူဆောင်ရန် သုံးမှာ ဖြစ်သည်။ သို့သော် logistic regression တွင် Scikit-learn မှာ parameter များ များစွာ အသုံးပြုနိုင်ပါသည်။ [parameter များကြည့်ပါ](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)။

အဓိကအားဖြင့် အရေးကြီးသော parameter နှစ်ခုရှိသည် - `multi_class` နှင့် `solver` - တိုင်းတာပြီး Scikit-learn ကို logistic regression လုပ်ရန် တောင်းဆိုသည့်အခါ သတ်မှတ်ရမည်။ `multi_class` သည် မတူညီသော လုပ်ဆောင်ချက်တစ်ခု လုပ်ဆောင်ပြီး၊ solver သည် ဘယ် algorithm ကို သုံးမယ်ဆိုတာ သတ်မှတ်သည်။ အားလုံးသော solver များသည် `multi_class` အားလုံးနှင့် တွဲဖက် အသုံးပြုလို့ မရပါ။

စာရွက်စာတမ်းအရ၊ multiclass တွင်၊ သင်ကြားရေး algorithm သည် -

- `multi_class` ကို `ovr` သတ်မှတ်ထားလျှင် **one-vs-rest (OvR) စနစ်ကို အသုံးပြုသည်**
- `multi_class` ကို `multinomial` သတ်မှတ်ထားလျှင် **cross-entropy loss ကို အသုံးပြုသည်**။ (လောလောဆယ် ‘lbfgs’, ‘sag’, ‘saga’ နဲ့ ‘newton-cg’ solver တွေရဲ့ အတွက်သာ `multinomial` ကို ထောက်ပံ့သည်)"

> 🎓 'scheme' ဆိုတာဟာ 'ovr' (one-vs-rest) သို့မဟုတ် 'multinomial' ဖြစ်နိုင်သည်။ Logistic regression သည် ဒုတိယအချိုးချင်း classification အတွက် ရည်ရွယ်ထားပြီး၎င်းဖြင့် multiclass classification ပြဿနာများကို ပိုမိုကောင်းမွန်စွာ ဖြေရှင်းနိုင်စေရန် အဲဒီစနစ်တွေကို အသုံးပြုသည်။ [source](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' သည် optimization ပြဿနာတွင် အသုံးပြုသော algorithm ကို သတ်မှတ်ထားသည်။ [source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn သည်  solver များသည် ကွာခြားသော ဒေတာဖွဲ့စည်းမှုအမျိုးအစားများကို မည်သို့ ကိုင်တွယ်သည်ဆိုတာကို ရှင်းပြရန် အောက်ပါ အတန်းဇယားကို ပေးထားသည်။

![solvers](../../../../translated_images/my/solvers.5fc648618529e627.webp)

## လေ့ကျင့်ခန်း - ဒေတာခွဲခြမ်းခြားခြင်း

မကြာသေးခင် သင်ခန်းစာတစ်ခုမှာ logistic regression ကို သင်ယူခဲ့တာဖြစ်လို့ ပထမဆုံး သင်ကြားမှုအတွက် logistic regression ကို အဓိကထားပြီး စမ်းသပ်ပါ။
`train_test_split()` ကို ခေါ်ဆို၍ သင်နှင့် စမ်းသပ် ဘားကိုခွဲပါ -

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## လေ့ကျင့်ခန်း - logistic regression ကို အသုံးပြုပါ

မိမိသည် multiclass ဖြစ်သောကြောင့် ဘယ် _scheme_ ကို အသုံးပြုမလဲ၊ ဘယ် _solver_ ကို သတ်မှတ်မလဲ ဆိုတာ ရွေးဖို့လိုပါသည်။ LogisticRegression ကို multi_class ကို `ovr` နှင့် solver ကို **liblinear** သတ်မှတ်၍ သင်ကြားပါ။

1. multi_class ကို `ovr` သတ်မှတ်ထားပြီး solver ကို `liblinear` သတ်မှတ်ထားသော logistic regression ကို ဖန်တီးပါ။

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ ပုံမှန်အားဖြင့်သတ်မှတ်ထားသော `lbfgs` ကဲ့သို့ အခြား solver ကို စမ်းသုံးကြည့်ပါ။

    > မှတ်ချက် - လိုအပ်လျှင် Pandas ၏ [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) function အသုံးပြုပြီး ဒေတာကို တန်းစီချခြင်း။

    တိကျမှုမှာ **၈၀% ကျော်** ဖြစ်ကြောင်း တွေ့နိုင်ပါသည်!

1. ဒေတာ အတန်းတစ်ခု (#50) ကို စမ်းသပ်ကာ ဒီမော်ဒယ်က ဘယ်လို သွားမလဲ ကြည့်နိုင်ပါသည် -

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    ရလဒ် ကို ဤသို့ ပြသသည် -

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ အတန်း အရေအတွက် ပြောင်းပြီး ရလဒ်တွေ စစ်ဆေးကြည့်ပါ။
1. Digging deeper, you can check for the accuracy of this prediction:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    The result is printed - Indian cuisine is its best guess, with good probability:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Can you explain why the model is pretty sure this is an Indian cuisine?

1. Get more detail by printing a classification report, as you did in the regression lessons:

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

ဒီသင်ခန်းစာမှာ သင်ဟာ သန့်ရှင်းပြီးပြင်ဆင်ထားတဲ့ ဒေတာကို အသုံးပြုပြီး ပစ္စည်းများအခြေပြု၍ နိုင်ငံတစ်ကာစားနပ်ရိက္ခာမျိုးကို ခန့်မှန်းနိုင်တဲ့ စက်မှုသင်ယူမှု မော်ဒယ်ကို တည်ဆောက်ခဲ့ပါသည်။ Scikit-learn က ဒေတာကို သတ်မှတ်ဖော်ကြားဖို့ ပေးသည့် အမျိုးအစားများအတွက် ရွေးချယ်စရာများစွာကို အချိန်ယူဖတ်ရှုပါ။ 'solver' အယူအဆိုနောက်က အကြောင်းအရာကို ပိုမိုနက်နဲစွာ လေ့လာပါ။

## [စာသင်ခန်း နှောင်း အကြံပေး စစ်ဆေးမှု](https://ff-quizzes.netlify.app/en/ml/)

## ပြန်လည်သုံးသပ်ခြင်းနှင့် ကိုယ်တိုင်လေ့လာခြင်း

Logistic regression နောက်က ဖြစ်ပေါ်လာတဲ့ နည်းဗေဒများကို [ဒီသင်ခန်းစာ](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf) တွင် နည်းပိုမို နက်နဲစွာ လေ့လာပါ။
## အလုပ်တာဝန်

[Solvers များကို လေ့လာပါ](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ကန်တော့ချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ဖြင့် ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေသော်လည်း စက်ရုပ်ဘာသာပြန်မှုများတွင် အမှားများ သို့မဟုတ် မှန်ကန်မှုမရှိမှုများ ဖြစ်ပေါ်နိုင်ပါသည်။ မူလစာရွက်စာတမ်းကို မူရင်းဘာသာစကားဖြင့်သာ တရားဝင်အရင်းအမြစ်အဖြစ် သတ်မှတ်ရန်လိုအပ်ပါသည်။ အရေးကြီးသောအချက်အလက်များအတွက် ကျွမ်းကျင်သော လူ့ဘာသာ ပြန်ဆိုမှုကို သုံးသင့်ပါသည်။ ဤဘာသာပြန်ချက်ကို အသုံးပြုမှုကြောင့် ဖြစ်ပေါ်လာသော မနားမသာနားလည်မှုများ သို့မဟုတ် အဓိပ္ပာယ်မှားချက်များအတွက် ကျွန်ုပ်တို့ အာမခံမှု မရှိပါ။
<!-- CO-OP TRANSLATOR DISCLAIMER END -->