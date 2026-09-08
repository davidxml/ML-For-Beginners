# အစားအသောက် အမျိုးအစား ခွဲခြားခြင်း 2

ဒီ ဒုတိယအတန်းခွဲခြားခြင်း ပညာသင်ခန်းစာမှာ သင်သည် ဂဏန်း ဒေတာများကို အမျိုးအစား ခွဲခြားနိုင်မည့် နည်းလမ်းများ ပိုမိုလေ့လာမည်ဖြစ်သည်။ သင်သည် အမျိုးအစားခွဲခြားသူ တစ်ခုပြီး တစ်ခုကို ရွေးချယ်ရာမှ ဖြစ်ပေါ်နိုင်သည့် ထိခိုက်မှုများကိုလည်း သိရှိမည်ဖြစ်သည်။

## [သင်ခန်းစာမတိုင်ခင် မေးခွန်းစစ်](https://ff-quizzes.netlify.app/en/ml/)

### ယခင်သင်ခန်းစာလိုအပ်ချက်

သင်သည် ယခင်သင်ခန်းစာများကို ပြီးမြောက်ပြီးဖြစ်ပြီး၊ ဒီ ၄-ခန်းသင်ခန်းစာ folder၏ ရှေ့ဆက် သက်သက်ရှိသော _cleaned_cuisines.csv_ ဟူသော သန့်ရှင်းထားသော ဒေတာစုစည်းမှုကို `data` ဖိုလ်ဒါအတွင်းရှိနေသည်ဟု ထင်မြင်ကြောင်း။ 

### အဆင်ပြေမှု

သင်၏ _notebook.ipynb_ ဖိုင်ကို သန့်ရှင်းထားသော ဒေတာနှင့် ဖွင့်ထားပြီး X နှင့် y dataframes သို့ ခွဲထားပြီး မော်ဒယ်တည်ဆောက်ခြင်းလုပ်ငန်းစဉ်အတွက် အသင့်ဖြစ်သည်။

## အမျိုးအစား ခွဲခြားခြင်း အမြေပုံ

ပြီးခဲ့သောအချိန်တွင် Microsoft ၏ cheat sheet ကို အသုံးပြု၍ ဒေတာ ခွဲခြားရာမှာ ရွေးချယ်နိုင်သည့် နည်းလမ်းမျိုးစုံကို သင်သိရှိခဲ့သည်။ Scikit-learn သည် ဆက်ဆံမှု ပိုမြင့်နှင့် အသေးစိတ်ကဲြသော cheat sheet ကို အသုံးပြုနိုင်ပြီး classifier များအား ပိုမို ကန့်သတ်နိုင်စေသည် (classifier အတွက် alternative term ဖြစ်သည်)။

![ML Map from Scikit-learn](../../../../translated_images/my/map.e963a6a51349425a.webp)
> အကြံပြုချက်- [ဤမြေပုံကို အွန်လိုင်းတွင် သွားရောက်ကြည့်ရှုပါ](https://scikit-learn.org/stable/tutorial/machine_learning_map/)၊ လမ်းကြောင်းအလိုက် နှိပ်၍ စာရွက်စာတမ်းများကို ဖတ်ရှုနိုင်ပါသည်။

### အစီအစဉ်

ဒီမြေပုံသည် သင်၏ ဒေတာကို လုံးလုံးလင်လင် သိရှိပြီးနောက်တွင် ဦးတည်ချက်ချရာတွင် အလွန်အသုံးဝင်သည်။

- မည်သည့်နောက်ထပ် sampling များ > 50ရှိသည်
- အမျိုးအစားတစ်ခုကို ခန့်မှန်းလိုသည်
- label အချက်အလက်ရှိသည်
- sampling များ < 100K ဖြစ်သည်
- ✨ Linear SVC ကို ရွေးချယ်နိုင်သည်
- အကယ်၍ မအောင်မြင်ပါက, ဂဏန်းဒေတာရှိသောကြောင့်
  - ✨ KNeighbors Classifier ကို စမ်းသပ်နိုင်သည်
    - မအောင်မြင်ပါက ✨ SVC နှင့် ✨ Ensemble Classifiers ကို စမ်းသပ်နိုင်သည်

ဒီလမ်းကြောင်းသည် လိုက်နာရန်အတွက် အလွန် အကျိုးဖြစ်ဖွယ် ဖြစ်သည်။

## လေ့ကျင့်ခန်း - ဒေတာကို ခွဲခြားမှု

ဒီလမ်းကြောင်းအရ သုံးရန်လိုအပ်သည့် libraries များကို ယနေ့ဆီစတင် အတင်ပါ။

1. လိုအပ်သော libraries များကို ထည့်သွင်းပါ-

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. သင်၏ သင်ကြားမှုနှင့် စမ်းသပ်မှု ဒေတာကို ခွဲပါ-

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC classifier

Support-Vector clustering (SVC) သည် Support-Vector machine များ၏ ကလေး ဖြစ်ပြီး ML နည်းပညာမျိုးဆက်တစ်ခုဖြစ်သည် (အောက်တွင် ပိုမိုလေ့လာပါ)။ ဤနည်း၊ သင်သည် 'kernel' ကို ရွေးချယ်ကာ labels များကို ဘယ်လို အုပ်စုဖွဲ့မည်ကိုဆုံးဖြတ်နိုင်သည်။ 'C' parameter သည် 'regularization' ကို ဆိုလိုပြီး သတ်မှတ်မှု parameter များ၏ သက်ရောက်မှုကို ထိန်းညှိပေးသည်။ kernel သည် [အမျိုးမျိုး](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) ရှိသော်လည်း; ဒီမှာ ဗဟိုထားသည်မှာ 'linear' ဖြစ်ပြီး linear SVC ကိုအသုံးချရန်ဖြစ်သည်။ Probability သည် မပေးထားပါက 'false' ဖြစ်သည်; ဒီမှာ ကို 'true' သတ်မှတ်ထားပြီး probability အခြေခံခန့်မှန်းမှုများပိုရရှိစေရန် ဖြစ်သည်။ random state ကို '0' သတ်မှတ်၍ ဒေတာတွေ စီမံချက်တည့် စောင့်ရှောက်ခြင်းအတွက်ဖြစ်သည်။

### လေ့ကျင့်ခန်း - Linear SVC ကို အသုံးပြုပါ

စတင်ရန် classifier array တစ်ခုကို ပြုလုပ်ပါ။ စမ်းသပ်မှုကြာလာသည့်အချိန်တွင် ဒီ array ထဲတွင် ဆက်လက် လုပ်ဆောင်သွားမည်။

1. Linear SVC ဖြင့် စတင်ပါ-

    ```python
    C = 10
    # ကွဲပြားတဲ့ classifier များကို ဖန်တီးပါ။
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Linear SVC ကို အသုံးပြု၍ မော်ဒယ်ကို လေ့ကျင့်ပြီး အစီရင်ခံစာ ထုတ်ရန်-

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    ရလဒ်က အလွန်ကောင်းမွန်သည်-

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## K-Neighbors classifier

K-Neighbors သည် "neighbors" မျိုးဆက် ထဲရှိ ML နည်းလမ်းတစ်ခုဖြစ်ပြီး supervised နှင့် unsupervised learning နှစ်မျိုးလုံး အသုံးပြုနိုင်သည်။ ဤနည်းလမ်းတွင် ဖတ်ရှုသူများ စုစည်းထားသည့် အချက်အလက်များ၏ အရေအတွက်ကို သတ်မှတ်ပြီး အချက်အလက်များကို ဤအချက်အလက်များနားတွင် စုစည်းကာ အခြေခံ၍ label များကို ခန့်မှန်းသည်။

### လေ့ကျင့်ခန်း - K-Neighbors classifier ကို အသုံးပြုပါ

ယခင် classifier သည် ဒေတာနှင့် အတူ ကောင်းမွန်စွာ လုပ်ဆောင်သည်၊ ဒါပေမယ့် တိကျမှန်ကန်မှု ပိုမိုကောင်းစေဖို့ လုပ်နိုင်သည်။ K-Neighbors classifier ကို စမ်းကြည့်ပါ။

1. သင်၏ classifier array တွင် (Linear SVC မှာ comma ထည့်ပါ) အတန်းတစ်ကြောင်း ထည့်ပါ-

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    ရလဒ်က နည်းနည်း ပိုမဆိုးသည်-

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors) အကြောင်း သင်ယူရန်

## Support Vector Classifier

Support-Vector classifier များသည် [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) မျိုးဆက် ML နည်းလမ်းတစ်ခုဖြစ်ပြီး classification နှင့် regression လုပ်ငန်းများတွင် အသုံးပြုသည်။ SVM များသည် "သင်ကြားမှုဥပမာများကို အကွက်အတွင်း အချက်အလက်များအဖြစ် မှတ်တမ်းတင်ပြီး" အမျိုးအစား နှစ်ခုကြား အကွာအဝေးကို အများဆုံး ပြုလုပ်သည်။ နောက်ထပ်ရလာသော ဒေတာများသည် ဒီအကွက်တွင် စံချိန်ထားပြီး ထိုအမျိုးအစား ခန့်မှန်းမှု ခံယူသည်။

### လေ့ကျင့်ခန်း - Support Vector Classifier ကို အသုံးပြုပါ

Support Vector Classifier နှင့် တိကျမှန်ကန်မှု ပိုမိုကောင်းစေဖို့ စမ်းကြည့်လိုက်ပါ။

1. K-Neighbors item မှ comma ထည့်ပြီး အောက်ပါလိုင်းကို ထည့်ပါ-

    ```python
    'SVC': SVC(),
    ```

    ရလဒ်က အလွန်ကောင်းမွန်သည်!

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm) အကြောင်း သင်ယူရန်

## Ensemble Classifiers

ယခင် စမ်းသပ်မှုက အလွန်ကောင်းမှုရှိသော်လည်း၊ လမ်းကြောင်းအား အဆုံးတိုင်ထိ လိုက်နာကြည့်ပါမည်။ 'Ensemble Classifiers', အထူးသဖြင့် Random Forest နှင့် AdaBoost ကို စမ်းသပ်ပါ-

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

ရလဒ်က အထူးသဖြင့် Random Forest မှာ အလွန်ကောင်းသည်-

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html) အကြောင်း သင်ယူပါ

ဒီ Machine Learning နည်းလမ်းသည် "အခြေခံ လေ့လာမှုနည်းလမ်းတစ်ချို့၏ ခန့်မှန်းချက်များကို ပေါင်းစည်းခြင်း" ဖြင့် မော်ဒယ်အရည်အသွေးကို တိုးတက်စေသည်။ ဥပမာမှာ Random Trees နှင့် AdaBoost ကို အသုံးပြုခဲ့သည်။

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest) သည် averaging နည်းလမ်းဖြစ်ပြီး random ထည့်သွင်းထားသော 'decision trees' များဆောက်ပြီး overfitting မဖြစ်အောင် ကြိုးပမ်းသည်။ n_estimators parameter သည် သစ်ပင် အရေအတွက်ဖြစ်သည်။

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) သည် dataset အတွက် classifier တစ်ခု အသုံးပြုကာ ထို classifier ၏ များစွာကို ထပ်မံခေါ်ယူကာ မမှန်ကန်စွာ ခွဲခြားထားသည့် အချက်အလက်များ အလေးချိန်ပေး ပြင်ဆင်ရန် ကြိုးပမ်းသည်။

---

## 🚀 စိန်ခေါ်မှု

နည်းလမ်းတိုင်းတွင် parameter များ အများကြီးရှိပြီး သင် tweak လုပ်နိုင်သည်။ parameter များ၏ default တန်ဖိုးများကို လေ့လာပြီး parameter tuning မှ မော်ဒယ် အရည်အသွေး တိုးတက်မှုအတွက် ဘာကို ဆိုလိုလိုက်လိမ့်မည်ဟု စဥ်းစားကြည့်ပါ။

## [သင်ခန်းစာပြီးနောက် မေးခွန်းစစ်](https://ff-quizzes.netlify.app/en/ml/)

## ပြန်လည်ကြည့်ရှုမှု & ကိုယ်တိုင်လေ့လာမှု

ဤသင်ခန်းစာများတွင် စကားလုံးအသုံးများနေသောကြောင့် [ဒီစာရင်း](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) တွင် ပါသည့် အသုံးဝင်သော 용어များကို အကောင်းဆုံး ပြန်လည် သုံးသပ်ပါ။

## အပ်ဒိတ်

[Parameter play](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**တရားဝင်အကြောင်းကြားချက်**  
ဤစာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှုဖြစ်သော [Co-op Translator](https://github.com/Azure/co-op-translator) အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှုအတွက် ကြိုးပမ်း၍ဖြစ်သော်လည်း၊ အလိုအလျောက် ဘာသာပြန်ခြင်းက အမှားများ သို့မဟုတ် မှန်ကန်မှုလျော့နည်းမှုများ ပါဝင်နိုင်ကြောင်း သတိပြုပါရစေ။ မူလစာတမ်းကို မူလဘာသာဖြင့်သာ အတည်ပြုရမည့် အရင်းမြစ်အဖြစ် ယူဆရမည်ဖြစ်သည်။ အရေးကြီးသော အချက်အလက်များအတွက် အတွေ့အကြုံရှိ လူ့ဘာသာပြန်ကျွမ်းကျင်သူမှ ဘာသာပြန်ခြင်းကို အကြံပြုပါသည်။ ဤဘာသာပြန်ချက်အား အသုံးပြုမှုကြောင့် ဖြစ်ပေါ်နိုင်သည့် နားလည်မှု မမှန်ခြင်းများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မဲ့ပါကြောင်း သတိပေးအပ်ပါသည်။
<!-- CO-OP TRANSLATOR DISCLAIMER END -->