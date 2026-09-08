# உணவுப்பான வகைப்பாட்டாளர்கள் 1

இந்த பாடத்தில், நீங்கள் சமீபத்தில் சேமித்த சமநிலை, சுத்தமான, அனைத்தும் உணவுப்பானங்கள் பற்றிய தரவுத்தொகுதியில் பயன்படுத்துவீர்கள்.

பல்வேறு வகைப்பாட்டாளர்களுடன் இந்த தரவுத்தொகுதியைப் பயன்படுத்தி, _ஒரு குறிப்பிட்ட தேசிய உணவுப்பானத்தை ஒரு குழு பொருட்களின் அடிப்படையில் முன்னறிவிப்பு செய்வீர்கள்_. இதைச் செய்யும்போது, வகைப்பாட்டுப் பணிகளுக்கு ஆல்காரிதங்களைக் எப்படி பயன்படுத்தலாம் என்பதைப் பற்றி மேலும் அறியலாம்.

## [முன்னங்கற்பாட امتحان](https://ff-quizzes.netlify.app/en/ml/)
# தயார் செய்தல்

நீங்கள் [பாடம் 1](../1-Introduction/README.md) முடித்துள்ளீர்கள் என நினைத்து, இந்த நான்கு பாடங்களுக்கு ரூட் `/data` கோப்பகத்தில் _cleaned_cuisines.csv_ கோப்பு இருப்பதை உறுதிப்படுத்திக்கொள்ளவும்.

## பயிற்சி - ஒரு தேசிய உணவுப்பானத்தை முன்னறிவி

1. இந்த பாடத்தின் _notebook.ipynb_ கோப்பகத்தில் பணியாற்றி, அந்த கோப்பையும் Pandas நூலகத்தையும் இறக்குமதி செய்க:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    தரவு இதுவே போன்றது:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. இப்போது, மேலும் சில நூலகங்களை இறக்குமதி செய்க:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. X மற்றும் y குறியீடுகளை இரண்டு தரவுத்தொகுதிகளாகப் பிரிக்கவும் பயிற்சிக்காக. `cuisine` என்ற பகுதி குறிச்சொற்கள் தரவுத்தொகுதியாக இருக்கலாம்:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    இது இதுபோல் இருக்கும்:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. அப்படியே `Unnamed: 0` மற்றும் `cuisine` என்ற பத்திகளை `drop()` மூலம் நீக்கவும். மீதமுள்ள தரவுகளை பயிற்சி பண்புகள் ஆகச் சேமிக்கவும்:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    உங்கள் பண்புகள் இதுபோல் தெரியும்:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

இப்பொழுது உங்கள் மாதிரியை பயிற்சி செய்ய தயாராக உள்ளீர்கள்!

## உங்கள் வகைப்பாட்டாளரைத் தேர்ந்தெடுத்தல்

தரவு சுத்தமானதும் பயிற்சிக்கத் தயாரானதும், நீங்கள் எந்த ஆல்காரிதத்தை பயன்படுத்த என்ன தீர்மானிக்க வேண்டும்.

Scikit-learn வகைப்பாட்டை கட்டுப்படுத்தப்பட்ட கற்றல் (Supervised Learning) என்ற பிரிவில் வைத்து, இதில் பல வகைக்கும் வழிகள் உண்டு. [இந்த வகைகள்](https://scikit-learn.org/stable/supervised_learning.html) முதலில் பார்ப்பதற்கு குழப்பமானவையாக இருக்கலாம். கீழ்க்காணும் முறைகள் அனைத்தும் வகைப்பாடு தொழில்நுட்பங்களை உள்ளடக்கியவை:

- நேரியல் மாடல்கள் (Linear Models)
- ஆதரவு வெக்டார் இயந்திரங்கள் (Support Vector Machines)
- புள்ளி குறைபாடு இறக்குமதி (Stochastic Gradient Descent)
- மிக அருகிய அயலவர்கள் (Nearest Neighbors)
- காசியம் செயலிகள் (Gaussian Processes)
- முடிவு மரங்கள் (Decision Trees)
- கொலு முறைகள் (voting Classifier)
- பலவகை மற்றும் பலவெளி ஆல்காரிதங்கள் (multiclass and multilabel classification, multiclass-multioutput classification)

> நீங்கள் தரவை வகைப்படுத்த [நியூரல் நெட்வொர்க்குகளை](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification) தரவுப் பகுப்பாய்வுக்கு பயன்படுத்தலாம், ஆனால் அது இந்த பாடத்தின் வரம்புக்கு வெளியே உள்ளது.

### எந்த வகைப்பாட்டாளரை தேர்ந்தெடுக்கலாம்?

எந்த வகைப்பாட்டாளரை நீங்கள் தேர்ந்தெடுக்க வேண்டும்? பலவற்றை ஓடவிட்டு சிறந்த முடிவைக் காண்பது ஒரு வழி. Scikit-learn ஒரு [அருகில்-அருகு ஒப்பீடு](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) கொடுக்கும், இதில் KNeighbors, SVC இருவிதமாக, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB மற்றும் QuadraticDiscrinationAnalysis ஆகியவை ஒப்பிடப்பட்டுள்ளன கீழ்வருமாறு காட்டப்படுகின்றன:

![வகைப்பாட்டாளர்கள் ஒப்பீடு](../../../../translated_images/ta/comparison.edfab56193a85e7f.webp)
> Scikit-learn ஆவணத்தில் உருவாக்கப்பட்ட வரைபடங்கள்

> AutoML இந்தப் பிரச்சனையை மெக்ரோமாகச் சரி செய்யும், இந்த ஒப்பீடுகளை மேघத்தில் ஓட்டி, உங்கள் தரவுக்கு சிறந்த ஆல்காரிதத்தைத் தேர்வு செய்ய உதவும். அதை [இங்கு](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott) முயற்சிக்கவும்.

### ஒரு சிறந்த நடைமுறை

வாயிலாக கணித்துப்போகவதைவிட, இந்த பதிவிறக்கக்கூடிய [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) உள்ள ஐடியாக்களை பின்பற்றுவது சிறந்தது. இங்கே, நமது பலவகை பிரச்சனையில், சில தேர்வுகள் உள்ளன:

![பலவகை பிரச்சனைகளுக்கு Cheatsheet](../../../../translated_images/ta/cheatsheet.07a475ea444d2223.webp)
> மைக்ரோசாஃப்ட்டின் ஆல்காரிதம் Cheat Sheet இல் பலவகை வகைப்பாட்டு விருப்பங்கள் விவரிக்கப்பட்டுள்ள பகுதி

✅ இந்த cheat sheet ஐ பதிவிறக்கம் செய்து, அச்சிட்டு உங்கள் சுவர்க்கு ஒட்டவும்!

### காரணக் கூறல்

நம் கட்டுப்பாடுகளின் அடிப்படையில் பல்வேறு அணுகுமுறைகளை ஆராய்வோம்:

- **நியூரல் நெட்வொர்க்குகள் மிக அதிக பொருள்கள் கொண்டவை**. நம்முடைய சுத்தமான, ஆனாலும் குறைந்த அளவு தரவுத்தொகுதியையும், நொட்புக் குறிப்பு பதிவேடுகள் வழியாக பயிற்சி செய்கின்ற நிலையில் அது கைப்பற்ற முடியாது.
- **இரு-வகை வகைப்பாட்டாளரைப் பயன்படுத்தவில்லை**. ஆகையால் one-vs-all முறையைப் பயன்படுத்த தேவையில்லை.
- **முடிவு மரம் அல்லது லாஜிஸ்டிக் ரிக்ரெஷன் வேலை செய்யலாம்**. முடிவு மரம் ஒன்று வேலை செய்யக்கூடும் அல்லது பலவகை தரவுக்கு லாஜிஸ்டிக் ரிக்ரெஷன்.
- **பலவகை கூட்டு முடிவு மரங்கள் வேறு பிரச்சனைக்கு உகந்தவை**. பல்லாண்டு தூண்டுதலுடன் கூடிய முடிவு மரங்கள் அதிகரிப்பு வேலைகளுக்கு மிகச் சிறந்தவை, எனவே நமக்கு பொருத்தமல்ல.

### Scikit-learn பயன்படுத்துதல்

நாம் Scikit-learn ஐ நமது தரவு பகுப்பாய்வுக்கு பயன்படுத்தப் போகிறோம். ஆனால், Scikit-learn இல் லாஜிஸ்டிக் ரிக்ரெஷனுக்கு பலவிதமான முறைகள் உள்ளன. [கொடுக்க வேண்டிய சீட்டைகள்](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) பாருங்கள்.  

அந்த வகையில் இரண்டு முக்கிய சீட்டைகள் இருக்கின்றன - `multi_class` மற்றும் `solver` - இவற்றை Scikit-learn லாஜிஸ்டிக் ரிக்ரெஷன் செய்யும்போது நாம் குறிப்பிடவேண்டும். `multi_class` ஒரு நடத்தையை வகைப்படுத்தும். `solver` என்பதை எந்த ஆல்காரிதம் பயன்படுத்த வேண்டும் என்பதாகும். அனைத்து solver களும் எல்லா `multi_class` உருமாற்றங்களுக்கும் பொருந்தவில்லை.

ஆவணங்கள் படி, பலவகை வழியிலான பயிற்சி ஆல்காரிதம்:

- `multi_class` `ovr` என்றால் ஒருவகம்-எதிர் (one-vs-rest) திட்டத்தைப் பயன்படுத்தும்
- `multi_class` `multinomial` என்றால், குறுக்கேற்ற சதிப்பு இழப்பு (cross-entropy loss) ஐப் பயன்படுத்தும். (`multinomial` விருப்பம் தற்போது ‘lbfgs’, ‘sag’, ‘saga’ மற்றும் ‘newton-cg’ சொல்வர்களினால் மட்டுமே ஆதரிக்கப்படுகிறது.)

> 🎓 இங்கே 'திட்டம்' என்றால் 'ovr' (ஒருவகம்-எதிர்) அல்லது 'multinomial' ஆக இருக்கலாம். லாஜிஸ்டிக் ரிக்ரெஷன் அடிப்படையில் இருநிலை வகுப்பு மெஷினாக உருவாக்கப்பட்டது, இவை பல்லாண்டு வகைப்பாட்டு பணிகளை சிறப்பாகச் செய்ய உதவும். [மூலம்](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 செயலி (solver) என்பது "பெருமளவு பிரச்சனையில் பயன்படுத்தும் ஆல்காரிதம்" என்று வரையறுக்கப்பட்டுள்ளது. [மூலம்](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)

Scikit-learn கீழ்காணும் அட்டவணையைக் கொடுத்து, சொல்வர்கள் எப்படி பல்வேறு தரவு அமைப்புகளின் சவால்களை கையாள்கின்றன என்பதை விளக்குகிறது:

![சொல்வர்கள்](../../../../translated_images/ta/solvers.5fc648618529e627.webp)

## பயிற்சி - தரவைப் பிரி

நீங்கள் சமீபத்தில் ஒரு பாடத்தில் லாஜிஸ்டிக் ரிக்ரெஷனைப் பயன்படுத்தி கற்றிருந்ததைக் கருத்தில் கொண்டால், முதலில் இதனை முயற்சிக்கலாம். `train_test_split()` அழைத்து தரவை பயிற்சிக்கும் மற்றும் சோதனைக் குழுக்களாக பிரிக்கவும்:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## பயிற்சி - லாஜிஸ்டிக் ரிக்ரெஷன் பயன்பாடு

நீங்கள் பலவகை வழிமுறையில் இருப்பதால், எந்த _திட்டத்தையும்_ மற்றும் எந்த _சொல்வரையும்_ பயன்படுத்த வேண்டும் என்பதைத் தேர்ந்தெடுக்க வேண்டும். பலவகை அமைப்புடன் **liblinear** சொல்வரை பயன்படுத்தி LogisticRegression ஐ பயிற்சி செய்யவும்.

1. multi_class ஐ `ovr` ஆகவும் solver ஐ `liblinear` ஆகவும் அமைத்து ஒரு லாஜிஸ்டிக் ரிக்ரெஷன் உருவாக்கவும்:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ பலமுறை பயன்படுத்தப்படும் `lbfgs` போன்ற வேறு சொல்வர்களையும் முயற்சிக்கவும்

    > கவனிக்க, தேவையான போது Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) செயலியைத் தரவை சீராக்க பயன்படுத்தவும்.

    செம்மையான துல்லியம் **80%** க்கும் மேல்!

1. ஒரு வரிசை (#50) மூலம் இந்த மாதிரியை சோதனை செய்து பார்க்கலாம்:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    முடிவு அச்சிடப்படும்:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ வேறு வரிசை எண்ணை முயற்சித்துப் முடிவுகளைச் சரிபார்க்கவும்
1. மேலும் ஆழமாக ஆராய்ந்து, இந்த முன்கூட்டிய கணிப்பின் துல்லியத்தை நீங்கள் சரிபார்க்கலாம்:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    முடிவை அச்சிடப்பட்டுள்ளது - இந்திய உணவு என்பது அதன் சிறந்த முன்னறிவு, நல்ல சாத்தியத்துடன்:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ இந்த மodel் இந்திய உணவு என்று pretty உறுதியாக இருக்கிறதென்று நீங்கள் ஏன் விளக்க முடியுமா?

1. நீங்கள் ரெகிரஷன் பாடங்களில் செய்தது போல வகைப்படுத்தல் அறிக்கையை அச்சிடுவதன் மூலம் கூடுதல் விவரங்களை பெறுங்கள்:

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

## 🚀சவால்

இந்த பாடத்தில், நீங்கள் உங்கள் சுத்திகரிக்கப்பட்ட தரவை பயன்படுத்தி ஒரு இயந்திரக் கற்றல் மாதிரியை கட்டியுள்ளீர்கள், இது ஒரு தொடர் பொருட்கள் அடிப்படையில் தேசிய உணவுப்பாடத்தை முன்னறிவிக்க முடியும். தரவை வகைப்படுத்தும் பல விருப்பங்களை Scikit-learn வழங்குகிறது என்பதைக் கவனித்து வாசிக்க சில நேரம் ஒதுக்குங்கள். பின்னணி உள்ள விடயங்களை புரிந்து கொள்ள 'solver' என்ற கருத்தில் மேலும் ஆழமாக பயின்று பாருங்கள்.

## [பாட விரிவாக்கக் குயிஸ்](https://ff-quizzes.netlify.app/en/ml/)

## மதிப்பாய்வு மற்றும் சுயபடைப்பு

[இந்தப் பாடத்தில்](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf) லாஜிஸ்டிக் ரெகிரஷனின் பின்னணியில் உள்ள கணிதத்தை மேலும் ஆய்வு செய்க
## பணிப்புரை

[சால்வர்களைப் படியுங்கள்](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**தகவல் அறிவிப்பு**:
இந்த ஆவணம் AI மொழிபெயர்ப்பு சேவை [Co-op Translator](https://github.com/Azure/co-op-translator) மூலம் மொழிபெயர்க்கப்பட்டது. நாங்கள் துல்லியத்திற்காக முயலினாலும், தானியங்கி மொழிபெயர்ப்புகளில் பிழைகள் அல்லது தவறுகள் இருக்கக் கூடும் என்பதை கவனத்தில் கொள்ளவும். அசல் ஆவணம் அதன் ஒரிஜினல் மொழியில் தான் அதிகாரப்பூர்வ ஆதாரமாக கருதப்பட வேண்டும். முக்கியமான தகவலுக்கு, தொழில்முறை மனித மொழிபெயர்ப்பு பரிந்துரைக்கப்படுகிறது. இந்த மொழிபெயர்ப்பின் பயன்பாட்டினால் ஏற்படும் எந்தவொரு தவறான புரிதல்கள் அல்லது தவறான விளக்கங்களுக்காக நாங்கள் பொறுப்பேற்கவில்லை.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->