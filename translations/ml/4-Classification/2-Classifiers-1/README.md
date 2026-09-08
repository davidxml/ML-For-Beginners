# ക്യൂസിൻ ക്ലാസിഫയറുകൾ 1

ഈ പാഠത്തിൽ, നിങ്ങൾ ക്യൂസിനുകളെക്കുറിച്ച് സുതാര്യമായ, സഞ്ചിതമായ ഡാറ്റകൾ നിറഞ്ഞ് പൂർത്തിയായ Dataset ഉപയോഗിക്കും.

ഒരു ഗ്രൂപ്പ് ഘടകങ്ങളുടെ അടിസ്ഥാനത്തിൽ ഒരു ന شپുക രാഷ്‌ട്രീയം ക്യൂസിൻ പ്രവചിക്കാൻ നിങ്ങൾ ഈ Dataset വിവിധ ക്ലാസിഫയറുകളുമായി ഉപയോഗിക്കും. ഇത് നടത്തുമ്പോൾ, ക്ലാസിഫിക്കേഷൻ ടാസ്ക്കുകൾക്കായി ആलगൊരിതങ്ങള пайдаланിക്കപ്പെടാവുന്ന ചില മാർഗങ്ങളെക്കുറിച്ച് നിങ്ങൾ കൂടുതൽ പഠിക്കും.

## [പ്രീ-ലെക്ചർ ക്വിസ്](https://ff-quizzes.netlify.app/en/ml/)
# തയ്യാറെടുപ്പ്

[Lesson 1](../1-Introduction/README.md) പൂർത്തീകരിച്ചിട്ടുണ്ടെന്ന് കരുതി, ഈ നാല് പാഠങ്ങൾക്കായി _cleaned_cuisines.csv_ ഫയൽ റൂട്ട് `/data` ഫോൾഡറിൽ ഉണ്ടെന്ന് ഉറപ്പാക്കുക.

## വ്യായാമം - ഒരു റാഷ്‌ട്ര ക്യൂസിൻ പ്രവചിക്കുക

1. ഈ പാഠത്തിലെ _notebook.ipynb_ ഫോൾഡറിലേക്കു ജോലി ചെയ്ത്, Pandas ലൈബ്രറിയോടൊപ്പം ആ ഫയൽ ഇറക്കുമതി ചെയ്യുക:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    ഡാറ്റ ഇങ്ങനെ കാണപ്പെടുന്നു:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. ഇനി, കൂടുതൽ ചില ലൈബ്രറികൾ ഇറക്കുമതി ചെയ്യുക:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. X, y കോര്‍ഡിനേറ്റുകൾ രണ്ട് Dataframe ആയി പരിശീലനത്തിനായി വിഭജിക്കുക. `cuisine` ലേബലുകൾ ഉള്ള Dataframe ആയിരിക്കാം:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    ഇത് ഇങ്ങനെ പ്രത്യക്ഷപ്പെടും:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. ആ `Unnamed: 0` കോളവും `cuisine` കോളവും `drop()` വഴി തള്ളുക. ശേഷിക്കുന്ന ഡാറ്റ ട്രെയിനിങ്ങിനുള്ള ഫീച്ചറുകളായി സംരക്ഷിക്കുക:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    നിങ്ങളുടെ ഫീച്ചറുകൾ ഇങ്ങനെ കാണപ്പെടും:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

ഇപ്പോൾ മോഡൽ ട്രെയിനുചെയ്യാൻ നിങ്ങൾ സജ്ജമാണ്!

## നിങ്ങളുടെ ക്ലാസിഫയർ തിരഞ്ഞെടുക്കൽ

നിങ്ങളുടെ ഡാറ്റ ക്ലീൻ ആയും പരിശീലനത്തിനുമായി തയ്യാറായതിനുശേഷം, ഏത് ആൽഗോരിതം ഉപയോഗിക്കാമെന്ന് തീരുമാനിക്കേണ്ടതാണ്.

Scikit-learn ക്ലാസിഫിക്കേഷനെ സൂപ്പർവൈസ്ഡ് ലേണിംഗിനായി ഗണിക്കുന്നു, അതിന്റെ കീഴിൽ വിവിധ ക്ലാസിഫിക്കേഷൻ മാർഗങ്ങൾ കാണാൻ കഴിയും. [വിധമതങ്ങൾ](https://scikit-learn.org/stable/supervised_learning.html) ആദ്യദൃശ്യത്തിൽ അല്പം ആശയക്കുഴപ്പം വരുത്താം. താഴെപ്പറയുന്ന മാർഗ്ഗങ്ങൾ എല്ലാം ക്ലാസിഫിക്കേഷൻ സാങ്കേതികവിദ്യകൾ ഉൾക്കൊള്ളുന്നു:

- ലിനിയർ മോഡലുകൾ
- സപ്പോർട്ട് വെക്ടർ മെഷീൻസ്
- സ്റ്റോക്കാസ്റ്റിക് ഗ്രേഡിയന്റ് ഡേസന്റ്
- നെയറസ്റ്റ് നെഞ്ചിബേഴ്സ്
- ഗോസിയൻ പ്രോസസ്സ്
- ഡെസിഷൻ ട്രീസ്
- എൻസെംബിൾ മെത്തഡ്സ് (വോട്ടിംഗ് ക്ലാസിഫയർ)
- മൾട്ടിക്ലാസ്, മൾട്ടിഔട്ട്പുട്ട് ആൽഗോരിതങ്ങൾ (മൾട്ടിക്ലാസ്, മൾട്ടിലേബൽ ക്ലാസിഫിക്കേഷൻ, മൾട്ടിക്ലാസ്-മൾട്ടിഔട്ട്പുട്ട് ക്ലാസിഫിക്കേഷൻ)

> നിങ്ങൾക്ക് [ന്യൂറൽ നെറ്റ്വർക്കുകൾ ഉപയോഗിച്ച് ഡാറ്റ ക്ലാസിഫൈ ചെയ്യാൻ](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification) സാധിക്കും, പക്ഷേ അത് ഈ പാഠത്തിന്റെ പരിധിക്ക് പുറത്താണ്.

### ഏത് ക്ലാസിഫയർ തിരഞ്ഞെടുക്കണം?

ഏതു ക്ലാസിഫയർ തിരഞ്ഞെടുക്കണമെന്നു കണ്ടെത്താൻ പലതും പ്രവർത്തിപ്പിച്ച് നല്ല ഫലത്തിന് നോക്കുക സാധാരണ വഴി ആണ്. Scikit-learn ഒരു [ഒപ്പം-ഒപ്പം താരതമ്യം](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) Dataset ഉപയോഗിച്ച് KNeighbors, SVC രണ്ട് രീതികളും, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB, QuadraticDiscrinationAnalysis എന്നിവയുടെ ഫലങ്ങൾ ചിത്രീകരിച്ചിരിക്കുന്നു:

![comparison of classifiers](../../../../translated_images/ml/comparison.edfab56193a85e7f.webp)
> പ്ലോട്ടുകൾ Scikit-learn ഡോക്യുമെന്റേഷനിൽ നിന്ന്

> AutoML ഈ പ്രശ്നം ക്ലൗഡിൽ ഓടിച്ചുകൊണ്ട് പരിഹരിക്കുന്നു, നിങ്ങളുടെ ഡാറ്റയ്ക്ക് ഏറ്റവും അനുയോജ്യമായ ആൽഗോരിതം തിരഞ്ഞെടുക്കാൻ അനുവദിക്കുന്നു. ഇത് [ഇവിടെ](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott) പരീക്ഷിക്കുക

### മെച്ചപ്പെട്ട സമീപനം

മുന്നറിയിപ്പ് ചെയ്യാതെ കരുതിയുള്ള പകരം, ഈ ഡൗൺലോഡുചെയ്യാവുന്ന [ML ചീറ്റ് ഷീറ്റ്](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) ര ചുവടെ പറഞ്ഞിരിക്കുന്ന ആശയങ്ങൾ പാലിക്കുക. ഇവിടെ, നമ്മുടെ മൾട്ടിക്ലാസ് പ്രശ്നത്തിന് പലയിടത്തായി തിരഞ്ഞെടുക്കാനാകും:

![cheatsheet for multiclass problems](../../../../translated_images/ml/cheatsheet.07a475ea444d2223.webp)
> മൈക്രോസോഫ്റ്റിന്റെ Algorithm Cheat Sheet-ന്റെ ഒരു വകഭാഗം, മൾട്ടിക്ലാസ് ക്ലാസിഫിക്കേഷൻ ഓപ്ഷനുകൾ വിശദീകരിക്കുന്നു

✅ ഈ ചീറ്റ് ഷീറ്റ് ഡൗൺലോഡ് ചെയ്ത് പ്രിന്റ് ചെയ്ത് നിങ്ങളുടെ ഭിത്തിയിൽ അത് തൂക്കുക!

### വിവേകം

നമുക്ക് പല സമീപനങ്ങളും ഉപയോക്തൃ നിർബന്ധങ്ങൾക്ക് അനുസരിച്ചു ഗണിക്കാമോ നോക്കാം:

- **ന്യൂറൽ നെറ്റ്വർക്കുകൾ വളരെ ഭാരമാണ**് ന. നമ്മുടെ ക്ലീൻ എന്നാൽ കുറഞ്ഞ ഡാറ്റ സെറ്റ് ഉണ്ട്, ലോക്കലായി നോട്ട്‌ബുക്കുകൾ വഴി ട്രെയ്‌നിംഗ് നടത്തുന്നപക്ഷം ന്യൂറൽ നെറ്റ്‌വർകുകൾ ഭാരമാണ്.
- **രണ്ട് ലേബൽ ക്ലാസിഫയറുകൾ ഇല്ല**. അതുകൊണ്ട് ഓൺ-വേഴ്സ്-ആൾ ഒരുക്കം ഒഴിവാക്കാം.
- **ഡെസിഷൻ ട്രീ അല്ലെങ്കിൽ ലോഗിസ്റ്റിക് റിഗ്രഷൻ പ്രയോജനം കാണിക്കും**. മൾട്ടിക്ലാസ് ഡാറ്റയ്ക്ക് ഡെസിഷൻ ട്രീ അല്ലെങ്കിൽ ലോഗിസ്റ്റിക് റിഗ്രഷൻ ഉപയോഗിക്കാവുന്നവിധം.
- **മൾട്ടിക്ലാസ് ബൂസ്റ്റഡ് ഡെസിഷൻ ട്രീ വ്യത്യസ്ത പ്രശ്നത്തിന്**. മൾട്ടിക്ലാസ് ബൂസ്റ്റഡ് ഡെസിഷൻ ട്രീ പരാമിതിയില്ലാത്ത ടാസ്കുകൾക്കായി ആണ് പറ്റിയത്, പോലെയുള്ള റാങ്കിംഗുകൾ നിർമ്മിക്കുന്നതിനായി, അതിനാൽ അത് നമ്മുടെ പ്രയോഗത്തിന് ഉപയോഗപ്രദമല്ല.

### Scikit-learn ഉപയോഗിക്കൽ

ഡാറ്റ വിശകലനം ചെയ്യാൻ Scikit-learn ഉപയോഗിക്കും. എന്നാൽ Scikit-learn ലോഗിസ്റ്റിക് റിഗ്രഷൻ ഉപയോഗിക്കാൻ പല വഴികളും ഉണ്ട്. [പാരാമീറ്ററുകൾ](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) പരിശോധിക്കുക.

പ്രധാനമായും `multi_class` അതും `solver` എന്നവയാണ് നിർണായക പാരാമീറ്ററുകൾ. Scikit-learn-നെ ലോഗിസ്റ്റിക് റിഗ്രഷൻ നടത്താൻ കോൾ ചെയ്തത് ഈ രണ്ട് പാരാമീറ്ററുകൾ വ്യക്തമാക്കണം. `multi_class` മൂല്യം ഒരു പ്രത്യേക പെരുമാറ്റം ഏർപ്പെടുത്തുന്നു. solver എന്നത് ഉപയോഗിക്കുന്ന ആൽഗോരിതമാണ്. എല്ലാ solver-കളും എല്ലാ `multi_class` മൂല്യങ്ങളോടും പൊരുത്തപ്പെടുവാൻ കഴിയാറില്ല.

ഡോക്‌സ് പ്രകാരം, മൾട്ടിക്ലാസ് കാര്യത്തിൽ, പരിശീലന ആൽഗോരിതം:

- **'ovr' ആണെങ്കിൽ one-vs-rest (OvR) സ്കീം ഉപയോഗിക്കുന്നു**.
- **'multinomial' ആണെങ്കിൽ ക്രോസ്-എൻട്രോപ്പി ലോസ് ഉപയോഗിക്കുന്നു**. (`multinomial` ഇപ്പോൾ 'lbfgs', 'sag', 'saga', 'newton-cg' solver-കൾക്കു മാത്രമേ പിന്തുണയുള്ളു.)

> 🎓 ഇവിടെ 'സ്കീം' 'ovr' (ഒന്ന്-വേഴ്സ്-റെസ്റ്റ്) അല്ലെങ്കിൽ 'മൾട്ടിനോമിയൽ' ആകാം. ലോഗിസ്റ്റിക് റിഗ്രഷൻ സാധാരണയായി ബൈനറി ക്ലാസിഫിക്കേഷനിനു രൂപപ്പെടുത്തിയിട്ടുള്ളതിനാൽ, ഇതുവഴി മൾട്ടിക്ലാസ് ടാസ്കുകൾ മെച്ചപ്പെട്ടവയായി കൈകാര്യം ചെയ്യപ്പെടുന്നു. [മൂലം](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 solver എന്ന് പറഞ്ഞാൽ "ഓപ്റ്റിമൈസേഷൻ പ്രശ്നത്തിൽ ഉപയോഗിക്കുന്ന ആൽഗോരിതം" എന്നാണ്. [മൂലം](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn solver-കൾ ഓരോ തരത്തിലുള്ള ഡാറ്റ വിവിധ വെല്ലുവിളികൾ എങ്ങനെ കൈകാര്യം ചെയ്യുന്നുവെന്നതിനുള്ള പട്ടിക നൽകുന്നു:

![solvers](../../../../translated_images/ml/solvers.5fc648618529e627.webp)

## വ്യായാമം - ഡാറ്റ വിഭജിക്കുക

നിങ്ങൾക്ക് മുമ്പത്തെ പാഠത്തിൽ ലോഗിസ്റ്റിക് റിഗ്രഷൻ പഠിച്ചതിനാൽ ആദ്യ ട്രെയിനിങ്ങിനായി അതിൽ ശ്രദ്ധ കേന്ദ്രീകരിക്കുക.
`train_test_split()` വിളിച്ച് ഡാറ്റ ട്രെയിനുകളും ടെസ്റ്റിങ്ങും ആയി വിഭജിക്കുക:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## വ്യായാമം - ലോഗിസ്റ്റിക് റിഗ്രഷൻ പ്രയോഗിക്കുക

മൾട്ടിക്ലാസ് കേസ് ഉപയോഗിക്കുന്നതുകൊണ്ട്, ഏത് _സ്കീം_ ഉപയോഗിക്കേണ്ടതും ഏത് _സോൾവർ_ സജ്ജീകരിക്കേണ്ടതും തിരഞ്ഞെടുക്കേണ്ടതാണ്. LogisticRegression ഉപയോഗിച്ച് multi_class 'ovr' ആയും solver 'liblinear' ആയും സജ്ജമാക്കി تربീൻ ചെയ്യുക.

1. multi_class പാരാമീറ്റർ `ovr` ആയും, solver `liblinear` ആയും സെറ്റ് ചെയ്ത് ലോഗിസ്റ്റിക് റിഗ്രഷൻ ഉണ്ടാക്കുക:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ ഭിന്നമായ ഒരു സോൾവർ പരീക്ഷിക്കുക, ഉദാഹരണത്തിന് `lbfgs`, സാധാരണ ഗതിയിൽ ഇത് ഡീഫോൾട്രായി സജ്ജീകരിച്ചിരിക്കുന്നതാണ്

    > ശ്രദ്ധിക്കുക, pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) ഫംഗ്ഷൻ ആവശ്യമുള്ളപ്പോൾ ഡാറ്റ flatten ചെയ്യുന്നതിന് ഉപയോഗിക്കുക.

    കൃത്യത 80% ലധികം നല്ലതാണ്!

1. ഡാറ്റയിലെ ഒരു പദവി (#50) ടെസ്റ്റ് ചെയ്ത് ഈ മോഡൽ പ്രവർത്തനം പരിശോധിക്കാം:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    ഫലം അച്ചടിപ്പിച്ചു കാണിക്കുന്നു:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ വ്യത്യസ്ത ഒരു പദവി ഉപയോഗിച്ച് പരീക്ഷിച്ച് ഫലങ്ങൾ പരിശോധിക്കുക
1. കൂടുതൽ ആഴത്തിലുള്ള പരിശോധനയിൽ, ഈ പ്രവചനം സമ്മതിക്കുന്നതിന്റെ കൃത്യത പരിശോധിക്കാം:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    ഫലം മুদ্রിക്കപ്പെട്ടിരിക്കുന്നു - ഇന്ത്യയുടെ പാചക ശൈലി ഇതിന്റെ മികച്ച സഹായം, നല്ല സാധ്യതയോടെ:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ മോഡൽ എന്തുകൊണ്ട് ഈ പാചക ശൈലി ഇന്ത്യൻ ആണെന്ന് ഉറപ്പുള്ളതായി നിങ്ങൾക്ക് വിശദീകരിക്കാമോ?

1. റെഗ്രഷൻ പാഠങ്ങളിലെ പോലെ ക്ലാസിഫിക്കേഷൻ റിപ്പോർട്ട് പ്രിന്റ് ചെയ്ത് കൂടുതൽ വിശദാംശങ്ങൾ നേടുക:

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

## 🚀ചലഞ്ച്

ഈ പാഠത്തിൽ, ഒരുപാട് ഘടകങ്ങൾ അടിസ്ഥാനമാക്കി ഒരു ദേശീയ പാചക ശൈലി പ്രവചിക്കാൻ കഴിവുള്ള മെഷീൻ ലേർണിംഗ് മോഡൽ നിങ്ങൾ കഴിഞ്ഞു നിർമ്മിച്ചു. ഡാറ്റ ക്ലാസിഫൈ ചെയ്യാൻ Scikit-learn നൽകുന്ന വിവിധ ഓപ്ഷനുകൾ വായിക്കാൻ സമയം എടുത്ത് പഠിക്കുക. 'solver' എന്ന ആശയത്തെ കൂടുതൽ ആഴത്തിൽ അറിയാൻ ശ്രമിക്കുക, പിന്നിലെ പ്രക്രിയകൾ അറിയാൻ.

## [പാഠശേഷി ക്വിസ്](https://ff-quizzes.netlify.app/en/ml/)

## അവലോകനം & സ്വയം പഠനം

ലോജിസ്റ്റിക് റെഗ്രഷിൻ പിന്നിലുള്ള ഗണിതത്തിൽ ഈ [പാഠത്തിൽ](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf) കൂടുതൽ ആഴത്തിൽ വെട്ടിവെക്കുക.

## അസൈൻമെന്റ്

[സോൾവേഴ്‌സിനെ പഠിക്കുക](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ഡിസ്ക്ലെയിമർ**:  
ഈ ദਸਤാവേസ് AI വിവർത്തന സേവനം [Co-op Translator](https://github.com/Azure/co-op-translator) ഉപയോഗിച്ച് വിവർത്തനം ചെയ്തതാണ്. നാം കൃത്യതക്കായി പരിശ്രമിക്കുന്നുവെങ്കിലും, സ്വയംപ്രവർത്തിക്കുന്ന വിവർത്തനങ്ങളിൽ പിഴവുകളോ അപാകതകളോ ഉണ്ടായിരിക്കാമെന്ന് ദയവായി മനസ്സിലാക്കുക. യഥാർത്ഥ ഭാഷയിലുള്ള ദസ്താവേസ് ഉത്തരവാദിത്വപ്പെടുത്തിയ ഉറവിടമായി കണക്കാക്കേണ്ടതാണ്. നിർണായക വിവരങ്ങൾക്കായി പ്രൊഫഷണൽ മനുഷ്യ വിവർത്തനം നിർദ്ദേശിക്കുന്നു. ഈ വിവർത്തനം ഉപയോഗിക്കുന്നതിനാൽ ഉണ്ടാകുന്ന ഏതെങ്കിലും തെറ്റിദ്ധാരണകൾക്കും തെറ്റായ വ്യാഖ്യാനങ്ങൾക്കും നാം ഉത്തരവാദികളല്ല.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->