# Cuisine classifiers 1

For dis lesson, you go use di dataset wey you save from di last lesson wey full wit balanced, clean data wey dey all about cuisines.

You go use dis dataset wit plenty classifiers dem to _predict one kain national cuisine based on one group of ingredients_. While you dey do am, you go learn more about some of di ways wey algorithms fit take help for classification tasks.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)
# Preparation

Assuming say you don finish [Lesson 1](../1-Introduction/README.md), make sure say one _cleaned_cuisines.csv_ file dey for di root `/data` folder for dis four lessons.

## Exercise - predict one national cuisine

1. For dis lesson's _notebook.ipynb_ folder, import dat file plus di Pandas library:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Di data look like dis:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Now, import more libraries:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Divide X and y coordinates into two dataframes for training. `cuisine` fit be labels dataframe:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    E go look like dis:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Drop dat `Unnamed: 0` column and di `cuisine` column by using `drop()`. Save di rest data as trainable features:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Your features go look like dis:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Now you ready to train your model!

## Choosing your classifier

Now wey your data clean and don ready for training, you gats make decision which algorithm you go use for di work. 

Scikit-learn put classification under Supervised Learning, and for dat category you go find plenty ways to classify. [Di variety](https://scikit-learn.org/stable/supervised_learning.html) dey confuse small for first look. Di methods wey follow dey include classification techniques:

- Linear Models
- Support Vector Machines
- Stochastic Gradient Descent
- Nearest Neighbors
- Gaussian Processes
- Decision Trees
- Ensemble methods (voting Classifier)
- Multiclass and multioutput algorithms (multiclass and multilabel classification, multiclass-multioutput classification)

> You fit also use [neural networks to classify data](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), but dat one no dey inside dis lesson scope.

### Which classifier to choose?

So, which classifier you go choose? Most times, you fit run several classifiers and see wey one get better result. Scikit-learn get [side-by-side comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) for one created dataset, wey compare KNeighbors, SVC two ways, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB and QuadraticDiscrinationAnalysis, and dem show di results as visual:

![comparison of classifiers](../../../../translated_images/pcm/comparison.edfab56193a85e7f.webp)
> Charts wey Scikit-learn documentation generate

> AutoML fit solve dis kind wahala well well by to run these comparisons for cloud, so dat you fit choose di best algorithm for your data. Try am [here](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Better approach

Better pass just guess work na to follow di ideas wey dey inside dis downloadable [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). For here, we see sey for our multiclass problem, we get some choices:

![cheatsheet for multiclass problems](../../../../translated_images/pcm/cheatsheet.07a475ea444d2223.webp)
> One part of Microsoft Algorithm Cheat Sheet, wey show multiclass classification options

✅ Download dis cheat sheet, print am come hang am for wall!

### Reasoning

Make we try reason our way through different approaches based on wetin we get:

- **Neural networks too heavy**. Based on our clean but small dataset, plus the fact say we dey run training local for notebooks, neural networks too heavy for dis task.
- **No two-class classifier**. We no dey use two-class classifier, so we no go use one-vs-all. 
- **Decision tree or logistic regression fit work**. Decision tree fit work, or logistic regression for multiclass data. 
- **Multiclass Boosted Decision Trees solve other kinds problem**. Dis multiclass boosted decision tree best for nonparametric tasks, e.g. tasks wey dey build rankings, so e no concern us.

### Using Scikit-learn 

We go use Scikit-learn analyze our data. But, many ways dey to use logistic regression for Scikit-learn. Look di [parameters wey you fit pass](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Basically, two parameters matter - `multi_class` and `solver` - we need to specify when we ask Scikit-learn to do logistic regression. Di value of `multi_class` determine how e go behave. Di solver na wetin algorithm to use be dat. No all solvers fit with all `multi_class` values.

According to docs, for multiclass case, di training algorithm:

- **Use one-vs-rest (OvR) scheme**, if `multi_class` option set to `ovr`
- **Use cross-entropy loss**, if `multi_class` option set to `multinomial`. (Currently `multinomial` option dey supported only by ‘lbfgs’, ‘sag’, ‘saga’ and ‘newton-cg’ solvers.)"

> 🎓 Di 'scheme' fit be 'ovr' (one-vs-rest) or 'multinomial'. Because logistic regression na to support binary classification, these schemes help am better handle multiclass classification tasks. [source](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 Di 'solver' na "di algorithm to use inside di optimization problem". [source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn get table to explain how solvers dey handle different challenges from many kinds data structures:

![solvers](../../../../translated_images/pcm/solvers.5fc648618529e627.webp)

## Exercise - split di data

We fit focus on logistic regression for our first training because you just learn am for previous lesson.
Split your data into training and testing groups by calling `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Exercise - use logistic regression

Since you dey use multiclass case, you gats choose which _scheme_ to use and which _solver_ to set. Use LogisticRegression with multiclass setting and **liblinear** solver for train.

1. Create logistic regression wit multi_class set to `ovr` and solver set to `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Try different solver like `lbfgs`, wey dem dey use as default many times

    > Note, use Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) function to flatten your data when e needed.

    Di accuracy good pass **80%**!

1. You fit see dis model dey work by testing one row of data (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Di result go print:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Try different row number come check di results
1. Diggin deep, you fit check the accuracy of dis prediction:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Di result e print - Indian cuisine na di best guess, wit good probability:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ You fit explain why di model sure sey na Indian cuisine dis be?

1. Get more detail by printing classification report, like how you do for regression lessons:

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

For dis lesson, you use your clean data to build machine learning model wey fit predict national cuisine based on series of ingredients dem. Take some time read through all di options wey Scikit-learn provide to classify data. Dig deep inside di concept of 'solver' to understand wetin dey happen behind di scenes.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Dig small more inside di math behind logistic regression inside [this lesson](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Assignment 

[Study the solvers](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:
Dis document dem don translate am wit AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). Even tho we dey try make am correct, abeg sabi say automated translations fit get errors or mistakes. Di original document for im own language na di correct source. For important matter, e better make person wey sabi do professional translation handle am. We no go take responsibility for any misunderstanding or wrong meaning wey fit show from dis translation.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->