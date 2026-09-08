# Cuisine classifiers 2

Inside dis second classification lesson, you go explore more ways to classify numeric data. You go also learn about wetin fit happen if you choose one classifier pass the oda one.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Prerequisite

We dey assume say you don finish di previous lessons and you get cleaned dataset for your `data` folder wey dem call _cleaned_cuisines.csv_ for di root of dis 4-lesson folder.

### Preparation

We don load your _notebook.ipynb_ file wit di cleaned dataset and we don divide am into X and y dataframes, ready for di model building process.

## A classification map

Before now, you don learn about di different options wey you get wen you dey classify data using Microsoft cheat sheet. Scikit-learn get similar, but more detailed cheat sheet wey fit help you narrow down your estimators (another word for classifiers):

![ML Map from Scikit-learn](../../../../translated_images/pcm/map.e963a6a51349425a.webp)
> Tip: [visit dis map online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) and click along di path to read documentation.

### The plan

Dis map go help wella if you done sabi your data well, because you fit 'walk' along e paths go make decision:

- We get >50 samples
- We want predict category
- We get labeled data
- We get less than 100K samples
- ✨ We fit choose Linear SVC
- If dis no work, since we get numeric data
    - We fit try ✨ KNeighbors Classifier 
      - If dat one no work, try ✨ SVC and ✨ Ensemble Classifiers

Dis na better way wey you fit follow.

## Exercise - split the data

Follow dis path, we suppose start by import some libraries to use.

1. Import di libraries wey you need:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Split your training and test data:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC classifier

Support-Vector clustering (SVC) na pikin from di Support-Vector machines family of ML techniques (you fit learn more about dem below). For dis method, you fit choose 'kernel' to decide how you go cluster di labels. Di 'C' parameter mean 'regularization' wey dey control how parameters go influence di model. Di kernel fit be one of [plenty](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); for here we set am to 'linear' make we use linear SVC. Probability set to 'false' by default; but here we set am to 'true' to get probability estimates. We set random state to '0' to shuffle di data so that we fit get probabilities.

### Exercise - apply a linear SVC

Start by creating array of classifiers. You go dey add to dis array as we dey test.

1. Start with Linear SVC:

    ```python
    C = 10
    # Make different classifier dem.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Train your model using di Linear SVC and print report:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Di result good:

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

K-Neighbors na part of di "neighbors" family of ML methods, wey fit dey used for both supervised and unsupervised learning. For dis method, dem go set how many points before and gather data around dem points so that you fit predict labels wey fit generalize for di data.

### Exercise - apply the K-Neighbors classifier

Di previous classifier good and e work well for di data, but maybe we fit get better accuracy. Try K-Neighbors classifier.

1. Add one line to your classifier array (put comma after the Linear SVC item):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Di result just small worse:

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

    ✅ Learn about [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vector classifiers na part of di [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) family of ML methods wey dem dey use for classification and regression tasks. SVMs "go map training examples go points for space" to make distance between two categories max. Data wey come after go also map inside dis space so that dem fit predict di category.

### Exercise - apply a Support Vector Classifier

Make we try small better accuracy with Support Vector Classifier.

1. Add comma after K-Neighbors item then add dis line:

    ```python
    'SVC': SVC(),
    ```

    Di result good well!

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

    ✅ Learn about [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble Classifiers

Make we follow di path reach last, even though di previous test good well. Make we try some 'Ensemble Classifiers, especially Random Forest and AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Di result correct wella, especially for Random Forest:

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

✅ Learn about [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Dis Machine Learning method "go join di predictions from many base estimators" to make di model better. For our example, we use Random Trees and AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), averaging method, dey build 'forest' of 'decision trees' wey get random nature to avoid overfitting. Di n_estimators parameter na di number of trees.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) dey fit classifier to dataset then e fit many copies of dat classifier to di same dataset. E dey focus for weights of items wey classifier no classify well and e fit adjust di fit for next classifier to correct am.

---

## 🚀Challenge

Each of these techniques get plenti parameters wey you fit change. Research how each one their default parameters be and reason how changing dem fit affect di model quality.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Plenty big big words dey dis lessons, so take small time review [dis list](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) of useful terms!

## Assignment 

[Parameter play](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis document don translate wit AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). Even though we dey try make am correct, abeg sabi say automated translation fit get some errors or mistake. Di original document for im own language na di correct source. If na important information, make you use professional human translation. We no go take responsibility for any wrong understanding or mistake wey fit happen because of this translation.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->