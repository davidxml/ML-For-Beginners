# Cuisine classifiers 1

ਇਸ ਪਾਠ ਵਿੱਚ, ਤੁਸੀਂ ਉਹ ਡੇਟਾਸੇਟ ਵਰਤੋਂਗੇ ਜੋ ਤੁਸੀਂ ਪਿਛਲੇ ਪਾਠ ਤੋਂ ਸੰਭਾਲਿਆ ਸੀ, ਜਿਸ ਵਿੱਚ ਕਿਊਜ਼ੀਨਾਂ ਬਾਰੇ ਸੰਤੁਲਿਤ, ਸਾਫ਼ ਡੇਟਾ ਹੈ।

ਤੁਸੀਂ ਇਸ ਡੇਟਾਸੇਟ ਨੂੰ ਵੱਖ-ਵੱਖ ਕਲਾਸੀਫਾਇਰਾਂ ਨਾਲ ਵਰਤੋਂਗੇ ਤਾਂ ਜੋ _ਦਿੱਤੇ ਗਏ ਸਮੱਗਰੀਆਂ ਦੇ ਸਮੂਹ ਦੇ ਆਧਾਰ 'ਤੇ ਰਾਸ਼ਟਰੀ ਰਸੋਈ ਦੀ ਭਵਿੱਖਬਾਣੀ ਕੀਤੀ ਜਾ ਸਕੇ_। ਇਸ ਦੌਰਾਨ, ਤੁਸੀਂ ਜਾਣੋਗੇ ਕਿ ਕਿਵੇਂ ਅਲਗੋਰਿਦਮ ਕਲਾਸੀਫਿਕੇਸ਼ਨ ਕੰਮਾਂ ਲਈ ਵਰਤੇ ਜਾ ਸਕਦੇ ਹਨ।

## [ਪ੍ਰੀ-ਲੈਕਚਰ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/)
# ਤਿਆਰੀ

ਮੰਨ ਲਓ ਕਿ ਤੁਸੀਂ [Lesson 1](../1-Introduction/README.md) ਪੂਰਾ ਕਰ ਲਿਆ ਹੈ, ਤਾਂ ਇਹ ਯਕੀਨੀ ਬਣਾਓ ਕਿ ਇੱਕ _cleaned_cuisines.csv_ ਫਾਈਲ ਰੂਟ `/data` ਫੋਲਡਰ ਵਿੱਚ ਮੌਜੂਦ ਹੈ ਇਹਨਾਂ ਚਾਰ ਪਾਠਾਂ ਲਈ।

## ਅਭਿਆਸ - ਇੱਕ ਰਾਸ਼ਟਰੀ ਰਸੋਈ ਦੀ ਭਵਿੱਖਬਾਣੀ ਕਰੋ

1. ਇਸ ਪਾਠ ਦੇ _notebook.ipynb_ ਫੋਲਡਰ ਵਿੱਚ ਕੰਮ ਕਰਦੇ ਹੋਏ, ਉਸ ਫਾਈਲ ਨੂੰ ਅਤੇ Pandas ਲਾਇਬ੍ਰੇਰੀ ਨੂੰ ਇੰਪੋਰਟ ਕਰੋ:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    ਡੇਟਾ ਇਸ ਤਰ੍ਹਾਂ ਦਿੱਸਦਾ ਹੈ:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. ਹੁਣ, ਕੁਝ ਹੋਰ ਲਾਇਬ੍ਰੇਰੀਆਂ ਇੰਪੋਰਟ ਕਰੋ:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. X ਅਤੇ y ਕੋਆਰਡੀਨੇਟਾਂ ਨੂੰ ਦੋ ਡੇਟਾ ਫ੍ਰੇਮਾਂ ਵਿੱਚ ਵੰਡੋ। `cuisine` ਲੇਬਲ ਡੇਟਾ ਫ੍ਰੇਮ ਹੋ ਸਕਦਾ ਹੈ:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    ਇਹ ਇਸ ਤਰ੍ਹਾਂ ਲੱਗੇਗਾ:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. `Unnamed: 0` ਕਾਲਮ ਅਤੇ `cuisine` ਕਾਲਮ ਨੂੰ `drop()` ਕਾਲ ਕਰਕੇ ਹਟਾਓ। ਬਾਕੀ ਡੇਟਾ ਨੂੰ ਟ੍ਰੇnable ਫੀਚਰਾਂ ਵਜੋਂ ਸੁਰੱਖਿਅਤ ਕਰੋ:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    ਤੁਹਾਡੇ ਫੀਚਰ ਏਵੇਂ ਦਿੱਸਦੇ ਹਨ:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

ਹੁਣ ਤੁਸੀਂ ਆਪਣਾ ਮਾਡਲ ਟ੍ਰੇਨ ਕਰਨ ਲਈ ਤਿਆਰ ਹੋ!

## ਆਪਣੇ ਕਲਾਸੀਫਾਇਰ ਦੀ ਚੋਣ

ਹੁਣ ਜਦੋਂ ਕਿ ਤੁਹਾਡੇ ਡੇਟਾ ਨੂੰ ਸਾਫ਼ ਅਤੇ ਟ੍ਰੇਨਿੰਗ ਲਈ ਤਿਆਰ ਕੀਤਾ ਗਿਆ ਹੈ, ਤੁਹਾਨੂੰ ਇਹ ਫੈਸਲਾ ਕਰਨਾ ਹੈ ਕਿ ਕੰਮ ਲਈ ਕਿਹੜਾ ਅਲਗੋਰਿਦਮ ਵਰਤਣਾ ਹੈ।

Scikit-learn ਵਿੱਚ ਕਲਾਸੀਫਿਕੇਸ਼ਨ ਨੂੰ Supervised Learning ਦੇ ਅੰਦਰ ਰੱਖਿਆ ਗਿਆ ਹੈ, ਅਤੇ ਇਸ ਸ਼੍ਰੇਣੀ ਵਿੱਚ ਤੁਸੀਂ ਕਈ ਤਰੀਕੇ ਲੱਭੋਗੇ। [ਇਸ ਦੀ ਵੱਧਤਰੀ](https://scikit-learn.org/stable/supervised_learning.html) ਪਹਿਲੀ ਨਜ਼ਰ ਵਿੱਚ ਕਾਫ਼ੀ ਉਲਝਣ ਵਾਲੀ ਹੈ। ਹੇਠਾਂ ਦਿੱਤੇ ਤਰੀਕੇ ਸਾਰਿਆਂ ਕਲਾਸੀਫਿਕੇਸ਼ਨ ਤਕਨੀਕਾਂ ਵਿੱਚ ਸ਼ਾਮਲ ਹਨ:

- ਲੀਨੀਅਰ ਮਾਡਲ
- ਸਪੋਰਟ ਵੇਕਟਰ ਮਸ਼ੀਨਜ਼
- ਸਟੋਕਾਸਟਿਕ ਗਰੇਡੀਐਂਟ ਡੈਸੈਂਟ
- ਨੇਰੇਸਟ ਨੇਬਰਸ
- ਗਾਊਸਿਅਨ ਪ੍ਰਕਿਰਿਆਵਾਂ
- ਫੈਸਲਾ ਟ੍ਰੀਜ਼
- ਐਂਸੇਮਬਲ ਤਰੀਕੇ (ਵੋਟਿੰਗ ਕਲਾਸੀਫਾਇਰ)
- ਮਲਟੀਕਲਾਸ ਅਤੇ ਮਲਟੀਆਉਟਪੁੱਟ ਅਲਗੋਰਿਦਮ (ਮਲਟੀਕਲਾਸ ਅਤੇ ਮਲਟੀਲੈਬਲ ਕਲਾਸੀਫਿਕੇਸ਼ਨ, ਮਲਟੀਕਲਾਸ-ਮਲਟੀਆਉਟਪੁੱਟ ਕਲਾਸੀਫਿਕੇਸ਼ਨ)

> ਤੁਸੀਂ [ਨਿਊਰਲ ਨੈੱਟਵਰਕਾਂ ਦੇ ਨਾਲ ਡੇਟਾ ਕਲਾਸੀਫਾਈ ਕਰ ਸਕਦੇ ਹੋ](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), ਪਰ ਉਹ ਇਸ ਪਾਠ ਦੇ ਦਾਇਰੇ ਤੋਂ ਬਾਹਰ ਹੈ।

### ਕਿਹੜਾ ਕਲਾਸੀਫਾਇਰ ਚੁਣਨਾ ਚਾਹੀਦਾ ਹੈ?

ਤਾਂ, ਤੁਸੀਂ ਕਿਹੜਾ ਕਲਾਸੀਫਾਇਰ ਚੁਣੋਗੇ? ਅਕਸਰ ਕਈ ਕਲਾਸੀਫਾਇਰ ਚਲਾ ਕੇ ਅਤੇ ਵਧੀਆ ਨਤੀਜੇ ਦੀ ਤਲਾਸ਼ ਕਰ ਕੇ ਟੈਸਟ ਕੀਤਾ ਜਾਂਦਾ ਹੈ। Scikit-learn ਇੱਕ [ਸਾਈਡ-ਬਾਈ-ਸਾਈਡ ਤੁਲਨਾ](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) ਦਿੰਦਾ ਹੈ ਇੱਕ ਬਣਾਏ ਗਏ ਡੇਟਾਸੇਟ 'ਤੇ, KNeighbors, SVC ਦੋ ਤਰੀਕਿਆਂ ਨਾਲ, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB ਅਤੇ QuadraticDiscrinationAnalysis ਨੂੰ ਤੁਲਨਾ ਕਰਕੇ ਨਤੀਜੇ ਵਿਜ਼ੂਅਲਾਈਜ਼ ਕੀਤੇ ਹਨ:

![comparison of classifiers](../../../../translated_images/pa/comparison.edfab56193a85e7f.webp)
> Scikit-learn ਦੀ ਡੌਕਯੂਮੈਂਟੇਸ਼ਨ ਵਿੱਚ ਬਣਾਏ ਗਏ ਪਲਾਟ

> AutoML ਇਹ ਸਮੱਸਿਆ ਕਲਾਊਡ ਵਿੱਚ ਚੱਲਾ ਕੇ ਸੁਚੱਜੇ ਤਰੀਕੇ ਨਾਲ ਹੱਲ ਕਰਦਾ ਹੈ, ਜਿਸ ਨਾਲ ਤੁਸੀਂ ਆਪਣੇ ਡੇਟਾ ਲਈ ਸਭ ਤੋਂ ਵਧੀਆ ਅਲਗੋਰਿਦਮ ਚੁਣ ਸਕਦੇ ਹੋ। [ਇੱਥੇ ਕੋਸ਼ਿਸ਼ ਕਰੋ](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### ਇੱਕ ਵਧੀਆ ਤਰੀਕਾ

ਅਣਪਛਾਤੇ ਅਨੁਮਾਨ ਲਗਾਉਣ ਨਾਲ ਵਧੀਆ ਤਰੀਕਾ ਹੈ ਇਸ ਡਾਉਨਲੋਡ ਯੋਗ [ML ਚੀਟਸ਼ੀਟ](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) ਦੇ ਵਿਚਾਰਾਂ ਦੀ ਪਾਲਣਾ ਕਰਨੀ। ਇੱਥੇ, ਅਸੀਂ ਪਤਾ ਲਾਉਂਦੇ ਹਾਂ ਕਿ ਸਾਡੇ ਮਲਟੀਕਲਾਸ ਸਮੱਸਿਆ ਲਈ ਸਾਡੇ ਕੋਲ ਕੁਝ ਚੋਣਾਂ ਹਨ:

![cheatsheet for multiclass problems](../../../../translated_images/pa/cheatsheet.07a475ea444d2223.webp)
> Microsoft ਦੇ Algorithm Cheat Sheet ਦਾ ਇੱਕ ਹਿੱਸਾ, ਮਲਟੀਕਲਾਸ ਕਲਾਸੀਫਿਕੇਸ਼ਨ ਵਿਕਲਪਾਂ ਦਾ ਵੇਰਵਾ

✅ ਇਸ ਚੀਟਸ਼ੀਟ ਨੂੰ ਡਾਊਨਲੋਡ ਕਰੋ, ਛਾਪੋ, ਅਤੇ ਆਪਣੀ ਦਿਵਾਰ ਤੇ ਲਗਾਓ!

### ਤਰਕ

ਆਓ ਵੇਖੀਏ ਕਿ ਅਸੀਂ ਵੱਖ-ਵੱਖ ਪਹਲੂਆਂ ਨੂੰ ਲੈਕੇ ਸਾਡੇ ਪਾਸ ਜਿੰਨੇ ਸੀਮਤੀਆਂ ਹਨ, ਆਪਣੇ ਤਰਕ ਨਾਲ ਕਿਵੇਂ ਜਾ ਸਕਦੇ ਹਾਂ:

- **ਨਿਊਰਲ ਨੈੱਟਵਰਕ ਬਹੁਤ ਭਾਰੀ ਹਨ।** ਸਾਡੇ ਸਾਫ਼, ਪਰ ਘੱਟ ਡੇਟਾਸੇਟ ਦੇ ਮਦਨਜ਼ਰ ਅਤੇ ਇਹ ਗੱਲ ਕਿ ਅਸੀਂ ਟ੍ਰੇਨਿੰਗ ਲੋਕਲ ਨੋਟਬੁੱਕਾਂ ਵਿੱਚ ਕਰ ਰਹੇ ਹਾਂ, ਨਿਊਰਲ ਨੈੱਟਵਰਕ ਇਸ ਕੰਮ ਲਈ ਬਹੁਤ ਭਾਰੀ ਹਨ।
- **ਕੋਈ ਦੋ-ਕਲਾਸ ਕਲਾਸੀਫਾਇਰ ਨਹੀਂ।** ਅਸੀਂ ਦੋ-ਕਲਾਸ ਕਲਾਸੀਫਾਇਰ ਵਰਤਦੇ ਨਹੀਂ, ਇਸ ਲਈ ਇੱਕ-ਵਰਸਸ-ਆਲ ਖਤਮ ਹੋ ਜਾਂਦਾ ਹੈ।
- **ਡਿਸਿਜ਼ਨ ਟ੍ਰੀ ਜਾਂ ਲੌਜਿਸਟਿਕ ਰਿਗ੍ਰੈਸ਼ਨ ਕੰਮ ਕਰ ਸਕਦੇ ਹਨ।** ਇੱਕ ਫੈਸਲਾ ਟ੍ਰੀ ਜਾਂ ਲੌਜਿਸਟਿਕ ਰਿਗ੍ਰੈਸ਼ਨ ਮਲਟੀਕਲਾਸ ਡੇਟਾ ਲਈ ਕੰਮ ਕਰ ਸਕਦਾ ਹੈ।
- **ਮਲਟੀਕਲਾਸ ਬੂਸਟਡ ਡਿਸਿਜ਼ਨ ਟ੍ਰੀਜ਼ ਇਕ ਹੋਰ ਸਮੱਸਿਆ ਹੱਲ ਕਰਦੇ ਹਨ।** ਮਲਟੀਕਲਾਸ ਬੂਸਟਡ ਡਿਸਿਜ਼ਨ ਟ੍ਰੀ ਨਾਂਪੈਰਾਮੈਟ੍ਰਿਕ ਕਾਰਜਾਂ ਲਈ ਸਭ ਤੋਂ ਉਚਿਤ ਹੈ, ਜਿਵੇਂ ਕਿ ਰੈਂਕਿੰਗ ਬਣਾਉਣ ਦੇ ਕਾਰਜ, ਇਸ ਲਈ ਸਾਡੇ ਲਈ ਇਹ ਲਾਭਦਾਇਕ ਨਹੀਂ ਹੈ।

### Scikit-learn ਵਰਤੋਂ

ਅਸੀਂ ਆਪਣਾ ਡੇਟਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰਨ ਲਈ Scikit-learn ਵਰਤਾਂਗੇ। ਹਾਲਾਂਕਿ, Scikit-learn ਵਿੱਚ ਲੌਜਿਸਟਿਕ ਰਿਗ੍ਰੈਸ਼ਨ ਨੂੰ ਵਰਤਣ ਦੇ ਕਾਫੀ ਤਰੀਕੇ ਹਨ। [ਪੈਰਾਮੀਟਰਾਂ ਦੇਖੋ](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) ਜੋ ਤੁਸੀਂ ਪਾਸ ਕਰ ਸਕਦੇ ਹੋ।  

ਅਸਲ ਵਿੱਚ, ਦੋ ਮੁੱਖ ਪੈਰਾਮੀਟਰ ਹਨ - `multi_class` ਅਤੇ `solver` - ਜਿਨ੍ਹਾਂ ਨੂੰ ਅਸੀਂ Scikit-learn ਵਿੱਚ ਲੌਜਿਸਟਿਕ ਰਿਗ੍ਰੈਸ਼ਨ ਕਰਵਾਉਣ ਦੇ ਸਮੇਂ ਸੈੱਟ ਕਰਨਾ ਪੈਂਦਾ ਹੈ। `multi_class` ਦਾ ਕੀਮਤ ਇੱਕ ਵਿਸ਼ੇਸ਼ ਵਰਤੋਂ ਲਾਉਂਦਾ ਹੈ। solver ਦਾ ਕੀਮਤ ਇਹ ਦੱਸਦਾ ਹੈ ਕਿ ਅਲਗੋਰਿਦਮ ਕਿਹੜਾ ਵਰਤਣਾ ਹੈ। ਸਾਰੇ solver ਸਾਰੇ `multi_class` ਵੈਲਿਊਜ਼ ਨਾਲ ਜੋੜੇ ਨਹੀਂ ਜਾ ਸਕਦੇ।

ਡਾਕਯੂਮੈਂਟਸ ਦੇ ਅਨੁਸਾਰ, ਮਲਟੀਕਲਾਸ ਵਿੱਚ, ਟ੍ਰੇਨਿੰਗ ਅਲਗੋਰਿਦਮ:

- **one-vs-rest (OvR) ਸਕੀਮ ਵਰਤਦਾ ਹੈ**, ਜੇ `multi_class` ਵਿਕਲਪ `ovr` ਤੇ ਸੈੱਟ ਹੈ
- **ਕ੍ਰਾਸ-ਐਂਟਰੋਪੀ ਲਾਸ ਵਰਤਦਾ ਹੈ**, ਜੇ `multi_class` ਵਿਕਲਪ `multinomial` ਤੇ ਸੈੱਟ ਹੈ। (ਹੁਣੇ ਲਈ `multinomial` ਵਿਕਲਪ ਸਿਰਫ਼ ‘lbfgs’, ‘sag’, ‘saga’ ਅਤੇ ‘newton-cg’ solvers ਦੁਆਰਾ ਸਮਰਥਿਤ ਹੈ.)"

> 🎓 ਇੱਥੇ 'ਸਕੀਮ' ਜਾ ਤਾਂ 'ovr' (one-vs-rest) ਜਾਂ 'multinomial' ਹੋ ਸਕਦਾ ਹੈ। ਕਿਉਂਕਿ ਲੌਜਿਸਟਿਕ ਰਿਗ੍ਰੈਸ਼ਨ ਅਸਲ ਵਿੱਚ ਬਾਈਨਰੀ ਕਲਾਸੀਫਿਕੇਸ਼ਨ ਲਈ ਬਣਾਇਆ ਗਿਆ ਹੈ, ਇਨ੍ਹਾਂ ਸਕੀਮਾਂ ਨਾਲ ਇਹ ਮਲਟੀਕਲਾਸ ਕਲਾਸੀਫਿਕੇਸ਼ਨ ਕੰਮਾਂ ਨੂੰ ਬਿਹਤਰ ਸੰਭਾਲਦਾ ਹੈ। [ਸਰੋਤ](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' ਨੂੰ "ਆਪਟਿਮਾਈਜ਼ੇਸ਼ਨ ਸਮੱਸਿਆ ਵਿੱਚ ਵਰਤਿਆ ਜਾਣ ਵਾਲਾ ਅਲਗੋਰਿਦਮ" ਵਜੋਂ ਪਰਿਭਾਸ਼ਿਤ ਕੀਤਾ ਗਿਆ ਹੈ। [ਸਰੋਤ](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn ਇਹ ਟੇਬਲ ਦਿੰਦਾ ਹੈ ਕਿ ਕਿਵੇਂ ਸਾਲਵਰ ਵੱਖਰੇ ਤਰ੍ਹਾਂ ਦੇ ਡੇਟਾ ਸਟ੍ਰਕਚਰਾਂ ਦੀਆਂ ਸਮੱਸਿਆਵਾਂ ਨੂੰ ਹੱਲ ਕਰਦੇ ਹਨ:

![solvers](../../../../translated_images/pa/solvers.5fc648618529e627.webp)

## ਅਭਿਆਸ - ਡੇਟਾ ਵੰਡੋ

ਅਸੀਂ ਆਪਣੀ ਪਹਿਲੀ ਟ੍ਰੇਨਿੰਗ ਕੋਸ਼ਿਸ਼ ਲਈ ਲੌਜਿਸਟਿਕ ਰਿਗ੍ਰੈਸ਼ਨ ‘ਤੇ ਧਿਆਨ ਕੇਂਦਰਿਤ ਕਰ ਸਕਦੇ ਹਾਂ ਕਿਉਂਕਿ ਤੁਸੀਂ ਇਹ ਪਹਿਲਾ ਪਾਠ ਵਿੱਚ ਹਾਲ-ਹਾਲੇ ਸਿੱਖਿਆ ਸੀ।  
`train_test_split()` ਕਾਲ ਕਰਕੇ ਆਪਣੇ ਡੇਟਾ ਨੂੰ ਟ੍ਰੇਨਿੰਗ ਅਤੇ ਟੈਸਟਿੰਗ ਗਰੁੱਪਾਂ ਵਿੱਚ ਵੰਡੋ:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## ਅਭਿਆਸ - ਲੌਜਿਸਟਿਕ ਰਿਗ੍ਰੈਸ਼ਨ ਲਾਗੂ ਕਰੋ

ਕਿਉਂਕਿ ਤੁਸੀਂ ਮਲਟੀਕਲਾਸ ਕੇਸ ਵਰਤ ਰਹੇ ਹੋ, ਤੁਹਾਨੂੰ ਇਹ ਚੁਣਨਾ ਹੈ ਕਿ ਕਿਸ _ਸਕੀਮ_ ਨੂੰ ਵਰਤਣਾ ਹੈ ਅਤੇ ਕਿਹੜਾ _solver_ ਸੈੱਟ ਕਰਨਾ ਹੈ। LogisticRegression ਨੂੰ ਮਲਟੀਕਲਾਸ ਸੈਟਿੰਗ ਅਤੇ **liblinear** solver ਨਾਲ ਟ੍ਰੇਨ ਕਰੋ।

1. multi_class `ovr` ਤੇ ਅਤੇ solver `liblinear` ਤੇ ਸੈੱਟ ਕਰਕੇ ਇੱਕ ਲੌਜਿਸਟਿਕ ਰਿਗ੍ਰੈਸ਼ਨ ਬਣਾਓ:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ ਇੱਕ ਹੋਰ solver `lbfgs` ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ, ਜੋ ਅਕਸਰ ਡਿਫਾਲਟ ਹੁੰਦਾ ਹੈ

    > ਨੋਟ ਕਰੋ, ਜਦੋਂ ਜ਼ਰੂਰਤ ਹੋਵੇ ਤਾਂ ਆਪਣਾ ਡੇਟਾ ਫਲੈੱਟ ਕਰਨ ਲਈ Pandas ਦੀ [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) ਫੰਕਸ਼ਨ ਵਰਤੋਂ।

    ਸਹੀਤਾ 80% ਤੋਂ ਉਪਰ ਚੰਗੀ ਹੈ!

1. ਤੁਸੀਂ ਇਹ ਮਾਡਲ ਕਾਰਗੁਜ਼ਾਰੀ ਨੂੰ ਇੱਕ ਡੇਟਾ ਦੀ ਲਾਈਨ (#50) ਦੀ ਜਾਂਚ ਕਰਕੇ ਵੇਖ ਸਕਦੇ ਹੋ:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    ਨਤੀਜਾ ਪ੍ਰਿੰਟ ਕੀਤਾ ਜਾਂਦਾ ਹੈ:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ ਇੱਕ ਵੱਖਰਾ ਕਤਾਰ ਨੰਬਰ ਚੁਣੋ ਅਤੇ ਨਤੀਜੇ ਚੈੱਕ ਕਰੋ
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

## 🚀ਚੁਣੌਤੀ

ਇਸ ਪਾਠ ਵਿੱਚ, ਤੁਸੀਂ ਆਪਣਾ ਸਾਫ ਕੀਤਾ ਡਾਟਾ ਇਸਤੇਮਾਲ ਕਰਕੇ ਇੱਕ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਮਾਡਲ ਬਣਾਇਆ ਜੋ ਕਿ ਸਮੱਗਰੀਆਂ ਦੀ ਇੱਕ ਸੀਰੀਜ਼ ਦੇ ਅਧਾਰ 'ਤੇ ਰਾਸ਼ਟਰੀ ਖਾਣਾ ਭਣਾ ਕਰਨ ਦੀ ਭਵਿੱਖਬਾਣੀ ਕਰ ਸਕਦਾ ਹੈ। ਡਾਟਾ ਨੂੰ ਵਰਗੀਕ੍ਰਿਤ ਕਰਨ ਲਈ ਸਕਾਈਕਿਟ-ਲਰਨ ਦੁਆਰਾ ਦਿੱਤੀਆਂ ਕਈ ਵਿਕਲਪਾਂ ਨੂੰ ਪੜ੍ਹਨ ਲਈ ਕੁਝ ਸਮਾਂ ਲਓ। 'solver' ਦੀ ਸੰਕਲਪਨਾ ਵਿੱਚ ਹੋਰ ਗਹਿਰਾਈ ਨਾਲ ਜਾਓ ਤਾਂ ਜੋ ਇਹ ਸਮਝਿਆ ਜਾ ਸਕੇ ਕਿ ਹੀਨਦੋੜੇ ਪਿੱਛੇ ਕੀ ਹੁੰਦਾ ਹੈ।

## [ਪੋਸਟ-ਲੇਕਚਰ ਕੁਇਜ਼](https://ff-quizzes.netlify.app/en/ml/)

## ਸਮੀਖਿਆ ਅਤੇ ਸਵੈ-ਅਧਿਐਨ

ਲੋਜਿਸਟਿਕ ਰਿਗ੍ਰੈਸ਼ਨ ਦੇ ਪਿਛੋਕੜ ਵਿਚ ਗਣਿਤ ਨੂੰ ਹੋਰ ਵਧੀਕ ਸਮਝਣ ਲਈ [ਇਸ ਪਾਠ](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf) ਵਿੱਚ ਖੋਜ ਕਰੋ।

## ਅਸਾਈਨਮੈਂਟ 

[ਸੋਲਵਰਾਂ ਦਾ ਅਧਿਐਨ ਕਰੋ](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ਅਸਵੀਕਰਣ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ ਆਈਏਈ ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਿਵੇਂ ਕਿ ਅਸੀਂ ਸਹੀਤਾ ਲਈ ਕੋਸ਼ਿਸ਼ ਕਰਦੇ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ ਕਿ ਆਟੋਮੇਟਿਡ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸਮਰਥਤਾਵਾਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਇਸ ਦੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਅਸਲੀ ਦਸਤਾਵੇਜ਼ ਤਕੜਾ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਣ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਅਨੁਵਾਦ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਹੋਣ ਵਾਲੀਆਂ ਕਿਸੇ ਵੀ ਗਲਤਫਹਮੀ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆਵਾਂ ਲਈ ਅਸੀਂ ਜ਼ਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।
<!-- CO-OP TRANSLATOR DISCLAIMER END -->