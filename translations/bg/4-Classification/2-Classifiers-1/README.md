# Класификатори на кухни 1

В този урок ще използвате набора от данни, който запазихте от последния урок, пълен с балансирани, чисти данни, свързани с кухни.

Ще използвате този набор от данни с различни класификатори, за да _предскажете дадена национална кухня въз основа на група съставки_. Докато го правите, ще научите повече за някои от начините, по които алгоритмите могат да се използват за задачи по класификация.

## [Предварителен тест преди лекцията](https://ff-quizzes.netlify.app/en/ml/)
# Подготовка

Предполагайки, че сте завършили [Урок 1](../1-Introduction/README.md), уверете се, че файлът _cleaned_cuisines.csv_ съществува в главната папка `/data` за тези четири урока.

## Упражнение - предсказване на национална кухня

1. Работейки в папката с _notebook.ipynb_ за този урок, импортирайте този файл заедно с библиотеката Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Данните изглеждат така:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Сега импортирайте още няколко библиотеки:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Разделете координатите X и y в два датафрейма за обучение. `cuisine` може да бъде датафреймът с етикети:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Ще изглежда така:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Изхвърлете колоните `Unnamed: 0` и `cuisine`, като използвате `drop()`. Запазете останалите данни като функции за обучение:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Вашите функции изглеждат така:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Сега сте готови да обучите вашия модел!

## Избор на класификатор

След като данните ви са чисти и готови за обучение, трябва да решите кой алгоритъм да използвате.

Scikit-learn групира класификацията под Контролирано обучение, а в тази категория ще намерите много начини за класифициране. [Разнообразието](https://scikit-learn.org/stable/supervised_learning.html) е доста объркващо на пръв поглед. Следните методи включват техники за класификация:

- Линейни модели
- Машини с опорни вектори
- Стохастичен градиентен спад
- Най-близки съседи
- Гаусови процеси
- Дървета за вземане на решения
- Методи ансамбли (гласуващ класификатор)
- Мултикласови и мултиизходни алгоритми (мултикласова и мултиетикетна класификация, мултикласова мултиизходна класификация)

> Можете също да използвате [невронни мрежи за класифициране на данни](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), но това е извън обхвата на този урок.

### Кой класификатор да изберете?

И така, кой класификатор да изберете? Често е полезно да изпробвате няколко и да видите за добър резултат. Scikit-learn предлага [паралелно сравнение](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) върху създаден набор от данни, сравнявайки KNeighbors, SVC по два начина, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB и QuadraticDiscrinationAnalysis, като резултатите са визуализирани:

![сравнение на класификатори](../../../../translated_images/bg/comparison.edfab56193a85e7f.webp)
> Графики, генерирани в документацията на Scikit-learn

> AutoML решава този проблем елегантно, като изпълнява тези сравнения в облака, позволявайки ви да изберете най-добрия алгоритъм за вашите данни. Опитайте [тук](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### По-добър подход

По-добър начин от безразборното гадаене е да следвате идеите в този изтегляем [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Тук откриваме, че за нашия мултикласов проблем имаме някои избори:

![чийтшит за мултикласови проблеми](../../../../translated_images/bg/cheatsheet.07a475ea444d2223.webp)
> Част от списъка с алгоритми на Microsoft, описваща опции за мултикласова класификация

✅ Изтеглете този чийтшит, разпечатайте го и го закачете на стената си!

### Обосновка

Нека да видим дали можем да разсъдим различни подходи с оглед на ограниченията, които имаме:

- **Невронните мрежи са твърде тежки**. Като имаме чист, но минимален набор от данни и факта, че обучението се изпълнява локално чрез тетрадки, невронните мрежи са твърде тежки за тази задача.
- **Без двукласови класификатори**. Не използваме двукласови класификатори, което изключва one-vs-all.
- **Дърво за решения или логистична регресия могат да работят**. Дървото за решения може да свърши работа, или логистичната регресия за мултикласови данни.
- **Мултикласовите засилени дървета за решения решават друг проблем**. Мултикласовото засилено дърво е най-подходящо за непараметрични задачи, например задачи за изграждане на ранжиране, така че не е полезно за нас.

### Използване на Scikit-learn

Ще използваме Scikit-learn за анализ на нашите данни. Въпреки това има много начини за използване на логистична регресия в Scikit-learn. Вижте [параметрите, които трябва да зададете](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

По същество има два важни параметъра - `multi_class` и `solver` - които трябва да зададем, когато искаме Scikit-learn да изпълни логистична регресия. Стойността на `multi_class` прилага определено поведение. Стойността на solver е кой алгоритъм да се използва. Не всички solver-и могат да се сдвояват с всички `multi_class` стойности.

Според документацията, в мултикласовия случай, алгоритъмът за обучение:

- **Използва схемата one-vs-rest (OvR)**, ако опцията `multi_class` е настроена на `ovr`
- **Използва крос-ентропийна загуба**, ако опцията `multi_class` е настроена на `multinomial`. (В момента опцията `multinomial` се поддържа само от solver-ите ‘lbfgs’, ‘sag’, ‘saga’ и ‘newton-cg’)."

> 🎓 "Схемата" тук може да бъде 'ovr' (one-vs-rest) или 'multinomial'. Тъй като логистичната регресия е създадена предимно за бинарна класификация, тези схеми позволяват по-добро справяне с мултикласови задачи. [източник](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 "solver" е дефиниран като "алгоритъмът, който се използва в оптимизационната задача". [източник](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn предлага тази таблица, за да обясни как solver-ите се справят с различните предизвикателства, представени от различни видове структури на данни:

![solver-и](../../../../translated_images/bg/solvers.5fc648618529e627.webp)

## Упражнение - разделяне на данните

Можем да се фокусираме върху логистичната регресия за първия опит за обучение, тъй като наскоро научихте за нея в предишен урок.
Разделете данните си на тренировъчни и тестови групи, като извикате `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Упражнение - приложете логистична регресия

Тъй като използвате мултикласовия случай, трябва да изберете коя _схема_ да използвате и кой _solver_ да зададете. Използвайте LogisticRegression с мултикласова настройка и **liblinear** solver за обучение.

1. Създайте логистична регресия с `multi_class` настроено на `ovr` и solver - `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Опитайте различен solver като `lbfgs`, който често е по подразбиране

    > Забележка: използвайте функцията [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) на Pandas, за да развиете данните си, когато е необходимо.

    Точността е добра - над **80%**!

1. Можете да видите този модел в действие, като тествате ред №50:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Резултатът се отпечатва:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Опитайте с друг номер на ред и проверете резултатите
1. Копайки по-надълбоко, можете да проверите точността на тази прогноза:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Резултатът е отпечатан - индийската кухня е най-добрата му предположение, с добра вероятност:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Можете ли да обясните защо моделът е доста сигурен, че това е индийска кухня?

1. Получете повече подробности, като отпечатате отчет за класификацията, както направихте в уроците по регресия:

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

## 🚀Предизвикателство

В този урок използвахте почистените си данни, за да изградите модел за машинно обучение, който може да предскаже национална кухня въз основа на поредица от съставки. Отделете време да разгледате многото опции, които Scikit-learn предоставя за класифициране на данни. Проучете по-задълбочено концепцията за 'solver', за да разберете какво се случва зад кулисите.

## [Квиз след лекцията](https://ff-quizzes.netlify.app/en/ml/)

## Преглед и Самостоятелно обучение

Разгледайте по-подробно математиката зад логистичната регресия в [този урок](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Задание

[Изучете solver-ите](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Отказ от отговорност**:  
Този документ е преведен с помощта на AI преводаческа услуга [Co-op Translator](https://github.com/Azure/co-op-translator). Въпреки че се стремим към точност, моля, имайте предвид, че автоматизираните преводи могат да съдържат грешки или неточности. Оригиналният документ на неговия роден език трябва да се счита за авторитетен източник. За критична информация се препоръчва професионален човешки превод. Ние не носим отговорност за всякакви недоразумения или неправилни тълкувания, произтичащи от използването на този превод.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->