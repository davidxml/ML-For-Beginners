# Класифікатори кухонь 1

У цьому уроці ви використаєте набір даних, який зберегли з попереднього уроку, повний збалансованих, чистих даних про кухні.

Ви використаєте цей набір даних із різними класифікаторами, щоб _передбачити національну кухню на основі набору інгредієнтів_. Під час цього ви дізнаєтеся більше про деякі способи використання алгоритмів для задач класифікації.

## [Передлекційний тест](https://ff-quizzes.netlify.app/en/ml/)
# Підготовка

Припускаючи, що ви виконали [Урок 1](../1-Introduction/README.md), переконайтеся, що файл _cleaned_cuisines.csv_ існує в кореневій папці `/data` протягом цих чотирьох уроків.

## Вправа – передбачення національної кухні

1. Працюючи у папці з цим уроком _notebook.ipynb_, імпортуйте цей файл разом із бібліотекою Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Дані виглядають так:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Тепер імпортуйте ще кілька бібліотек:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Розділіть координати X і y на два датафрейми для навчання. `cuisine` може бути датафреймом міток:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Це виглядатиме так:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Відкиньте стовпець `Unnamed: 0` і стовпець `cuisine`, викликавши `drop()`. Збережіть решту даних як ознаки для навчання:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Ваші ознаки виглядають так:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Тепер ви готові навчати свою модель!

## Вибір класифікатора

Тепер, коли ваші дані чисті та готові до навчання, вам потрібно вирішити, який алгоритм використовувати для задачі.

Scikit-learn об'єднує класифікацію під категорією Навчання з вчителем (Supervised Learning), і в цій категорії ви знайдете багато способів класифікації. [Різноманітність](https://scikit-learn.org/stable/supervised_learning.html) на перший погляд досить вражаюча. Наступні методи включають техніки класифікації:

- Лінійні моделі
- Підтримкові векторні машини
- Стохастичний градієнтний спуск
- Найближчі сусіди
- Гаусові процеси
- Дерева рішень
- Метод ансамблів (voting Classifier)
- Алгоритми для багатокласової та мультививідної класифікації (multiclass і multilabel classification, multiclass-multioutput classification)

> Ви також можете використовувати [нейронні мережі для класифікації даних](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), але це поза межами цього уроку.

### Який класифікатор обрати?

Отже, який класифікатор слід вибрати? Часто запуск декількох і пошук гарного результату — це спосіб тестування. Scikit-learn пропонує [порівняння один поруч з одним](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) на створеному наборі даних, порівнюючи KNeighbors, SVC двома способами, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB і QuadraticDiscrinationAnalysis, демонструючи результати у візуалізованому вигляді:

![порівняння класифікаторів](../../../../translated_images/uk/comparison.edfab56193a85e7f.webp)
> Графіки створені на документації Scikit-learn

> AutoML вирішує цю проблему акуратно, запускаючи ці порівняння у хмарі, що дозволяє обрати найкращий алгоритм для ваших даних. Спробуйте [тут](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Кращий підхід

Однак кращий спосіб ніж гадаючи навмання — це слідувати ідеям з цього завантажувального [ML Cheatsheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Тут ми бачимо, що для нашої задачі багатокласової класифікації ми маємо кілька варіантів:

![шпаргалка для багатокласових задач](../../../../translated_images/uk/cheatsheet.07a475ea444d2223.webp)
> Частина алгоритмічного шпаргалки Microsoft, що описує варіанти багатокласової класифікації

✅ Завантажте цю шпаргалку, роздрукуйте її та прикріпіть на стіну!

### Обґрунтування

Подивимося, чи можемо ми логічно підходити до вибору підходів з урахуванням наших обмежень:

- **Нейронні мережі занадто важкі**. Враховуючи наш чистий, але мінімальний набір даних і той факт, що навчання відбувається локально у блокнотах, нейронні мережі є занадто ресурсомісткими для цієї задачі.
- **Без двокласового класифікатора**. Ми не використовуємо двокласовий класифікатор, тому виключаємо схему один проти усіх (one-vs-all).
- **Може підійти дерево рішень або логістична регресія**. Можливо дерево рішень, або логістична регресія для багатокласових даних.
- **Багатокласові бустінгові дерева прийнятні для інших задач**. Багатокласове бустінгове дерево найкраще пасує для непараметричних задач, наприклад, для побудови ранжувань, тому для нас не підходить.

### Використання Scikit-learn

Ми будемо використовувати Scikit-learn для аналізу наших даних. Однак існує багато способів використовувати логістичну регресію у Scikit-learn. Ознайомтеся з [параметрами, що можна передавати](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

По суті, є два важливі параметри — `multi_class` та `solver` — які треба вказати, коли ви просите Scikit-learn провести логістичну регресію. Параметр `multi_class` задає певну поведінку. Параметр `solver` визначає, який алгоритм використовувати. Не всі solver-ки можуть поєднуватись з усіма значеннями `multi_class`.

Згідно з документацією, для багатокласового випадку алгоритм навчання:

- **Використовує схему Один проти решти (OvR)**, якщо параметр `multi_class` встановлено в `ovr`
- **Використовує крос-ентропійне відхилення**, якщо параметр `multi_class` встановлено в `multinomial`. (Зараз опція `multinomial` підтримується лише solver-ами 'lbfgs', 'sag', 'saga' і 'newton-cg')."

> 🎓 'Схема' тут може бути або 'ovr' (один проти решти), або 'multinomial'. Оскільки логістична регресія насамперед розроблена для бінарної класифікації, ці схеми дають їй можливість краще працювати з багатокласовими задачами. [джерело](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' визначається як "алгоритм для використання у задачі оптимізації". [джерело](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn пропонує цю таблицю, щоб пояснити, як solver-и справляються з різними викликами, які виникають у різних типах структур даних:

![solver-и](../../../../translated_images/uk/solvers.5fc648618529e627.webp)

## Вправа – розділення даних

Ми можемо зосередитись на логістичній регресії для нашої першої спроби навчання, оскільки ви нещодавно вивчили її у попередньому уроці.
Розділіть свої дані на тренувальну і тестову групи, викликавши `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Вправа – застосування логістичної регресії

Оскільки ви використовуєте багатокласовий випадок, потрібно вибрати, яку _схему_ застосувати і який _solver_ встановити. Використовуйте LogisticRegression з параметром multi_class і solver-ом **liblinear** для навчання.

1. Створіть логістичну регресію з multi_class `ovr` і solver-ом `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Спробуйте інший solver, наприклад, `lbfgs`, який часто встановлений за замовчуванням

    > Зауважте, використовуйте функцію Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) для згладжування даних, коли це потрібно.

    Точність досить добра — понад **80%**!

1. Ви можете побачити цю модель у дії, протестувавши один рядок даних (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Результат виводиться:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Спробуйте інший номер рядка і перевірте результати
1. Заглиблюючись, ви можете перевірити точність цього передбачення:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Результат надруковано - індійська кухня є найбільш імовірним варіантом:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Чи можете ви пояснити, чому модель досить впевнена, що це індійська кухня?

1. Отримайте більше деталей, надрукувавши звіт про класифікацію, як ви робили в уроках регресії:

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

## 🚀Виклик

У цьому уроці ви використовували свої очищені дані для побудови моделі машинного навчання, яка може передбачити національну кухню на основі набору інгредієнтів. Приділіть час, щоб ознайомитися з багатьма варіантами, які пропонує Scikit-learn для класифікації даних. Заглибтесь у концепцію 'solver', щоб зрозуміти, що відбувається за лаштунками.

## [Пост-лекційний тест](https://ff-quizzes.netlify.app/en/ml/)

## Огляд та Самостійне вивчення

Заглибтеся в математику за логістичною регресією у [цьому уроці](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Завдання

[Вивчіть solvers](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Відмова від відповідальності**:
Цей документ було перекладено за допомогою сервісу автоматичного перекладу [Co-op Translator](https://github.com/Azure/co-op-translator). Хоча ми прагнемо до точності, будь ласка, майте на увазі, що автоматичні переклади можуть містити помилки або неточності. Оригінальний документ його рідною мовою слід вважати авторитетним джерелом. Для критичної інформації рекомендується професійний переклад людиною. Ми не несемо відповідальності за будь-які непорозуміння або неправильні тлумачення, що виникли внаслідок використання цього перекладу.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->