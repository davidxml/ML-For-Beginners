# Класификатори кухиња 1

У овој лекцији ћете користити скуп података који сте сачували из претходне лекције, пун уравнотежених, чистих података о кухињама.

Користићете овај скуп података са разним класификаторима да бисте _предвидели дато национално јело на основу групе састојака_. При томе ћете научити више о неким од начина на које се алгоритми могу искористити за задатке класификације.

## [Претпредавачки квиз](https://ff-quizzes.netlify.app/en/ml/)
# Припрема

Под претпоставком да сте завршили [Лекцију 1](../1-Introduction/README.md), уверите се да фајл _cleaned_cuisines.csv_ постоји у главном `/data` фолдеру за ове четири лекције.

## Вежба - предвиди националну кухињу

1. Радите у фолдеру ове лекције _notebook.ipynb_, учитајте тај фајл заједно са библиотеком Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Подaци изгледају овако:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Сада, увезите још неколико библиотека:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Поделите X и y координате у два датафрејма за тренинг. `cuisine` може бити датафрејм етикета:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Изгледаће овако:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Избаците колоне `Unnamed: 0` и `cuisine` позивом `drop()`. Остатак података сачувајте као могуће карактеристике за обуку:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Ваше карактеристике изгледају овако:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Сада сте спремни да обучите модел!

## Избор класификатора

Сада када су ваши подаци чисти и спремни за обуку, морате одлучити који алгоритам да користите.

Scikit-learn групише класификацију под Надзирано учење, и у тој категорији ћете пронаћи много начина да се врши класификација. [Разноврсност](https://scikit-learn.org/stable/supervised_learning.html) је на први поглед прилично збуњујућа. Следеће методе укључују технике класификације:

- Линеарни модели
- Support Vector Machines (Супротне вредносне машине)
- Стохастички градијентни спуст
- Најближи суседи
- Гаусови процеси
- Одлучивачка стабла
- Методе ансамбла (гласачки класификатор)
- Мултикласа и мултиаутпут алгоритми (мултикласа и мултилеијбл класификација, мултикласа-мултиаутпут класификација)

> Можете такође користити [неуронске мреже за класификацију података](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), али то је изван опсега ове лекције.

### Који класификатор одабрати?

Дакле, који класификатор изабрати? Често, пробајући неколико и тражећи добар резултат је начин да се тестира. Scikit-learn нуди [упоредну табелу](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) на направљеном скупу података, поредећи KNeighbors, SVC на два начина, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB и QuadraticDiscrinationAnalysis, показујући резултате визуализоване: 

![поређење класификатора](../../../../translated_images/sr/comparison.edfab56193a85e7f.webp)
> Графикони генерисани у документацији Scikit-learn

> AutoML решава овај проблем елегантно покретањем ових поређења у облаку, омогућавајући вам да изаберете најбољи алгоритам за ваше податке. Испробајте [овде](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Бољи приступ

Бољи начин од насумичног погађања је да пратите идеје са овог склопивог [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Овде откривамо да за наш мултикласа проблем имамо неке изборе:

![читава листа за мултикласа проблеме](../../../../translated_images/sr/cheatsheet.07a475ea444d2223.webp)
> Одељак Microsoft-овог листа са преваром алгоритама, који детаљно описује опције мултикласа класификације

✅ Преузмите овај лист, одштампајте га и обесите на зид!

### Рaзлози

Погледајмо да ли можемо разложити различите приступе узимајући у обзир ограничења која имамо:

- **Неуронске мреже су превише захтевне**. Имајући у виду наш чист, али минималан скуп података и чињеницу да изводимо обуку локално преко нотебоока, неуронске мреже су пречесто за овај задатак.
- **Нема класификатора са два класа**. Не користимо двокласни класификатор, што искључује one-vs-all приступ.
- **Одлучујуће стабло или логистичка регресија могу функционисати**. Одлучивачко стабло или логистичка регресија за мултикласа податке могу бити добри.
- **Мултикласа Појачана Одлучивачка стабла решавају други проблем**. Мултикласа појачана одлучивачка стабла су најприкладнија за непараметричке задатке, нпр. задатке дизајниране за прављење ранкинга, те нам нису корисна.

### Коришћење Scikit-learn-а

Користићемо Scikit-learn за анализу наших података. Међутим, постоји више начина да се у Scikit-learn-у изведе логистичка регресија. Погледајте [параметре које треба проследити](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

У суштини, постоје два важна параметра - `multi_class` и `solver` - које треба одредити када тражите од Scikit-learn-а да изведе логистичку регресију. Вредност `multi_class` подешава одређено понашање алгоритма. Вредност `solver` одређује који алгоритам ће се користити. Ниједан solver се не може користити са свим врстама `multi_class` вредности.

Према документацији, у случају мултикласе:

- **Користи се шема један наспрам свих (OvR)**, ако је опција `multi_class` подешена на `ovr`
- **Користи се крос-ентропијски губитак**, ако је опција `multi_class` подешена на `multinomial`. (Тренутно су опције `multinomial` подржане само од стране solver-а ‘lbfgs’, ‘sag’, ‘saga’ и ‘newton-cg’)."

> 🎓 'Схема' овде може бити 'ovr' (један-напротив-свих) или 'multinomial'. Пошто је логистичка регресија првобитно дизајнирана за бинарну класификацију, ове шеме јој омогућавају боље руковање мултикласа задацима. [извор](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' је дефинисан као "алгоритам који се користи у оптимизационом проблему". [извор](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn нуди ову табелу да објасни како solver-и решавају различите изазове које представљају различите структуре података:

![solver-и](../../../../translated_images/sr/solvers.5fc648618529e627.webp)

## Вежба - подели податке

Можемо се фокусирати на логистичку регресију за наш први тренинг покушај јер сте недавно учили о њој у претходној лекцији.
Поделите податке на групе за обуку и тестирање позивом `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Вежба - примени логистичку регресију

Пошто користите мултикласа случај, морате одабрати коју _шему_ да користите и који _solver_ да подесите. Користите LogisticRegression са подешеним multi_class и solver-ом **liblinear** за обуку.

1. Направите логистичку регресију са `multi_class` подешеним на `ovr` и solver-ом подешеним на `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Испробајте други solver као `lbfgs`, који је често подразумеван

    > Напомена: користите Pandas-ову функцију [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) да спљоштите податке када је потребно.

    Тачност је добра, преко **80%**!

1. Можете видети овај модел у акцији тестирајући један ред података (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Резултат је исписан:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Испробајте другачији број реда и проверите резултате
1. Дужећи дубље, можете проверити тачност ове прогнозе:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Резултат је одштампан - Индијска кухиња је највероватнији избор, са добром вероватноћом:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Можете ли објаснити зашто је модел прилично сигуран да је ово индијска кухиња?

1. Добијте више детаља тако што ћете одштампати извештај о класификацији, као што сте радили на часовима регресије:

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

## 🚀Изазов

У овом часу користили сте своје очишћене податке за изградњу модела машинског учења који може да предвиди националну кухињу на основу скупа састојака. Узмите мало времена да прочитате бројне опције које Scikit-learn пружа за класификацију података. Дубље истражите појам 'solver' да бисте разумели шта се дешава иза сцене.

## [Квиз након предавања](https://ff-quizzes.netlify.app/en/ml/)

## Преглед и Самостално учење

Истражите мало више математику иза логистичке регресије у [овом часу](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Задатак

[Проучите solver-е](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Изјава о одрицању одговорности**:
Овај документ је преведен коришћењем AI услуге за превођење [Co-op Translator](https://github.com/Azure/co-op-translator). Иако се трудимо да превод буде прецизан, имајте у виду да аутоматизовани преводи могу садржати грешке или нетачности. Оригинални документ на његовом изворном језику треба сматрати ауторитетом. За критичне информације препоручује се професионални превод од стране људског преводиоца. Нисмо одговорни за било каква неразумевања или погрешне интерпретације које могу настати коришћењем овог превода.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->