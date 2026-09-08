# Cuisine classifiers 1

在本课中，您将使用上一课中保存的关于菜系的平衡、干净数据集。

您将使用该数据集和各种分类器来_根据一组食材预测给定的国家菜系_。在此过程中，您将进一步了解算法在分类任务中的一些应用方式。

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)
# 准备工作

假设您已经完成了[第1课](../1-Introduction/README.md)，请确保在这四节课的根目录 `/data` 文件夹下存在一个_cleaned_cuisines.csv_文件。

## 练习 - 预测国家菜系

1. 在本课的_notebook.ipynb_文件夹中，导入该文件以及 Pandas 库：

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    数据如下所示：

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. 现在，导入更多库：

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. 将X和y变量划分为两个数据框用于训练。`cuisine` 可以作为标签数据框：

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    它看起来像这样：

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. 删除 `Unnamed: 0` 和 `cuisine` 列，调用 `drop()`。将剩余数据另存为训练特征：

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    你的特征看起来像这样：

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

现在，您已经准备好训练模型了！

## 选择分类器

既然数据已经清理好并准备好训练，您需要决定使用哪种算法。

Scikit-learn 将分类归类为监督学习，在该类别下您会发现许多分类方法。[种类繁多](https://scikit-learn.org/stable/supervised_learning.html) 初看令人眼花缭乱。以下方法都包括分类技术：

- 线性模型
- 支持向量机
- 随机梯度下降
- 最近邻居
- 高斯过程
- 决策树
- 集成方法（投票分类器）
- 多类和多输出算法（多类和多标签分类，多类-多输出分类）

> 您也可以使用[神经网络进行分类](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification)，但这超出本课范围。

### 选择哪个分类器？

那么，应该选择哪个分类器？通常，运行多个分类器然后寻找好的结果是测试方法之一。Scikit-learn 提供了一个[并排比较](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)的示例数据集，对比了 KNeighbors、两种 SVC、GaussianProcessClassifier、DecisionTreeClassifier、RandomForestClassifier、MLPClassifier、AdaBoostClassifier、GaussianNB 和 QuadraticDiscrinationAnalysis，并以图形方式展示结果：

![comparison of classifiers](../../../../translated_images/zh-CN/comparison.edfab56193a85e7f.webp)
> 由 Scikit-learn 文档生成的图表

> AutoML 通过在云端运行这些比较问题，帮你轻松解决此问题，助你为数据选择最佳算法。试试[这里](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### 更好的方法

不过，比胡乱猜测更好的方法是参考这个可下载的[机器学习备忘单](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott)。在这里，我们发现针对多类问题，有一些选择：

![cheatsheet for multiclass problems](../../../../translated_images/zh-CN/cheatsheet.07a475ea444d2223.webp)
> 微软算法备忘单中的一部分，详述多类分类选项

✅ 下载此备忘单，打印并挂在墙上！

### 推理

让我们看是否可以根据已有的限制推理不同的方法：

- <strong>神经网络过于庞大</strong>。鉴于我们干净但数据量少的数据集，以及局域笔记本的本地训练，神经网络过于庞大，不适合此任务。
- <strong>不是两类分类器</strong>。我们不使用二分类器，所以排除了一对多方法。
- <strong>决策树或逻辑回归可能适用</strong>。决策树可能行得通，或者使用逻辑回归处理多类数据。
- <strong>多类提升决策树解决不同问题</strong>。多类提升决策树更适合非参数任务，例如设计排名的任务，因此对我们并不适用。

### 使用 Scikit-learn

我们将使用 Scikit-learn 来分析数据。但 Scikit-learn 中逻辑回归有多种用法。请看看它的[参数说明](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)。

实质上有两个重要参数需要指定：`multi_class` 和 `solver`，分别用于控制多分类方案和求解算法。并非所有 `solver` 都能与所有 `multi_class` 组合使用。

文档说明，在多类情况下，训练算法：

- **使用一对多（OvR）方案**，如果 `multi_class` 设为 `ovr`
- <strong>使用交叉熵损失</strong>，如果 `multi_class` 设为 `multinomial`。（当前 `multinomial` 仅被 `'lbfgs'`, `'sag'`, `'saga'`, `'newton-cg'` 求解器支持）"

> 🎓 此处的“scheme”（方案）可以是“ovr”（一对多）或“multinomial”（多项式）。由于逻辑回归本质上设计为二分类，这些方案帮助它更好地处理多类分类任务。[来源](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 “solver” 指的是“在优化问题中使用的算法”。[来源](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)

Scikit-learn 提供下表说明不同求解器如何处理不同数据结构带来的挑战：

![solvers](../../../../translated_images/zh-CN/solvers.5fc648618529e627.webp)

## 练习 - 拆分数据

由于您最近在前一课学习了逻辑回归，我们可专注于逻辑回归作为首次训练尝试。调用 `train_test_split()` 将数据拆分为训练组和测试组：

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## 练习 - 应用逻辑回归

由于使用的是多分类情况，需要选择使用哪个_方案_以及设置哪个_求解器_。使用带有多类设置和<strong>liblinear</strong>求解器的 LogisticRegression 进行训练。

1. 创建一个 `multi_class` 设为 `ovr`，`solver` 设为 `liblinear` 的逻辑回归：

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ 尝试使用 `lbfgs` 这类常被设为默认的求解器

    > 注意，当需要时，使用 Pandas 的[`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html)函数将数据扁平化。

    准确率超过 **80%**，效果不错！

1. 通过测试一行数据（第50行）来查看该模型的实际表现：

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    打印结果如下：

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ 尝试使用不同行号并检查结果
1. 深入挖掘，您可以检查此预测的准确性：

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    结果被打印出来——印度菜是模型的最佳猜测，且概率较高：

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ 你能解释为什么模型相当确信这是印度菜吗？

1. 通过打印分类报告获取更多细节，就像你在回归课程中所做的那样：

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

## 🚀挑战

在本课中，您使用清理过的数据构建了一个机器学习模型，该模型可以根据一系列配料预测国家菜系。花些时间阅读 Scikit-learn 提供的众多数据分类选项。深入了解“solver”的概念，以理解其背后的工作原理。

## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 复习与自学

深入研究[这节课](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)中逻辑回归背后的数学原理  
## 作业

[学习解算器](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免责声明**：  
本文档使用 AI 翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 进行翻译。尽管我们努力确保准确性，但请注意，自动翻译可能包含错误或不准确之处。原始文档的母语版本应被视为权威来源。对于重要信息，建议采用专业人工翻译。对于因使用本翻译而引起的任何误解或误释，我们不承担任何责任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->