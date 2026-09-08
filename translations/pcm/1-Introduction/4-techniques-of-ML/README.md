# Techniques of Machine Learning

Di process wey dey build, use, and maintain machine learning models and di data dem use na different kin process from plenty oda development workflows. For dis lesson, we go clear di process, and show di main techniques wey you need sabi. You go:

- Understand di processes wey machine learning dey follow for high level.
- Explore base concepts like 'models', 'predictions', and 'training data'.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Click di image up dere for short video wey go run through dis lesson.

## Introduction

For high level, di skill of creating machine learning (ML) processes get plenty steps:

1. **Decide on the question**. Most ML processes dey start by asking question wey conditional program or rules-based engine no fit answer easily. These questions dey usually about predictions wey base on data wey dem collect.
2. **Collect and prepare data**. To fit answer your question, you need data. Di quality and sometimes di quantity of your data go determine how well you fit answer your question. To visualize data na important part of dis phase. Dis phase also get splitting di data into training and testing group to build model.
3. **Choose a training method**. Based on your question and di kind data wey you get, you gats choose how you go train model to best represent your data and make correct predictions. Dis na di part of ML process wey need special skill and sometimes plenty trial and error.
4. **Train the model**. Use your training data, you go use algorithms train di model to recognize patterns inside di data. Di model fit get internal weights wey you fit adjust to focus on some parts of di data pass oda parts to build better model.
5. **Evaluate the model**. Use data wey di model never see before (your testing data) to check how di model dey perform.
6. **Parameter tuning**. Based on how di model perform, you fit repeat di process using different parameters or variables wey control di algorithms behaviour wey train di model.
7. **Predict**. Use new inputs to test how correct your model predictions be.

## Wetin question to ask

Computa sabi find hidden patterns inside data well well. Dis tin dey help researchers wey get questions about domain wey no easy to answer by conditional rules engine. For example, data scientist fit fit handcrafted rules for mortality between smokers and non-smokers for actuarial task.

But if many other variables enter di matter, ML model fit balance well to predict future mortality from past health history. Another example be forecasting weather for April for one place based on latitude, longitude, climate change, near ocean, jet stream patterns, and more.

✅ Dis [slide deck](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) about weather models get historical view of how dem dey use ML for weather analysis.

## Pre-building tasks

Before you start to build model, you gots to finish some tasks. To test your question and form hypothesis from model predictions, you need to identify and arrange some elements.

### Data

To fit answer your question for sure, you need plenty data wey correct kind. Two tins you gots to do for here:

- **Collect data**. Remember di lesson on fairness for data analysis, collect your data carefully. Know where di data come from, any biases inside, and write down where e originate.
- **Prepare data**. Data preparation get many steps. You fit combine data and normalize if e come from different sources. You fit improve data quality and quantity with methods like changing strings to numbers (like we do for [Clustering](../../5-Clustering/1-Visualize/README.md)). You fit make new data from original data (like we do for [Classification](../../4-Classification/1-Introduction/README.md)). You fit clean and edit data (like before [Web App](../../3-Web-App/README.md) lesson). Finally, you fit randomize and shuffle am, depend on your training techniques.

✅ After you don collect and process your data, check if di shape of am fit answer your question. Sometimes data no go perform well for your task, like we see for our [Clustering](../../5-Clustering/1-Visualize/README.md) lessons!

### Features and Target

[Feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) na measurable property for your data. For many dataset, e dey show as column header like 'date' 'size' or 'color'. Your feature variable, most time dem call am `X` for code, na di input variable wey you go use train model.

Target na wetin you dey try predict. Target, wey dem dey call `y` for code, na di answer to di question you dey ask your data: for December, which **color** pumpkins go cheap? for San Francisco, which neighborhoods get best real estate **price**? Sometime target fit also be label attribute.

### Selecting your feature variable

🎓 **Feature Selection and Feature Extraction** How you go sabi which variable to choose to build model? You fit do feature selection or feature extraction to choose correct variables wey model go perform well. Dem no mean di same: "Feature extraction dey create new features from original features functions, but feature selection na to pick subset of features." ([source](https://wikipedia.org/wiki/Feature_selection))

### Visualize your data

One power data scientist get na how to visualize data with good libraries like Seaborn or MatPlotLib. To show your data visually fit help you find hidden connections wey you fit use. Your visualizations fit also help find bias or unbalanced data (like we see for [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Split your dataset

Before training, you gots split your dataset into two or more parts wey no equal size but still represent di data well.

- **Training**. Dis part of dataset na for train your model. E be di majority of original dataset.
- **Testing**. Test dataset na independent data group, often from original data, wey you use check how model dey perform.
- **Validating**. Validation set na smaller independent sample wey you use tune model hyperparameters or architecture to improve model. Depending on your data size and question, you fit no need build this third group (like we talk for [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md)).

## Building a model

Use your training data, your aim na to build model, or statistical representation of your data, with algorithms to **train** am. Training model mean you expose am to data so e fit make assumptions about patterns e finds, confirms, accept or reject.

### Decide on a training method

Based on your question and your data type, you go select method to train am. If you check [Scikit-learn's documentation](https://scikit-learn.org/stable/user_guide.html) - we dey use am for this course - you fit explore many ways to train model. Depending on your experience, you fit try many methods before you build best model. Data scientists dey evaluate model by feeding am unseen data, look how accurate e be, check bias or quality wahala, then pick best training method.

### Train a model

With your training data, you ready to 'fit' am build model. For many ML libraries, you go find code 'model.fit' - this na when you send your feature variable as array (usually 'X') and target variable (usually 'y').

### Evaluate the model

After training complete (e fit take many rounds or 'epochs' to train big model), you fit check model quality using test data to see how am perform. Dis data na part of original data wey model never analyze before. You fit print table of metrics for your model quality.

🎓 **Model fitting**

For machine learning, model fitting mean how accurate model function dey wen e dey try analyze data wey e never see before.

🎓 **Underfitting** and **overfitting** na common problems wey spoil model quality, as model fit no fit well or e fit too well. This one fit cause model to make predictions wey too close or too far from training data. Overfit model dey predict training data well because e sabi di detail and noise well well. Underfit model no accurate because e no fit analyze training data or unseen data correctly.

![overfitting model](../../../../translated_images/pcm/overfitting.1c132d92bfd93cb6.webp)
> Infographic by [Jen Looper](https://twitter.com/jenlooper)

## Parameter tuning

After your first training finish, watch model quality and think about improving am by adjusting 'hyperparameters'. Read more about dis process [for documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Prediction

Na dis time you go use fresh data test your model accuracy. For 'applied' ML setting, where you dey build web tools to use model for production, dis tin fit involve gathering user input (like pressing button) to set variable and send am go model for inference or evaluation.

For dis lessons, you go learn how to use these steps to prepare, build, test, evaluate, and predict - all di skills wey data scientists get and more, as you dey waka your journey to become 'full stack' ML engineer.

---

## 🚀Challenge

Draw flow chart wey show steps wey ML practitioner dey follow. Where you dey now for di process? Which part you think go hard? Which part go easy for you?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Search online for interviews with data scientists wey talk about their daily work. Here be [one](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Assignment

[Interview a data scientist](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis dokument don get translate by AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we dey try make am correct, abeg sabi say automated translation fit get mistake or no too correct. The original dokument wey dey im own language na im be correct source. For important information, make person wey sabi human translation do am. We no go responsible if any misunderstanding or wrong meaning show because of this translation.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->