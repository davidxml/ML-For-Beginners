# מסווגי מטבחים 1

בשעור זה, תשתמש במאגר הנתונים ששמרת מהשיעור הקודם מלא בנתונים נקיים ומאוזנים אודות מטבחים.

תשתמש במאגר זה עם מגוון מסווגים כדי _לחזות מטבח לאומי נתון בהתבסס על קבוצה של מרכיבים_. בעת ביצוע זאת, תלמד עוד על כמה מהדרכים בהן ניתן לנצל אלגוריתמים לביצוע משימות סיווג.

## [מבחן טרום-הרצאה](https://ff-quizzes.netlify.app/en/ml/)
# הכנה

בהנחה שסיימת את [שיעור 1](../1-Introduction/README.md), וודא שקובץ _cleaned_cuisines.csv_ קיים בתיקיית השורש `/data` עבור ארבעת השיעורים הללו.

## תרגיל - חיזוי מטבח לאומי

1. בעבודה בתיקיית _notebook.ipynb_ של שיעור זה, ייבא את הקובץ יחד עם ספריית Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    הנתונים נראים כך:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. עכשיו, ייבא מספר ספריות נוספות:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. חלק את הקואורדינטות X ו-y לשני DataFrame-ים עבור האימון. `cuisine` יכול להיות ה-DataFrame של התוויות:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    זה ייראה כך:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. הסר את העמודות `Unnamed: 0` ו-`cuisine` באמצעות קריאה ל-`drop()`. שמור את שאר הנתונים כמאפייני אימון:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    המאפיינים שלך נראים כך:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

כעת אתה מוכן לאמן את המודל שלך!

## בחירת המסווג

כעת שהנתונים שלך נקיים ומוכנים לאימון, עליך להחליט איזה אלגוריתם להשתמש למטרה.

Scikit-learn מסווג את הסיווג תחת לימת מפקח, ובקטגוריה זו תמצא דרכים רבות לסווג. [המגוון](https://scikit-learn.org/stable/supervised_learning.html) מבלבל למדי במבט ראשון. השיטות הבאות כוללות טכניקות סיווג:

- מודלים לינאריים  
- מכונות וקטור תמיכה (SVM)  
- ירידת גרדיאנט סטוכסטית  
- השכנים הקרובים ביותר  
- תהליכים גאוסיים  
- עצי החלטה  
- שיטות אסמבלא (מסווג הצבעות)  
- אלגוריתמים לריבוי מחלקות וריבוי תפוקות (סיווג Multiclass ו-multilabel, סיווג multiclass-multioutput)

> ניתן גם להשתמש ב[רשתות עצביות לסיווג נתונים](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), אך זה מחוץ לתחום שיעור זה.

### איזה מסווג לבחור?

אז, איזה מסווג כדאי לבחור? לעיתים קרובות, הרצת כמה מהם וחיפוש אחר תוצאה טובה היא דרך לבדוק. Scikit-learn מציעה [השוואה מקבילה](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) על מאגר נתונים שנוצר, המשווה בין KNeighbors, SVC בשתי דרכים, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB ו-QuadraticDiscrinationAnalysis, ומציגה את התוצאות בצורה ויזואלית: 

![השוואת מסווגים](../../../../translated_images/he/comparison.edfab56193a85e7f.webp)  
> תרשימים שנוצרו בתיעוד של Scikit-learn

> AutoML פותר בעיה זו באופן אלגנטי על ידי ביצוע השוואות אלה בענן, ומאפשר לך לבחור את האלגוריתם הטוב ביותר עבור הנתונים שלך. נסה את זה [כאן](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### גישה טובה יותר

גישה טובה יותר מאשר ניחוש פרוע, היא לעקוב אחר הרעיונות שבגליון [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) הניתן להורדה. כאן, אנו מגלים שלבעיית הריבוי מחלקות שלנו יש כמה אפשרויות:

![גליון רמאויות לבעיות ריבוי מחלקות](../../../../translated_images/he/cheatsheet.07a475ea444d2223.webp)  
> קטע מגליון הרמאויות של מיקרוסופט, המתאר אפשרויות לסיווג ריבוי מחלקות

✅ הורד את גיליון הרמאויות הזה, הדפס אותו ותלה אותו על הקיר שלך!

### שיקול דעת

בוא נראה אם נוכל לנמק את הדרך דרך גישות שונות בהתחשב במגבלות שיש לנו:

- **רשתות עצביות כבדות מדי**. בהתבסס על מאגר הנתונים הנקי, אך המינימלי שלנו, ועל העובדה שאנו עושים אימון מקומי דרך פנקסי הערות (notebooks), רשתות עצביות הן כבדות מדי למשימה זו.
- **אין מסווג דו-מחלקתי**. אנחנו לא משתמשים במסווג דו-מחלקתי, ולכן שוללים את השיטה one-vs-all.
- **עץ החלטה או רגרסיה לוגיסטית יכולים להתאים**. עץ החלטה עשוי להתאים, או רגרסיה לוגיסטית לנתונים מרובי מחלקות.
- **עצים מחוזקים לריבוי מחלקות פותרים בעיה שונה**. עץ החלטה מחוזק לריבוי מחלקות מתאים בעיקר למשימות לא פרמטריות, לדוגמה משימות שנועדו לבנות דירוגים, ולכן אינו שימושי במקרה שלנו.

### שימוש ב-Scikit-learn

נשתמש ב-Scikit-learn לניתוח הנתונים שלנו. עם זאת, יש דרכים רבות להשתמש ברגרסיה לוגיסטית ב-Scikit-learn. עיין בפרמטרים שיש להעביר [כאן](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

בעיקרון קיימים שני פרמטרים חשובים - `multi_class` ו-`solver` - שעלינו להגדיר כאשר מבקשים מ-Scikit-learn לבצע רגרסיה לוגיסטית. הערך של `multi_class` מכתיב התנהגות מסוימת. ערך `solver` הוא האלגוריתם שיש להשתמש בו. לא כל ה-solvers יכולים להיות משולבים עם כל ערכי `multi_class` .

לפי התיעוד, במקרה של ריבוי מחלקות, אלגוריתם האימון:

- **משתמש בסכמת one-vs-rest (OvR)** אם אפשרות `multi_class` מוגדרת כ-`ovr`
- **משתמש בהפסד cross-entropy**, אם אפשרות `multi_class` מוגדרת ל-`multinomial` (כיום אפשרות `multinomial` נתמכת רק על ידי solvers מסוג ‘lbfgs’, ‘sag’, ‘saga’ ו-‘newton-cg’)."

> 🎓 ה'סכמה' כאן יכולה להיות 'ovr' (one-vs-rest) או 'multinomial'. כיוון שרגרסיה לוגיסטית תוכננה במקור לתמוך בסיווג בינארי, סכמות אלו מאפשרות לה להתמודד טוב יותר עם משימות סיווג ריבוי מחלקות. [מקור](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 ה-'solver' מוגדר כ"אלגוריתם לשימוש בפתרון בעיית האופטימיזציה". [מקור](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn מציעה טבלה המסבירה איך שיטות הפיתרון מתמודדות עם אתגרים שונים המוצגים על ידי סוגי מבני נתונים שונים:

![solvers](../../../../translated_images/he/solvers.5fc648618529e627.webp)

## תרגיל - פצל את הנתונים

נוכל להתמקד ברגרסיה לוגיסטית לניסוי האימון הראשון שלנו מכיוון שלמדת רק לאחרונה עליה בשיעור קודם.  
חתוך את הנתונים שלך לקבוצות אימון ובדיקה עם קריאה ל-`train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## תרגיל - השתמש ברגרסיה לוגיסטית

מכיוון שאתה משתמש במקרה של ריבוי מחלקות, עליך לבחור איזו _סכמה_ להשתמש ואיזה _solver_ להגדיר. השתמש ב-LogisticRegression עם הגדרת ריבוי מחלקות ו-**liblinear** בתור ה-solver לאימון.

1. צור רגרסיה לוגיסטית עם multi_class מוגדר כ-`ovr` וה-solver מוגדר כ-`liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ נסה solver שונה כמו `lbfgs`, שלרוב מוגדר כברירת מחדל

    > שים לב, השתמש בפונקציה [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) של Pandas כדי לשטח את הנתונים בעת הצורך.

    הדיוק טוב ומעל **80%**!

1. תוכל לראות את המודל פועל על ידי בדיקת שורה אחת (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    התוצאה מודפסת:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ נסה מספר שורה שונה ובדוק את התוצאות
1. חופרים לעומק, ניתן לבדוק את הדיוק של התחזית הזו:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    התוצאה מודפסת - המטבח ההודי הוא הניחוש הטוב ביותר, עם הסתברות טובה:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ האם תוכל להסביר מדוע המודל די בטוח שזה מטבח הודי?

1. קבל פרטים נוספים על ידי הדפסת דוח סיווג, כפי שעשית בשיעורי הרגרסיה:

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

## 🚀אתגר

בשיעור זה, השתמשת בנתונים המסוננים שלך לבניית מודל למידת מכונה שיכול לנבא מטבח לאומי בהתבסס על סדרת רכיבים. קח קצת זמן לקרוא על האפשרויות הרבות ש-Scikit-learn מספקת לסיווג נתונים. חקור לעומק את מושג ה'פותר' (solver) כדי להבין מה מתרחש מאחורי הקלעים.

## [חידון לאחר ההרצאה](https://ff-quizzes.netlify.app/en/ml/)

## סקירה ולמידה עצמית

חפר קצת יותר לעומק למתמטיקה שמאחורי רגרסיה לוגיסטית ב[השיעור הזה](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## משימה 

[למד את הפותרים](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**כתב ויתור**:  
מסמך זה תורגם בעזרת שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדייק, יש להיות מודעים לכך שתרגומים אוטומטיים עלולים להכיל שגיאות או אי-דיוקים. המסמך המקורי בשפת המקור שלו הוא המקור המוסמך. למידע קריטי מומלץ להשתמש בתרגום מקצועי אנושי. איננו אחראים לאי-הבנות או לפרשנויות שגויות הנגרמות משימוש בתרגום זה.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->