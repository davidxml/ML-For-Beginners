# ตัวจำแนกอาหารประจำชาติ 1

ในบทเรียนนี้ คุณจะใช้ชุดข้อมูลที่คุณบันทึกไว้จากบทเรียนก่อนหน้า ซึ่งเต็มไปด้วยข้อมูลที่สมดุลและสะอาดทั้งหมดเกี่ยวกับอาหารประจำชาติ

คุณจะใช้ชุดข้อมูลนี้กับตัวจำแนกประเภทหลากหลายแบบเพื่อ _ทำนายอาหารประจำชาติที่กำหนดโดยอิงจากกลุ่มของส่วนผสม_ ในขณะทำเช่นนั้น คุณจะได้เรียนรู้เพิ่มเติมเกี่ยวกับวิธีต่างๆ ที่อัลกอริทึมสามารถนำมาใช้ในงานจำแนกประเภทได้

## [แบบทดสอบก่อนบรรยาย](https://ff-quizzes.netlify.app/en/ml/)
# การเตรียมตัว

สมมติว่าคุณทำ [บทเรียน 1](../1-Introduction/README.md) เสร็จแล้ว ให้ตรวจสอบให้แน่ใจว่าไฟล์ _cleaned_cuisines.csv_ มีอยู่ในโฟลเดอร์ราก `/data` สำหรับบทเรียนเหล่านี้สี่บท

## แบบฝึกหัด - ทำนายอาหารประจำชาติ

1. ทำงานในไฟล์ _notebook.ipynb_ ของบทเรียนนี้ ให้นำเข้าไฟล์นั้นพร้อมกับไลบรารี Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    ข้อมูลจะมีลักษณะดังนี้:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. ตอนนี้ ให้นำเข้าไลบรารีเพิ่มเติมหลายตัว:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. แบ่งแกน X และ y เป็นสอง dataframes สำหรับการฝึกอบรม โดย ที่ `cuisine` จะเป็น dataframe ของป้ายกำกับ:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    จะมีลักษณะดังนี้:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. ลบคอลัมน์ `Unnamed: 0` และคอลัมน์ `cuisine` โดยใช้คำสั่ง `drop()` แล้วบันทึกข้อมูลที่เหลือเป็นคุณลักษณะที่จะใช้ฝึกอบรม:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    คุณลักษณะของคุณจะมีลักษณะดังนี้:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

ตอนนี้คุณพร้อมที่จะฝึกแบบจำลองของคุณแล้ว!

## การเลือกตัวจำแนกของคุณ

เมื่อข้อมูลของคุณสะอาดและพร้อมสำหรับการฝึกอบรมแล้ว คุณต้องตัดสินใจว่าจะใช้อัลกอริทึมใดในการทำงานนี้

Scikit-learn จัดกลุ่มงานจำแนกประเภทภายใต้การเรียนรู้แบบมีผู้ดูแล (Supervised Learning) และในประเภทนี้คุณจะพบวิธีหลายวิธีในการจำแนกประเภท [ความหลากหลาย](https://scikit-learn.org/stable/supervised_learning.html) นั้นดูน่าหลงใหลในครั้งแรก วิธีการต่อไปนี้ล้วนมีเทคนิคการจำแนกประเภท:

- แบบจำลองเชิงเส้น
- Support Vector Machines
- Stochastic Gradient Descent
- Nearest Neighbors
- Gaussian Processes
- ต้นไม้ตัดสินใจ (Decision Trees)
- วิธีกลุ่ม (Ensemble methods เช่น voting Classifier)
- อัลกอริทึมมัลติคลาสและมัลติเอาต์พุต (multiclass และ multilabel classification, multiclass-multioutput classification)

> คุณยังสามารถใช้ [เครือข่ายประสาทเทียมเพื่อจำแนกข้อมูล](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification) แต่สิ่งนั้นอยู่นอกขอบเขตของบทเรียนนี้

### ควรเลือกตัวจำแนกประเภทใด?

แล้วควรเลือกตัวจำแนกประเภทไหน? โดยปกติแล้ว การทดลองใช้หลายตัวแล้วมองหาผลลัพธ์ที่ดีเป็นวิธีทดสอบ Scikit-learn มีการ [เปรียบเทียบเคียงข้างกัน](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) บนชุดข้อมูลที่สร้างขึ้น โดยเปรียบเทียบ KNeighbors, SVC สองวิธี, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB และ QuadraticDiscrinationAnalysis พร้อมแสดงผลลัพธ์ในรูปแบบกราฟ:

![comparison of classifiers](../../../../translated_images/th/comparison.edfab56193a85e7f.webp)
> กราฟที่สร้างขึ้นในเอกสารของ Scikit-learn

> AutoML แก้ปัญหานี้ได้อย่างลงตัวโดยการรันการเปรียบเทียบเหล่านี้ในระบบคลาวด์ ช่วยให้คุณเลือกอัลกอริทึมที่ดีที่สุดสำหรับข้อมูลของคุณ ลองได้ที่นี่ [here](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### วิธีที่ดีกว่า

วิธีที่ดีกว่าการเดาแบบสุ่ม คือการปฏิบัติตามแนวคิดใน [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) ที่ดาวน์โหลดได้ ที่นี่เราค้นพบว่า สำหรับปัญหามัลติคลาสของเรา เรามีตัวเลือกบางอย่าง:

![cheatsheet for multiclass problems](../../../../translated_images/th/cheatsheet.07a475ea444d2223.webp)
> ส่วนหนึ่งของ Algorithm Cheat Sheet ของ Microsoft ซึ่งอธิบายตัวเลือกสำหรับการจำแนกประเภทมัลติคลาส

✅ ดาวน์โหลดแผ่นช่วยจำนี้ พิมพ์ออกมา และแขวนไว้บนผนังของคุณ!

### การให้เหตุผล

ลองดูว่าเราจะให้เหตุผลเกี่ยวกับวิธีการต่าง ๆ ตามข้อจำกัดที่เรามีได้อย่างไร:

- **เครือข่ายประสาทเทียมหนักเกินไป** กล่าวคือชุดข้อมูลของเราสะอาดแต่น้อย และการฝึกสอนทำในเครื่องผ่านโน้ตบุ๊ก เครือข่ายประสาทเทียมจึงหนักเกินไปสำหรับงานนี้
- **ไม่มีตัวจำแนกสองคลาส** เราไม่ได้ใช้ตัวจำแนกสองคลาส ดังนั้นจึงตัดเมธอด one-vs-all ออก
- **ต้นไม้ตัดสินใจหรือลอจิสติกส์รีเกรสชันอาจใช้ได้** ต้นไม้ตัดสินใจอาจใช้ได้ หรือ ลอจิสติกส์รีเกรสชันสำหรับข้อมูลมัลติคลาส
- **มัลติคลาส บูสท์เทด ต้นไม้ตัดสินใจแก้ปัญหาที่ต่างออกไป** มัลติคลาสบูสท์เทดต้นไม้ตัดสินใจเหมาะสมที่สุดกับงานที่ไม่ใช่แบบพารามิเทริก เช่น งานที่ออกแบบมาเพื่อสร้างการจัดอันดับ ซึ่งจึงไม่เหมาะกับเรา

### การใช้ Scikit-learn

เราจะใช้ Scikit-learn ในการวิเคราะห์ข้อมูลของเรา อย่างไรก็ตาม มีหลายวิธีในการใช้ลอจิสติกส์รีเกรสชันใน Scikit-learn ดูที่ [พารามิเตอร์ที่ต้องส่ง](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)

โดยทั่วไป มีพารามิเตอร์สำคัญสองตัว - `multi_class` และ `solver` - ที่เราต้องระบุ เมื่อเราขอให้ Scikit-learn ทำการลอจิสติกส์รีเกรสชัน ค่า `multi_class` จะใช้พฤติกรรมเฉพาะ ส่วนค่า `solver` คืออัลกอริทึมที่ใช้ ไม่ใช่ทุก solver จะจับคู่กับค่า `multi_class` ได้ทั้งหมด

ตามเอกสาร ในกรณีมัลติคลาส อัลกอริทึมฝึกอบรม:

- **ใช้แผนผัง one-vs-rest (OvR)** หากตัวเลือก `multi_class` ตั้งค่าเป็น `ovr`
- **ใช้ cross-entropy loss** หากตัวเลือก `multi_class` ตั้งค่าเป็น `multinomial` (ขณะนี้ตัวเลือก `multinomial` รองรับเฉพาะ solver ‘lbfgs’, ‘sag’, ‘saga’ และ ‘newton-cg’ เท่านั้น)"

> 🎓 ‘แผนผัง’ ในที่นี้หมายถึง 'ovr' (one-vs-rest) หรือ 'multinomial' เนื่องจากลอจิสติกส์รีเกรสชันออกแบบมาเพื่อรองรับการจำแนกประเภทแบบไบนารี เทคนิคนั้นช่วยให้มันจัดการกับงานจำแนกประเภทมัลติคลาสได้ดียิ่งขึ้น [ที่มา](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 ‘solver’ ถูกกำหนดเป็น "อัลกอริทึมที่ใช้ในปัญหาการเพิ่มประสิทธิภาพ" [ที่มา](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn มีตารางนี้เพื่ออธิบายว่า solver แต่ละตัวจัดการกับความท้าทายของโครงสร้างข้อมูลประเภทต่าง ๆ อย่างไร:

![solvers](../../../../translated_images/th/solvers.5fc648618529e627.webp)

## แบบฝึกหัด - แบ่งชุดข้อมูล

เราสามารถเน้นที่ลอจิสติกส์รีเกรสชันสำหรับการทดลองฝึกครั้งแรกของเรา เนื่องจากคุณเพิ่งเรียนรู้เกี่ยวกับสิ่งนี้ในบทเรียนก่อนหน้า แบ่งข้อมูลของคุณออกเป็นกลุ่มฝึกและทดสอบโดยเรียก `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## แบบฝึกหัด - ใช้ลอจิสติกส์รีเกรสชัน

เนื่องจากคุณใช้กรณีมัลติคลาส คุณต้องเลือกว่าจะใช้ _แผนผัง_ อะไรและจะตั้งค่า _solver_ แบบใด ใช้ LogisticRegression โดยตั้งค่า multi_class และใช้ solver **liblinear** เพื่อฝึก

1. สร้างลอจิสติกส์รีเกรสชันโดยตั้งค่า multi_class เป็น `ovr` และตั้งค่า solver เป็น `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ ลองใช้ solver อื่น เช่น `lbfgs` ซึ่งมักถูกตั้งเป็นค่าดีฟอลต์

    > หมายเหตุ ใช้ฟังก์ชัน [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) ของ Pandas เพื่อแปลงข้อมูลให้เป็นแบบแถวเดียวเมื่อจำเป็น

    ความแม่นยำดีมากเกินกว่า **80%**!

1. คุณสามารถดูโมเดลนี้ทำงานโดยทดสอบข้อมูลแถวหนึ่ง (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    ผลลัพธ์จะถูกพิมพ์ออกมา:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ ลองแถวอื่นและตรวจสอบผลลัพธ์ดู
1. สำรวจให้ลึกขึ้น คุณสามารถตรวจสอบความแม่นยำของการทำนายนี้ได้:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    ผลลัพธ์จะถูกพิมพ์ออกมา - อาหารอินเดียเป็นการทำนายที่ดีที่สุด โดยมีความน่าจะเป็นที่ดี:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ คุณอธิบายได้ไหมว่าทำไมโมเดลถึงมั่นใจว่าคืออาหารอินเดีย?

1. รับรายละเอียดเพิ่มเติมโดยการพิมพ์รายงานการจำแนกประเภท ดังที่คุณทำในบทเรียนการวิเคราะห์การถดถอย:

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

## 🚀ท้าทาย

ในบทเรียนนี้ คุณใช้ข้อมูลที่ทำความสะอาดแล้วเพื่อสร้างโมเดลแมชชีนเลิร์นนิงที่สามารถทำนายอาหารประจำชาติจากชุดส่วนผสมต่างๆ ใช้เวลาอ่านผ่านตัวเลือกมากมายที่ Scikit-learn มีให้เพื่อจำแนกข้อมูล ทำความเข้าใจให้ลึกซึ้งเกี่ยวกับแนวคิด 'solver' เพื่อเข้าใจสิ่งที่เกิดขึ้นเบื้องหลัง

## [แบบทดสอบหลังบรรยาย](https://ff-quizzes.netlify.app/en/ml/)

## ทบทวน & ศึกษาด้วยตัวเอง

ขุดลึกเข้าไปในคณิตศาสตร์เบื้องหลังการถดถอยลอจิสติกใน [บทเรียนนี้](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## การบ้าน

[ศึกษาตัวแก้ปัญหา](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) ขณะที่เราพยายามให้ความถูกต้อง โปรดทราบว่าการแปลอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่แม่นยำ เอกสารต้นฉบับในภาษาต้นฉบับควรถูกพิจารณาเป็นแหล่งข้อมูลที่ถูกต้อง สำหรับข้อมูลที่สำคัญ แนะนำให้ใช้การแปลโดยมืออาชีพ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความผิดใด ๆ ที่เกิดจากการใช้การแปลนี้
<!-- CO-OP TRANSLATOR DISCLAIMER END -->