# Phân loại ẩm thực 1

Trong bài học này, bạn sẽ sử dụng bộ dữ liệu bạn đã lưu từ bài học trước, đầy đủ dữ liệu cân bằng, sạch về các loại ẩm thực.

Bạn sẽ sử dụng bộ dữ liệu này với nhiều bộ phân loại khác nhau để _dự đoán một nền ẩm thực quốc gia dựa trên nhóm nguyên liệu_. Trong khi làm việc này, bạn sẽ học thêm về một số cách mà các thuật toán có thể được tận dụng cho các tác vụ phân loại.

## [Bài kiểm tra trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)
# Chuẩn bị

Giả sử bạn đã hoàn thành [Bài học 1](../1-Introduction/README.md), hãy đảm bảo rằng một tập tin _cleaned_cuisines.csv_ tồn tại trong thư mục gốc `/data` cho bốn bài học này.

## Bài tập - dự đoán một nền ẩm thực quốc gia

1. Làm việc trong thư mục _notebook.ipynb_ của bài học này, nhập tập tin đó cùng với thư viện Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Dữ liệu trông như sau:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Bây giờ, nhập thêm một số thư viện khác:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Chia tọa độ X và y thành hai dataframe cho việc huấn luyện. `cuisine` có thể là dataframe nhãn:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Nó sẽ trông như thế này:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Loại bỏ cột `Unnamed: 0` và cột `cuisine`, gọi `drop()`. Lưu phần còn lại của dữ liệu làm các đặc trưng để huấn luyện:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Các đặc trưng của bạn sẽ trông như sau:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Bây giờ bạn đã sẵn sàng để huấn luyện mô hình!

## Chọn bộ phân loại của bạn

Bây giờ dữ liệu của bạn đã sạch và sẵn sàng cho việc huấn luyện, bạn phải quyết định thuật toán nào để sử dụng cho công việc này.

Scikit-learn nhóm các nhiệm vụ phân loại dưới Học có giám sát, và trong danh mục này bạn sẽ tìm thấy nhiều cách để phân loại. [Sự đa dạng](https://scikit-learn.org/stable/supervised_learning.html) có thể gây khó hiểu khi nhìn lần đầu. Các phương pháp sau đây đều bao gồm các kỹ thuật phân loại:

- Mô hình tuyến tính
- Máy vector hỗ trợ
- Gradient Descent Stochastic
- Hàng xóm gần nhất
- Quá trình Gaussian
- Cây quyết định
- Phương pháp tập hợp (voting Classifier)
- Thuật toán đa lớp và đa đầu ra (phân loại đa lớp và đa nhãn, phân loại đa lớp-đa đầu ra)

> Bạn cũng có thể sử dụng [mạng nơ-ron để phân loại dữ liệu](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), nhưng điều đó nằm ngoài phạm vi của bài học này.

### Nên chọn bộ phân loại nào?

Vậy nên chọn bộ phân loại nào? Thường thì, chạy thử một số và tìm kết quả tốt là một cách để thử nghiệm. Scikit-learn cung cấp một [bảng so sánh song song](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) trên một bộ dữ liệu tạo sẵn, so sánh KNeighbors, SVC hai cách, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB và QuadraticDiscrinationAnalysis, hiển thị kết quả bằng biểu đồ:

![so sánh các bộ phân loại](../../../../translated_images/vi/comparison.edfab56193a85e7f.webp)
> Biểu đồ được tạo ra trong tài liệu của Scikit-learn

> AutoML giải quyết vấn đề này rất gọn gàng bằng cách chạy những so sánh này trong đám mây, giúp bạn chọn thuật toán tốt nhất cho dữ liệu của mình. Hãy thử [tại đây](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Một cách tiếp cận tốt hơn

Một cách tốt hơn là không đoán bừa, bạn có thể theo các ý tưởng trong [bảng cheatsheet ML có thể tải xuống](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Ở đây, chúng ta khám phá ra là, với bài toán đa lớp của mình, chúng ta có một số lựa chọn:

![cheatsheet cho các bài toán đa lớp](../../../../translated_images/vi/cheatsheet.07a475ea444d2223.webp)
> Một phần của Bảng cheatsheet Thuật toán của Microsoft, chi tiết các lựa chọn phân loại đa lớp

✅ Tải bảng cheatsheet này xuống, in ra và treo trên tường của bạn!

### Lý do lựa chọn

Hãy xem chúng ta có thể lý luận qua các phương án khác nhau dựa trên các ràng buộc hiện có:

- **Mạng nơ-ron quá nặng**. Với bộ dữ liệu sạch nhưng đơn giản của chúng ta, và thực tế rằng chúng ta đang chạy huấn luyện cục bộ qua notebook, mạng nơ-ron quá nặng cho nhiệm vụ này.
- **Không dùng bộ phân loại hai lớp**. Chúng ta không sử dụng bộ phân loại hai lớp, nên loại bỏ phương pháp một-chọi-tất cả.
- **Cây quyết định hoặc hồi quy logistic có thể hiệu quả**. Cây quyết định có thể làm việc, hoặc hồi quy logistic cho dữ liệu đa lớp.
- **Cây quyết định đa lớp tăng cường giải quyết vấn đề khác**. Cây quyết định tăng cường đa lớp thích hợp nhất cho các tác vụ phi tham số, ví dụ các tác vụ xây dựng thứ hạng, nên không phù hợp cho chúng ta.

### Sử dụng Scikit-learn

Chúng ta sẽ sử dụng Scikit-learn để phân tích dữ liệu của mình. Tuy nhiên, có nhiều cách sử dụng hồi quy logistic trong Scikit-learn. Hãy xem xét [tham số để truyền vào](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Về cơ bản có hai tham số quan trọng - `multi_class` và `solver` - mà chúng ta cần xác định khi yêu cầu Scikit-learn thực hiện hồi quy logistic. Giá trị của `multi_class` áp dụng một hành vi nhất định. Giá trị của `solver` là thuật toán sẽ sử dụng. Không phải mọi solver đều phù hợp với tất cả giá trị `multi_class`.

Theo tài liệu, ở trường hợp đa lớp, thuật toán huấn luyện:

- **Sử dụng sơ đồ một-chọi-rest (OvR)**, nếu tùy chọn `multi_class` được đặt là `ovr`
- **Sử dụng mất mát entropy chéo (cross-entropy loss)**, nếu tùy chọn `multi_class` được đặt là `multinomial`. (Hiện tại tùy chọn `multinomial` chỉ được hỗ trợ bởi các solver ‘lbfgs’, ‘sag’, ‘saga’ và ‘newton-cg’.)"

> 🎓 ‘scheme’ ở đây có thể là ‘ovr’ (một-chọi-rest) hoặc ‘multinomial’. Vì hồi quy logistic được thiết kế chủ yếu cho phân loại nhị phân, những sơ đồ này giúp nó xử lý tốt hơn các bài toán phân loại đa lớp. [nguồn](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 ‘solver’ được định nghĩa là "thuật toán để sử dụng trong bài toán tối ưu hóa". [nguồn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn cung cấp bảng này để giải thích cách các solver xử lý những thách thức khác nhau được gây ra bởi các cấu trúc dữ liệu khác nhau:

![solvers](../../../../translated_images/vi/solvers.5fc648618529e627.webp)

## Bài tập - tách dữ liệu

Chúng ta có thể tập trung vào hồi quy logistic cho lần thử huấn luyện đầu tiên vì bạn mới học về phương pháp này trong bài học trước.
Chia dữ liệu của bạn thành nhóm huấn luyện và kiểm thử bằng cách gọi `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Bài tập - áp dụng hồi quy logistic

Vì bạn đang sử dụng trường hợp đa lớp, bạn cần chọn _scheme_ nào để dùng và _solver_ nào để đặt. Sử dụng LogisticRegression với thiết lập đa lớp và solver **liblinear** để huấn luyện.

1. Tạo một hồi quy logistic với multi_class đặt là `ovr` và solver đặt là `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Thử một solver khác như `lbfgs`, thường được đặt làm mặc định

    > Lưu ý, sử dụng hàm [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) của Pandas để dẹt dữ liệu khi cần.

    Độ chính xác tốt trên **80%**!

1. Bạn có thể xem mô hình này hoạt động bằng cách thử với một hàng dữ liệu (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Kết quả được in ra:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Thử với một số hàng khác và kiểm tra kết quả
1. Đào sâu hơn, bạn có thể kiểm tra độ chính xác của dự đoán này:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Kết quả được in ra - Ẩm thực Ấn Độ là dự đoán tốt nhất, với xác suất cao:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Bạn có thể giải thích tại sao mô hình lại khá chắc chắn đây là ẩm thực Ấn Độ không?

1. Lấy thêm chi tiết bằng cách in báo cáo phân loại, như bạn đã làm trong các bài học hồi quy:

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

## 🚀Thử thách

Trong bài học này, bạn đã sử dụng dữ liệu đã làm sạch để xây dựng một mô hình học máy có thể dự đoán một nền ẩm thực quốc gia dựa trên một loạt các nguyên liệu. Hãy dành chút thời gian để đọc qua nhiều tùy chọn mà Scikit-learn cung cấp để phân loại dữ liệu. Đào sâu hơn về khái niệm 'solver' để hiểu những gì diễn ra phía sau hậu trường.

## [Bài kiểm tra sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Đào sâu hơn về toán học phía sau hồi quy logistic trong [bài học này](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Bài tập

[Nghiên cứu các solvers](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ gốc nên được xem là nguồn chính thức. Đối với thông tin quan trọng, nên sử dụng dịch thuật chuyên nghiệp do con người thực hiện. Chúng tôi không chịu trách nhiệm về bất kỳ hiểu lầm hoặc giải thích sai nào phát sinh từ việc sử dụng bản dịch này.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->