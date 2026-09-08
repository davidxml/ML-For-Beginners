# Bộ phân loại ẩm thực 2

Trong bài học phân loại thứ hai này, bạn sẽ khám phá thêm nhiều cách để phân loại dữ liệu số. Bạn cũng sẽ tìm hiểu về những hệ quả khi chọn một bộ phân loại thay vì bộ phân loại khác.

## [Quiz trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

### Điều kiện tiên quyết

Chúng tôi giả định rằng bạn đã hoàn thành các bài học trước và có một bộ dữ liệu đã làm sạch trong thư mục `data` của bạn có tên là _cleaned_cuisines.csv_ trong thư mục gốc của bộ 4 bài học này.

### Chuẩn bị

Chúng tôi đã tải tệp _notebook.ipynb_ của bạn với bộ dữ liệu đã làm sạch và chia nó thành các dataframe X và y, sẵn sàng cho quá trình xây dựng mô hình.

## Bản đồ phân loại

Trước đây, bạn đã học về các tùy chọn khác nhau khi phân loại dữ liệu bằng cheat sheet của Microsoft. Scikit-learn cung cấp một cheat sheet tương tự nhưng chi tiết hơn, giúp thu hẹp các bộ ước lượng của bạn (cũng gọi là bộ phân loại):

![ML Map from Scikit-learn](../../../../translated_images/vi/map.e963a6a51349425a.webp)
> Tip: [truy cập bản đồ này trực tuyến](https://scikit-learn.org/stable/tutorial/machine_learning_map/) và nhấp dọc theo các đường đi để đọc tài liệu.

### Kế hoạch

Bản đồ này rất hữu ích khi bạn đã nắm rõ dữ liệu của mình, vì bạn có thể ‘đi’ theo các đường dẫn của nó đến một quyết định:

- Chúng ta có >50 mẫu
- Chúng ta muốn dự đoán một danh mục
- Chúng ta có dữ liệu được gán nhãn
- Chúng ta có ít hơn 100K mẫu
- ✨ Chúng ta có thể chọn Linear SVC
- Nếu điều đó không hiệu quả, vì chúng ta có dữ liệu số
    - Chúng ta có thể thử ✨ KNeighbors Classifier
      - Nếu vẫn không hiệu quả, thử ✨ SVC và ✨ Ensemble Classifiers

Đây là một con đường rất hữu ích để theo dõi.

## Bài tập - chia dữ liệu

Theo con đường này, chúng ta nên bắt đầu bằng cách nhập một số thư viện để sử dụng.

1. Nhập các thư viện cần thiết:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Chia dữ liệu huấn luyện và kiểm tra của bạn:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Bộ phân loại Linear SVC

Support-Vector clustering (SVC) là một loại trong gia đình kỹ thuật máy học Support-Vector machines (tìm hiểu thêm về chúng bên dưới). Trong phương pháp này, bạn có thể chọn một ‘kernel’ để quyết định cách nhóm nhãn. Tham số ‘C’ liên quan đến ‘regularization’ giúp điều chỉnh ảnh hưởng của các tham số. Kernel có thể là một trong [nhiều](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); ở đây chúng ta đặt là ‘linear’ để đảm bảo sử dụng linear SVC. Giá trị probability mặc định là 'false'; ở đây chúng ta đặt thành 'true' để thu thập ước lượng xác suất. Chúng ta đặt random state là '0' để xáo trộn dữ liệu cho việc tính xác suất.

### Bài tập - áp dụng linear SVC

Bắt đầu bằng việc tạo một mảng các bộ phân loại. Bạn sẽ thêm dần vào mảng này khi chúng ta thử nghiệm.

1. Bắt đầu với Linear SVC:

    ```python
    C = 10
    # Tạo các bộ phân loại khác nhau.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Huấn luyện mô hình của bạn bằng Linear SVC và in ra báo cáo:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Kết quả khá tốt:

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

## Bộ phân loại K-Neighbors

K-Neighbors là một phần của gia đình phương pháp "neighbors" trong ML, có thể được sử dụng cho cả học có giám sát và không giám sát. Trong phương pháp này, một số điểm được xác định trước được tạo ra và dữ liệu được gom quanh các điểm này sao cho các nhãn tổng quát có thể được dự đoán cho dữ liệu.

### Bài tập - áp dụng bộ phân loại K-Neighbors

Bộ phân loại trước đó hoạt động tốt và thích hợp với dữ liệu, nhưng có thể chúng ta có thể cải thiện độ chính xác hơn. Thử dùng bộ phân loại K-Neighbors.

1. Thêm một dòng vào mảng bộ phân loại của bạn (thêm dấu phẩy sau mục Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Kết quả hơi kém hơn một chút:

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

    ✅ Tìm hiểu về [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Bộ phân loại Support Vector

Support-Vector classifiers là một phần của gia đình phương pháp [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) trong ML, được sử dụng cho các nhiệm vụ phân loại và hồi quy. SVM "ánh xạ các ví dụ huấn luyện thành các điểm trong không gian" để tối đa hóa khoảng cách giữa hai danh mục. Dữ liệu kế tiếp được ánh xạ vào không gian này để dự đoán danh mục của chúng.

### Bài tập - áp dụng bộ phân loại Support Vector

Chúng ta thử cải thiện độ chính xác một chút bằng bộ phân loại Support Vector.

1. Thêm dấu phẩy sau mục K-Neighbors, rồi thêm dòng này:

    ```python
    'SVC': SVC(),
    ```

    Kết quả khá tốt!

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

    ✅ Tìm hiểu về [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Bộ phân loại Ensemble

Hãy tiếp tục theo đường dẫn đến cuối cùng, mặc dù thử nghiệm trước khá tốt. Hãy thử một số 'Bộ phân loại Ensemble', cụ thể là Random Forest và AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Kết quả rất tốt, đặc biệt với Random Forest:

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

✅ Tìm hiểu về [Bộ phân loại Ensemble](https://scikit-learn.org/stable/modules/ensemble.html)

Phương pháp học máy này "kết hợp dự đoán của nhiều bộ ước lượng cơ sở" để cải thiện chất lượng mô hình. Trong ví dụ của chúng ta, chúng ta sử dụng Random Trees và AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), một phương pháp trung bình, xây dựng một 'rừng' các 'cây quyết định' được thêm ngẫu nhiên để tránh overfitting. Tham số n_estimators được đặt bằng số cây.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) phù hợp một bộ phân loại với bộ dữ liệu, sau đó phù hợp các bản sao của bộ phân loại đó lên cùng bộ dữ liệu. Phương pháp này tập trung vào trọng số các mục được phân loại sai và điều chỉnh phần phù hợp cho bộ phân loại tiếp theo để sửa lỗi.

---

## 🚀Thử thách

Mỗi kỹ thuật này có rất nhiều tham số bạn có thể điều chỉnh. Hãy nghiên cứu tham số mặc định của từng kỹ thuật và suy nghĩ về việc điều chỉnh các tham số này sẽ ảnh hưởng như thế nào đến chất lượng mô hình.

## [Quiz sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Có rất nhiều thuật ngữ trong các bài học này, vì vậy hãy dành một phút để xem lại [danh sách này](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) các thuật ngữ hữu ích!

## Bài tập

[Chơi với tham số](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi nỗ lực đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Văn bản gốc bằng ngôn ngữ gốc nên được xem là nguồn thông tin chính thức. Đối với các thông tin quan trọng, nên sử dụng dịch vụ dịch thuật chuyên nghiệp của con người. Chúng tôi không chịu trách nhiệm về bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->