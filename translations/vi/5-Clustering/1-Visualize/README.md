# Giới thiệu về phân nhóm

Phân nhóm là một loại [Học không giám sát](https://wikipedia.org/wiki/Unsupervised_learning) giả định rằng một bộ dữ liệu không có nhãn hoặc đầu vào của nó không được ghép với các đầu ra định trước. Nó sử dụng các thuật toán khác nhau để sắp xếp dữ liệu chưa được gán nhãn và cung cấp các nhóm theo các mẫu mà nó phát hiện trong dữ liệu.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Nhấp vào hình ảnh trên để xem video. Khi bạn đang học máy học với phân nhóm, hãy thưởng thức một số bản nhạc Dance Hall Nigeria - đây là một bài hát được đánh giá cao từ năm 2014 bởi PSquare.

## [Bài kiểm tra trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

### Giới thiệu

[Phân nhóm](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) rất hữu ích cho việc khám phá dữ liệu. Hãy xem nó có thể giúp phát hiện các xu hướng và mẫu trong cách khán giả Nigeria tiêu thụ âm nhạc như thế nào.

✅ Dành một phút để suy nghĩ về các ứng dụng của phân nhóm. Trong đời thực, phân nhóm xảy ra bất cứ khi nào bạn có một đống quần áo cần phân loại quần áo của các thành viên trong gia đình bạn 🧦👕👖🩲. Trong khoa học dữ liệu, phân nhóm xảy ra khi cố gắng phân tích sở thích của người dùng, hoặc xác định các đặc điểm của bất kỳ bộ dữ liệu chưa được gán nhãn nào. Phân nhóm, theo một cách nào đó, giúp hiểu được sự hỗn độn, giống như ngăn kéo đựng tất vậy.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Nhấp vào hình ảnh trên để xem video: John Guttag của MIT giới thiệu về phân nhóm

Trong một môi trường chuyên nghiệp, phân nhóm có thể được sử dụng để xác định các điều như phân đoạn thị trường, xác định nhóm tuổi nào mua các mặt hàng nào, ví dụ. Một ứng dụng khác là phát hiện bất thường, có thể để phát hiện gian lận từ một bộ dữ liệu giao dịch thẻ tín dụng. Hoặc bạn có thể dùng phân nhóm để xác định các khối u trong một loạt các bản quét y tế.

✅ Hãy nghĩ một phút về cách bạn có thể đã gặp phân nhóm 'trong thực tế', trong lĩnh vực ngân hàng, thương mại điện tử hoặc kinh doanh.

> 🎓 Thú vị là, phân tích phân nhóm có nguồn gốc từ các lĩnh vực Nhân chủng học và Tâm lý học vào những năm 1930. Bạn có thể tưởng tượng nó đã được sử dụng như thế nào không?

Ngoài ra, bạn có thể dùng nó để nhóm các kết quả tìm kiếm - theo các liên kết mua sắm, hình ảnh, hoặc đánh giá, ví dụ. Phân nhóm hữu ích khi bạn có một bộ dữ liệu lớn mà bạn muốn giảm thiểu và thực hiện phân tích chi tiết hơn, vì vậy kỹ thuật này có thể được sử dụng để tìm hiểu về dữ liệu trước khi xây dựng các mô hình khác.

✅ Khi dữ liệu của bạn được tổ chức thành các cụm, bạn sẽ gán nó một mã cụm, và kỹ thuật này có thể hữu ích khi bảo vệ quyền riêng tư của bộ dữ liệu; bạn có thể thay vào đó tham chiếu một điểm dữ liệu bằng mã cụm của nó, thay vì bằng dữ liệu nhận dạng rõ ràng hơn. Bạn có thể nghĩ đến những lý do khác tại sao bạn sẽ tham chiếu mã cụm thay vì các thành phần khác của cụm để nhận diện nó?

Hãy đào sâu hiểu biết của bạn về các kỹ thuật phân nhóm trong [chương trình học](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott) này.
## Bắt đầu với phân nhóm

[Scikit-learn cung cấp một loạt](https://scikit-learn.org/stable/modules/clustering.html) các phương pháp để thực hiện phân nhóm. Loại mà bạn chọn sẽ phụ thuộc vào trường hợp sử dụng của bạn. Theo tài liệu, mỗi phương pháp có các lợi ích khác nhau. Dưới đây là bảng đơn giản hóa các phương pháp được Scikit-learn hỗ trợ và các trường hợp sử dụng phù hợp:

| Tên phương pháp             | Trường hợp sử dụng                                                    |
| :-------------------------- | :------------------------------------------------------------------- |
| K-Means                     | mục đích chung, suy diễn                                              |
| Truyền truyền độ tương tự   | nhiều cụm, không đồng đều, suy diễn                                   |
| Mean-shift                  | nhiều cụm, không đồng đều, suy diễn                                   |
| Phân nhóm phổ               | ít cụm, đều, truyền đề                                               |
| Phân nhóm phân cấp Ward     | nhiều cụm, có ràng buộc, truyền đề                                  |
| Phân nhóm kết hợp           | nhiều, có ràng buộc, khoảng cách phi Euclid, truyền đề               |
| DBSCAN                      | hình học phi phẳng, cụm không đều, truyền đề                        |
| OPTICS                      | hình học phi phẳng, cụm không đều với mật độ biến đổi, truyền đề    |
| Hỗn hợp Gaussian            | hình học phẳng, suy diễn                                             |
| BIRCH                       | bộ dữ liệu lớn có ngoại lệ, suy diễn                                 |

> 🎓 Cách chúng ta tạo các cụm có liên quan rất lớn đến cách chúng ta gom các điểm dữ liệu thành nhóm. Hãy cùng làm rõ một số thuật ngữ:
>
> 🎓 ['Truyền đề' vs. 'suy diễn'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Suy luận truyền đề được rút ra từ các trường hợp huấn luyện quan sát được ánh xạ đến các trường hợp kiểm tra cụ thể. Suy luận suy diễn được rút ra từ các trường hợp huấn luyện ánh xạ tới các quy tắc chung và chỉ sau đó áp dụng cho các trường hợp kiểm tra.
> 
> Ví dụ: Giả sử bạn có một bộ dữ liệu chỉ được gán nhãn một phần. Một số thứ là 'đĩa ghi âm', một số là 'đĩa CD', và một số thì trống. Nhiệm vụ của bạn là gán nhãn cho những trường trống. Nếu bạn chọn phương pháp suy diễn, bạn sẽ huấn luyện một mô hình để tìm 'đĩa ghi âm' và 'đĩa CD', rồi áp dụng nhãn đó cho dữ liệu chưa được gán nhãn. Phương pháp này sẽ gặp khó khăn khi phân loại các thứ thực sự là 'băng cassette'. Ngược lại, phương pháp truyền đề xử lý dữ liệu chưa biết hiệu quả hơn vì nó nhóm các mục giống nhau lại với nhau và sau đó áp dụng nhãn cho nhóm đó. Trong trường hợp này, các cụm có thể phản ánh 'đồ vật âm nhạc tròn' và 'đồ vật âm nhạc vuông'.
> 
> 🎓 ['Hình học phi phẳng' vs. 'hình học phẳng'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Bắt nguồn từ thuật ngữ toán học, hình học phi phẳng và hình học phẳng đề cập đến cách đo khoảng cách giữa các điểm bằng các phương pháp hình học 'phẳng' ([Euclid](https://wikipedia.org/wiki/Euclidean_geometry)) hoặc 'phi phẳng' (phi Euclid).
>
> 'Phẳng' trong ngữ cảnh này là hình học Euclid (một phần được dạy là hình học phẳng), và phi phẳng chỉ hình học phi Euclid. Hình học có liên quan gì đến học máy? Vì cả hai lĩnh vực đều bắt nguồn từ toán học, nên cần một cách chung để đo khoảng cách giữa các điểm trong cụm, và điều này có thể được thực hiện theo cách 'phẳng' hoặc 'phi phẳng' tùy thuộc vào tính chất của dữ liệu. [Khoảng cách Euclid](https://wikipedia.org/wiki/Euclidean_distance) được đo bằng chiều dài đoạn thẳng giữa hai điểm. [Khoảng cách phi Euclid](https://wikipedia.org/wiki/Non-Euclidean_geometry) được đo theo một đường cong. Nếu dữ liệu của bạn, khi trực quan hóa, có vẻ như không tồn tại trên một mặt phẳng, bạn có thể cần dùng thuật toán chuyên biệt để xử lý nó.
>
![Infographic hình học phẳng vs phi phẳng](../../../../translated_images/vi/flat-nonflat.d1c8c6e2a96110c1.webp)
> Đồ họa thông tin bởi [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Khoảng cách'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Các cụm được xác định bởi ma trận khoảng cách của chúng, ví dụ khoảng cách giữa các điểm. Khoảng cách này có thể được đo theo một vài cách. Các cụm Euclid được xác định bằng trung bình giá trị các điểm, và có một 'tâm cụm' hay điểm trung tâm. Khoảng cách do đó được đo tới điểm tâm cụm đó. Khoảng cách phi Euclid đề cập đến 'clustroid', điểm gần nhất với các điểm khác. Clustroid lại có thể được xác định theo nhiều cách.
> 
> 🎓 ['Có ràng buộc'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Phân nhóm có ràng buộc](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) giới thiệu học 'bán giám sát' vào phương pháp không giám sát này. Các mối quan hệ giữa các điểm được đánh dấu là 'không thể liên kết' hoặc 'phải liên kết', do đó một số quy tắc được áp đặt lên bộ dữ liệu.
>
> Ví dụ: Nếu một thuật toán được thả tự do trên một lô dữ liệu chưa hoặc bán nhãn, các cụm mà nó tạo ra có thể có chất lượng kém. Như ví dụ trên, các cụm có thể nhóm 'đồ vật âm nhạc tròn', 'đồ vật âm nhạc vuông', 'đồ vật hình tam giác' và 'bánh quy'. Nếu được cung cấp một số ràng buộc, hoặc quy tắc để tuân theo ("mặt hàng phải làm bằng nhựa", "mặt hàng cần có khả năng tạo ra âm nhạc"), điều này có thể giúp 'ràng buộc' thuật toán để đưa ra lựa chọn tốt hơn.
> 
> 🎓 'Mật độ'
> 
> Dữ liệu được coi là 'nhiễu' thì được xem là 'mật độ cao'. Khoảng cách giữa các điểm trong mỗi cụm khi kiểm tra có thể chứng minh là đặc biệt hoặc ít đặc, hay 'đông đúc' và dữ liệu này cần được phân tích bằng phương pháp phân nhóm phù hợp. [Bài viết này](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) minh họa sự khác biệt giữa việc sử dụng thuật toán phân nhóm K-Means với HDBSCAN để khám phá bộ dữ liệu nhiễu có mật độ cụm không đồng đều.

## Các thuật toán phân nhóm

Có hơn 100 thuật toán phân nhóm, và việc sử dụng phụ thuộc vào bản chất của dữ liệu hiện có. Hãy thảo luận một số thuật toán chính:

- **Phân nhóm phân cấp**. Nếu một đối tượng được phân loại bằng khoảng cách đến một đối tượng gần đó, thay vì đối tượng xa hơn, các cụm được hình thành dựa trên khoảng cách của các thành viên đến và đi từ các đối tượng khác. Phân nhóm kết hợp của Scikit-learn là phân cấp.

   ![Đồ họa thông tin phân nhóm phân cấp](../../../../translated_images/vi/hierarchical.bf59403aa43c8c47.webp)
   > Đồ họa thông tin bởi [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Phân nhóm tâm cụm**. Thuật toán phổ biến này yêu cầu lựa chọn 'k', số cụm cần tạo, sau đó thuật toán xác định điểm trung tâm của một cụm và gom dữ liệu xung quanh điểm đó. [Phân nhóm K-means](https://wikipedia.org/wiki/K-means_clustering) là một phiên bản phổ biến của phân nhóm tâm cụm. Trung tâm được xác định bằng trung bình gần nhất, do đó có tên gọi. Khoảng cách bình phương từ cụm được giảm thiểu.

   ![Đồ họa thông tin phân nhóm tâm cụm](../../../../translated_images/vi/centroid.097fde836cf6c918.webp)
   > Đồ họa thông tin bởi [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Phân nhóm dựa trên phân phối**. Dựa trên mô hình thống kê, phân nhóm dựa trên phân phối tập trung vào xác định xác suất một điểm dữ liệu thuộc về một cụm nào đó, và phân bố điểm đó tương ứng. Phương pháp hợp Gaussian thuộc loại này.

- **Phân nhóm dựa trên mật độ**. Các điểm dữ liệu được gán vào cụm dựa trên mật độ của chúng, hoặc sự nhóm lại xung quanh nhau. Các điểm dữ liệu xa nhóm được coi là ngoại lệ hoặc nhiễu. DBSCAN, Mean-shift và OPTICS thuộc loại phân nhóm này.

- **Phân nhóm dựa trên lưới**. Đối với bộ dữ liệu đa chiều, một lưới được tạo ra và dữ liệu được chia vào các ô của lưới, từ đó tạo thành các cụm.

## Bài tập - phân nhóm dữ liệu của bạn

Phân nhóm như một kỹ thuật được hỗ trợ rất nhiều bằng cách trực quan hóa thích hợp, vì vậy hãy bắt đầu bằng cách trực quan hóa dữ liệu âm nhạc của chúng ta. Bài tập này sẽ giúp chúng ta quyết định phương pháp phân nhóm nào nên được sử dụng hiệu quả nhất cho tính chất của dữ liệu này.

1. Mở file [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) trong thư mục này.

1. Nhập gói `Seaborn` để trực quan hóa dữ liệu tốt.

    ```python
    !pip install seaborn
    ```

1. Thêm dữ liệu bài hát từ [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Tải một dataframe với một số dữ liệu về các bài hát. Hãy chuẩn bị khám phá dữ liệu này bằng cách nhập các thư viện và in ra dữ liệu:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Kiểm tra một vài dòng dữ liệu đầu tiên:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Lấy một số thông tin về dataframe, gọi `info()`:

    ```python
    df.info()
    ```

   Kết quả trông như sau:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 530 entries, 0 to 529
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   name              530 non-null    object 
     1   album             530 non-null    object 
     2   artist            530 non-null    object 
     3   artist_top_genre  530 non-null    object 
     4   release_date      530 non-null    int64  
     5   length            530 non-null    int64  
     6   popularity        530 non-null    int64  
     7   danceability      530 non-null    float64
     8   acousticness      530 non-null    float64
     9   energy            530 non-null    float64
     10  instrumentalness  530 non-null    float64
     11  liveness          530 non-null    float64
     12  loudness          530 non-null    float64
     13  speechiness       530 non-null    float64
     14  tempo             530 non-null    float64
     15  time_signature    530 non-null    int64  
    dtypes: float64(8), int64(4), object(4)
    memory usage: 66.4+ KB
    ```

1. Kiểm tra lại giá trị null, bằng cách gọi `isnull()` và xác minh tổng là 0:

    ```python
    df.isnull().sum()
    ```

    Trông ổn:

    ```output
    name                0
    album               0
    artist              0
    artist_top_genre    0
    release_date        0
    length              0
    popularity          0
    danceability        0
    acousticness        0
    energy              0
    instrumentalness    0
    liveness            0
    loudness            0
    speechiness         0
    tempo               0
    time_signature      0
    dtype: int64
    ```

1. Mô tả dữ liệu:

    ```python
    df.describe()
    ```

    |       | release_date | length      | popularity | danceability | acousticness | energy   | instrumentalness | liveness | loudness  | speechiness | tempo      | time_signature |
    | ----- | ------------ | ----------- | ---------- | ------------ | ------------ | -------- | ---------------- | -------- | --------- | ----------- | ---------- | -------------- |
    | count | 530          | 530         | 530        | 530          | 530          | 530      | 530              | 530      | 530       | 530         | 530        | 530            |
    | mean  | 2015.390566  | 222298.1698 | 17.507547  | 0.741619     | 0.265412     | 0.760623 | 0.016305         | 0.147308 | -4.953011 | 0.130748    | 116.487864 | 3.986792       |
    | std   | 3.131688     | 39696.82226 | 18.992212  | 0.117522     | 0.208342     | 0.148533 | 0.090321         | 0.123588 | 2.464186  | 0.092939    | 23.518601  | 0.333701       |
    | min   | 1998         | 89488       | 0          | 0.255        | 0.000665     | 0.111    | 0                | 0.0283   | -19.362   | 0.0278      | 61.695     | 3              |
    | 25%   | 2014         | 199305      | 0          | 0.681        | 0.089525     | 0.669    | 0                | 0.07565  | -6.29875  | 0.0591      | 102.96125  | 4              |
    | 50%   | 2016         | 218509      | 13         | 0.761        | 0.2205       | 0.7845   | 0.000004         | 0.1035   | -4.5585   | 0.09795     | 112.7145   | 4              |
    | 75%   | 2017         | 242098.5    | 31         | 0.8295       | 0.403        | 0.87575  | 0.000234         | 0.164    | -3.331    | 0.177       | 125.03925  | 4              |
    | max   | 2020         | 511738      | 73         | 0.966        | 0.954        | 0.995    | 0.91             | 0.811    | 0.582     | 0.514       | 206.007    | 5              |

> 🤔 Nếu chúng ta đang làm việc với phân cụm, một phương pháp không giám sát không yêu cầu dữ liệu có nhãn, tại sao chúng ta lại hiển thị dữ liệu này với các nhãn? Trong giai đoạn khám phá dữ liệu, chúng rất hữu ích, nhưng chúng không cần thiết cho các thuật toán phân cụm hoạt động. Bạn cũng có thể loại bỏ tiêu đề cột và tham khảo dữ liệu bằng số cột.

Hãy nhìn vào các giá trị chung của dữ liệu. Lưu ý rằng popularity có thể là '0', điều này cho thấy các bài hát không có thứ hạng. Chúng ta sẽ sớm loại bỏ những trường hợp đó.

1. Sử dụng barplot để tìm ra các thể loại phổ biến nhất:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![phổ biến nhất](../../../../translated_images/vi/popular.9c48d84b3386705f.webp)

✅ Nếu bạn muốn xem nhiều giá trị hàng đầu hơn, hãy thay đổi `[:5]` thành giá trị lớn hơn, hoặc loại bỏ nó để xem tất cả.

Lưu ý, khi thể loại hàng đầu được mô tả là 'Missing', tức là Spotify không phân loại được, vì vậy hãy loại bỏ nó.

1. Loại bỏ dữ liệu bị thiếu bằng cách lọc ra

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Bây giờ kiểm tra lại các thể loại:

    ![phổ biến nhất](../../../../translated_images/vi/all-genres.1d56ef06cefbfcd6.webp)

1. Cho đến nay, ba thể loại hàng đầu chiếm ưu thế trong bộ dữ liệu này. Hãy tập trung vào `afro dancehall`, `afropop`, và `nigerian pop`, đồng thời lọc bộ dữ liệu để loại bỏ bất kỳ giá trị popularity nào bằng 0 (có nghĩa là nó không được phân loại với mức độ phổ biến trong bộ dữ liệu và có thể được coi là nhiễu cho mục đích của chúng ta):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Thực hiện một thử nghiệm nhanh để xem liệu dữ liệu có tương quan mạnh mẽ đặc biệt nào hay không:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![tương quan](../../../../translated_images/vi/correlation.a9356bb798f5eea5.webp)

    Mối tương quan duy nhất mạnh mẽ là giữa `energy` và `loudness`, điều này không quá ngạc nhiên, vì nhạc to thường khá năng lượng. Ngoài ra, các mối tương quan khá yếu. Sẽ rất thú vị để xem thuật toán phân cụm có thể tạo ra gì từ dữ liệu này.

    > 🎓 Lưu ý rằng tương quan không ngụ ý nguyên nhân! Chúng ta có bằng chứng về tương quan nhưng không có bằng chứng về nguyên nhân. Một [trang web thú vị](https://tylervigen.com/spurious-correlations) có một số hình ảnh nhấn mạnh điểm này.

Liệu có sự hội tụ trong bộ dữ liệu này quanh mức độ phổ biến và khả năng khiêu vũ của một bài hát không? Một FacetGrid cho thấy có các vòng đồng tâm xếp hàng, bất kể thể loại. Có thể là thị hiếu của người Nigeria hội tụ ở mức độ khiêu vũ nhất định cho thể loại này?

✅ Thử các điểm dữ liệu khác nhau (energy, loudness, speechiness) và nhiều hoặc khác các thể loại âm nhạc. Bạn có thể khám phá được gì? Hãy xem bảng `df.describe()` để thấy sự phân bố chung của các điểm dữ liệu.

### Bài tập - phân bố dữ liệu

Ba thể loại này có khác biệt đáng kể trong cách nhận thức về khả năng khiêu vũ của chúng, dựa trên độ phổ biến không?

1. Kiểm tra phân bố dữ liệu của ba thể loại hàng đầu về mức độ phổ biến và khả năng khiêu vũ theo trục x và y đã cho.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Bạn có thể khám phá các vòng đồng tâm quanh một điểm hội tụ chung, cho thấy phân bố các điểm.

    > 🎓 Lưu ý rằng ví dụ này sử dụng biểu đồ KDE (Ước lượng Mật độ Hạt nhân) thể hiện dữ liệu bằng một đường cong mật độ xác suất liên tục. Điều này cho phép chúng ta diễn giải dữ liệu khi làm việc với nhiều phân phối.

    Nói chung, ba thể loại này căn chỉnh lỏng lẻo về mức độ phổ biến và khả năng khiêu vũ. Việc xác định các nhóm trong dữ liệu căn chỉnh lỏng lẻo này sẽ là một thách thức:

    ![phân bố](../../../../translated_images/vi/distribution.9be11df42356ca95.webp)

1. Tạo một biểu đồ scatter:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Biểu đồ scatter cùng các trục cho thấy một mẫu hội tụ tương tự

    ![Facetgrid](../../../../translated_images/vi/facetgrid.9b2e65ce707eba1f.webp)

Nói chung, đối với phân cụm, bạn có thể sử dụng biểu đồ scatter để hiển thị các nhóm dữ liệu, vì vậy việc thành thạo loại hình minh họa này rất hữu ích. Trong bài học tiếp theo, chúng ta sẽ lấy dữ liệu đã lọc này và sử dụng phân cụm k-means để khám phá các nhóm trong dữ liệu này có vẻ chồng chéo nhau theo những cách thú vị.

---

## 🚀Thử thách

Để chuẩn bị cho bài học tiếp theo, hãy tạo một biểu đồ về các thuật toán phân cụm khác nhau mà bạn có thể khám phá và sử dụng trong môi trường sản xuất. Các thuật toán phân cụm muốn giải quyết những loại vấn đề nào?

## [Bài kiểm tra sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Trước khi bạn áp dụng các thuật toán phân cụm, như chúng ta đã học, tốt nhất là hiểu bản chất của bộ dữ liệu của bạn. Đọc thêm về chủ đề này [tại đây](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Bài viết hữu ích này](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) sẽ hướng dẫn bạn qua các cách mà các thuật toán phân cụm khác nhau hoạt động, dựa trên các hình dạng dữ liệu khác nhau.

## Bài tập

[Nghiên cứu các hình thức trực quan hóa khác cho phân cụm](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Tuyên bố miễn trừ trách nhiệm**:
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng bản dịch tự động có thể chứa lỗi hoặc sai sót. Tài liệu gốc bằng ngôn ngữ gốc nên được coi là nguồn tin chính thức. Đối với thông tin quan trọng, nên sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm về bất kỳ hiểu lầm hoặc giải thích sai nào phát sinh từ việc sử dụng bản dịch này.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->