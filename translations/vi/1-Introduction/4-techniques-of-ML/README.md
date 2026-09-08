# Kỹ thuật của Máy học

Quá trình xây dựng, sử dụng và duy trì các mô hình máy học cùng dữ liệu mà chúng sử dụng là một quy trình rất khác biệt so với nhiều quy trình phát triển khác. Trong bài học này, chúng ta sẽ làm sáng tỏ quy trình và phác thảo các kỹ thuật chính mà bạn cần biết. Bạn sẽ:

- Hiểu được các quy trình làm nền tảng cho máy học ở mức cao.
- Khám phá các khái niệm cơ bản như 'mô hình', 'dự đoán', và 'dữ liệu huấn luyện'.

## [Bài kiểm tra trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Nhấn vào hình ảnh ở trên để xem video ngắn hướng dẫn qua bài học này.

## Giới thiệu

Ở mức độ tổng quát, công việc tạo ra các quy trình máy học (ML) bao gồm một số bước:

1. **Xác định câu hỏi**. Hầu hết các quy trình ML bắt đầu bằng việc đặt ra một câu hỏi không thể trả lời bằng một chương trình điều kiện đơn giản hoặc một công cụ dựa trên quy tắc. Những câu hỏi này thường xoay quanh các dự đoán dựa trên một tập hợp dữ liệu.
2. **Thu thập và chuẩn bị dữ liệu**. Để có thể trả lời câu hỏi, bạn cần dữ liệu. Chất lượng và, đôi khi, số lượng dữ liệu của bạn sẽ quyết định mức độ tốt bạn có thể trả lời câu hỏi ban đầu. Việc trực quan hóa dữ liệu là một khía cạnh quan trọng của giai đoạn này. Giai đoạn này cũng bao gồm chia dữ liệu thành nhóm huấn luyện và kiểm tra để xây dựng mô hình.
3. **Chọn phương pháp huấn luyện**. Tùy vào câu hỏi và tính chất của dữ liệu, bạn cần chọn cách huấn luyện một mô hình tốt nhất để phản ánh dữ liệu và đưa ra dự đoán chính xác với nó. Đây là phần trong quy trình ML của bạn đòi hỏi chuyên môn cụ thể và thường là một lượng lớn thử nghiệm.
4. **Huấn luyện mô hình**. Sử dụng dữ liệu huấn luyện, bạn sẽ sử dụng nhiều thuật toán khác nhau để huấn luyện một mô hình nhận diện các mô hình trong dữ liệu. Mô hình có thể sử dụng các trọng số nội bộ có thể điều chỉnh để ưu tiên phần dữ liệu nào đó nhằm xây dựng mô hình tốt hơn.
5. **Đánh giá mô hình**. Bạn sử dụng dữ liệu chưa từng thấy trước đây (dữ liệu kiểm tra) từ bộ dữ liệu đã thu thập để xem mô hình hoạt động ra sao.
6. **Điều chỉnh tham số**. Dựa trên hiệu suất của mô hình, bạn có thể thực hiện lại quy trình sử dụng các tham số khác nhau, hoặc biến số, kiểm soát hành vi của các thuật toán được dùng để huấn luyện mô hình.
7. **Dự đoán**. Sử dụng dữ liệu đầu vào mới để kiểm tra độ chính xác của mô hình.

## Câu hỏi cần đặt ra

Máy tính đặc biệt giỏi trong việc phát hiện các mẫu ẩn trong dữ liệu. Khả năng này rất hữu ích cho các nhà nghiên cứu có câu hỏi về một lĩnh vực nhất định mà không thể dễ dàng trả lời bằng cách tạo ra một công cụ quy tắc dựa trên điều kiện. Ví dụ, với một nhiệm vụ thống kê bảo hiểm, một nhà khoa học dữ liệu có thể xây dựng các quy tắc thủ công xoay quanh tỷ lệ tử vong của người hút thuốc so với người không hút thuốc.

Tuy nhiên, khi nhiều biến số khác được đưa vào phép tính, một mô hình ML có thể chứng minh hiệu quả hơn trong việc dự đoán tỷ lệ tử vong tương lai dựa trên lịch sử sức khỏe trong quá khứ. Một ví dụ vui hơn có thể là đưa ra dự đoán thời tiết cho tháng Tư tại một vị trí nhất định dựa trên dữ liệu gồm vĩ độ, kinh độ, biến đổi khí hậu, vị trí gần biển, các mẫu của dòng jet stream, và nhiều yếu tố khác.

✅ Bộ slide này [slide deck](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) về mô hình thời tiết cung cấp một góc nhìn lịch sử về việc sử dụng ML trong phân tích thời tiết.  

## Các nhiệm vụ trước khi xây dựng

Trước khi bắt đầu xây dựng mô hình, có một số nhiệm vụ bạn cần hoàn thành. Để kiểm tra câu hỏi và hình thành giả thuyết dựa trên dự đoán của mô hình, bạn cần xác định và cấu hình một số yếu tố.

### Dữ liệu

Để có thể trả lời câu hỏi của bạn với bất kỳ độ chắc chắn nào, bạn cần một lượng dữ liệu đủ tốt và đúng loại. Có hai việc bạn cần làm ở giai đoạn này:

- **Thu thập dữ liệu**. Ghi nhớ bài học trước về sự công bằng trong phân tích dữ liệu, hãy thu thập dữ liệu cẩn thận. Hiểu rõ nguồn gốc của dữ liệu này, bất kỳ thành kiến tiềm ẩn nào có thể tồn tại, và ghi chép nguồn gốc của nó.
- **Chuẩn bị dữ liệu**. Có một số bước trong quy trình chuẩn bị dữ liệu. Bạn có thể cần tổng hợp dữ liệu và chuẩn hóa nếu nó đến từ nhiều nguồn khác nhau. Bạn có thể cải thiện chất lượng và số lượng dữ liệu thông qua các phương pháp khác nhau như chuyển đổi chuỗi thành số (như chúng ta làm trong [Clustering](../../5-Clustering/1-Visualize/README.md)). Bạn cũng có thể tạo ra dữ liệu mới, dựa trên dữ liệu gốc (như chúng ta làm trong [Classification](../../4-Classification/1-Introduction/README.md)). Bạn có thể làm sạch và chỉnh sửa dữ liệu (như chúng ta sẽ làm trước bài học [Web App](../../3-Web-App/README.md)). Cuối cùng, bạn có thể cần làm ngẫu nhiên và trộn dữ liệu, tùy thuộc vào kỹ thuật huấn luyện của bạn.

✅ Sau khi thu thập và xử lý dữ liệu, hãy dành chút thời gian để xem liệu hình dạng dữ liệu có cho phép bạn giải quyết câu hỏi dự định hay không. Có thể dữ liệu sẽ không hoạt động tốt trong nhiệm vụ của bạn, như chúng ta phát hiện trong các bài học [Clustering](../../5-Clustering/1-Visualize/README.md)!

### Thuộc tính và Mục tiêu

Một [feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) là một thuộc tính có thể đo lường của dữ liệu. Trong nhiều bộ dữ liệu, nó được biểu thị dưới dạng tiêu đề cột như 'date' 'size' hoặc 'color'. Biến đặc trưng của bạn, thường được biểu diễn là `X` trong mã, biểu thị biến đầu vào sẽ được dùng để huấn luyện mô hình.

Một mục tiêu là một thứ mà bạn đang cố gắng dự đoán. Mục tiêu, thường biểu diễn là `y` trong mã, biểu thị câu trả lời cho câu hỏi bạn đặt ra với dữ liệu của mình: vào tháng Mười Hai, quả bí ngô có **màu sắc** nào sẽ rẻ nhất? ở San Francisco, khu vực nào sẽ có **giá** bất động sản tốt nhất? Đôi khi mục tiêu cũng được gọi là thuộc tính nhãn.

### Chọn biến đặc trưng của bạn

🎓 **Chọn lựa thuộc tính và Trích xuất thuộc tính** Làm thế nào để biết biến nào nên chọn khi xây dựng mô hình? Bạn có thể sẽ trải qua một quy trình chọn thuộc tính hoặc trích xuất thuộc tính để chọn các biến đúng cho mô hình hiệu quả nhất. Tuy nhiên, chúng không giống nhau: "Trích xuất thuộc tính tạo ra các thuộc tính mới từ các hàm của thuộc tính gốc, trong khi chọn thuộc tính trả về một tập con các thuộc tính." ([nguồn](https://wikipedia.org/wiki/Feature_selection))

### Trực quan hóa dữ liệu của bạn

Một khía cạnh quan trọng trong bộ công cụ của nhà khoa học dữ liệu là khả năng trực quan hóa dữ liệu bằng nhiều thư viện tuyệt vời như Seaborn hoặc MatPlotLib. Biểu diễn dữ liệu của bạn một cách trực quan có thể giúp bạn phát hiện các mối tương quan ẩn mà bạn có thể tận dụng. Các trực quan hóa của bạn cũng có thể giúp bạn phát hiện thành kiến hoặc dữ liệu không cân bằng (như chúng ta khám phá trong [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Chia bộ dữ liệu

Trước khi huấn luyện, bạn cần chia bộ dữ liệu thành hai hoặc nhiều phần có kích thước không đều nhưng vẫn đại diện tốt cho dữ liệu.

- **Huấn luyện**. Phần này của bộ dữ liệu được dùng để phù hợp với mô hình, huấn luyện nó. Phần này chiếm đa số trong bộ dữ liệu gốc.
- **Kiểm tra**. Một bộ dữ liệu kiểm tra là một nhóm dữ liệu độc lập, thường lấy từ dữ liệu gốc, mà bạn dùng để xác nhận hiệu suất của mô hình đã xây dựng.
- **Xác thực**. Bộ dữ liệu xác thực là một nhóm mẫu độc lập nhỏ hơn mà bạn dùng để điều chỉnh các siêu tham số, hoặc kiến trúc của mô hình, nhằm cải thiện mô hình. Tùy thuộc vào kích thước dữ liệu và câu hỏi bạn đang đặt ra, bạn có thể không cần xây dựng bộ thứ ba này (như chúng ta lưu ý trong [Dự báo chuỗi thời gian](../../7-TimeSeries/1-Introduction/README.md)).

## Xây dựng mô hình

Sử dụng dữ liệu huấn luyện, mục tiêu của bạn là xây dựng một mô hình, hoặc một biểu diễn thống kê của dữ liệu, sử dụng các thuật toán khác nhau để **huấn luyện** nó. Việc huấn luyện mô hình đưa nó vào tiếp xúc với dữ liệu và cho phép nó đưa ra giả định về các mẫu được nhận diện, xác nhận, và chấp nhận hoặc từ chối.

### Chọn phương pháp huấn luyện

Tùy vào câu hỏi và tính chất của dữ liệu, bạn sẽ chọn một phương pháp để huấn luyện. Khi xem qua [tài liệu của Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - thư viện chúng ta sử dụng trong khóa học này - bạn có thể khám phá nhiều cách huấn luyện mô hình. Tùy vào kinh nghiệm, bạn có thể phải thử nhiều phương pháp khác nhau để xây dựng mô hình tốt nhất. Bạn có thể sẽ trải qua một quy trình mà các nhà khoa học dữ liệu đánh giá hiệu suất của mô hình bằng cách cho mô hình thấy dữ liệu chưa từng nhìn thấy, kiểm tra độ chính xác, thành kiến, và các vấn đề làm giảm chất lượng khác, rồi chọn phương pháp huấn luyện phù hợp nhất với nhiệm vụ.

### Huấn luyện mô hình

Với dữ liệu huấn luyện, bạn sẵn sàng 'fit' nó để tạo ra mô hình. Bạn sẽ nhận thấy rằng trong nhiều thư viện ML thường có dòng mã 'model.fit' - đó chính là lúc bạn gửi vào biến đặc trưng dưới dạng mảng các giá trị (thường là 'X') và biến mục tiêu (thường là 'y').

### Đánh giá mô hình

Khi quy trình huấn luyện hoàn tất (có thể mất nhiều lần lặp, hoặc 'epoch', để huấn luyện một mô hình lớn), bạn có thể đánh giá chất lượng mô hình bằng việc sử dụng dữ liệu kiểm tra đo hiệu suất của nó. Dữ liệu này là một phần nhỏ của dữ liệu gốc mà mô hình chưa từng phân tích trước đây. Bạn có thể in ra một bảng các chỉ số về chất lượng mô hình.

🎓 **Fitting mô hình**

Trong ngữ cảnh máy học, fitting mô hình đề cập đến độ chính xác của hàm số cơ sở của mô hình khi nó cố gắng phân tích dữ liệu mà nó chưa quen thuộc.

🎓 **Underfitting** và **overfitting** là những vấn đề phổ biến làm giảm chất lượng mô hình, khi mô hình fit không đủ tốt hoặc quá tốt. Điều này khiến mô hình đưa ra dự đoán quá sát hoặc quá lỏng với dữ liệu huấn luyện. Mô hình overfit dự đoán dữ liệu huấn luyện quá tốt vì nó đã học quá kỹ các chi tiết và nhiễu của dữ liệu. Mô hình underfit không chính xác vì nó không thể phân tích chính xác dữ liệu huấn luyện hoặc dữ liệu chưa từng thấy.

![overfitting model](../../../../translated_images/vi/overfitting.1c132d92bfd93cb6.webp)
> Đồ họa thông tin bởi [Jen Looper](https://twitter.com/jenlooper)

## Điều chỉnh tham số

Khi huấn luyện ban đầu hoàn tất, quan sát chất lượng mô hình và cân nhắc cải thiện nó bằng cách điều chỉnh các 'siêu tham số' của nó. Đọc thêm về quy trình này [trong tài liệu](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Dự đoán

Đây là thời điểm bạn có thể sử dụng dữ liệu hoàn toàn mới để kiểm tra độ chính xác của mô hình. Trong bối cảnh ML 'ứng dụng', nơi bạn xây dựng các tài nguyên web để sử dụng mô hình trong sản xuất, quy trình này có thể bao gồm việc thu thập dữ liệu đầu vào từ người dùng (ví dụ nhấn nút) để đặt một biến và gửi nó tới mô hình để suy luận hoặc đánh giá.

Trong các bài học này, bạn sẽ khám phá cách sử dụng các bước này để chuẩn bị, xây dựng, kiểm tra, đánh giá, và dự đoán - tất cả những thao tác của một nhà khoa học dữ liệu và hơn thế nữa, khi bạn tiến bộ trên hành trình trở thành kỹ sư ML 'full stack'.

---

## 🚀Thử thách

Vẽ một sơ đồ quy trình phản ánh các bước của một nhà thực hành ML. Bạn đang thấy mình ở bước nào trong quy trình? Bạn dự đoán sẽ gặp khó khăn ở đâu? Điều gì có vẻ dễ với bạn?

## [Bài kiểm tra sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Tìm kiếm trên mạng các phỏng vấn với các nhà khoa học dữ liệu nói về công việc hàng ngày của họ. Đây là [một phỏng vấn](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Bài tập

[Phỏng vấn một nhà khoa học dữ liệu](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Tuyên bố từ chối trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng bản dịch tự động có thể chứa lỗi hoặc sai sót. Tài liệu gốc bằng ngôn ngữ mẹ đẻ của nó nên được coi là nguồn chính xác và có thẩm quyền. Đối với các thông tin quan trọng, nên sử dụng dịch vụ dịch thuật chuyên nghiệp của con người. Chúng tôi không chịu trách nhiệm về bất kỳ hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->