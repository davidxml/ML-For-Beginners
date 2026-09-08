# Giới thiệu về học máy

## [Bài quiz trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for beginners - Introduction to Machine Learning for Beginners](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML for beginners - Introduction to Machine Learning for Beginners")

> 🎥 Nhấn vào hình ảnh trên để xem video ngắn hướng dẫn bài học này.

Chào mừng bạn đến với khóa học về học máy cổ điển dành cho người mới bắt đầu! Cho dù bạn hoàn toàn mới với chủ đề này, hay là một người thực hành ML có kinh nghiệm muốn làm mới kiến thức ở một lĩnh vực, chúng tôi rất vui được chào đón bạn tham gia! Chúng tôi muốn tạo ra một điểm khởi đầu thân thiện cho việc học ML của bạn và rất mong nhận được, phản hồi, và tiếp thu [phản hồi](https://github.com/microsoft/ML-For-Beginners/discussions) của bạn.

[![Introduction to ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction to ML")

> 🎥 Nhấn vào hình ảnh trên để xem video: John Guttag của MIT giới thiệu về học máy

---
## Bắt đầu với học máy

Trước khi bắt đầu với chương trình học này, bạn cần chuẩn bị máy tính và sẵn sàng chạy các notebook cục bộ.

- **Cấu hình máy của bạn với các video này**. Sử dụng các liên kết dưới đây để học [cách cài đặt Python](https://youtu.be/CXZYvNRIAKM) trên hệ thống của bạn và [cài đặt môi trường soạn thảo mã](https://youtu.be/EU8eayHWoZg) để phát triển.
- **Học Python**. Bạn cũng nên có hiểu biết cơ bản về [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), một ngôn ngữ lập trình hữu ích cho các nhà khoa học dữ liệu mà chúng ta sẽ dùng trong khóa học này.
- **Học Node.js và JavaScript**. Chúng tôi cũng sử dụng JavaScript vài lần trong khóa học khi xây dựng các ứng dụng web, vì vậy bạn cần cài đặt [node](https://nodejs.org) và [npm](https://www.npmjs.com/), cũng như có sẵn [Visual Studio Code](https://code.visualstudio.com/) để phát triển cả Python và JavaScript.
- **Tạo tài khoản GitHub**. Nếu bạn tìm đến chúng tôi ở đây trên [GitHub](https://github.com), có thể bạn đã có tài khoản, nhưng nếu chưa thì hãy tạo một tài khoản và sau đó fork chương trình học này để sử dụng riêng. (Bạn cũng có thể đánh dấu sao cho chúng tôi nhé 😊)
- **Tìm hiểu Scikit-learn**. Làm quen với [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), một bộ thư viện ML mà chúng tôi tham khảo trong các bài học này.

---
## Học máy là gì?

Thuật ngữ 'học máy' là một trong những thuật ngữ phổ biến và được sử dụng nhiều nhất hiện nay. Có khả năng khá cao là bạn đã ít nhất một lần nghe thuật ngữ này nếu bạn có một số kiến thức về công nghệ, bất kể bạn làm việc trong lĩnh vực nào. Tuy nhiên, cơ chế của học máy là điều bí ẩn đối với đa số mọi người. Đối với người mới bắt đầu học máy, chủ đề này đôi khi có thể cảm thấy choáng ngợp. Do đó, điều quan trọng là hiểu học máy thực sự là gì, và học về nó từng bước, thông qua các ví dụ thực tế.

---
## Đường cong "Hype"

![ml hype curve](../../../../translated_images/vi/hype.07183d711a17aafe.webp)

> Google Trends cho thấy 'đường cong hype' gần đây của thuật ngữ 'machine learning'

---
## Một vũ trụ bí ẩn

Chúng ta sống trong một vũ trụ đầy những bí ẩn thú vị. Những nhà khoa học vĩ đại như Stephen Hawking, Albert Einstein, và nhiều người khác đã dành cả cuộc đời mình để tìm kiếm thông tin có ý nghĩa nhằm khám phá những bí ẩn của thế giới xung quanh chúng ta. Đây là điều kiện con người học hỏi: Một đứa trẻ học những điều mới và khám phá cấu trúc của thế giới của chúng theo từng năm khi lớn lên thành người trưởng thành.

---
## Bộ não của trẻ em

Bộ não và các giác quan của trẻ nhận biết các sự vật xung quanh và từ từ học các mẫu ẩn của cuộc sống giúp trẻ tạo ra các quy tắc logic để nhận dạng các mẫu đã học. Quá trình học của bộ não con người khiến con người trở thành sinh vật tinh vi nhất trên thế giới này. Học liên tục bằng cách khám phá các mẫu ẩn rồi đổi mới dựa trên những mẫu đó giúp chúng ta làm cho bản thân trở nên tốt hơn và tốt hơn trong suốt cuộc đời. Khả năng học này và năng lực tiến hóa có liên quan đến một khái niệm gọi là [độ dẻo não](https://www.simplypsychology.org/brain-plasticity.html). Bề ngoài, chúng ta có thể vẽ một số điểm tương đồng truyền cảm hứng giữa quá trình học của bộ não con người và các khái niệm trong học máy.

---
## Bộ não con người

[Bộ não con người](https://www.livescience.com/29365-human-brain.html) nhận biết các sự vật trong thế giới thực, xử lý thông tin nhận biết được, đưa ra quyết định hợp lý và thực hiện các hành động dựa trên hoàn cảnh. Đây chính là những gì ta gọi là hành xử thông minh. Khi chúng ta lập trình một mô hình hành xử thông minh cho máy, đó gọi là trí tuệ nhân tạo (AI).

---
## Một số thuật ngữ

Mặc dù các thuật ngữ có thể gây nhầm lẫn, học máy (ML) là một phần quan trọng của trí tuệ nhân tạo. **ML liên quan đến việc sử dụng các thuật toán chuyên biệt để khai thác thông tin có ý nghĩa và tìm các mẫu ẩn từ dữ liệu để hỗ trợ quá trình ra quyết định hợp lý**.

---
## AI, ML, Học sâu

![AI, ML, deep learning, data science](../../../../translated_images/vi/ai-ml-ds.537ea441b124ebf6.webp)

> Sơ đồ thể hiện mối quan hệ giữa AI, ML, học sâu và khoa học dữ liệu. Đồ họa thông tin bởi [Jen Looper](https://twitter.com/jenlooper) lấy cảm hứng từ [bức tranh này](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Các khái niệm sẽ học

Trong chương trình học này, chúng ta sẽ chỉ trình bày các khái niệm cốt lõi của học máy mà người mới bắt đầu cần biết. Chúng ta gọi đó là 'học máy cổ điển' chủ yếu sử dụng Scikit-learn, một thư viện xuất sắc mà nhiều học sinh dùng để học các kiến thức cơ bản. Để hiểu các khái niệm rộng hơn về trí tuệ nhân tạo hoặc học sâu, kiến thức nền chắc chắn về học máy là điều không thể thiếu, và chúng tôi xin được cung cấp điều đó tại đây.

---
## Trong khóa học này bạn sẽ học:

- các khái niệm cốt lõi của học máy
- lịch sử của ML
- ML và tính công bằng
- các kỹ thuật hồi quy ML
- các kỹ thuật phân loại ML
- các kỹ thuật tập cụm ML
- các kỹ thuật xử lý ngôn ngữ tự nhiên ML
- các kỹ thuật dự báo chuỗi thời gian ML
- học tăng cường
- các ứng dụng thực tế cho ML

---
## Những gì chúng ta sẽ không học

- học sâu
- mạng nơ-ron
- trí tuệ nhân tạo (AI)

Để mang lại trải nghiệm học tốt hơn, chúng ta sẽ tránh các phức tạp của mạng nơ-ron, học sâu - xây dựng mô hình nhiều lớp sử dụng mạng nơ-ron - và AI, những chủ đề này sẽ được thảo luận trong một chương trình học khác. Chúng tôi cũng sẽ cung cấp một chương trình khoa học dữ liệu sắp tới để tập trung vào khía cạnh lớn hơn của lĩnh vực này.

---
## Tại sao nên học học máy?

Từ góc nhìn hệ thống, học máy được định nghĩa là việc tạo ra các hệ thống tự động có thể học các mẫu ẩn từ dữ liệu để hỗ trợ đưa ra các quyết định thông minh.

Động lực này được lấy cảm hứng một cách lỏng lẻo từ cách bộ não con người học một số điều dựa trên dữ liệu nó nhận biết từ thế giới bên ngoài.

✅ Hãy nghĩ trong một phút tại sao một doanh nghiệp lại muốn thử sử dụng chiến lược học máy thay vì tạo một bộ quy tắc được lập trình cứng nhắc.

---
## Tại sao chất lượng dữ liệu quan trọng

Dữ liệu chất lượng cao cải thiện hiệu quả mô hình. Dữ liệu kém chất lượng hoặc nhiễu có thể dẫn đến dự đoán không chính xác, ngay cả khi sử dụng các thuật toán học máy tiên tiến.

---
## Ứng dụng của học máy

Ứng dụng của học máy hiện nay gần như ở khắp mọi nơi, và phổ biến như dữ liệu đang lưu thông trong xã hội chúng ta, được tạo ra bởi điện thoại thông minh, các thiết bị kết nối và các hệ thống khác. Xét về tiềm năng khổng lồ của các thuật toán học máy hiện đại, các nhà nghiên cứu đã khám phá khả năng của chúng trong việc giải quyết các vấn đề thực tế đa chiều và đa ngành với nhiều kết quả tích cực.

---
## Ví dụ về ứng dụng ML

**Bạn có thể sử dụng học máy theo nhiều cách**:

- Để dự đoán khả năng bệnh từ lịch sử hoặc báo cáo y tế của bệnh nhân.
- Dùng dữ liệu thời tiết để dự báo các sự kiện thời tiết.
- Để hiểu tâm trạng của một đoạn văn bản.
- Để phát hiện tin giả nhằm ngăn chặn sự lan truyền tuyên truyền.

Tài chính, kinh tế, khoa học trái đất, không gian, kỹ thuật y sinh, khoa học nhận thức và thậm chí các lĩnh vực nhân văn đã áp dụng học máy để giải quyết các vấn đề nặng về xử lý dữ liệu trong lĩnh vực của mình.

---
## Kết luận

Học máy tự động hóa quá trình khám phá mẫu bằng cách tìm ra các hiểu biết có ý nghĩa từ dữ liệu thực tế hoặc dữ liệu được tạo ra. Nó đã chứng minh giá trị cao trong các ứng dụng kinh doanh, y tế và tài chính, cùng nhiều lĩnh vực khác.

Trong tương lai gần, việc hiểu các kiến thức cơ bản về học máy sẽ trở thành điều bắt buộc đối với mọi người từ bất kỳ lĩnh vực nào do sự phổ biến rộng rãi của nó.

---
# 🚀 Thử thách

Phác thảo, trên giấy hoặc sử dụng ứng dụng trực tuyến như [Excalidraw](https://excalidraw.com/), hiểu biết của bạn về sự khác biệt giữa AI, ML, học sâu, và khoa học dữ liệu. Thêm một vài ý tưởng về các vấn đề mà mỗi kỹ thuật này giỏi giải quyết.

# [Bài quiz sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

---
# Ôn tập & Tự học

Để tìm hiểu thêm về cách bạn có thể làm việc với các thuật toán ML trên đám mây, hãy theo dõi [Lộ trình học](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Tham gia [Lộ trình học](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) về kiến thức cơ bản của ML.

---
# Bài tập

[Khởi động và chạy](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Tuyên bố miễn trừ trách nhiệm**:
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng bản dịch tự động có thể chứa lỗi hoặc sai sót. Tài liệu gốc bằng ngôn ngữ gốc nên được coi là nguồn tin chính thức. Đối với thông tin quan trọng, nên sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm về bất kỳ hiểu lầm hoặc giải thích sai nào phát sinh từ việc sử dụng bản dịch này.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->