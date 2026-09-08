# Makine Öğrenimi Teknikleri

Makine öğrenimi modelleri ve kullandıkları verileri oluşturma, kullanma ve sürdürme süreci, birçok diğer geliştirme iş akışından çok farklı bir süreçtir. Bu derste, süreci sadeleştirecek ve bilmeniz gereken temel tekniklerin ana hatlarını çizeceğiz. Şunları yapacaksınız:

- Makine öğreniminin temelini oluşturan süreçleri yüksek seviyede anlayacaksınız.
- 'Modeller', 'tahminler' ve 'eğitim verisi' gibi temel kavramları keşfedeceksiniz.

## [Ders öncesi quiz](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Bu derste ilerlemek için yukarıdaki görsele tıklayın.

## Giriş

Yüksek seviyede, makine öğrenimi (ML) süreçleri oluşturma sanatı birkaç adımdan oluşur:

1. **Soruyu belirleyin**. Çoğu ML süreci, basit bir koşullu program veya kurallara dayalı motorla cevaplanamayan bir soruyu sorarak başlar. Bu sorular genellikle bir veri koleksiyonuna dayanan tahminlerle ilgilidir.
2. **Veri toplayın ve hazırlayın**. Sorunuzu cevaplayabilmek için veriye ihtiyacınız vardır. Verinizin kalitesi ve zaman zaman miktarı, ilk sorunuza ne kadar iyi cevap verebileceğinizi belirler. Veriyi görselleştirmek bu aşamanın önemli bir parçasıdır. Bu aşama ayrıca, bir modeli oluşturmak için veriyi eğitim ve test gruplarına ayırmayı da içerir.
3. **Bir eğitim yöntemi seçin**. Sorunuza ve verinizin doğasına bağlı olarak, verinizi en iyi şekilde yansıtacak ve doğru tahminler yapacak bir modeli nasıl eğitmek istediğinizi seçmeniz gerekir. Bu, ML sürecinizin belirli uzmanlık gerektiren ve genellikle önemli miktarda deney gerektiren kısmıdır.
4. **Modeli eğitin**. Eğitim verilerinizi kullanarak, verideki kalıpları tanımak için çeşitli algoritmalarla bir modeli eğiteceksiniz. Model, verinin bazı bölümlerini diğerlerine göre daha öncelikli kılmak için ayarlanabilen iç ağırlıkları kullanabilir.
5. **Modeli değerlendirin**. Daha önce hiç görmediğiniz veriyi (test verilerinizi) kullanarak modelin performansını kontrol edersiniz.
6. **Parametre ayarlaması yapın**. Modelinizin performansına dayanarak, modeli eğitmek için kullanılan algoritmaların davranışını kontrol eden farklı parametrelerle veya değişkenlerle süreci yeniden yapabilirsiniz.
7. **Tahmin yapın**. Modelinizin doğruluğunu test etmek için yeni girişler kullanın.

## Hangi soruyu sormalı

Bilgisayarlar, veride gizli kalmış kalıpları keşfetme konusunda özellikle iyidir. Bu fayda, belirli bir alanla ilgili soruları koşullu kurallara dayalı bir motor oluşturmakla kolayca cevaplanamayan araştırmacılar için çok değerlidir. Örneğin, bir aktüeryal görevde, bir veri bilimci sigara içenler ve içmeyenlerin ölüm oranı hakkında el yapımı kurallar oluşturabilir.

Ancak, birçok başka değişken bu denklemde yer aldığında, ML modeli geçmiş sağlık geçmişine dayalı gelecekteki ölüm oranlarını tahmin etmek için daha verimli olabilir. Daha neşeli bir örnek, enlem, boylam, iklim değişikliği, okyanusa yakınlık, jet akımlarının desenleri ve daha fazlasını içeren verilere dayalı olarak belirli bir konumda Nisan ayı hava tahminleri yapmak olabilir.

✅ Bu [sunum dizisi](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) hava modelleri hakkında ML kullanımı için tarihsel bir perspektif sunar.

## Ön hazırlık görevleri

Modelinizi oluşturmaya başlamadan önce tamamlamanız gereken birkaç görev vardır. Sorunuzu test etmek ve modelin tahminlerine dayanan bir hipotez oluşturmak için çeşitli unsurları tanımlamanız ve yapılandırmanız gerekir.

### Veri

Sorunuza belirli bir kesinlikle cevap verebilmek için doğru türde yeterli miktarda veriye ihtiyacınız vardır. Bu noktada yapmanız gereken iki şey vardır:

- **Veri toplayın**. Veri analizinde adalet hakkında önceki dersi aklınızda tutarak verinizi dikkatlice toplayın. Bu verinin kaynaklarının farkında olun, sahip olabileceği herhangi bir yerleşik önyargıyı göz önünde bulundurun ve kökenini belgeleyin.
- **Veriyi hazırlayın**. Veri hazırlama sürecinde birkaç adım vardır. Veriler çeşitli kaynaklardan geliyorsa onları toplamanız ve normalleştirmeniz gerekebilir. Verinin kalitesini ve miktarını, [Kümeleme](../../5-Clustering/1-Visualize/README.md) dersinde yaptığımız gibi dizeleri sayılara dönüştürmek gibi çeşitli yöntemlerle artırabilirsiniz. Ayrıca, orijinal veriye dayalı yeni veriler oluşturabilirsiniz ([Sınıflandırma](../../4-Classification/1-Introduction/README.md) dersinde yaptığımız gibi). Veriyi temizleyebilir ve düzenleyebilirsiniz ([Web Uygulaması](../../3-Web-App/README.md) dersinden önceki gibi). Son olarak, eğitim tekniklerinize bağlı olarak veriyi rastgele karıştırmanız ve karıştırmanız gerekebilir.

✅ Verinizi topladıktan ve işledikten sonra, şeklinin sorunuza cevap vermenize izin verip vermediğine bakmak için bir an durun. Verinin görevde iyi performans göstermeyebileceği gibi durumlarla [Kümeleme](../../5-Clustering/1-Visualize/README.md) derslerinde karşılaşıyoruz!

### Özellikler ve Hedef

[Özellik](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection), verinizin ölçülebilir bir niteliğidir. Pek çok veri setinde bu, 'tarih', 'boyut' veya 'renk' gibi bir sütun başlığı olarak ifade edilir. Özellik değişkeniniz, genellikle kodda `X` ile gösterilir, bir modeli eğitmek için kullanılan giriş değişkenini temsil eder.

Hedef, tahmin etmeye çalıştığınız şeydir. Hedef, genellikle kodda `y` ile gösterilir, verinizden sormaya çalıştığınız sorunun cevabını temsil eder: Aralık ayında en ucuz balkabağı **rengi** ne olacak? San Francisco'da en iyi gayrimenkul **fiyatına** sahip mahalleler hangileri olacak? Bazen hedef bir etiket (label) özniteliği olarak da adlandırılır.

### Özellik değişkeninizi seçmek

🎓 **Özellik Seçimi ve Özellik Çıkarımı** Bir model kurarken hangi değişkeni seçeceğinizi nasıl biliyorsunuz? Muhtemelen en etkili modeli oluşturmak için doğru değişkenleri seçmek üzere bir özellik seçimi veya özellik çıkarımı sürecinden geçeceksiniz. Ancak bu ikisi aynı şey değildir: "Özellik çıkarımı, orijinal özelliklerin fonksiyonlarından yeni özellikler oluştururken, özellik seçimi özelliklerin bir alt kümesini döndürür." ([kaynak](https://wikipedia.org/wiki/Feature_selection))

### Verinizi görselleştirin

Bir veri bilimcisinin araç setinin önemli bir yönü, Seaborn veya MatPlotLib gibi birkaç mükemmel kütüphane kullanarak veriyi görselleştirme gücüdür. Verinizi görsel olarak temsil etmek, kullanabileceğiniz gizli korelasyonları ortaya çıkarabilir. Görselleştirmeleriniz ayrıca önyargıyı veya dengesiz veriyi ortaya çıkarabilir (bunu [Sınıflandırma](../../4-Classification/2-Classifiers-1/README.md) dersinde keşfediyoruz).

### Veri setinizi bölün

Eğitime başlamadan önce, veri setinizi eşit olmayan ancak veriyi iyi temsil eden iki veya daha fazla parçaya bölmeniz gerekir.

- **Eğitim**. Veri setinin bu kısmı, modeli eğitmek için kullanılır. Bu set orijinal veri setinin çoğunluğunu oluşturur.
- **Test**. Test veri seti, genellikle orijinal veriden toplanan bağımsız bir veri grubudur ve oluşturulan modelin performansını doğrulamak için kullanılır.
- **Doğrulama**. Doğrulama seti, modeli iyileştirmek için modelin hiperparametrelerini veya mimarisini ayarlamak için kullanılan daha küçük bağımsız bir örnek grubudur. Verinizin boyutuna ve sorduğunuz soruya bağlı olarak bu üçüncü seti oluşturmanız gerekmeyebilir ([Zaman Serisi Tahmini](../../7-TimeSeries/1-Introduction/README.md) dersinde not ettiğimiz gibi).

## Model oluşturma

Eğitim verinizi kullanarak, modeli **eğitmek** için çeşitli algoritmalar kullanarak verinizin istatistiksel bir temsilini oluşturmayı hedeflersiniz. Bir modeli eğitmek, onu veriye maruz bırakır ve keşfettiği, doğruladığı ve kabul ya da reddettiği algılanan kalıplar hakkında varsayımlarda bulunmasını sağlar.

### Bir eğitim yöntemi seçin

Sorunuza ve verinizin doğasına bağlı olarak, onu eğitmek için bir yöntem seçeceksiniz. Bu derste kullandığımız [Scikit-learn dokümantasyonu](https://scikit-learn.org/stable/user_guide.html)'nu adım adım inceleyerek bir modeli eğitmenin birçok yolunu keşfedebilirsiniz. Deneyiminize bağlı olarak, en iyi modeli oluşturmak için birkaç farklı yöntemi denemeniz gerekebilir. Veri bilimcilerin, modele hiç görmediği veriyi vererek performansını değerlendirdiği, doğruluğu, önyargıyı ve diğer kaliteyi düşüren sorunları kontrol ettiği ve görev için en uygun eğitim yöntemini seçtiği bir süreci muhtemelen yaşayacaksınız.

### Modeli eğitin

Eğitim verinizle hazırlanmış olarak, modeli oluşturmak için 'fit' işlemini gerçekleştirmeye hazırsınız. Pek çok ML kütüphanesinde 'model.fit' kodunu göreceksiniz - bu aşamada genellikle özellik değişkeninizi (`X`) ve hedef değişkeninizi (`y`) değerler dizisi olarak gönderirsiniz.

### Modeli değerlendirin

Eğitim süreci tamamlandıktan sonra (büyük bir modeli eğitmek için birçok yineleme veya 'epoch' gerekebilir), test verilerini kullanarak modelin kalitesini değerlendirebilirsiniz. Bu veriler, modelin daha önce analiz etmediği orijinal verinin bir alt kümesidir. Modelinizin kalitesi hakkında metriklerden oluşan bir tablo yazdırabilirsiniz.

🎓 **Model uyumu**

Makine öğrenimi bağlamında model uyumu, modelin bilinmeyen verileri analiz etmeye çalışırken altında yatan fonksiyonunun doğruluğunu ifade eder.

🎓 **Eksik uyum** ve **aşırı uyum** yaygın sorunlardır ve modelin kalitesini düşürür; model ya yeterince iyi uymamış ya da aşırı iyi uymuş olur. Bu, modelin tahminlerini eğitim verisiyle ya çok sıkı ya da çok gevşek hizalanmış şekilde yapmasına neden olur. Aşırı uyumlu model, verinin detaylarını ve gürültüyü çok iyi öğrendiği için eğitim verisini çok iyi tahmin eder. Eksik uyumlu model ise ne eğitim verisini ne de daha önce 'görmediği' veriyi doğru analiz edebilir.

![aşırı uyumlu model](../../../../translated_images/tr/overfitting.1c132d92bfd93cb6.webp)
> Bilgi grafiği: [Jen Looper](https://twitter.com/jenlooper)

## Parametre ayarlaması

İlk eğitiminiz tamamlandıktan sonra, modelin kalitesini gözlemleyin ve 'hiperparametrelerini' ayarlayarak geliştirmeyi düşünün. Sürecin daha fazlası için [dokümantasyona bakabilirsiniz](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Tahmin

Bu, modelinizin doğruluğunu tamamen yeni verilerle test edebileceğiniz andır. Üretimde modeli kullanmak üzere web varlıkları oluşturduğunuz 'uygulamalı' ML ortamında, bu süreç modelin çıkarım veya değerlendirme için değişkeni ayarlayıp modele göndermek üzere (örneğin bir düğmeye basılması gibi) kullanıcı girdisi toplama içerebilir.

Bu derslerde, bir veri bilimcisinin tüm jestlerini ve daha fazlasını — model hazırlamayı, oluşturmayı, test etmeyi, değerlendirmeyi ve tahmin yapmayı — öğrenerek 'full stack' ML mühendisi olma yolunda ilerleyeceksiniz.

---

## 🚀Meydan Okuma

Bir ML uygulayıcısının adımlarını yansıtan bir akış şeması çizin. Şu anda sürecin hangi aşamasında olduğunuzu düşünüyorsunuz? Hangi aşamada zorlanacağınızı tahmin ediyorsunuz? Size kolay gelen şeyler neler?

## [Ders sonrası quiz](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme & Kendi Kendine Çalışma

Günlük işlerinden bahseden veri bilimcilerle yapılmış çevrimiçi röportajlar arayın. İşte [biri](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Ödev

[Bir veri bilimci ile röportaj yapın](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Feragatname**:  
Bu belge, AI çeviri servisi [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba sarf etsek de, otomatik çevirilerin hatalar veya yanlışlıklar içerebileceğinin farkında olunuz. Orijinal belge, kendi ana dilindeki haliyle yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanılmasıyla ortaya çıkabilecek yanlış anlamalar veya yanlış yorumlamalar nedeniyle sorumluluk kabul edilmeyecektir.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->