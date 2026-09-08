# Makine öğrenimine giriş

## [Ders öncesi quiz](https://ff-quizzes.netlify.app/en/ml/)

---

[![Başlangıç seviyesindekiler için ML - Makine Öğrenimine Giriş](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "Başlangıç seviyesindekiler için ML - Makine Öğrenimine Giriş")

> 🎥 Bu dersten geçen kısa video için yukarıdaki görsele tıklayın.

Başlangıç seviyesindekiler için klasik makine öğrenimi kursuna hoş geldiniz! Bu konuya tamamen yeniyseniz ya da alanınızı tazelemek isteyen deneyimli bir ML uygulayıcısıysanız, bize katıldığınız için mutluyuz! Makine öğrenimi çalışmanız için dostane bir başlangıç noktası oluşturmak istiyoruz ve [geribildirimlerinizi](https://github.com/microsoft/ML-For-Beginners/discussions) değerlendirmeye, cevaplamaya ve dahil etmeye memnun oluruz.

[![Makine Öğrenimine Giriş](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Makine Öğrenimine Giriş")

> 🎥 Aşağıdaki görsele tıklayarak bir video izleyin: MIT'den John Guttag makine öğrenimini tanıtıyor

---
## Makine öğrenimine başlamak

Bu müfredata başlamadan önce, bilgisayarınızın yerel olarak not defterlerini çalıştırmaya hazır olması gerekir.

- **Makinenizi bu videolarla yapılandırın**. Sisteminizde [Python nasıl kurulur](https://youtu.be/CXZYvNRIAKM) öğrenmek ve geliştirme için bir [metin editörü nasıl kurulur](https://youtu.be/EU8eayHWoZg) öğrenmek için aşağıdaki bağlantıları kullanın.
- **Python öğrenin**. Bu derste kullandığımız, veri bilimciler için yararlı bir programlama dili olan [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott) hakkında temel bir anlayışa sahip olmak da önerilir.
- **Node.js ve JavaScript öğrenin**. Bu derste web uygulamaları oluştururken birkaç kez JavaScript de kullanıyoruz, bu nedenle [node](https://nodejs.org) ve [npm](https://www.npmjs.com/) yüklü olmalı ve hem Python hem de JavaScript geliştirme için [Visual Studio Code](https://code.visualstudio.com/) hazır olmalıdır.
- **Bir GitHub hesabı oluşturun**. Bizi burada [GitHub](https://github.com) üzerinde bulduğunuz için zaten bir hesabınız olabilir, ancak yoksa bir hesap oluşturun ve bu müfredatı kendi kullanımınız için fork edin. (Bize yıldız vermekten çekinmeyin 😊)
- **Scikit-learn'u keşfedin**. Bu derslerde referans verdiğimiz bir makine öğrenimi kütüphaneleri seti olan [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) ile tanışın.

---
## Makine öğrenimi nedir?

'Makine öğrenimi' terimi, bugün en popüler ve en sık kullanılan terimlerden biridir. Teknolojiyle bir şekilde aşinalığınız varsa, hangi alanda çalışıyor olursanız olun, bu terimi en az bir kez duymuş olma olasılığınız oldukça yüksektir. Bununla birlikte, makine öğreniminin mekanikleri çoğu kişi için gizemlidir. Makine öğrenimine yeni başlayanlar için konu bazen bunaltıcı gelebilir. Bu nedenle, makine öğreniminin ne olduğunu tam anlamak ve onu pratik örneklerle adım adım öğrenmek önemlidir.

---
## Hype eğrisi

![ml hype curve](../../../../translated_images/tr/hype.07183d711a17aafe.webp)

> Google Trends, 'makine öğrenimi' teriminin yakın zamandaki 'hype eğrisini' gösteriyor

---
## Gizemli bir evren

Büyüleyici gizemlerle dolu bir evrende yaşıyoruz. Stephen Hawking, Albert Einstein gibi büyük bilim insanları, etrafımızdaki dünyanın gizemlerini ortaya çıkaran anlamlı bilgiler aramak için hayatlarını adamışlardır. Bu öğrenme hali doğrudandır: bir çocuk yeni şeyler öğrenir ve yetişkinliğe doğru büyürken dünyasının yapısını yıl yıl keşfeder.

---
## Çocuğun beyni

Bir çocuğun beyni ve duyuları çevresindeki gerçekleri algılar ve yaşamın gizli kalıplarını öğrenir; bunlar, çocuğun öğrendiği kalıpları tanımlamak için mantıklı kurallar oluşturmasına yardımcı olur. İnsan beyninin öğrenme süreci, insanları bu dünyanın en sofistike canlıları yapar. Gizli kalıpları keşfederek ve sonra bu kalıplar üzerinde yenilik yaparak sürekli öğrenmek, kendimizi yaşam boyu daha iyi yapmamızı sağlar. Bu öğrenme kapasitesi ve gelişen yetenek, [beyin plastisitesi](https://www.simplypsychology.org/brain-plasticity.html) adı verilen bir kavramla ilişkilidir. Dıştan bakıldığında, insan beyninin öğrenme süreci ve makine öğrenimi kavramları arasında bazı motive edici benzerlikler çizilebilir.

---
## İnsan beyni

[İnsan beyni](https://www.livescience.com/29365-human-brain.html), gerçek dünyadan şeyleri algılar, algılanan bilgiyi işler, rasyonel kararlar verir ve duruma göre belirli eylemler gerçekleştirir. Buna biz zeki davranmak diyoruz. Zeki davranış sürecinin bir benzerini bir makineye programladığımızda, buna yapay zeka (AI) denir.

---
## Bazı terimler

Terimler karıştırılsa da, makine öğrenimi (ML) yapay zekanın önemli bir alt kümesidir. **ML, algılanan verilerden anlamlı bilgiler keşfetmek ve rasyonel karar verme sürecini desteklemek için gizli kalıpları bulmak üzere özel algoritmalar kullanmakla ilgilenir**.

---
## AI, ML, Derin Öğrenme

![AI, ML, derin öğrenme, veri bilimi](../../../../translated_images/tr/ai-ml-ds.537ea441b124ebf6.webp)

> AI, ML, derin öğrenme ve veri bilimi arasındaki ilişkileri gösteren bir diagram. [Jen Looper](https://twitter.com/jenlooper) tarafından oluşturulmuş, [bu grafik](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining) temel alınmıştır.

---
## Ele alınacak kavramlar

Bu müfredatta, bir başlangıç seviyesinin bilmesi gereken sadece makine öğreniminin temel kavramlarını ele alacağız. Birçok öğrencinin temel becerileri öğrenmek için kullandığı mükemmel bir kütüphane olan Scikit-learn kullanarak esas olarak 'klasik makine öğrenimi'ni işliyoruz. Yapay zekanın ya da derin öğrenmenin daha geniş kavramlarını anlamak için makine öğreniminde sağlam temel bilgi şarttır, bunu burada size sunmak istiyoruz.

---
## Bu derste öğrenecekleriniz:

- makine öğreniminin temel kavramları
- ML tarihçesi
- ML ve adalet
- regresyon ML teknikleri
- sınıflandırma ML teknikleri
- kümeleme ML teknikleri
- doğal dil işleme ML teknikleri
- zaman serisi tahmin ML teknikleri
- pekiştirmeli öğrenme
- ML'nin gerçek dünya uygulamaları

---
## Ele alınmayacaklar

- derin öğrenme
- sinir ağları
- AI

Daha iyi bir öğrenme deneyimi sağlamak için sinir ağlarının karmaşıklığından, 'derin öğrenme' - sinir ağları kullanılarak çok katmanlı model oluşturma - ve AI'dan kaçınacağız; bunları başka bir müfredatta ele alacağız. Ayrıca bu daha büyük alanın veri bilimi yönüne odaklanan ilerleyen bir data science müfredatı sunmayı planlıyoruz.

---
## Neden makine öğrenimi çalışmalısınız?

Sistem perspektifinden makine öğrenimi, zekice karar vermeye yardımcı olmak için verilerden gizli kalıpları öğrenebilen otomatik sistemlerin oluşturulması olarak tanımlanır.

Bu motivasyon, insan beyninin dış dünyadan algıladığı verilere dayanarak bazı şeyleri nasıl öğrendiğiyle gevşekçe ilham alınmıştır.

✅ Bir işletmenin neden katı kurallarla çalışan bir motor yaratmak yerine makine öğrenimi stratejileri kullanmak isteyebileceğini bir dakika düşünün.

---
## Veri kalitesi neden önemlidir?

Yüksek kaliteli veri model performansını artırır. Kötü veya gürültülü veriler, gelişmiş makine öğrenimi algoritmaları kullanılsa bile yanlış tahminlere yol açabilir.

---
## Makine öğrenimi uygulamaları

Makine öğrenimi uygulamaları artık hemen her yerde ve akıllı telefonlarımız, bağlı cihazlarımız ve diğer sistemlerimiz tarafından üretilen veri kadar yaygın. En gelişmiş makine öğrenimi algoritmalarının muazzam potansiyelini göz önünde bulundurarak, araştırmacılar çok boyutlu ve çok disiplinli gerçek yaşam problemlerini büyük olumlu sonuçlarla çözme yeteneklerini keşfetmektedir.

---
## Uygulamalı ML örnekleri

**Makine öğrenimini birçok şekilde kullanabilirsiniz**:

- Bir hastanın tıbbi geçmişi veya raporlarından hastalığın olasılığını tahmin etmek.
- Hava durumu verilerini kullanarak hava olaylarını tahmin etmek.
- Bir metnin duygu durumunu anlamak.
- Yanlış haberleri tespit ederek propaganda yayılmasını durdurmak.

Finans, ekonomi, yer bilimleri, uzay keşfi, biyomedikal mühendislik, bilişsel bilimler ve hatta beşeri bilimler alanları, kendi alanlarındaki zorlu, veri işleme yoğun sorunları çözmek için makine öğrenimini uyarlamıştır.

---
## Sonuç

Makine öğrenimi, gerçek dünya veya üretilmiş verilerden anlamlı içgörüler bularak kalıp keşfi sürecini otomatikleştirir. İş, sağlık ve finans gibi alanlarda kendini çok değerli kanıtlamıştır.

Yakın gelecekte, yaygın kullanımı nedeniyle makine öğreniminin temellerini anlamak herhangi bir alandan insanlar için zorunlu hale gelecektir.

---
# 🚀 Meydan okuma

AI, ML, derin öğrenme ve veri bilimi arasındaki farkları kağıda veya [Excalidraw](https://excalidraw.com/) gibi çevrimiçi bir uygulama kullanarak tasvir edin. Bu tekniklerin her birinin hangi sorunları çözmekte iyi olduğuna dair bazı fikirler ekleyin.

# [Ders sonrası quiz](https://ff-quizzes.netlify.app/en/ml/)

---
# İnceleme & Kendi Kendine Çalışma

Bulutta ML algoritmaları ile nasıl çalışabileceğinizi öğrenmek için bu [Öğrenme Yolu](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott)'nu takip edin.

ML'nin temelleri hakkında bir [Öğrenme Yolu](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott)'na katılın.

---
# Ödev

[Başlayın ve çalıştırın](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Feragatname**:
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba sarf etsek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayınız. Orijinal belge, kendi dilinde yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımı sonucu ortaya çıkabilecek yanlış anlamalardan veya yanlış yorumlamalardan sorumlu değiliz.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->