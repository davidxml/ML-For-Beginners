# Kümelemeye Giriş

Kümeleme, bir veri kümesinin etiketlenmediğini veya girdilerin önceden tanımlanmış çıktılarla eşleştirilmediğini varsayan bir [Denetimsiz Öğrenme](https://wikipedia.org/wiki/Unsupervised_learning) türüdür. Etiketlenmemiş verileri çeşitli algoritmalarla tarar ve veride algıladığı desenlere göre gruplamalar sağlar.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Video için yukarıdaki görüntüye tıklayın. Kümeleme ile makine öğrenmesini incelerken, bazı Nijeryalı Dance Hall parçalarının tadını çıkarın - bu, PSquare tarafından 2014 yılında yüksek puan alan bir şarkıdır.

## [Ön ders sınavı](https://ff-quizzes.netlify.app/en/ml/)

### Giriş

[Kümeleme](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) veri keşfi için çok faydalıdır. Nijeryalı dinleyicilerin müziği tüketme biçimindeki eğilimleri ve desenleri keşfetmeye yardımcı olup olmadığını görelim.

✅ Kümelemenin kullanım alanları hakkında bir dakika düşünün. Gerçek hayatta, bir çamaşır yığını ve aile üyelerinizin kıyafetlerini ayırmanız gerektiğinde kümeleme olur 🧦👕👖🩲. Veri bilimine gelince, kümeleme bir kullanıcının tercihlerini analiz etmeye veya herhangi bir etiketlenmemiş veri kümesinin özelliklerini belirlemeye çalışırken olur. Kümeleme, bir bakıma, kaosu anlamaya yardımcı olur; çorap çekmecesi gibi.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Video için yukarıdaki görüntüye tıklayın: MIT'den John Guttag kümelemeyi tanıtıyor

Profesyonel bir ortamda, kümeleme piyasa segmentasyonu, hangi yaş gruplarının hangi ürünleri satın aldığını belirlemek gibi şeylerde kullanılabilir. Başka bir kullanım örneği olarak, dolandırıcılığı tespit etmek için kredi kartı işlemlerinin bulunduğu bir veri kümesinde anomali tespiti yapılabilir. Ya da bir grup tıbbi taramada tümörleri belirlemek için kümeleme kullanılabilir.

✅ Bankacılık, e-ticaret veya iş ortamında "doğada" kümelemeyle nasıl karşılaşmış olabileceğinizi bir dakika düşünün.

> 🎓 İlginçtir ki, küme analizleri 1930'larda Antropoloji ve Psikoloji alanlarında ortaya çıkmıştır. Nasıl kullanılmış olabileceğini hayal edebiliyor musunuz?

Alternatif olarak, örneğin alışveriş bağlantıları, resimler veya yorumlar gibi arama sonuçlarını gruplamak için kullanabilirsiniz. Kümeleme, büyük bir veri kümeniz olduğunda ve daha detaylı analiz yapmak istediğinizde faydalıdır, böylece diğer modeller oluşturulmadan önce veriler hakkında öğrenme yapılabilir.

✅ Verileriniz kümelere organize edildikten sonra, küme kimliği atarsınız ve bu teknik, bir veri kümesinin gizliliğini korumada faydalı olabilir; bir veri noktasına daha fazla açıklayıcı tanımlayıcı veri yerine küme kimliğiyle referans verebilirsiniz. Bir küme kimliğine, kümeyi tanımlamak için diğer unsurlardan daha çok neden başvurmak isteyebileceğinize dair başka nedenler düşünebiliyor musunuz?

Kümeleme tekniklerini bu [Öğrenme modülünde](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott) derinlemesine inceleyin.

## Kümelemeye Başlamak

[Scikit-learn, kümeleme yapmak için geniş bir yöntem dizisi sunar](https://scikit-learn.org/stable/modules/clustering.html). Seçeceğiniz tür kullanım durumunuza bağlıdır. Dokümantasyona göre, her yöntemin çeşitli faydaları vardır. İşte Scikit-learn tarafından desteklenen yöntemlerin ve uygun kullanım durumlarının basit bir tablosu:

| Yöntem Adı                  | Kullanım Durumu                                                      |
| :--------------------------- | :------------------------------------------------------------------ |
| K-Ortalamalar               | genel amaçlı, tümevarımsal                                          |
| Affinity propagation         | çok sayıda, düzensiz kümeler, tümevarımsal                         |
| Mean-shift                   | çok sayıda, düzensiz kümeler, tümevarımsal                         |
| Spektral kümeleme           | az sayıda, düzenli kümeler, dönüştürücü                             |
| Ward hiyerarşik kümeleme    | çok sayıda, kısıtlanmış kümeler, dönüştürücü                       |
| Agglomerative kümeleme      | çok sayıda, kısıtlanmış, Öklidyen olmayan mesafeler, dönüştürücü   |
| DBSCAN                      | düz olmayan geometri, düzensiz kümeler, dönüştürücü                |
| OPTICS                      | düz olmayan geometri, değişken yoğunluklu düzensiz kümeler, dönüştürücü |
| Gauss karışımları           | düz geometri, tümevarımsal                                         |
| BIRCH                       | aykırı değer içeren büyük veri seti, tümevarımsal                   |

> 🎓 Kümeler oluşturma şeklimiz, veri noktalarını gruplara nasıl topladığımızla çok ilgilidir. Bazı terimleri açıklayalım:
>
> 🎓 ['Dönüştürücü' vs. 'Tümevarımsal'](https://wikipedia.org/wiki/Transduction_(machine_learning))
>
> Dönüştürücü çıkarım, belirli test durumlarına eşlenen gözlemlenmiş eğitim örneklerinden türetilir. Tümevarımsal çıkarım ise öncelikle genel kurallara eşlenen eğitim örneklerinden türetilir ve sonra bu kurallar test örneklerine uygulanır.
>
> Bir örnek: Etiketleri kısmen bulunan bir veri kümeniz olsun. Bazıları 'plak', bazıları 'cd', bazıları boş. Göreviniz boşlara etiket vermek. Tümevarımsal yaklaşımı seçerseniz, 'plak' ve 'cd' arayan bir model eğitirsiniz ve bu etiketleri etiketlenmemiş verilere uygularsınız. Bu yöntem, aslında 'kaset' olanları sınıflandırmakta zorlanır. Dönüştürücü yaklaşım ise bilinmeyen bu verileri, benzer öğeleri bir araya getirip gruplandırarak ve ardından gruba etiket atayarak daha etkili işler. Bu durumda kümeler 'yuvarlak müzik şeyleri' ve 'kare müzik şeyleri' şeklinde olabilir.
>
> 🎓 ['Düz' vs. 'Düz olmayan' geometri](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
>
> Matematiksel terimlerden türetilmiş, düz ve düz olmayan geometri, noktalar arasındaki mesafelerin ya 'düz' ([Öklidyen](https://wikipedia.org/wiki/Euclidean_geometry)) veya 'düz olmayan' (Öklidyen olmayan) geometrik yöntemlerle ölçülmesini ifade eder.
>
> Buradaki 'düz', Öklidyen geometriyi (bir kısmı 'düzlem' geometri olarak öğretilir) ifade eder, düz olmayan ise Öklidyen olmayan geometridir. Geometrinin makine öğrenmesi ile ne ilgisi var? Her iki alan da matematiğe dayandığından, kümelerdeki noktalar arasındaki mesafeleri ölçmek için ortak bir yol olmalıdır ve bu, verinin doğasına bağlı olarak düz veya düz olmayan şekilde yapılabilir. [Öklidyen mesafeler](https://wikipedia.org/wiki/Euclidean_distance), iki nokta arasındaki doğru parçasının uzunluğudur. [Öklidyen olmayan mesafeler](https://wikipedia.org/wiki/Non-Euclidean_geometry) ise bir eğri boyunca ölçülür. Veriniz, görselleştirildiğinde bir düzlemde değilse, bunu işlemek için özel bir algoritma gerekebilir.
>
![Düz ve Düz Olmayan Geometri Bilgi Grafiği](../../../../translated_images/tr/flat-nonflat.d1c8c6e2a96110c1.webp)
> Bilgi grafiği: [Dasani Madipalli](https://twitter.com/dasani_decoded)
>
> 🎓 ['Mesafeler'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
>
> Kümeler, mesafe matrisleri ile tanımlanır; örneğin noktalar arasındaki mesafeler. Bu mesafe birkaç şekilde ölçülebilir. Öklidyen kümeler, nokta değerlerinin ortalaması ile tanımlanır ve bir 'merkez' noktası (centroid) içerir. Mesafeler bu merkeze olan uzaklıkla ölçülür. Öklidyen olmayan mesafeler ise 'kümeidroid' denen, diğer noktalara en yakın nokta ile tanımlanır. Kümeidroidler çeşitli şekillerde tanımlanabilir.
>
> 🎓 ['Kısıtlı'](https://wikipedia.org/wiki/Constrained_clustering)
>
> [Kısıtlı Kümeleme](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf), denetimsiz yönteme 'yarı denetimli' öğrenmeyi tanıtır. Noktalar arasındaki ilişkiler 'bağlanamaz' veya 'zorunlu bağlanır' olarak işaretlenerek veri setine bazı kurallar getirilir.
>
> Bir örnek: Bir algoritma, etiketsiz veya yarı etiketli bir veri kümesine serbest bırakılırsa, ürettiği kümeler düşük kalitede olabilir. Yukarıdaki örnekte kümeler 'yuvarlak müzik şeyleri', 'kare müzik şeyleri', 'üçgen şeyler' ve 'kurabiyeler' şeklinde gruplanabilir. Bazı kısıtlamalar veya izlenecek kurallar verilirse ("ürün plastiğe yapılmalı", "ürün müzik üretebilmeli"), bu algoritmanın daha iyi seçimler yapmasını sağlar.
>
> 🎓 'Yoğunluk'
>
> 'Gürültülü' veri, 'yoğun' olarak kabul edilir. Kümelerindeki noktalar arasındaki mesafeler incelendiğinde daha seyrek veya daha yoğun, yani 'kalabalık' olabilir ve bu nedenle veri uygun kümeleme yöntemiyle analiz edilmelidir. [Bu makale](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html), gürültülü ve düzensiz küme yoğunluğuna sahip bir veri setini keşfetmek için K-Ortalamalar kümeleme ile HDBSCAN algoritmalarının farkını göstermektedir.

## Kümeleme Algoritmaları

100’den fazla kümeleme algoritması bulunmaktadır ve kullanımları mevcut verinin doğasına bağlıdır. Bazı büyük algoritmalara bakalım:

- **Hiyerarşik kümeleme**. Bir nesne, daha uzak olan yerine yakın bir nesneye göre sınıflandırılırsa, kümeler üyelerinin diğer nesnelere olan mesafesine dayanarak oluşur. Scikit-learn’un aglomeratif kümelemesi hiyerarşiktir.

   ![Hiyerarşik kümeleme Bilgi Grafiği](../../../../translated_images/tr/hierarchical.bf59403aa43c8c47.webp)
   > Bilgi grafiği: [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Merkez noktası kümelemesi**. Bu popüler algoritma, oluşturulacak küme sayısı 'k' seçimini gerektirir, ardından algoritma kümenin merkez noktasını belirler ve verileri bu noktanın etrafında toplar. [K-ortalama kümelenmesi](https://wikipedia.org/wiki/K-means_clustering), merkez noktası kümelemenin popüler bir versiyonudur. Merkez, en yakın ortalamaya göre belirlenir, bu yüzden adı böyledir. Kümeden olan karesel uzaklık minimize edilir.

   ![Merkez noktası kümeleme Bilgi Grafiği](../../../../translated_images/tr/centroid.097fde836cf6c918.webp)
   > Bilgi grafiği: [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Dağılıma dayalı kümeleme**. İstatistiksel modellemeye dayanan bu tür, bir veri noktasının kümeye ait olma olasılığını belirlemeye ve ona göre atamaya odaklanır. Gauss karışımı yöntemleri bu türe aittir.

- **Yoğunluğa dayalı kümeleme**. Veri noktaları, kendi aralarındaki yoğunluklarına veya birbirlerinin etrafında gruplanmalarına göre kümelere atanır. Grubun çok uzağındaki veri noktaları aykırı değer veya gürültü olarak kabul edilir. DBSCAN, Mean-shift ve OPTICS bu tür kümelemeye örnektir.

- **Kafes tabanlı kümeleme**. Çok boyutlu veri setleri için bir kafes oluşturulur ve veriler kafesin hücrelerine bölünerek kümeler oluşturulur.

## Alıştırma - Verinizi Kümeleyin

Kümeleme tekniği, uygun görselleştirmeyle çok desteklenir; bu yüzden müzik verimizi görselleştirmekle başlayalım. Bu alıştırma, verinin doğasına göre hangi kümeleme yöntemini en iyi şekilde kullanmamız gerektiğine karar vermemize yardımcı olacak.

1. Bu klasördeki [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) dosyasını açın.

1. İyi veri görselleştirmesi için `Seaborn` paketini içe aktarın.

    ```python
    !pip install seaborn
    ```

1. [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv) dosyasından şarkı verilerini ekleyin. Şarkılar hakkında bazı verilerle bir dataframe yükleyin. Kütüphaneleri içe aktararak ve verileri dökerek keşfe hazır olun:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Verinin ilk birkaç satırını kontrol edin:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Dataframe hakkında biraz bilgi alın, `info()` çağırarak:

    ```python
    df.info()
    ```

   Çıktı şu şekilde görünür:

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

1. Null değerler için iki kez kontrol yapın, `isnull()` çağırarak toplamın 0 olduğunu doğrulayın:

    ```python
    df.isnull().sum()
    ```

    Sorun görünmüyor:

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

1. Veriyi tanımlayın:

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

> 🤔 Eğer üzerinde çalıştığımız kümeleme, etiketlenmiş veriye ihtiyaç duymayan denetimsiz bir yöntem ise, neden bu verileri etiketlerle gösteriyoruz? Veri keşif aşamasında bunlar kullanışlıdır, ancak kümeleme algoritmalarının çalışması için gerekli değillerdir. Sütun başlıklarını kaldırabilir ve veriye sütun numarası ile başvurabilirsiniz.

Verinin genel değerlerine bakın. Popülerlik değerinin '0' olabileceğini unutmayın, bu da sıralaması olmayan şarkıları gösterir. Bu tür kayıtları kısa sürede kaldıracağız.

1. En popüler türleri bulmak için bir barplot kullanın:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/tr/popular.9c48d84b3386705f.webp)

✅ Daha fazla üst değer görmek isterseniz, `[:5]` ifadesini daha büyük bir değer ile değiştirebilir veya tamamen kaldırabilirsiniz.

Unutmayın, en popüler tür 'Missing' olarak tanımlanıyorsa, Spotify'ın onu sınıflandırmadığı anlamına gelir; bu yüzden bundan kurtulalım.

1. Eksik verilerden kurtulmak için filtre uygulayın:

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Şimdi türlere tekrar bakın:

    ![most popular](../../../../translated_images/tr/all-genres.1d56ef06cefbfcd6.webp)

1. Açıkça, en üst üç tür bu veri setine hakim. `afro dancehall`, `afropop` ve `nigerian pop` türlerine odaklanalım, ayrıca popülerlik değeri 0 olanları filtreleyelim (bu veride popülerlik ile sınıflandırılmamış ve amaçlarımız için gürültü olarak kabul edilebilir):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Verinin herhangi kuvvetli bir şekilde korelasyon gösterip göstermediğini hızlıca test edin:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/tr/correlation.a9356bb798f5eea5.webp)

    Tek güçlü korelasyon `energy` ve `loudness` arasında, bu da çok şaşırtıcı değil çünkü yüksek sesli müzikler genellikle oldukça enerjiktir. Diğer korelasyonlar nispeten zayıf. Bir kümeleme algoritmasının bu veriden ne çıkaracağını görmek ilginç olacak.

    > 🎓 Korelasyon nedensellik anlamına gelmez! Korelasyon kanıtımız var ama nedensellik kanıtımız yok. [Komik bir web sitesi](https://tylervigen.com/spurious-correlations) bu noktayı vurgulayan görseller içeriyor.

Bu dataset'te şarkının algılanan popülerliği ile dans edilebilirlik arasında bir yakınsama var mı? Bir FacetGrid, türden bağımsız olarak hizalanan iç içe halkalar olduğunu gösteriyor. Bu tür için Nijerya zevklerinin belirli bir dans edilebilirlik seviyesinde yakınsaması olabilir mi?

✅ Farklı veri noktaları (energy, loudness, speechiness) ve daha fazla veya farklı müzik türleri deneyin. Neler keşfedebilirsiniz? Veri noktalarının genel yayılımını görmek için `df.describe()` tablosuna bakın.

### Egzersiz - veri dağılımı

Bu üç tür, popülerliklerine göre dans edilebilirlik algısında anlamlı farklılık gösteriyor mu?

1. Üç en iyi türün popülerlik ve dans edilebilirlik veri dağılımını, verilen x ve y eksenlerinde inceleyin.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Genel bir yakınsama noktasının etrafında iç içe halkalar keşfedebilirsiniz, bu da noktaların dağılımını gösterir.

    > 🎓 Bu örnek, veriyi sürekli bir olasılık yoğunluk eğrisi kullanarak temsil eden bir KDE (Kernel Yoğunluk Tahmini) grafiği kullanır. Bu, birden fazla dağılımla çalışırken veriyi yorumlamamızı sağlar.

    Genel olarak, üç tür popülerlik ve dans edilebilirlik açısından gevşek bir şekilde hizalanmıştır. Bu gevşek hizalanmış veride kümeleri belirlemek zor olacaktır:

    ![distribution](../../../../translated_images/tr/distribution.9be11df42356ca95.webp)

1. Bir scatter plot (dağılım grafiği) oluşturun:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Aynı eksenlerin scatterplot'u benzer bir yakınsama deseni gösteriyor

    ![Facetgrid](../../../../translated_images/tr/facetgrid.9b2e65ce707eba1f.webp)

Genel olarak, kümeleme için veri kümelerini göstermek amacıyla scatterplotlar kullanılabilir, bu tür görselleştirmede ustalaşmak çok faydalıdır. Sonraki derste, bu filtrelenmiş veriyi kullanarak k-means kümeleme algoritmasıyla bu veride ilgi çekici şekilde örtüşen gruplar keşfedeceğiz.

---

## 🚀Meydan Okuma

Bir sonraki derse hazırlık olarak, üretim ortamında keşfedip kullanabileceğiniz çeşitli kümeleme algoritmaları hakkında bir grafik hazırlayın. Kümeleme hangi tür problemleri çözmeye çalışıyor?

## [Ders Sonrası Quiz](https://ff-quizzes.netlify.app/en/ml/)

## İnceleme & Kendi Kendine Çalışma

Kümeleme algoritmalarını uygulamadan önce, öğrendiğimiz gibi, veri setinizin doğasını anlamak iyi bir fikirdir. Bu konu hakkında daha fazla bilgi edinmek için [buraya](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html) bakabilirsiniz.

[Bu faydalı makale](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) farklı veri şekillerine göre çeşitli kümeleme algoritmalarının nasıl davrandığını açıklamaktadır.

## Ödev

[Kümeleme için diğer görselleştirmeleri araştırın](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Feragatname**:
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba sarf etsek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayınız. Orijinal belge, kendi dilinde yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımı sonucu ortaya çıkabilecek yanlış anlamalardan veya yanlış yorumlamalardan sorumlu değiliz.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->