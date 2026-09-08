# Mutfak sınıflandırıcıları 1

Bu derste, önceki dersten kaydettiğiniz, dengeli ve temiz verilerle dolu mutfaklar hakkındaki veri kümesini kullanacaksınız.

Bu veri kümesini, _bir grup malzemeye dayalı olarak belirli bir ulusal mutfağı tahmin etmek_ için çeşitli sınıflandırıcılarla kullanacaksınız. Bunu yaparken, algoritmaların sınıflandırma görevleri için nasıl kullanılabileceğine dair daha fazla bilgi edineceksiniz.

## [Ders öncesi quiz](https://ff-quizzes.netlify.app/en/ml/)
# Hazırlık

[1. Dersi](../1-Introduction/README.md) tamamladığınızı varsayarak, kök `/data` klasöründe bu dört ders için bir _cleaned_cuisines.csv_ dosyasının bulunduğundan emin olun.

## Alıştırma - ulusal mutfağı tahmin etme

1. Bu dersin _notebook.ipynb_ klasöründe çalışarak, o dosyayı ve Pandas kütüphanesini içe aktarın:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Veri şu şekilde görünüyor:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Şimdi birkaç kütüphane daha içe aktarın:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Eğitim için X ve y koordinatlarını iki veri çerçevesine bölün. `cuisine` etiketler veri çerçevesi olabilir:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Şu şekilde görünür:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. `Unnamed: 0` sütununu ve `cuisine` sütununu `drop()` ile düşürün. Kalan veriyi eğitilebilir özellikler olarak kaydedin:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Özellikleriniz şöyle görünür:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Artık modelinizi eğitmeye hazırsınız!

## Sınıflandırıcıyı seçmek

Veriniz temizlenip eğitim için hazır olduğuna göre, işi yapmak için hangi algoritmanın kullanılacağına karar vermeniz gerekiyor.

Scikit-learn, sınıflandırmayı Denetimli Öğrenme altında gruplar ve bu kategoride sınıflandırmak için birçok yol bulabilirsiniz. [Çeşitlilik](https://scikit-learn.org/stable/supervised_learning.html) ilk bakışta oldukça kafa karıştırıcıdır. Aşağıdaki yöntemler hepsi sınıflandırma tekniklerini içerir:

- Doğrusal Modeller
- Destek Vektör Makineleri
- Stokastik Gradyan İnişi
- En Yakın Komşular
- Gaussian İşlemleri
- Karar Ağaçları
- Ensemble yöntemleri (oylama sınıflandırıcısı)
- Çok sınıflı ve çoklu çıktı algoritmaları (çok sınıflı ve çok etiketli sınıflandırma, çok sınıflı-çoklu çıktı sınıflandırma)

> [Sinir ağlarını da veri sınıflandırmak için kullanabilirsiniz](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), ancak bu dersin kapsamı dışındadır.

### Hangi sınıflandırıcıyı seçmeli?

Peki, hangi sınıflandırıcıyı seçmelisiniz? Çoğunlukla, birkaçını çalıştırıp iyi bir sonuç aramak test yapmanın bir yoludur. Scikit-learn, oluşturulmuş bir veri seti üzerinde KNeighbors, SVC iki şekilde, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB ve QuadraticDiscrinationAnalysis'ı karşılaştıran ve sonuçları görselleştiren bir [yan yana karşılaştırma](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) sunar:

![sınıflandırıcı karşılaştırması](../../../../translated_images/tr/comparison.edfab56193a85e7f.webp)
> Scikit-learn dokümantasyonunda oluşturulmuş grafikler

> AutoML, bu karşılaştırmaları bulutta çalıştırarak veriniz için en iyi algoritmayı seçmenize olanak tanıyarak bu problemi kolayca çözer. [Burada deneyin](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Daha iyi bir yaklaşım

Ancak, rastgele tahmin etmekten daha iyi bir yol, indirilip kullanılabilecek bu [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) üzerindeki fikirleri takip etmektir. Burada, çok sınıflı problemimiz için bazı seçeneklere sahibiz:

![çok sınıflı problemler için cheat sheet](../../../../translated_images/tr/cheatsheet.07a475ea444d2223.webp)
> Microsoft’un Algoritma Cheat Sheet’inin bir kısmı, çok sınıflı sınıflandırma seçeneklerine dair detaylar

✅ Bu cheat sheet’i indirin, yazdırın ve duvarınıza asın!

### Mantık yürütme

Elimizdeki kısıtlamalar göz önüne alındığında farklı yaklaşımlar üzerine mantık yürütelim:

- **Sinir ağları çok ağırdır.** Temiz ancak minimal veri setimiz ve eğitimi yerel olarak defterlerde yaptığımız gerçeği göz önüne alındığında, sinir ağları bu görev için çok ağırdır.
- **İkili sınıflandırıcı yok.** İkili sınıflandırıcı kullanmıyoruz, bu yüzden bir-vs-hepsi kuralı dışlanır.
- **Karar ağacı veya lojistik regresyon işe yarayabilir.** Karar ağacı veya çok sınıflı veri için lojistik regresyon çalışabilir.
- **Çok sınıflı güçlendirilmiş karar ağaçları farklı bir problemi çözer.** Çok sınıflı güçlendirilmiş karar ağacı, sıralamalar oluşturmak gibi parametrik olmayan görevler için daha uygundur, bu yüzden bize faydası yoktur.

### Scikit-learn kullanımı

Verimizi analiz etmek için Scikit-learn kullanacağız. Ancak, Scikit-learn’de lojistik regresyon kullanmanın birçok yolu vardır. [Geçirilebilecek parametrelere](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) bakın.

Temelde, iki önemli parametre vardır - `multi_class` ve `solver` - bunları belirtmemiz gerekir, Scikit-learn’den lojistik regresyon yapmasını istediğimizde. `multi_class` değeri belirli bir davranışı uygular. `solver` değeri ise kullanılacak algoritmadır. Tüm çözücüler tüm `multi_class` değerleriyle eşleşmez.

Dokümanlara göre, çok sınıflı durumda, eğitim algoritması:

- `multi_class` seçeneği `ovr` olarak ayarlandıysa **bir-vs-hepsi (OvR) şemasını kullanır**
- `multi_class` seçeneği `multinomial` olarak ayarlandıysa **çapraz entropi kaybını kullanır**. (Şu anda `multinomial` seçeneği sadece ‘lbfgs’, ‘sag’, ‘saga’ ve ‘newton-cg’ çözücüleri tarafından desteklenmektedir.)"

> 🎓 Buradaki 'şema' ‘ovr’ (bir-vs-hepsi) veya ‘multinomial’ olabilir. Lojistik regresyon esasen ikili sınıflandırmayı desteklemek için tasarlandığından, bu şemalar çok sınıflı sınıflandırma görevlerini daha iyi ele almasını sağlar. [kaynak](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Çözücü' optimizasyon problemlerinde kullanılacak algoritma olarak tanımlanır. [kaynak](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn, çözücülerin farklı veri yapılarının sunduğu zorlukları nasıl işlediğini açıklamak için şu tabloyu sunar:

![çözücüler](../../../../translated_images/tr/solvers.5fc648618529e627.webp)

## Alıştırma - veriyi bölme

Önceki derste lojistik regresyondan öğrendiğiniz için, ilk eğitim denememizde lojistik regresyona odaklanabiliriz.
Verinizi `train_test_split()` çağırarak eğitim ve test gruplarına bölün:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Alıştırma - lojistik regresyon uygulama

Çok sınıflı durumu kullandığınızdan, hangi _şema_ ile çalışacağınızı ve hangi _çözücü_ 'yü ayarlayacağınızı seçmeniz gerekiyor. Çok sınıflı ayarla ve **liblinear** çözücüsü ile LogisticRegression kullanarak eğitin.

1. `multi_class` değeri `ovr` ve çözücü `liblinear` olacak şekilde bir lojistik regresyon oluşturun:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Çoğunlukla varsayılan olarak ayarlanan `lbfgs` gibi farklı bir çözücü deneyin

    > İhtiyaç duyduğunuzda verilerinizi yassılaştırmak için Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) fonksiyonunu kullanın.

    Doğruluk **%80**'in üzerindedir!

1. Bu modeli bir veri satırını (#50) test ederek iş başında görebilirsiniz:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Sonuç yazdırılır:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Farklı bir satır numarası deneyip sonuçları kontrol edin
1. Daha derine inerek, bu tahminin doğruluğunu kontrol edebilirsiniz:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Sonuç yazdırılır - Hint mutfağı en iyi tahmin olarak, iyi bir olasılıkla:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Modelin bunun Hint mutfağı olduğundan neden oldukça emin olduğunu açıklayabilir misiniz?

1. Regresyon derslerinde yaptığınız gibi bir sınıflandırma raporu yazdırarak daha fazla detay alın:

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

## 🚀Challenge

Bu derste, temizlenmiş verilerinizi kullanarak bir dizi malzemeye dayanarak ulusal bir mutfağı tahmin edebilen bir makine öğrenimi modeli oluşturdunuz. Verileri sınıflandırmak için Scikit-learn'ün sunduğu birçok seçeneği okumak için biraz zaman ayırın. Sahne arkasında neler olup bittiğini anlamak için 'solver' kavramını daha derinlemesine inceleyin.

## [Ders sonrası quiz](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme & Kendi Kendine Çalışma

[Lojistik regresyonun matematiğini](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf) biraz daha derinlemesine inceleyin.
## Ödev 

[Çözücüleri inceleyin](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba sarf etsek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayınız. Orijinal belge, bulunduğu ana dilde yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanılması sonucu ortaya çıkabilecek yanlış anlamalar veya yorum hatalarından sorumlu değiliz.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->