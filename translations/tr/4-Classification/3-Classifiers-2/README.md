# Mutfak sınıflandırıcıları 2

Bu ikinci sınıflandırma dersinde, sayısal verileri sınıflandırmanın daha fazla yolunu keşfedeceksiniz. Ayrıca, bir sınıflandırıcıyı diğerine tercih etmenin sonuçlarını öğreneceksiniz.

## [Ders öncesi quiz](https://ff-quizzes.netlify.app/en/ml/)

### Önkoşul

Önceki dersleri tamamladığınızı ve temizlenmiş bir veri setine sahip olduğunuzu varsayıyoruz. Bu veri seti, bu 4 derslik klasörün kökünde `data` klasöründe _cleaned_cuisines.csv_ olarak yer alıyor.

### Hazırlık

_notebook.ipynb_ dosyanız temizlenmiş veri seti ile yüklendi ve model oluşturma süreci için X ve y veri çerçevelerine bölündü.

## Bir sınıflandırma haritası

Önceden, Microsoft'un hızlı başvuru sayfasını kullanarak veri sınıflandırmada sahip olduğunuz çeşitli seçenekler hakkında bilgi edindiniz. Scikit-learn benzer ancak daha ayrıntılı bir hızlı başvuru sunar ve bu, tahmin edicilerinizi (sınıflandırıcıların başka bir terimi) daha da daraltmanıza yardımcı olabilir:

![Scikit-learn'den ML Haritası](../../../../translated_images/tr/map.e963a6a51349425a.webp)
> İpucu: [bu haritayı çevrimiçi ziyaret edin](https://scikit-learn.org/stable/tutorial/machine_learning_map/) ve dokümantasyona okumak için yol boyunca tıklayın.

### Plan

Bu harita, verilerinizi net olarak anladığınızda çok faydalıdır, çünkü karar vermek için yollarında 'yürüyebilirsiniz':

- 50'den fazla örneğimiz var
- Bir kategori tahmin etmek istiyoruz
- Etiketli verilerimiz var
- 100K'dan daha az örnek var
- ✨ Lineer SVC seçebiliriz
- Eğer bu işe yaramazsa, sayısal verimiz olduğundan
    - ✨ KNeighbors Sınıflandırıcıyı deneyebiliriz
      - Eğer bu da işe yaramazsa, ✨ SVC ve ✨ Topluluk Sınıflandırıcılarını deneyin

Takip etmek için çok faydalı bir yol.

## Alıştırma - veriyi böl

Bu yolu izleyerek kullanmak için bazı kütüphaneleri içe aktarmayla başlamalıyız.

1. Gerekli kütüphaneleri içe aktarın:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

2. Eğitim ve test verilerinizi bölün:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Lineer SVC sınıflandırıcı

Destek Vektör Kümeleme (SVC), ML tekniklerinin Destek Vektör makineleri ailesinin bir üyesidir (aşağıda bunlar hakkında daha fazla bilgi edinin). Bu yöntemde, etiketleri nasıl kümeleneceğine karar vermek için bir 'kernel' seçebilirsiniz. 'C' parametresi, parametrelerin etkisini düzenleyen 'regularizasyon'u ifade eder. Kernel [çeşitli](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) olabilir; burada lineer SVC'den yararlanmak için 'linear' olarak ayarladık. Probability varsayılan olarak 'false' tur; burada olasılık tahminlerini toplamak için 'true' olarak ayarladık. Verileri karıştırmak için random state '0' olarak ayarlandı.

### Alıştırma - lineer SVC uygula

Öncelikle bir sınıflandırıcılar dizisi oluşturun. Test ettikçe bu diziye kademeli olarak ekleme yapacaksınız.

1. Lineer SVC ile başlayın:

    ```python
    C = 10
    # Farklı sınıflandırıcılar oluşturun.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Modelinizi Lineer SVC kullanarak eğitin ve bir rapor yazdırın:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Sonuç oldukça iyi:

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

## K-Komşu sınıflandırıcı

K-Komşu, hem denetimli hem de denetimsiz öğrenmede kullanılabilen "komşular" ailesinin bir parçasıdır. Bu yöntemde, önceden tanımlanmış sayıda nokta oluşturulur ve veriler bu noktaların etrafında toplanır, böylece veri için genelleştirilmiş etiketler tahmin edilebilir.

### Alıştırma - K-Komşu sınıflandırıcıyı uygula

Önceki sınıflandırıcı iyiydi ve veri ile iyi çalıştı, ancak belki daha iyi doğruluk elde edebiliriz. Bir K-Komşu sınıflandırıcı deneyin.

1. Sınıflandırıcı dizinize bir satır ekleyin (Lineer SVC maddesinden sonra virgül koyun):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Sonuç biraz daha kötü:

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

    ✅ [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors) hakkında bilgi edinin

## Destek Vektör Sınıflandırıcı

Destek Vektör sınıflandırıcıları, sınıflandırma ve regresyon görevleri için kullanılan ML yöntemlerinin [Destek Vektör Makinesi](https://wikipedia.org/wiki/Support-vector_machine) ailesinin bir parçasıdır. SVM'ler "eğitim örneklerini uzaydaki noktalara eşler" ve iki kategori arasındaki mesafeyi maksimize eder. Sonraki veriler bu uzaya eşlenir ve kategorileri tahmin edilir.

### Alıştırma - Destek Vektör Sınıflandırıcı uygula

Destek Vektör Sınıflandırıcı ile biraz daha iyi doğruluk elde etmeye çalışalım.

1. K-Komşu maddesinden sonra virgül koyun ve sonra bu satırı ekleyin:

    ```python
    'SVC': SVC(),
    ```

    Sonuç oldukça iyi!

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

    ✅ [Destek Vektörler](https://scikit-learn.org/stable/modules/svm.html#svm) hakkında bilgi edinin

## Topluluk Sınıflandırıcıları

Önceki test oldukça iyi olmasına rağmen, yolu sonuna kadar takip edelim. Bazı 'Topluluk Sınıflandırıcıları' deneyelim, özellikle Random Forest ve AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Sonuç çok iyi, özellikle Random Forest için:

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

✅ [Topluluk Sınıflandırıcıları](https://scikit-learn.org/stable/modules/ensemble.html) hakkında bilgi edinin

Bu Makine Öğrenimi yöntemi, "birkaç temel tahmin edicinin tahminlerini birleştirerek" model kalitesini artırır. Örneğimizde Rastgele Ağaçlar ve AdaBoost kullandık.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), bir ortalama yöntemi, aşırı öğrenmeyi önlemek için rastgelelikle donatılmış 'karar ağaçları' 'ormanı' oluşturur. n_estimators parametresi ağaç sayısına ayarlanır.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html), bir sınıflandırıcıyı verisetine uyar ve sonra bu sınıflandırıcının kopyalarını aynı verisetine uyar. Yanlış sınıflandırılmış öğelerin ağırlıklarına odaklanır ve sonraki sınıflandırıcının uymasını düzeltmek için ayarlar.

---

## 🚀Meydan Okuma

Bu tekniklerin her birinin ayarlanabilecek çok sayıda parametresi vardır. Her birinin varsayılan parametrelerini araştırın ve bu parametrelerin değiştirilmesinin model kalitesi için ne anlama geleceğini düşünün.

## [Ders sonrası quiz](https://ff-quizzes.netlify.app/en/ml/)

## Tekrar & Kendi Kendine Çalışma

Bu derslerde çok fazla jargon var, bu yüzden faydalı terimler [bu listeyi](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) gözden geçirmek için bir dakika ayırın!

## Ödev

[Parametre oyunu](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Feragatname**:
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hatalar veya yanlışlıklar içerebileceğini lütfen göz önünde bulundurun. Orijinal belge, kendi ana dilinde yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımı sonucu ortaya çıkabilecek herhangi bir yanlış anlama veya yanlış yorumdan sorumlu değiliz.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->