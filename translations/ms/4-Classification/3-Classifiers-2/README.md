# Pengelas masakan 2

Dalam pelajaran pengelasan kedua ini, anda akan meneroka lebih banyak cara untuk mengelas data berangka. Anda juga akan mempelajari kesan pemilihan satu pengelas berbanding yang lain.

## [Kuiz pra-ceramah](https://ff-quizzes.netlify.app/en/ml/)

### Prasyarat

Kami menganggap anda telah menyelesaikan pelajaran sebelum ini dan mempunyai set data yang telah dibersihkan dalam folder `data` anda yang dipanggil _cleaned_cuisines.csv_ di akar folder 4 pelajaran ini.

### Persediaan

Kami telah memuatkan fail _notebook.ipynb_ anda dengan set data yang dibersihkan dan telah membahagikannya kepada dataframe X dan y, sedia untuk proses pembinaan model.

## Peta pengelasan

Sebelum ini, anda telah belajar tentang pelbagai pilihan yang anda ada apabila mengelaskan data menggunakan helaian cheat Microsoft. Scikit-learn menawarkan helaian cheat yang serupa, tetapi lebih terperinci yang boleh membantu mempersempitkan lagi estimator anda (istilah lain untuk pengelas):

![Peta ML dari Scikit-learn](../../../../translated_images/ms/map.e963a6a51349425a.webp)
> Tip: [lawati peta ini secara dalam talian](https://scikit-learn.org/stable/tutorial/machine_learning_map/) dan klik sepanjang laluan untuk membaca dokumentasi.

### Pelan

Peta ini sangat berguna setelah anda memahami data anda dengan jelas, kerana anda boleh ‘berjalan’ sepanjang laluan untuk membuat keputusan:

- Kami mempunyai >50 sampel
- Kami ingin meramalkan kategori
- Kami mempunyai data yang dilabel
- Kami mempunyai kurang daripada 100K sampel
- ✨ Kami boleh memilih Linear SVC
- Jika itu tidak berfungsi, kerana kami mempunyai data berangka
    - Kami boleh cuba ✨ KNeighbors Classifier
      - Jika itu tidak berfungsi, cuba ✨ SVC dan ✨ Ensemble Classifiers

Ini adalah laluan yang sangat berguna untuk diikuti.

## Latihan - bahagikan data

Mengikuti laluan ini, kita harus mulakan dengan mengimport beberapa perpustakaan yang diperlukan.

1. Import perpustakaan yang diperlukan:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Bahagikan data latihan dan ujian anda:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Pengelas Linear SVC

Support-Vector clustering (SVC) adalah salah satu jenis dalam keluarga mesin Support-Vector teknik ML (ketahui lebih lanjut tentang ini di bawah). Dalam kaedah ini, anda boleh memilih ‘kernel’ untuk menentukan bagaimana mengelompokkan label. Parameter ‘C’ merujuk kepada ‘regularisasi’ yang mengawal pengaruh parameter. Kernel boleh menjadi salah satu daripada [beberapa](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); di sini kami menetapkannya kepada ‘linear’ untuk memastikan kami memanfaatkan Linear SVC. Kebarangkalian lalai adalah ‘false’; di sini kami tetapkan kepada ‘true’ untuk mengumpul anggaran kebarangkalian. Kami tetapkan random state kepada ‘0’ untuk mengocak data supaya mendapat kebarangkalian.

### Latihan - gunakan Linear SVC

Mula dengan mencipta satu senarai pengelas. Anda akan menambah secara berperingkat kepada senarai ini semasa kita menguji.

1. Mula dengan Linear SVC:

    ```python
    C = 10
    # Buat penyaing yang berbeza.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Latih model anda menggunakan Linear SVC dan cetak laporan:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Keputusannya agak baik:

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

## Pengelas K-Neighbors

K-Neighbors adalah sebahagian daripada keluarga ‘neighbors’ dalam kaedah ML, yang boleh digunakan untuk pembelajaran terkawal dan tidak terkawal. Dalam kaedah ini, bilangan titik telah ditetapkan dan data dikumpulkan di sekitar titik-titik ini supaya label yang digeneralisasi boleh diramalkan untuk data tersebut.

### Latihan - gunakan pengelas K-Neighbors

Pengelas sebelum ini baik dan berfungsi dengan baik dengan data, tetapi mungkin kita boleh dapat ketepatan yang lebih baik. Cuba pengelas K-Neighbors.

1. Tambah satu baris ke senarai pengelas anda (tambah koma selepas item Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Keputusannya sedikit lebih buruk:

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

    ✅ Pelajari tentang [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Pengelas Support Vector

Pengelas Support-Vector adalah sebahagian daripada keluarga [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) kaedah ML yang digunakan untuk tugasan pengelasan dan regresi. SVM "memetakan contoh latihan ke titik dalam ruang" untuk memaksimumkan jarak antara dua kategori. Data seterusnya dipetakan ke dalam ruang ini supaya kategori mereka boleh diramalkan.

### Latihan - gunakan Pengelas Support Vector

Mari cuba untuk ketepatan yang sedikit lebih baik dengan Pengelas Support Vector.

1. Tambah koma selepas item K-Neighbors, dan kemudian tambah baris ini:

    ```python
    'SVC': SVC(),
    ```

    Keputusannya sangat baik!

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

    ✅ Pelajari tentang [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Pengelas Ensembel

Mari ikuti laluan hingga ke penghujung, walaupun ujian sebelum ini sudah cukup baik. Mari cuba beberapa ‘Pengelas Ensembel’, khususnya Random Forest dan AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Keputusannya sangat baik, terutamanya untuk Random Forest:

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

✅ Pelajari tentang [Pengelas Ensembel](https://scikit-learn.org/stable/modules/ensemble.html)

Kaedah Pembelajaran Mesin ini "menggabungkan ramalan beberapa estimator asas" untuk meningkatkan kualiti model. Dalam contoh kami, kami menggunakan Random Trees dan AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), kaedah purata, membina ‘hutan’ ‘pokok keputusan’ yang dipenuhi dengan rawak untuk mengelakkan overfitting. Parameter n_estimators ditetapkan kepada bilangan pokok.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) menyesuaikan satu pengelas kepada set data dan kemudian menyesuaikan salinan pengelas itu kepada set data yang sama. Ia menumpukan pada berat bagi item yang salah diklasifikasikan dan melaraskan keserasian untuk pengelas seterusnya untuk membetulkan.

---

## 🚀Cabaran

Setiap teknik ini mempunyai banyak parameter yang boleh anda laraskan. Selidiki parameter lalai setiap satu dan fikirkan apa maksud melaraskan parameter ini untuk kualiti model.

## [Kuiz pasca-ceramah](https://ff-quizzes.netlify.app/en/ml/)

## Semakan & Pembelajaran Sendiri

Terdapat banyak jargon dalam pelajaran ini, jadi ambil masa untuk menyemak [senarai ini](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) istilah berguna!

## Tugasan

[Parameter play](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila maklum bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber utama yang sahih. Untuk maklumat penting, terjemahan profesional oleh manusia adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->