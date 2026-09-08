# Pengklasifikasi masakan 1

Dalam pelajaran ini, Anda akan menggunakan dataset yang Anda simpan dari pelajaran terakhir yang penuh dengan data seimbang dan bersih tentang masakan.

Anda akan menggunakan dataset ini dengan berbagai klasifikasi untuk _memprediksi masakan nasional tertentu berdasarkan kumpulan bahan_. Saat melakukannya, Anda akan mempelajari lebih lanjut tentang beberapa cara algoritma dapat digunakan untuk tugas klasifikasi.

## [Kuis sebelum kuliah](https://ff-quizzes.netlify.app/en/ml/)
# Persiapan

Dengan asumsi Anda telah menyelesaikan [Pelajaran 1](../1-Introduction/README.md), pastikan file _cleaned_cuisines.csv_ ada di folder root `/data` untuk keempat pelajaran ini.

## Latihan - prediksi masakan nasional

1. Bekerja di folder _notebook.ipynb_ pada pelajaran ini, impor file itu beserta pustaka Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```
  
    Data terlihat seperti ini:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |

1. Sekarang, impor beberapa pustaka lagi:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```
  
1. Bagi koordinat X dan y menjadi dua dataframe untuk pelatihan. `cuisine` bisa menjadi dataframe label:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```
  
    Ini akan terlihat seperti ini:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```
  
1. Hapus kolom `Unnamed: 0` dan kolom `cuisine` dengan memanggil `drop()`. Simpan sisa data sebagai fitur yang dapat dilatih:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```
  
    Fitur Anda terlihat seperti ini:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Sekarang Anda siap melatih model Anda!

## Memilih pengklasifikasi

Sekarang data Anda bersih dan siap untuk pelatihan, Anda harus memutuskan algoritma mana yang digunakan.

Scikit-learn mengelompokkan klasifikasi di bawah Pembelajaran Terawasi, dan dalam kategori itu Anda akan menemukan banyak cara untuk mengklasifikasikan. [Keragamannya](https://scikit-learn.org/stable/supervised_learning.html) agak membingungkan pada pandangan pertama. Metode-metode berikut ini semua termasuk teknik klasifikasi:

- Model Linear
- Mesin Vektor Pendukung
- Stochastic Gradient Descent
- Tetangga Terdekat
- Proses Gaussian
- Pohon Keputusan
- Metode Ensemble (voting Classifier)
- Algoritma multiclass dan multioutput (klasifikasi multiclass dan multilabel, klasifikasi multiclass-multioutput)

> Anda juga dapat menggunakan [jaringan saraf untuk mengklasifikasikan data](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), tetapi itu di luar cakupan pelajaran ini.

### Pengklasifikasi yang harus dipilih?

Jadi, pengklasifikasi mana yang harus Anda pilih? Seringkali, mencoba beberapa dan mencari hasil yang baik adalah cara untuk mengetes. Scikit-learn menawarkan [perbandingan berdampingan](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) pada dataset yang dibuat, membandingkan KNeighbors, SVC dua cara, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB dan QuadraticDiscrinationAnalysis, dengan hasil yang divisualisasikan:

![perbandingan pengklasifikasi](../../../../translated_images/id/comparison.edfab56193a85e7f.webp)  
> Grafik yang dihasilkan pada dokumentasi Scikit-learn

> AutoML menyelesaikan masalah ini dengan rapi dengan menjalankan perbandingan ini di cloud, memungkinkan Anda memilih algoritma terbaik untuk data Anda. Coba [di sini](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Pendekatan yang lebih baik

Cara yang lebih baik daripada menebak sembarangan, adalah mengikuti ide pada [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) yang dapat diunduh ini. Di sini, kita menemukan bahwa, untuk masalah multiclass kita, kita memiliki beberapa pilihan:

![cheatsheet untuk masalah multiclass](../../../../translated_images/id/cheatsheet.07a475ea444d2223.webp)  
> Sebagian dari Algorithm Cheat Sheet Microsoft, merinci opsi klasifikasi multiclass

✅ Unduh cheat sheet ini, cetak, dan pajang di dinding Anda!

### Alasan

Mari kita lihat apakah kita bisa menganalisis berbagai pendekatan dengan mempertimbangkan batasan yang ada:

- **Jaringan saraf terlalu berat**. Mengingat dataset kami yang bersih tapi minimal, dan fakta bahwa pelatihan dijalankan secara lokal melalui notebook, jaringan saraf terlalu berat untuk tugas ini.
- **Tidak menggunakan pengklasifikasi dua kelas**. Kita tidak menggunakan pengklasifikasi dua kelas, jadi ini menyingkirkan one-vs-all.
- **Pohon keputusan atau regresi logistik bisa bekerja**. Pohon keputusan mungkin bekerja, atau regresi logistik untuk data multiclass.
- **Pohon keputusan boosted multiclass menyelesaikan masalah lain**. Pohon keputusan boosted multiclass paling cocok untuk tugas nonparametrik, misalnya tugas yang dirancang untuk membuat peringkat, sehingga tidak berguna bagi kita.

### Menggunakan Scikit-learn

Kita akan menggunakan Scikit-learn untuk menganalisis data kita. Namun, ada banyak cara untuk menggunakan regresi logistik dalam Scikit-learn. Lihatlah [parameter yang harus diberikan](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Intinya ada dua parameter penting - `multi_class` dan `solver` - yang perlu kita tentukan saat meminta Scikit-learn melakukan regresi logistik. Nilai `multi_class` menerapkan perilaku tertentu. Nilai solver adalah algoritma yang digunakan. Tidak semua solver bisa digabungkan dengan semua nilai `multi_class`.

Menurut dokumentasi, dalam kasus multiclass, algoritma pelatihan:

- **Menggunakan skema one-vs-rest (OvR)**, jika opsi `multi_class` disetel ke `ovr`
- **Menggunakan loss cross-entropy**, jika opsi `multi_class` disetel ke `multinomial`. (Saat ini opsi `multinomial` didukung hanya oleh solver ‘lbfgs’, ‘sag’, ‘saga’ dan ‘newton-cg’)."

> 🎓 'Skema' di sini bisa 'ovr' (one-vs-rest) atau 'multinomial'. Karena regresi logistik sebenarnya dirancang untuk klasifikasi biner, skema ini memungkinkan untuk menangani tugas klasifikasi multiclass dengan lebih baik. [sumber](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' didefinisikan sebagai "algoritma yang digunakan dalam masalah optimisasi". [sumber](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn menyediakan tabel ini untuk menjelaskan bagaimana solver menangani berbagai tantangan yang muncul dari berbagai jenis struktur data:

![solver](../../../../translated_images/id/solvers.5fc648618529e627.webp)

## Latihan - bagi data

Kita dapat fokus pada regresi logistik untuk percobaan pelatihan pertama karena Anda baru saja mempelajarinya di pelajaran sebelumnya.  
Bagi data Anda menjadi grup pelatihan dan pengujian dengan memanggil `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```
  
## Latihan - terapkan regresi logistik

Karena Anda menggunakan kasus multiclass, Anda perlu memilih skema apa yang digunakan dan solver apa yang diatur. Gunakan LogisticRegression dengan pengaturan multiclass dan solver **liblinear** untuk pelatihan.

1. Buat regresi logistik dengan `multi_class` disetel ke `ovr` dan solver disetel ke `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```
  
    ✅ Coba solver berbeda seperti `lbfgs`, yang biasanya disetel sebagai default

    > Catatan, gunakan fungsi Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) untuk meratakan data Anda bila diperlukan.

    Akurasi bagus, lebih dari **80%**!

1. Anda dapat melihat model ini beraksi dengan menguji satu baris data (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```
  
    Hasilnya dicetak:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```
  
   ✅ Coba nomor baris yang berbeda dan periksa hasilnya
1. Menggali lebih dalam, Anda dapat memeriksa keakuratan prediksi ini:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Hasil dicetak - masakan India adalah tebakan terbaiknya, dengan probabilitas yang baik:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Dapatkah Anda menjelaskan mengapa model cukup yakin ini adalah masakan India?

1. Dapatkan lebih banyak detail dengan mencetak laporan klasifikasi, seperti yang Anda lakukan dalam pelajaran regresi:

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

## 🚀Tantangan

Dalam pelajaran ini, Anda menggunakan data yang telah dibersihkan untuk membangun model machine learning yang dapat memprediksi masakan nasional berdasarkan serangkaian bahan. Luangkan waktu untuk membaca berbagai pilihan yang disediakan Scikit-learn untuk mengklasifikasikan data. Dalami konsep 'solver' untuk memahami apa yang terjadi di balik layar.

## [Kuis pasca kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Belajar Mandiri

Dalami sedikit lebih jauh matematika di balik regresi logistik dalam [pelajaran ini](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Tugas 

[Pelajari solver](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berupaya untuk akurasi, harap diperhatikan bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang berwenang. Untuk informasi penting, disarankan menggunakan terjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas salah pengertian atau penafsiran yang timbul dari penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->