# Pengelas masakan 1

Dalam pelajaran ini, anda akan menggunakan set data yang anda simpan daripada pelajaran terakhir yang penuh dengan data seimbang dan bersih semuanya mengenai masakan.

Anda akan menggunakan set data ini dengan pelbagai pengelas untuk _meramalkan masakan nasional yang diberikan berdasarkan kumpulan ramuan_. Semasa melakukannya, anda akan belajar lebih lanjut tentang beberapa cara di mana algoritma boleh dimanfaatkan untuk tugasan pengelasan.

## [Kuiz pra-ceramah](https://ff-quizzes.netlify.app/en/ml/)
# Persediaan

Dengan anggapan anda telah menyelesaikan [Pelajaran 1](../1-Introduction/README.md), pastikan fail _cleaned_cuisines.csv_ wujud dalam folder root `/data` untuk keempat-empat pelajaran ini.

## Latihan - ramalkan masakan nasional

1. Bekerja dalam folder _notebook.ipynb_ pelajaran ini, import fail itu bersama pustaka Pandas:

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
  

1. Sekarang, import beberapa pustaka lagi:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Bahagikan koordinat X dan y kepada dua dataframe untuk latihan. `cuisine` boleh menjadi dataframe label:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Ia akan kelihatan seperti ini:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Buang lajur `Unnamed: 0` dan lajur `cuisine` itu, dengan memanggil `drop()`. Simpan baki data sebagai ciri boleh latih:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Ciri-ciri anda kelihatan seperti ini:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Kini anda sudah bersedia untuk melatih model anda!

## Memilih pengelas anda

Setelah data anda bersih dan siap untuk latihan, anda perlu memutuskan algoritma mana yang digunakan untuk tugasan itu.

Scikit-learn mengkelaskan klasifikasi di bawah Pembelajaran Berpenyelia, dan dalam kategori itu anda akan menemui banyak cara untuk mengklasifikasikan. [Kepelbagaian](https://scikit-learn.org/stable/supervised_learning.html) agak mengelirukan pada pandangan pertama. Kaedah berikut semuanya termasuk teknik klasifikasi:

- Model Linear
- Mesin Sokongan Vektor
- Kecerunan Stokastik Turunan
- Jiran Terdekat
- Proses Gaussian
- Pokok Keputusan
- Kaedah Ensemble (Pengelas pengundian)
- Algoritma multiclass dan multioutput (klasifikasi multiclass dan multilabel, klasifikasi multiclass-multioutput)

> Anda juga boleh menggunakan [rangkaian neural untuk mengklasifikasikan data](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), tetapi itu di luar skop pelajaran ini.

### Pengelas mana untuk dipilih?

Jadi, pengelas mana yang harus anda pilih? Selalunya, menjalankan beberapa pengelas dan mencari hasil yang baik adalah cara untuk menguji. Scikit-learn menawarkan [perbandingan sebelah-menyebelah](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) pada set data yang dibuat, membandingkan KNeighbors, SVC dua cara, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB dan QuadraticDiscrinationAnalysis, menunjukkan keputusan secara visual: 

![perbandingan pengelas](../../../../translated_images/ms/comparison.edfab56193a85e7f.webp)
> Plot dijana pada dokumentasi Scikit-learn

> AutoML menyelesaikan masalah ini dengan kemas dengan menjalankan perbandingan ini di awan, membolehkan anda memilih algoritma terbaik untuk data anda. Cuba [di sini](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Pendekatan yang lebih baik

Cara yang lebih baik daripada meneka secara rawak, bagaimanapun, adalah mengikuti idea dalam [Lembaran Cheat ML](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) yang boleh dimuat turun ini. Di sini, kita dapati bahawa untuk masalah multiclass kita, kita mempunyai beberapa pilihan:

![lembaran cheat untuk masalah multiclass](../../../../translated_images/ms/cheatsheet.07a475ea444d2223.webp)
> Seksyen Lembaran Cheat Algoritma Microsoft, memperincikan pilihan klasifikasi multiclass

✅ Muat turun lembaran cheat ini, cetaknya, dan gantung di dinding anda!

### Penalaran

Mari lihat jika kita boleh beralasan melalui pendekatan yang berbeza memandangkan kekangan yang kita ada:

- **Rangkaian neural terlalu berat**. Dengan set data yang bersih tetapi minimum, dan fakta bahawa kita menjalankan latihan secara tempatan melalui notebook, rangkaian neural terlalu berat untuk tugasan ini.
- **Tiada pengelas dua kelas**. Kita tidak menggunakan pengelas dua kelas, jadi itu menolak satu-vs-semua.
- **Pokok keputusan atau regresi logistik boleh berfungsi**. Pokok keputusan mungkin berfungsi, atau regresi logistik untuk data multiclass.
- **Pokok Keputusan Didorong multiclass menyelesaikan masalah berbeza**. Pokok keputusan yang didorong multiclass paling sesuai untuk tugas bukan parametrik, misalnya tugasan yang direka untuk membina penggredan, jadi ia tidak berguna untuk kita.

### Menggunakan Scikit-learn 

Kita akan menggunakan Scikit-learn untuk menganalisis data kita. Walau bagaimanapun, terdapat banyak cara untuk menggunakan regresi logistik dalam Scikit-learn. Lihatlah [parameter yang perlu diserahkan](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Asasnya, terdapat dua parameter penting - `multi_class` dan `solver` - yang kita perlu tetapkan, apabila kita meminta Scikit-learn melakukan regresi logistik. Nilai `multi_class` menerapkan tingkah laku tertentu. Nilai `solver` adalah algoritma yang digunakan. Tidak semua solver boleh digabungkan dengan semua nilai `multi_class`.

Menurut dokumentasi, dalam kes multiclass, algoritma latihan:

- **Menggunakan skim one-vs-rest (OvR)**, jika pilihan `multi_class` ditetapkan kepada `ovr`
- **Menggunakan cross-entropy loss**, jika pilihan `multi_class` ditetapkan kepada `multinomial`. (Pilihan `multinomial` kini disokong hanya oleh solver ‘lbfgs’, ‘sag’, ‘saga’ dan ‘newton-cg’.)"

> 🎓 'Skim' di sini boleh sama ada 'ovr' (one-vs-rest) atau 'multinomial'. Oleh kerana regresi logistik sebenarnya direka untuk menyokong klasifikasi binari, skim ini membolehkan ia menangani tugasan klasifikasi multiclass dengan lebih baik. [sumber](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' ditakrifkan sebagai "algoritma yang digunakan dalam masalah pengoptimuman". [sumber](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn menawarkan jadual ini untuk menerangkan bagaimana solver mengendalikan cabaran yang berbeza yang dibentangkan oleh jenis struktur data yang berbeza:

![solver](../../../../translated_images/ms/solvers.5fc648618529e627.webp)

## Latihan - bahagi data

Kita boleh menumpukan kepada regresi logistik untuk percubaan latihan pertama kita kerana anda baru-baru ini belajar mengenainya dalam pelajaran sebelumnya.
Bahagikan data anda kepada kumpulan latihan dan ujian dengan memanggil `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Latihan - gunakan regresi logistik

Oleh kerana anda menggunakan kes multiclass, anda perlu memilih skim apa yang hendak digunakan dan solver apa yang hendak ditetapkan. Gunakan LogisticRegression dengan tetapan multiclass dan solver **liblinear** untuk melatih.

1. Cipta regresi logistik dengan multi_class ditetapkan kepada `ovr` dan solver ditetapkan kepada `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Cuba solver lain seperti `lbfgs`, yang sering ditetapkan sebagai lalai

    > Nota, gunakan fungsi Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) untuk meratakan data anda bila perlu.

    Ketepatan adalah baik, melebihi **80%**!

1. Anda boleh melihat model ini beraksi dengan menguji satu baris data (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Keputusan dicetak:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Cuba nombor baris berbeza dan periksa keputusan
1. Menggali lebih dalam, anda boleh memeriksa ketepatan ramalan ini:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Keputusan dicetak - Masakan India adalah tekaan terbaiknya, dengan kebarangkalian yang baik:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Bolehkah anda jelaskan mengapa model ini agak pasti ini adalah masakan India?

1. Dapatkan lebih banyak butiran dengan mencetak laporan klasifikasi, seperti yang anda lakukan dalam pelajaran regresi:

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

## 🚀Cabaran

Dalam pelajaran ini, anda menggunakan data yang telah dibersihkan untuk membina model pembelajaran mesin yang boleh meramalkan masakan kebangsaan berdasarkan siri bahan-bahan. Luangkan masa untuk membaca pelbagai pilihan yang disediakan oleh Scikit-learn untuk mengklasifikasikan data. Selidiki lebih jauh konsep 'solver' untuk memahami apa yang berlaku di belakang tabir.

## [Kuiz pasca kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Belajar Sendiri

Gali lebih dalam matematik di sebalik regresi logistik dalam [pelajaran ini](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Tugasan

[Pelajari solvers](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang sahih. Untuk maklumat penting, terjemahan profesional oleh manusia adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->