# Pengenalan kepada pengelompokan

Pengelompokan adalah satu jenis [Pembelajaran Tanpa Penyelia](https://wikipedia.org/wiki/Unsupervised_learning) yang menganggap bahawa set data tidak berlabel atau inputnya tidak dipadankan dengan output yang telah ditetapkan. Ia menggunakan pelbagai algoritma untuk menyusun data tanpa label dan memberikan pengelompokan mengikut corak yang dikesan dalam data tersebut.

[![No One Like You oleh PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You oleh PSquare")

> 🎥 Klik imej di atas untuk video. Semasa anda belajar pembelajaran mesin dengan pengelompokan, nikmati beberapa trek Dance Hall Nigeria - ini adalah lagu yang sangat dinilai dari tahun 2014 oleh PSquare.

## [Kuiz pra-ceramah](https://ff-quizzes.netlify.app/en/ml/)

### Pengenalan

[Pengelompokan](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) sangat berguna untuk penerokaan data. Mari kita lihat jika ia dapat membantu menemui tren dan corak dalam cara penonton Nigeria mengkonsumsi muzik.

✅ Luangkan masa sebentar untuk memikirkan kegunaan pengelompokan. Dalam kehidupan sebenar, pengelompokan berlaku setiap kali anda mempunyai timbunan pakaian kotor dan perlu menyusun pakaian ahli keluarga anda 🧦👕👖🩲. Dalam sains data, pengelompokan berlaku apabila cuba menganalisis keutamaan pengguna, atau menentukan ciri-ciri mana-mana set data tanpa label. Pengelompokan, secara tidak langsung, membantu memberi makna kepada kekacauan, seperti laci stoking.

[![Pengenalan kepada ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Pengenalan kepada Pengelompokan")

> 🎥 Klik imej di atas untuk video: John Guttag dari MIT memperkenalkan pengelompokan

Dalam persekitaran profesional, pengelompokan boleh digunakan untuk menentukan perkara seperti segmentasi pasaran, menentukan kumpulan umur yang membeli barang apa, contohnya. Satu kegunaan lain adalah pengesanan anomali, contohnya untuk mengesan penipuan daripada set data transaksi kad kredit. Atau anda mungkin menggunakan pengelompokan untuk menentukan tumor dalam siri imbasan perubatan.

✅ Fikirkan sebentar tentang bagaimana anda mungkin telah menemui pengelompokan 'di alam nyata', dalam persekitaran perbankan, e-dagang, atau perniagaan.

> 🎓 Menariknya, analisis kluster berasal daripada bidang Antropologi dan Psikologi pada tahun 1930-an. Bolehkah anda bayangkan bagaimana ia mungkin telah digunakan?

Sebagai alternatif, anda boleh menggunakannya untuk mengelompokkan hasil carian - mengikut pautan membeli-belah, imej, atau ulasan, misalnya. Pengelompokan berguna apabila anda mempunyai set data besar yang ingin anda kurangkan dan pada mana anda ingin melakukan analisis lebih terperinci, jadi teknik ini boleh digunakan untuk memahami data sebelum model lain dibina.

✅ Setelah data anda disusun dalam kluster, anda memberi ID kluster, dan teknik ini boleh berguna untuk menjaga privasi set data; anda boleh merujuk kepada titik data melalui ID kluster, bukan melalui data yang lebih mendedahkan yang boleh dikenal pasti. Bolehkah anda fikirkan sebab lain mengapa anda akan merujuk kepada ID kluster dan bukan elemen lain dalam kluster untuk mengenalinya?

Perdalam pemahaman anda tentang teknik pengelompokan dalam [modul Pembelajaran](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)
## Memulakan dengan pengelompokan

[Scikit-learn menawarkan pelbagai jenis](https://scikit-learn.org/stable/modules/clustering.html) kaedah untuk melakukan pengelompokan. Jenis yang anda pilih bergantung pada kes penggunaan anda. Menurut dokumentasi, setiap kaedah mempunyai pelbagai faedah. Berikut adalah jadual ringkas kaedah-kaedah yang disokong oleh Scikit-learn dan kes penggunaan yang sesuai:

| Nama kaedah                  | Kes penggunaan                                                        |
| :--------------------------- | :-------------------------------------------------------------------- |
| K-Means                      | tujuan umum, induktif                                                 |
| Penyebaran keakraban          | banyak, kluster tidak sekata, induktif                                |
| Mean-shift                   | banyak, kluster tidak sekata, induktif                                |
| Pengelompokan spektral       | sedikit, kluster sekata, transduktif                                 |
| Pengelompokan hierarki Ward  | banyak, kluster terhad, transduktif                                  |
| Pengelompokan aglomeratif    | banyak, terhad, jarak bukan Euclidean, transduktif                   |
| DBSCAN                       | geometri tidak rata, kluster tidak sekata, transduktif               |
| OPTICS                       | geometri tidak rata, kluster tidak sekata dengan ketumpatan berubah-ubah, transduktif |
| Campuran Gaussian            | geometri rata, induktif                                              |
| BIRCH                        | set data besar dengan nilai luar, induktif                           |

> 🎓 Cara kita mencipta kluster banyak bergantung pada cara kita mengumpulkan titik data ke dalam kumpulan. Mari kita jelaskan beberapa istilah:
>
> 🎓 ['Transduktif' vs. 'induktif'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Inferens transduktif diperoleh daripada kes latihan yang diperhatikan yang dipadankan dengan kes ujian tertentu. Inferens induktif diperoleh daripada kes latihan yang dipadankan dengan peraturan umum yang kemudiannya hanya digunakan untuk kes ujian.
> 
> Contoh: Bayangkan anda mempunyai set data yang hanya sebahagiannya berlabel. Sesetengah perkara adalah 'rekod', sesetengah 'cd', dan sesetengah kosong. Tugas anda adalah memberi label kepada yang kosong. Jika anda memilih pendekatan induktif, anda akan melatih model mencari 'rekod' dan 'cd', dan menggunakan label itu pada data tanpa label anda. Pendekatan ini akan menghadapi kesukaran mengklasifikasikan perkara yang sebenarnya adalah 'kaset'. Pendekatan transduktif pula mengendalikan data tidak diketahui ini dengan lebih berkesan kerana ia berusaha mengelompokkan item yang serupa bersama dan kemudian menerapkan label pada kumpulan tersebut. Dalam kes ini, kluster mungkin mencerminkan 'benda muzik bulat' dan 'benda muzik segi empat'.
> 
> 🎓 ['Geometri bukan rata' vs. 'rata'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Berasal dari istilah matematik, geometri bukan rata vs. rata merujuk kepada ukuran jarak antara titik sama ada dengan kaedah geometri 'rata' ([Euclidean](https://wikipedia.org/wiki/Euclidean_geometry)) atau 'bukan rata' (bukan Euclidean).
>
> 'Rata' dalam konteks ini merujuk kepada geometri Euclidean (bahagian daripadanya diajar sebagai geometri 'datar'), dan bukan rata merujuk kepada geometri bukan Euclidean. Apa kaitan geometri dengan pembelajaran mesin? Sebagai dua bidang yang berakar dalam matematik, mesti ada cara umum untuk mengukur jarak antara titik dalam kluster, dan itu boleh dilakukan secara 'rata' atau 'bukan rata', bergantung pada sifat data. [Jarak Euclidean](https://wikipedia.org/wiki/Euclidean_distance) diukur sebagai panjang segmen garis antara dua titik. [Jarak bukan Euclidean](https://wikipedia.org/wiki/Non-Euclidean_geometry) diukur sepanjang lengkungan. Jika data anda, apabila divisualisasikan, nampak seolah-olah tidak wujud pada satah, anda mungkin perlu menggunakan algoritma khas untuk mengendalikannya.
>
![Infografik Geometri Rata vs Bukan Rata](../../../../translated_images/ms/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infografik oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Jarak'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Kluster ditakrifkan oleh matriks jarak mereka, contohnya jarak antara titik. Jarak ini boleh diukur dalam beberapa cara. Kluster Euclidean ditakrifkan oleh purata nilai titik, dan mengandungi 'pusat' atau titik tengah. Jarak diukur dengan jarak ke pusat itu. Jarak Non-Euclidean merujuk kepada 'klustroid', titik yang paling dekat dengan titik-titik lain. Klustroid pula boleh ditakrifkan dalam pelbagai cara.
> 
> 🎓 ['Berkekangan'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Pengelompokan Berkekangan](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) memperkenalkan pembelajaran 'semi-terawas' ke dalam kaedah tanpa penyelia ini. Hubungan antara titik ditandakan sebagai 'tidak boleh pautkan' atau 'mesti pautkan' supaya beberapa peraturan dipaksa ke atas set data.
>
> Contoh: Jika algoritma dibebaskan pada set data tanpa label atau setengah berlabel, kluster yang dihasilkannya mungkin berkualiti rendah. Dalam contoh di atas, kluster mungkin mengelompokkan 'benda muzik bulat', 'benda muzik segi empat', 'benda segitiga' dan 'biskut'. Jika diberikan beberapa kekangan, atau peraturan untuk diikuti ("barang mesti diperbuat daripada plastik", "barang mesti boleh menghasilkan muzik") ini boleh membantu 'mengkekang' algoritma supaya membuat pilihan yang lebih baik.
> 
> 🎓 'Ketumpatan'
> 
> Data yang 'berbunyi bising' dianggap 'padat'. Jarak antara titik dalam setiap kluster mungkin, selepas pemeriksaan, lebih atau kurang padat, atau 'sesak' dan data ini perlu dianalisis dengan kaedah pengelompokan yang sesuai. [Artikel ini](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) menunjukkan perbezaan antara menggunakan pengelompokan K-Means vs. algoritma HDBSCAN untuk meneroka set data bising dengan ketumpatan kluster yang tidak sekata.

## Algoritma pengelompokan

Terdapat lebih daripada 100 algoritma pengelompokan, dan penggunaannya bergantung pada sifat data yang ada. Mari kita bincangkan beberapa yang utama:

- **Pengelompokan hierarki**. Jika objek diklasifikasikan berdasarkan kedekatannya dengan objek berhampiran, bukan dengan yang lebih jauh, kluster dibentuk berdasarkan jarak ahli kepada dan daripada objek lain. Pengelompokan aglomeratif Scikit-learn adalah hierarki.

   ![Infografik pengelompokan hierarki](../../../../translated_images/ms/hierarchical.bf59403aa43c8c47.webp)
   > Infografik oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Pengelompokan pusat**. Algoritma popular ini memerlukan pemilihan 'k', atau bilangan kluster untuk dibentuk, selepas itu algoritma menentukan titik pusat kluster dan mengumpul data di sekeliling titik itu. [Pengelompokan K-means](https://wikipedia.org/wiki/K-means_clustering) adalah versi popular pengelompokan pusat. Pusat ditentukan oleh min paling dekat, maka namanya. Jarak kuasa dua daripada kluster diminimumkan.

   ![Infografik pengelompokan pusat](../../../../translated_images/ms/centroid.097fde836cf6c918.webp)
   > Infografik oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Pengelompokan berdasarkan taburan**. Berasaskan pemodelan statistik, pengelompokan berdasarkan taburan menumpukan pada menentukan kebarangkalian bahawa titik data tergolong dalam kluster, dan menetapkannya mengikut itu. Kaedah campuran Gaussian tergolong dalam jenis ini.

- **Pengelompokan berdasarkan ketumpatan**. Titik data ditetapkan ke kluster berdasarkan ketumpatan mereka, atau pengelompokan mereka di sekitar satu sama lain. Titik data yang jauh dari kumpulan dianggap sebagai nilai luar atau bunyi bising. DBSCAN, Mean-shift dan OPTICS tergolong dalam jenis pengelompokan ini.

- **Pengelompokan berasaskan grid**. Untuk set data berbilang dimensi, grid dicipta dan data dibahagikan di antara sel-sel grid, lalu mencipta kluster.

## Latihan - kelompokan data anda

Pengelompokan sebagai teknik sangat dibantu oleh visualisasi yang betul, jadi mari kita mula dengan memvisualisasikan data muzik kita. Latihan ini akan membantu kita memutuskan kaedah pengelompokan mana yang paling sesuai digunakan untuk sifat data ini.

1. Buka fail [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) dalam folder ini.

1. Import pakej `Seaborn` untuk visualisasi data yang baik.

    ```python
    !pip install seaborn
    ```

1. Lampirkan data lagu dari [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Muatkan dataframe dengan beberapa data tentang lagu-lagu tersebut. Bersedia untuk meneroka data ini dengan mengimport perpustakaan dan memaparkan data:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Semak beberapa baris pertama data:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Dapatkan sedikit maklumat mengenai dataframe, dengan memanggil `info()`:

    ```python
    df.info()
    ```

   Outputnya kelihatan seperti ini:

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

1. Semak semula untuk nilai null, dengan memanggil `isnull()` dan mengesahkan jumlahnya adalah 0:

    ```python
    df.isnull().sum()
    ```

    Nampak bagus:

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

1. Huraikan data tersebut:

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

> 🤔 Jika kita bekerja dengan pengelompokan, sebuah kaedah tanpa penyeliaan yang tidak memerlukan data berlabel, mengapa kita menunjukkan data ini dengan label? Dalam fasa penerokaan data, ia sangat membantu, tetapi tidak perlu untuk algoritma pengelompokan berfungsi. Anda juga boleh mengeluarkan tajuk lajur dan merujuk kepada data mengikut nombor lajur.

Lihat nilai umum data. Perhatikan bahawa populariti boleh menjadi '0', yang menunjukkan lagu yang tiada kedudukan. Mari kita keluarkan yang tersebut sebentar lagi.

1. Gunakan barplot untuk mengetahui genre yang paling popular:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/ms/popular.9c48d84b3386705f.webp)

✅ Jika anda ingin melihat lebih banyak nilai teratas, ubah top `[:5]` ke nilai yang lebih besar, atau keluarkan ia untuk melihat semua.

Perhatikan, apabila genre teratas digambarkan sebagai 'Missing', itu bermaksud Spotify tidak mengklasifikasikannya, jadi mari kita buang ia.

1. Buang data yang hilang dengan menapisnya keluar

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Sekarang semak semula genre:

    ![most popular](../../../../translated_images/ms/all-genres.1d56ef06cefbfcd6.webp)

1. Setakat ini, tiga genre teratas menguasai dataset ini. Mari kita fokus pada `afro dancehall`, `afropop`, dan `nigerian pop`, tambahan pula tapis dataset untuk mengeluarkan apa-apa dengan nilai populariti 0 (bermaksud ia tidak diklasifikasikan dengan populariti dalam dataset dan boleh dianggap sebagai gangguan untuk tujuan kita):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Lakukan ujian pantas untuk melihat jika data berkorelasi dalam cara yang sangat kuat:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/ms/correlation.a9356bb798f5eea5.webp)

    Satu-satunya korelasi kuat adalah antara `energy` dan `loudness`, yang tidak menghairankan, memandangkan muzik kuat biasanya cukup bertenaga. Selain itu, korelasi agak lemah. Menarik untuk melihat apa yang boleh algoritma pengelompokan buat dengan data ini.

    > 🎓 Perhatikan bahawa korelasi tidak bermaksud kausaliti! Kita mempunyai bukti korelasi tetapi tiada bukti kausaliti. Sebuah [laman web yang menghiburkan](https://tylervigen.com/spurious-correlations) mempunyai beberapa visual yang menekankan perkara ini.

Adakah terdapat sebarang konvergensi dalam dataset ini mengenai populariti yang dirasai oleh lagu dan tarian? FacetGrid menunjukkan terdapat bulatan berserenjang yang sejajar, tanpa mengira genre. Adakah mungkin citarasa Nigeria bertemu pada tahap tarian tertentu untuk genre ini?

✅ Cuba titik data berbeza (energy, loudness, speechiness) dan lebih banyak atau berbeza genre muzik. Apa yang anda boleh temui? Lihat jadual `df.describe()` untuk melihat taburan umum titik data.

### Latihan - taburan data

Adakah ketiga-tiga genre ini berbeza dengan ketara dalam persepsi tarian mereka berdasarkan populariti?

1. Periksa taburan data untuk ketiga-tiga genre teratas bagi populariti dan tarian sepanjang paksi x dan y yang diberikan.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Anda boleh menemui bulatan berserenjang di sekitar titik konvergensi umum, menunjukkan taburan titik.

    > 🎓 Perhatikan bahawa contoh ini menggunakan graf KDE (Kernel Density Estimate) yang mewakili data menggunakan lengkung ketumpatan kebarangkalian berterusan. Ini membolehkan kita mentafsir data apabila bekerja dengan taburan berganda.

    Secara amnya, ketiga-tiga genre tersebut selari secara longgar dari segi populariti dan tarian. Menentukan kluster dalam data yang selaras secara longgar ini akan menjadi cabaran:

    ![distribution](../../../../translated_images/ms/distribution.9be11df42356ca95.webp)

1. Buat plot taburan:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Scatterplot bagi paksi yang sama menunjukkan corak konvergensi yang serupa

    ![Facetgrid](../../../../translated_images/ms/facetgrid.9b2e65ce707eba1f.webp)

Secara umum, untuk pengelompokan, anda boleh menggunakan scatterplots untuk menunjukkan kluster data, jadi menguasai jenis visualisasi ini sangat berguna. Dalam pelajaran seterusnya, kita akan menggunakan data yang telah ditapis ini dan menggunakan pengelompokan k-means untuk menemui kumpulan dalam data ini yang nampaknya bertindih dengan cara yang menarik.

---

## 🚀Cabaran

Sebagai persiapan untuk pelajaran seterusnya, buat carta tentang pelbagai algoritma pengelompokan yang mungkin anda temui dan gunakan dalam persekitaran produksi. Apakah jenis masalah yang cuba diselesaikan oleh pengelompokan?

## [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Pembelajaran Sendiri

Sebelum anda menerapkan algoritma pengelompokan, seperti yang telah kita pelajari, adalah idea yang baik untuk memahami sifat dataset anda. Baca lebih lanjut tentang topik ini [di sini](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Artikel yang berguna ini](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) membimbing anda melalui pelbagai cara bagaimana algoritma pengelompokan berkelakuan, mengikut bentuk data yang berbeza.

## Tugasan

[Penyelidikan visualisasi lain untuk pengelompokan](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk ketepatan, sila ambil maklum bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang sahih. Untuk maklumat penting, terjemahan oleh manusia profesional adalah disyorkan. Kami tidak bertanggungjawab terhadap sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->