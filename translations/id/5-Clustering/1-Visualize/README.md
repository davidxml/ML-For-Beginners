# Pengantar clustering

Clustering adalah salah satu jenis [Pembelajaran Tak Terawasi](https://wikipedia.org/wiki/Unsupervised_learning) yang mengasumsikan bahwa sebuah dataset tidak berlabel atau bahwa inputnya tidak dipasangkan dengan output yang telah ditentukan. Ia menggunakan berbagai algoritma untuk menyortir data yang tidak berlabel dan memberikan pengelompokan sesuai pola yang ia temukan dalam data tersebut. 

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Klik gambar di atas untuk video. Saat Anda belajar pembelajaran mesin dengan clustering, nikmati beberapa lagu Dance Hall Nigeria - ini adalah lagu yang sangat populer dari 2014 oleh PSquare.

## [Kuis pra-lecture](https://ff-quizzes.netlify.app/en/ml/)

### Pengantar

[Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) sangat berguna untuk eksplorasi data. Mari kita lihat apakah ini bisa membantu menemukan tren dan pola dalam cara audiens Nigeria mengonsumsi musik.

✅ Luangkan waktu sebentar untuk memikirkan kegunaan clustering. Dalam kehidupan nyata, clustering terjadi setiap kali Anda memiliki tumpukan cucian dan perlu menyortir pakaian anggota keluarga Anda 🧦👕👖🩲. Dalam ilmu data, clustering terjadi ketika mencoba menganalisis preferensi pengguna, atau menentukan karakteristik dari dataset yang tidak berlabel. Clustering, dalam suatu cara, membantu mengerti kekacauan, seperti laci kaus kaki.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Klik gambar di atas untuk video: John Guttag dari MIT memperkenalkan clustering

Dalam lingkungan profesional, clustering dapat digunakan untuk menentukan hal-hal seperti segmentasi pasar, menentukan kelompok usia mana yang membeli barang apa, misalnya. Penggunaan lain adalah deteksi anomali, misalnya untuk mendeteksi penipuan dari kumpulan data transaksi kartu kredit. Atau Anda bisa menggunakan clustering untuk menentukan tumor dalam sekumpulan scan medis.

✅ Pikirkan sesaat bagaimana Anda mungkin pernah menjumpai clustering 'di lapangan', dalam perbankan, e-commerce, atau lingkungan bisnis.

> 🎓 Menariknya, analisis cluster berasal dari bidang Antropologi dan Psikologi pada tahun 1930-an. Bisakah Anda membayangkan bagaimana penggunaannya?

Sebagai alternatif, Anda bisa menggunakannya untuk mengelompokkan hasil pencarian - berdasarkan tautan belanja, gambar, atau ulasan, misalnya. Clustering berguna ketika Anda memiliki dataset besar yang ingin Anda kurangi dan pada dataset tersebut ingin melakukan analisis yang lebih mendetail, sehingga teknik ini dapat digunakan untuk mengenal data sebelum model lain dibangun.

✅ Setelah data Anda terorganisir dalam cluster, Anda menetapkan id cluster padanya, dan teknik ini bisa berguna saat menjaga privasi dataset; Anda dapat merujuk ke titik data dengan id cluster, bukan dengan data yang lebih mengungkapkan identitas. Bisakah Anda memikirkan alasan lain mengapa Anda akan merujuk id cluster daripada elemen lain dari cluster untuk mengidentifikasinya?

Perdalam pemahaman Anda tentang teknik clustering di [modul Learn ini](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)
## Memulai dengan clustering

[Scikit-learn menawarkan banyak metode](https://scikit-learn.org/stable/modules/clustering.html) untuk melakukan clustering. Jenis yang Anda pilih akan tergantung pada kasus penggunaan Anda. Menurut dokumentasi, setiap metode memiliki berbagai manfaat. Berikut tabel sederhana dari metode yang didukung oleh Scikit-learn dan kasus penggunaan yang sesuai:

| Nama Metode                  | Kasus Penggunaan                                                       |
| :--------------------------- | :-------------------------------------------------------------------- |
| K-Means                      | tujuan umum, induktif                                                 |
| Propagasi afinitas           | banyak, cluster tidak merata, induktif                                |
| Mean-shift                   | banyak, cluster tidak merata, induktif                                |
| Clustering spektral          | sedikit, cluster merata, transduktif                                 |
| Clustering hierarki Ward     | banyak, cluster terbatas, transduktif                                 |
| Agglomerative clustering     | banyak, terbatas, jarak non Euclidean, transduktif                    |
| DBSCAN                       | geometri tidak datar, cluster tidak rata, transduktif                |
| OPTICS                       | geometri tidak datar, cluster tidak rata dengan kerapatan variabel, transduktif |
| Gaussian mixtures            | geometri datar, induktif                                              |
| BIRCH                        | dataset besar dengan outlier, induktif                                |

> 🎓 Cara kita membuat cluster sangat berkaitan dengan bagaimana kita mengumpulkan titik data ke dalam kelompok. Mari kita uraikan beberapa kosakata:
>
> 🎓 ['Transduktif' vs. 'induktif'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Inferensi transduktif berasal dari kasus pelatihan yang diamati yang memetakan ke kasus uji tertentu. Inferensi induktif berasal dari kasus pelatihan yang memetakan ke aturan umum yang baru kemudian diterapkan pada kasus uji.
> 
> Contoh: Bayangkan Anda memiliki dataset yang hanya sebagian berlabel. Beberapa adalah 'rekaman', beberapa 'cd', dan beberapa kosong. Tugas Anda adalah memberikan label pada bagian kosong tersebut. Jika Anda memilih pendekatan induktif, Anda melatih model untuk mencari 'rekaman' dan 'cd', kemudian menerapkan label tersebut pada data yang tidak berlabel. Pendekatan ini akan kesulitan mengklasifikasikan sesuatu yang sebenarnya adalah 'kaset'. Sebaliknya pendekatan transduktif menangani data yang tidak dikenal ini lebih efektif karena bekerja untuk mengelompokkan item yang serupa bersama dan kemudian menerapkan label pada kelompok tersebut. Dalam kasus ini, cluster mungkin mencerminkan 'benda musik bulat' dan 'benda musik kotak'.
> 
> 🎓 ['Geometri tidak datar' vs. 'datar'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Berasal dari terminologi matematika, geometri tidak datar vs. datar mengacu pada pengukuran jarak antara titik dengan menggunakan metode geometris 'datar' ([Euclidean](https://wikipedia.org/wiki/Euclidean_geometry)) atau 'tidak datar' (non-Euclidean).
>
>'Datar' dalam konteks ini mengacu pada geometri Euclidean (bagian-bagiannya diajarkan sebagai geometri 'bidang'), dan tidak datar mengacu pada geometri non-Euclidean. Apa hubungannya geometri dengan pembelajaran mesin? Sebagai dua bidang yang berakar pada matematika, harus ada cara umum mengukur jarak antara titik dalam cluster, dan itu bisa dilakukan secara 'datar' atau 'tidak datar', tergantung sifat data. [Jarak Euclidean](https://wikipedia.org/wiki/Euclidean_distance) diukur sebagai panjang segmen garis antara dua titik. [Jarak non-Euclidean](https://wikipedia.org/wiki/Non-Euclidean_geometry) diukur sepanjang kurva. Jika data Anda, bila divisualisasikan, tampak tidak berada pada bidang datar, Anda mungkin perlu menggunakan algoritma khusus untuk menanganinya.
>
![Flat vs Nonflat Geometry Infographic](../../../../translated_images/id/flat-nonflat.d1c8c6e2a96110c1.webp)
> Infografis oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Jarak'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Cluster didefinisikan oleh matriks jaraknya, misalnya jarak antar titik. Jarak ini bisa diukur dalam beberapa cara. Cluster Euclidean didefinisikan oleh rata-rata nilai titik, dan mengandung 'centroid' atau titik pusat. Jarak diukur dari titik ke centroid tersebut. Jarak non-Euclidean mengacu pada 'clustroid', titik yang paling dekat dengan titik lainnya. Clustroid sendiri dapat didefinisikan dengan berbagai cara.
> 
> 🎓 ['Terbatas'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Clustering Terbatas](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) memperkenalkan pembelajaran 'semi-terawasi' ke dalam metode tak terawasi ini. Hubungan antar titik diberi label 'tidak dapat terhubung' atau 'harus-terhubung' sehingga beberapa aturan dipaksakan pada dataset.
>
>Contoh: Jika algoritma dibiarkan bebas pada sekumpulan data yang tidak berlabel atau semi-berlabel, cluster yang dihasilkannya mungkin berkualitas rendah. Dalam contoh di atas, cluster mungkin mengelompokkan 'benda musik bulat', 'benda musik kotak', 'benda segitiga', dan 'kue'. Jika diberikan beberapa batasan, atau aturan untuk diikuti ("item harus terbuat dari plastik", "item harus bisa menghasilkan musik") hal ini dapat membantu 'membatasi' algoritma untuk membuat pilihan yang lebih baik.
> 
> 🎓 'Kepadatan'
> 
> Data yang 'bising' dianggap 'padat'. Jarak antara titik dalam masing-masing cluster dapat terbukti, setelah pemeriksaan, lebih atau kurang padat, atau 'ramai', sehingga data ini perlu dianalisa dengan metode clustering yang tepat. [Artikel ini](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) menunjukkan perbedaan antara menggunakan clustering K-Means vs. algoritma HDBSCAN untuk menjelajah dataset yang berisik dengan kepadatan cluster tidak rata.

## Algoritma clustering

Ada lebih dari 100 algoritma clustering, dan penggunaannya bergantung pada sifat data yang tersedia. Mari kita bahas beberapa yang utama:

- **Clustering hierarki**. Jika sebuah objek diklasifikasikan berdasarkan kedekatannya dengan objek terdekat, bukan dengan yang lebih jauh, cluster terbentuk berdasarkan jarak anggota mereka ke dan dari objek lain. Agglomerative clustering di Scikit-learn bersifat hierarki.

   ![Hierarchical clustering Infographic](../../../../translated_images/id/hierarchical.bf59403aa43c8c47.webp)
   > Infografis oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering centroid**. Algoritma populer ini memerlukan pemilihan 'k', atau jumlah cluster yang akan dibentuk, setelah itu algoritma menentukan titik pusat cluster dan mengumpulkan data di sekitarnya. [K-means clustering](https://wikipedia.org/wiki/K-means_clustering) adalah versi populer dari clustering centroid. Titik pusat ditentukan oleh rata-rata terdekat, jadi namanya demikian. Jarak kuadrat dari cluster diminimalkan.

   ![Centroid clustering Infographic](../../../../translated_images/id/centroid.097fde836cf6c918.webp)
   > Infografis oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering berbasis distribusi**. Berdasarkan pemodelan statistik, clustering berbasis distribusi berfokus pada menentukan probabilitas bahwa sebuah titik data termasuk dalam cluster, dan menetapkannya sesuai. Metode campuran Gaussian termasuk tipe ini.

- **Clustering berbasis kepadatan**. Titik data ditetapkan ke cluster berdasarkan kepadatannya, atau pengelompokan mereka satu sama lain. Titik data yang jauh dari kelompok dianggap outlier atau noise. DBSCAN, Mean-shift, dan OPTICS adalah jenis clustering ini.

- **Clustering berbasis grid**. Untuk dataset multi-dimensi, dibuat sebuah grid dan data dibagi ke dalam sel-sel grid tersebut, sehingga membentuk cluster.

## Latihan - cluster data Anda

Clustering sebagai teknik sangat terbantu oleh visualisasi yang tepat, jadi mari kita mulai dengan memvisualisasikan data musik kita. Latihan ini akan membantu kita memutuskan metode clustering mana yang paling efektif digunakan untuk sifat data ini.

1. Buka file [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) di folder ini.

1. Impor paket `Seaborn` untuk visualisasi data yang baik.

    ```python
    !pip install seaborn
    ```

1. Tambahkan data lagu dari [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Buat dataframe dengan beberapa data tentang lagu-lagu tersebut. Bersiaplah untuk mengeksplorasi data ini dengan mengimpor pustaka dan menampilkan data:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Periksa beberapa baris pertama data:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Dapatkan beberapa informasi tentang dataframe, dengan memanggil `info()`:

    ```python
    df.info()
    ```

   Outputnya terlihat seperti ini:

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

1. Periksa kembali nilai null, dengan memanggil `isnull()` dan memverifikasi jumlahnya adalah 0:

    ```python
    df.isnull().sum()
    ```

    Terlihat baik:

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

1. Deskripsikan data:

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

> 🤔 Jika kita bekerja dengan clustering, sebuah metode tanpa pengawasan yang tidak memerlukan data berlabel, mengapa kita menunjukkan data ini dengan label? Dalam fase eksplorasi data, label ini sangat berguna, tetapi mereka tidak diperlukan agar algoritma clustering dapat bekerja. Kamu juga bisa menghapus header kolom dan merujuk data berdasarkan nomor kolom.

Lihat nilai umum dari data. Perhatikan bahwa popularitas dapat bernilai '0', yang menunjukkan lagu yang tidak memiliki peringkat. Mari kita hapus nilai-nilai tersebut sebentar lagi.

1. Gunakan barplot untuk mengetahui genre yang paling populer:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/id/popular.9c48d84b3386705f.webp)

✅ Jika kamu ingin melihat nilai teratas lainnya, ubah jumlah `[:5]` ke nilai yang lebih besar, atau hapus agar bisa melihat semuanya.

Catatan, ketika genre teratas disebut sebagai 'Missing', itu berarti Spotify tidak mengklasifikasikannya, jadi mari kita hapus data tersebut.

1. Hilangkan data yang hilang dengan memfilternya

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Sekarang periksa kembali genre:

    ![most popular](../../../../translated_images/id/all-genres.1d56ef06cefbfcd6.webp)

1. Jauh lebih dominan tiga genre teratas dalam dataset ini. Mari kita fokus pada `afro dancehall`, `afropop`, dan `nigerian pop`, serta memfilter dataset untuk menghapus data dengan nilai popularitas 0 (artinya tidak diklasifikasikan dengan popularitas dalam dataset dan bisa dianggap sebagai noise untuk keperluan kita):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Lakukan tes cepat untuk melihat apakah data berkorelasi dengan cara yang sangat kuat:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/id/correlation.a9356bb798f5eea5.webp)

    Korelasi yang kuat hanya antara `energy` dan `loudness`, yang tidak mengherankan, mengingat musik keras biasanya cukup energik. Selain itu, korelasinya relatif lemah. Akan menarik untuk melihat apa yang bisa dibuat algoritma clustering dari data ini.

    > 🎓 Perlu dicatat bahwa korelasi tidak berarti sebab-akibat! Kita memiliki bukti korelasi tetapi bukan bukti sebab-akibat. Sebuah [situs web yang lucu](https://tylervigen.com/spurious-correlations) menampilkan visual yang menekankan poin ini.

Apakah ada konvergensi dalam dataset ini terkait persepsi popularitas lagu dan danceability? FacetGrid menunjukkan lingkaran konsentris yang sejajar, terlepas dari genre. Mungkinkah selera masyarakat Nigeria berkumpul pada tingkat tertentu dari danceability untuk genre ini?

✅ Coba titik data yang berbeda (energy, loudness, speechiness) dan genre musik yang lebih banyak atau berbeda. Apa yang bisa kamu temukan? Lihat tabel `df.describe()` untuk mengetahui penyebaran umum titik data.

### Latihan - distribusi data

Apakah ketiga genre ini secara signifikan berbeda dalam persepsi danceability mereka, berdasarkan popularitasnya?

1. Periksa distribusi data dari tiga genre teratas untuk popularitas dan danceability pada sumbu x dan y tertentu.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Kamu bisa menemukan lingkaran konsentris di sekitar titik konvergensi umum, yang menunjukkan distribusi titik.

    > 🎓 Perlu diketahui, contoh ini menggunakan grafik KDE (Kernel Density Estimate) yang mewakili data menggunakan kurva kepadatan probabilitas kontinu. Ini memungkinkan kita menginterpretasikan data ketika bekerja dengan berbagai distribusi.

    Secara umum, ketiga genre tersebut sejajar longgar dalam hal popularitas dan danceability. Menentukan klaster dalam data yang longgar sejajar ini akan menjadi tantangan:

    ![distribution](../../../../translated_images/id/distribution.9be11df42356ca95.webp)

1. Buat scatter plot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Scatterplot pada sumbu yang sama menunjukkan pola konvergensi yang serupa

    ![Facetgrid](../../../../translated_images/id/facetgrid.9b2e65ce707eba1f.webp)

Secara umum, untuk clustering, kamu bisa menggunakan scatterplot untuk menunjukkan klaster data, jadi menguasai jenis visualisasi ini sangat berguna. Pada pelajaran berikutnya, kita akan menggunakan data yang sudah difilter ini dan menggunakan clustering k-means untuk menemukan kelompok dalam data yang tampak tumpang tindih dengan cara yang menarik.

---

## 🚀Tantangan

Sebagai persiapan untuk pelajaran berikutnya, buatlah bagan tentang berbagai algoritma clustering yang mungkin kamu temukan dan gunakan dalam lingkungan produksi. Masalah seperti apa yang coba diatasi oleh clustering?

## [Kuis pasca kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Belajar Mandiri

Sebelum kamu menerapkan algoritma clustering, seperti yang telah kita pelajari, ada baiknya memahami sifat datasetmu. Baca lebih lanjut tentang topik ini [di sini](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Artikel yang berguna ini](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) memandu kamu melalui berbagai cara bagaimana algoritma clustering berbeda berperilaku, mengingat bentuk data yang berbeda.

## Tugas

[Teliti visualisasi lain untuk clustering](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:
Dokumen ini telah diterjemahkan menggunakan layanan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berupaya untuk mencapai akurasi, harap diketahui bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang sah. Untuk informasi penting, disarankan menggunakan terjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->