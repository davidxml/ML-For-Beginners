# Teknik Pembelajaran Mesin

Proses membangun, menggunakan, dan memelihara model pembelajaran mesin serta data yang mereka gunakan adalah proses yang sangat berbeda dari banyak alur kerja pengembangan lainnya. Dalam pelajaran ini, kita akan mengungkap prosesnya, dan menguraikan teknik utama yang perlu Anda ketahui. Anda akan:

- Memahami proses yang mendasari pembelajaran mesin pada tingkat tinggi.
- Menjelajahi konsep dasar seperti 'model', 'prediksi', dan 'data pelatihan'.

## [Kuis pra-ceramah](https://ff-quizzes.netlify.app/en/ml/)

[![ML untuk pemula - Teknik Pembelajaran Mesin](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML untuk pemula - Teknik Pembelajaran Mesin")

> 🎥 Klik gambar di atas untuk video singkat yang membahas pelajaran ini.

## Pendahuluan

Pada tingkat tinggi, keterampilan membuat proses pembelajaran mesin (ML) terdiri dari beberapa langkah:

1. **Tentukan pertanyaannya**. Sebagian besar proses ML dimulai dengan mengajukan pertanyaan yang tidak dapat dijawab oleh program kondisional sederhana atau mesin berbasis aturan. Pertanyaan-pertanyaan ini sering kali berkisar pada prediksi berdasarkan kumpulan data.
2. **Kumpulkan dan persiapkan data**. Untuk dapat menjawab pertanyaan Anda, Anda membutuhkan data. Kualitas dan, terkadang, kuantitas data Anda akan menentukan seberapa baik Anda dapat menjawab pertanyaan awal Anda. Visualisasi data adalah aspek penting dalam fase ini. Fase ini juga mencakup pembagian data menjadi kelompok pelatihan dan pengujian untuk membangun model.
3. **Pilih metode pelatihan**. Tergantung pada pertanyaan Anda dan sifat data Anda, Anda perlu memilih bagaimana Anda ingin melatih model agar paling merefleksikan data Anda dan membuat prediksi yang akurat terhadapnya. Ini adalah bagian dari proses ML Anda yang membutuhkan keahlian khusus dan, sering kali, sejumlah besar eksperimen.
4. **Latih model**. Dengan menggunakan data pelatihan Anda, Anda akan menggunakan berbagai algoritma untuk melatih model agar mengenali pola dalam data. Model mungkin memanfaatkan bobot internal yang dapat disesuaikan untuk memprioritaskan bagian tertentu dari data dibandingkan bagian lainnya guna membangun model yang lebih baik.
5. **Evaluasi model**. Anda menggunakan data yang belum pernah dilihat sebelumnya (data pengujian Anda) dari kumpulan data yang dikumpulkan untuk melihat bagaimana kinerja model.
6. **Penyetelan parameter**. Berdasarkan kinerja model Anda, Anda dapat mengulangi proses dengan menggunakan parameter berbeda, atau variabel, yang mengontrol perilaku algoritma yang digunakan untuk melatih model.
7. **Prediksi**. Gunakan input baru untuk menguji akurasi model Anda.

## Pertanyaan apa yang harus diajukan

Komputer sangat terampil dalam menemukan pola tersembunyi dalam data. Manfaat ini sangat membantu para peneliti yang memiliki pertanyaan tentang domain tertentu yang tidak dapat dengan mudah dijawab dengan membuat mesin aturan berbasis kondisi. Misalnya, dalam tugas aktuaria, seorang ilmuwan data mungkin dapat membuat aturan buatan tangan terkait kematian perokok vs bukan perokok.

Namun, saat banyak variabel lain dimasukkan ke dalam persamaan, model ML mungkin terbukti lebih efisien untuk memprediksi tingkat kematian di masa depan berdasarkan riwayat kesehatan masa lalu. Contoh yang lebih menyenangkan bisa berupa membuat prediksi cuaca untuk bulan April di suatu lokasi berdasarkan data yang mencakup lintang, bujur, perubahan iklim, kedekatan dengan laut, pola aliran jet, dan lainnya.

✅ [Slide deck ini](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) tentang model cuaca menawarkan perspektif historis penggunaan ML dalam analisis cuaca.  

## Tugas sebelum membangun

Sebelum mulai membangun model Anda, ada beberapa tugas yang harus diselesaikan. Untuk menguji pertanyaan Anda dan membentuk hipotesis berdasarkan prediksi model, Anda perlu mengidentifikasi dan mengonfigurasi beberapa elemen.

### Data

Untuk dapat menjawab pertanyaan Anda dengan kepastian tertentu, Anda perlu sejumlah data yang cukup dan tipe yang tepat. Ada dua hal yang perlu Anda lakukan pada tahap ini:

- **Kumpulkan data**. Dengan memperhatikan pelajaran sebelumnya tentang keadilan dalam analisis data, kumpulkan data Anda dengan hati-hati. Sadari sumber data ini, bias bawaan yang mungkin ada, dan dokumentasikan asal-usulnya.
- **Persiapkan data**. Ada beberapa langkah dalam proses persiapan data. Anda mungkin perlu menggabungkan data dan menormalkannya jika berasal dari sumber yang beragam. Anda dapat meningkatkan kualitas dan kuantitas data melalui berbagai metode seperti mengonversi string menjadi angka (seperti yang kita lakukan di [Clustering](../../5-Clustering/1-Visualize/README.md)). Anda juga dapat menghasilkan data baru berdasarkan data asli (seperti yang kita lakukan di [Classification](../../4-Classification/1-Introduction/README.md)). Anda dapat membersihkan dan mengedit data (seperti yang akan kita lakukan sebelum pelajaran [Web App](../../3-Web-App/README.md)). Akhirnya, Anda mungkin juga perlu mengacak dan mengocoknya, tergantung pada teknik pelatihan Anda.

✅ Setelah mengumpulkan dan memproses data, luangkan waktu untuk melihat apakah bentuk data tersebut memungkinkan Anda menjawab pertanyaan yang dimaksud. Bisa jadi data tersebut tidak berkinerja baik dalam tugas yang diberikan, seperti yang kita temui dalam pelajaran [Clustering](../../5-Clustering/1-Visualize/README.md)!

### Fitur dan Target

[Feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) adalah properti yang dapat diukur dari data Anda. Dalam banyak dataset, ini diekspresikan sebagai judul kolom seperti 'tanggal', 'ukuran', atau 'warna'. Variabel fitur Anda, biasanya direpresentasikan sebagai `X` dalam kode, mewakili variabel input yang akan digunakan untuk melatih model.

Target adalah hal yang Anda coba prediksi. Target, biasanya direpresentasikan sebagai `y` dalam kode, mewakili jawaban atas pertanyaan yang Anda ajukan terhadap data Anda: di bulan Desember, labu dengan **warna** apa yang akan termurah? di San Francisco, lingkungan mana yang memiliki **harga** real estate terbaik? Kadang-kadang target juga disebut atribut label.

### Memilih variabel fitur Anda

🎓 **Seleksi Fitur dan Ekstraksi Fitur** Bagaimana Anda tahu variabel mana yang dipilih saat membangun model? Anda mungkin akan melalui proses seleksi fitur atau ekstraksi fitur untuk memilih variabel yang tepat agar model paling optimal. Namun, keduanya tidak sama: "Ekstraksi fitur membuat fitur baru dari fungsi fitur asli, sedangkan seleksi fitur mengembalikan subset dari fitur tersebut." ([sumber](https://wikipedia.org/wiki/Feature_selection))

### Visualisasikan data Anda

Aspek penting dari toolkit ilmuwan data adalah kemampuan untuk memvisualisasikan data menggunakan beberapa perpustakaan hebat seperti Seaborn atau MatPlotLib. Mewakili data Anda secara visual dapat memungkinkan Anda menemukan korelasi tersembunyi yang dapat Anda manfaatkan. Visualisasi Anda juga bisa membantu mengungkap bias atau data yang tidak seimbang (seperti yang kita temukan dalam [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Pecah dataset Anda

Sebelum pelatihan, Anda perlu membagi dataset menjadi dua bagian atau lebih yang ukurannya tidak sama namun tetap mewakili data dengan baik.

- **Pelatihan**. Bagian dataset ini digunakan untuk melatih model Anda. Set ini merupakan mayoritas dari dataset asli.
- **Pengujian**. Dataset pengujian adalah kelompok data independen, sering diambil dari data asli, yang Anda gunakan untuk mengonfirmasi kinerja model yang dibangun.
- **Validasi**. Set validasi adalah kelompok contoh independen yang lebih kecil yang Anda gunakan untuk menyetel hiperparameter model, atau arsitektur, guna meningkatkan model. Tergantung ukuran data dan pertanyaan Anda, Anda mungkin tidak perlu membuat set ketiga ini (seperti yang kami catat di [Peramalan Deret Waktu](../../7-TimeSeries/1-Introduction/README.md)).

## Membangun model

Menggunakan data pelatihan Anda, tujuan Anda adalah membangun model, atau representasi statistik dari data Anda, menggunakan berbagai algoritma untuk **melatih**nya. Melatih model mengeksposnya pada data dan memungkinkan model membuat asumsi tentang pola yang disadari, divalidasi, kemudian diterima atau ditolak.

### Tentukan metode pelatihan

Tergantung pada pertanyaan dan sifat data Anda, Anda akan memilih metode untuk melatihnya. Dengan mempelajari dokumentasi [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - yang kita gunakan dalam kursus ini - Anda dapat menjelajahi banyak cara melatih model. Tergantung pengalaman Anda, Anda mungkin harus mencoba beberapa metode berbeda untuk membangun model terbaik. Anda kemungkinan akan melewati proses di mana ilmuwan data mengevaluasi kinerja model dengan memberinya data yang belum pernah dilihat, memeriksa akurasi, bias, dan masalah penurunan kualitas lainnya, serta memilih metode pelatihan yang paling tepat untuk tugas tersebut.

### Latih model

Dengan data pelatihan Anda, Anda siap untuk 'fit' guna membuat model. Anda akan melihat bahwa dalam banyak perpustakaan ML terdapat kode 'model.fit' - saat itulah Anda mengirimkan variabel fitur sebagai array nilai (biasanya 'X') dan variabel target (biasanya 'y').

### Evaluasi model

Setelah proses pelatihan selesai (bisa memakan banyak iterasi, atau 'epoch', untuk melatih model besar), Anda bisa mengevaluasi kualitas model dengan menggunakan data pengujian untuk mengukur kinerjanya. Data ini adalah subset dari data asli yang belum pernah dianalisis sebelumnya oleh model. Anda dapat mencetak tabel metrik tentang kualitas model Anda.

🎓 **Pemodelan fitting**

Dalam konteks pembelajaran mesin, pemodelan fitting mengacu pada akurasi fungsi dasar model saat mencoba menganalisis data yang belum dikenalinya.

🎓 **Underfitting** dan **overfitting** adalah masalah umum yang menurunkan kualitas model, karena model fittingnya terlalu buruk atau terlalu baik. Ini menyebabkan model membuat prediksi yang terlalu dekat atau terlalu longgar dengan data pelatihannya. Model overfit memprediksi data pelatihan terlalu baik karena sudah mempelajari detail dan noise data dengan sangat baik. Model underfit tidak akurat karena tidak dapat menganalisis data pelatihan maupun data yang belum pernah 'dilihat' dengan tepat.

![model overfitting](../../../../translated_images/id/overfitting.1c132d92bfd93cb6.webp)
> Infografis oleh [Jen Looper](https://twitter.com/jenlooper)

## Penyetelan parameter

Setelah pelatihan awal selesai, amati kualitas model dan pertimbangkan meningkatkan dengan mengubah 'hiperparameter'. Baca lebih lanjut tentang proses ini [di dokumentasi](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Prediksi

Ini adalah momen ketika Anda dapat menggunakan data baru sepenuhnya untuk menguji akurasi model Anda. Dalam pengaturan ML 'terapan', di mana Anda membangun aset web untuk menggunakan model dalam produksi, proses ini mungkin melibatkan pengambilan input pengguna (misalnya menekan tombol) untuk mengatur variabel dan mengirimkannya ke model untuk inferensi, atau evaluasi.

Dalam pelajaran-pelajaran ini, Anda akan menemukan cara menggunakan langkah-langkah ini untuk menyiapkan, membangun, menguji, mengevaluasi, dan memprediksi — semua gerakan seorang ilmuwan data dan lebih banyak lagi, seiring kemajuan Anda menjadi insinyur ML 'full stack'.

---

## 🚀Tantangan

Gambarlah diagram alir yang mencerminkan langkah-langkah seorang praktisi ML. Di mana Anda melihat diri Anda sekarang dalam proses ini? Di mana Anda memprediksi akan menghadapi kesulitan? Apa yang terasa mudah bagi Anda?

## [Kuis pasca-ceramah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Studi Mandiri

Cari wawancara online dengan ilmuwan data yang membahas pekerjaan sehari-hari mereka. Berikut [salah satunya](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Tugas

[Wawancarai seorang ilmuwan data](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk akurasi, harap diketahui bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang sah. Untuk informasi penting, disarankan menggunakan terjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau kesalahan tafsir yang timbul dari penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->