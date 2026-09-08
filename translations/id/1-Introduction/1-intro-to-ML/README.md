# Pengenalan pembelajaran mesin

## [Kuis sebelum kuliah](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML untuk pemula - Pengenalan Pembelajaran Mesin untuk Pemula](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML untuk pemula - Pengenalan Pembelajaran Mesin untuk Pemula")

> 🎥 Klik gambar di atas untuk video singkat yang membahas pelajaran ini.

Selamat datang di kursus pembelajaran mesin klasik untuk pemula ini! Apakah Anda benar-benar baru dalam topik ini, atau seorang praktisi ML berpengalaman yang ingin menyegarkan pengetahuan di suatu bidang, kami senang Anda bergabung! Kami ingin menciptakan tempat awal yang ramah untuk studi ML Anda dan kami bersedia mengevaluasi, merespon, dan menggabungkan [masukan](https://github.com/microsoft/ML-For-Beginners/discussions) Anda.

[![Pengenalan ke ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Pengenalan ke ML")

> 🎥 Klik gambar di atas untuk video: John Guttag dari MIT memperkenalkan pembelajaran mesin

---
## Memulai dengan pembelajaran mesin

Sebelum memulai kurikulum ini, Anda perlu menyiapkan komputer Anda agar siap menjalankan notebook secara lokal.

- **Konfigurasikan mesin Anda dengan video ini**. Gunakan tautan berikut untuk belajar [cara menginstal Python](https://youtu.be/CXZYvNRIAKM) di sistem Anda dan [menyiapkan editor teks](https://youtu.be/EU8eayHWoZg) untuk pengembangan.
- **Belajar Python**. Disarankan juga untuk memiliki pemahaman dasar tentang [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), bahasa pemrograman yang berguna bagi ilmuwan data yang kami gunakan di kursus ini.
- **Belajar Node.js dan JavaScript**. Kami juga menggunakan JavaScript beberapa kali dalam kursus ini saat membangun aplikasi web, jadi Anda perlu menginstal [node](https://nodejs.org) dan [npm](https://www.npmjs.com/), serta memiliki [Visual Studio Code](https://code.visualstudio.com/) untuk pengembangan baik Python maupun JavaScript.
- **Buat akun GitHub**. Karena Anda menemukan kami di [GitHub](https://github.com), mungkin Anda sudah memiliki akun, tapi jika belum, buatlah dan kemudian fork kurikulum ini untuk digunakan sendiri. (Jangan lupa untuk memberi bintang juga ya 😊)
- **Jelajahi Scikit-learn**. Biasakan diri Anda dengan [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), kumpulan pustaka ML yang kami rujuk dalam pelajaran ini.

---
## Apa itu pembelajaran mesin?

Istilah 'pembelajaran mesin' adalah salah satu istilah paling populer dan sering digunakan saat ini. Ada kemungkinan besar Anda pernah mendengar istilah ini setidaknya sekali jika Anda memiliki sedikit pengetahuan tentang teknologi, tidak peduli di bidang apa Anda bekerja. Namun mekanisme pembelajaran mesin adalah misteri bagi kebanyakan orang. Bagi pemula pembelajaran mesin, subjek ini kadang bisa terasa menakutkan. Oleh karena itu, penting untuk memahami apa sebenarnya pembelajaran mesin itu, dan mempelajarinya langkah demi langkah melalui contoh praktis.

---
## Kurva hype

![kurva hype ml](../../../../translated_images/id/hype.07183d711a17aafe.webp)

> Google Trends menunjukkan 'kurva hype' terbaru dari istilah 'pembelajaran mesin'

---
## Dunia yang penuh misteri

Kita hidup di alam semesta yang penuh misteri menarik. Ilmuwan hebat seperti Stephen Hawking, Albert Einstein, dan banyak lainnya telah mendedikasikan hidup mereka untuk mencari informasi bermakna yang mengungkap misteri dunia di sekitar kita. Ini adalah kondisi manusia dalam belajar: anak manusia mempelajari hal baru dan mengungkap struktur dunia mereka tahun demi tahun saat tumbuh menjadi dewasa.

---
## Otak anak

Otak dan indra anak merasakan fakta lingkungan mereka dan secara bertahap mempelajari pola-pola tersembunyi kehidupan yang membantu anak membuat aturan logis untuk mengidentifikasi pola yang dipelajari. Proses belajar otak manusia menjadikan manusia makhluk hidup paling canggih di dunia ini. Belajar terus-menerus dengan menemukan pola tersembunyi kemudian berinovasi pada pola tersebut memungkinkan kita menjadi lebih baik sepanjang hidup kita. Kapasitas belajar dan kemampuan berkembang ini terkait dengan konsep yang disebut [plastisitas otak](https://www.simplypsychology.org/brain-plasticity.html). Secara kasat mata, kita bisa menarik beberapa kesamaan motivasi antara proses belajar otak manusia dan konsep pembelajaran mesin.

---
## Otak manusia

[Otak manusia](https://www.livescience.com/29365-human-brain.html) menangkap hal dari dunia nyata, memproses informasi yang diterima, membuat keputusan rasional, dan melakukan tindakan tertentu berdasarkan keadaan. Inilah yang kita sebut berperilaku secara cerdas. Ketika kita memprogram tiruan dari proses perilaku cerdas tersebut ke mesin, itu disebut kecerdasan buatan (AI).

---
## Beberapa istilah

Meskipun istilahnya terkadang membingungkan, pembelajaran mesin (ML) adalah bagian penting dari kecerdasan buatan. **ML berhubungan dengan penggunaan algoritma khusus untuk menemukan informasi bermakna dan pola tersembunyi dari data yang diterima guna mendukung proses pengambilan keputusan rasional**.

---
## AI, ML, Deep Learning

![AI, ML, deep learning, data science](../../../../translated_images/id/ai-ml-ds.537ea441b124ebf6.webp)

> Diagram yang menunjukkan hubungan antara AI, ML, deep learning, dan data science. Infografis oleh [Jen Looper](https://twitter.com/jenlooper) terinspirasi dari [grafik ini](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Konsep yang akan dibahas

Dalam kurikulum ini, kami akan membahas hanya konsep inti pembelajaran mesin yang harus diketahui pemula. Kami membahas apa yang kami sebut 'pembelajaran mesin klasik' terutama menggunakan Scikit-learn, pustaka yang banyak digunakan pelajar untuk mempelajari dasar-dasarnya. Untuk memahami konsep kecerdasan buatan atau deep learning yang lebih luas, pengetahuan dasar pembelajaran mesin yang kuat sangat penting, dan karenanya kami ingin menyediakannya di sini.

---
## Dalam kursus ini Anda akan belajar:

- konsep inti pembelajaran mesin
- sejarah ML
- ML dan keadilan
- teknik regresi ML
- teknik klasifikasi ML
- teknik pengelompokan ML
- teknik pemrosesan bahasa alami ML
- teknik peramalan deret waktu ML
- pembelajaran penguatan
- aplikasi nyata ML

---
## Yang tidak akan kami bahas

- deep learning
- jaringan saraf
- AI

Untuk pengalaman belajar yang lebih baik, kami akan menghindari kompleksitas jaringan saraf, 'deep learning' - pembuatan model berlapis menggunakan jaringan saraf - dan AI, yang akan kami bahas dalam kurikulum berbeda. Kami juga akan menawarkan kurikulum ilmu data yang akan datang untuk fokus pada aspek tersebut dari bidang yang lebih luas ini.

---
## Mengapa mempelajari pembelajaran mesin?

Pembelajaran mesin, dari perspektif sistem, didefinisikan sebagai pembuatan sistem otomatis yang dapat mempelajari pola tersembunyi dari data untuk membantu membuat keputusan cerdas.

Motivasi ini secara longgar terinspirasi dari bagaimana otak manusia mempelajari hal tertentu berdasarkan data yang diterimanya dari dunia luar.

✅ Pikirkan sejenak mengapa suatu bisnis ingin mencoba menggunakan strategi pembelajaran mesin dibandingkan membuat mesin berbasis aturan yang dikodekan keras.

---
## Mengapa kualitas data penting

Data berkualitas tinggi meningkatkan performa model. Data yang buruk atau berisik dapat menyebabkan prediksi tidak akurat, bahkan ketika menggunakan algoritma pembelajaran mesin canggih.

---
## Aplikasi pembelajaran mesin

Aplikasi pembelajaran mesin sekarang hampir ada di mana-mana, dan sedemikian meluas seperti data yang mengalir di sekitar masyarakat kita, dihasilkan oleh ponsel pintar, perangkat terhubung, dan sistem lainnya. Mengingat potensi besar algoritma pembelajaran mesin mutakhir, para peneliti telah mengeksplorasi kemampuannya untuk menyelesaikan masalah nyata multidimensi dan multidisipliner dengan hasil positif yang besar.

---
## Contoh penerapan ML

**Anda dapat menggunakan pembelajaran mesin dengan berbagai cara**:

- Untuk memprediksi kemungkinan penyakit dari riwayat atau laporan medis pasien.
- Untuk memanfaatkan data cuaca untuk memprediksi kejadian cuaca.
- Untuk memahami sentimen sebuah teks.
- Untuk mendeteksi berita palsu guna menghentikan penyebaran propaganda.

Bidang keuangan, ekonomi, ilmu bumi, eksplorasi luar angkasa, rekayasa biomedis, ilmu kognitif, dan bahkan bidang humaniora telah mengadopsi pembelajaran mesin untuk menyelesaikan masalah berat pemrosesan data dalam domain mereka.

---
## Kesimpulan

Pembelajaran mesin mengotomatisasi proses penemuan pola dengan menemukan wawasan bermakna dari data dunia nyata atau data yang dihasilkan. Ini terbukti sangat berharga dalam bisnis, kesehatan, dan aplikasi keuangan, antara lain.

Dalam waktu dekat, memahami dasar pembelajaran mesin akan menjadi keharusan bagi orang dari bidang apa pun karena adopsinya yang luas.

---
# 🚀 Tantangan

Gambarlah, di atas kertas atau menggunakan aplikasi online seperti [Excalidraw](https://excalidraw.com/), pemahaman Anda tentang perbedaan antara AI, ML, deep learning, dan data science. Tambahkan beberapa ide masalah yang setiap teknik ini cocok untuk diselesaikan.

# [Kuis setelah kuliah](https://ff-quizzes.netlify.app/en/ml/)

---
# Ulasan & Belajar Mandiri

Untuk belajar lebih lanjut bagaimana Anda dapat bekerja dengan algoritma ML di cloud, ikuti [Jalur Pembelajaran](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Ikuti [Jalur Pembelajaran](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) tentang dasar-dasar ML.

---
# Tugas

[Mulai dan jalankan](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:
Dokumen ini telah diterjemahkan menggunakan layanan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berupaya untuk mencapai akurasi, harap diketahui bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang sah. Untuk informasi penting, disarankan menggunakan terjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->