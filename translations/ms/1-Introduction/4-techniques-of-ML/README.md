# Teknik Pembelajaran Mesin

Proses membina, menggunakan, dan menyelenggara model pembelajaran mesin dan data yang mereka gunakan adalah proses yang sangat berbeza daripada banyak aliran kerja pembangunan lain. Dalam pelajaran ini, kita akan menjelaskan proses ini, dan menggariskan teknik utama yang perlu anda ketahui. Anda akan:

- Memahami proses yang menjadi asas pembelajaran mesin pada tahap tinggi.
- Menerokai konsep asas seperti 'model', 'ramalan', dan 'data latihan'.

## [Kuiz pra-ceramah](https://ff-quizzes.netlify.app/en/ml/)

[![ML untuk pemula - Teknik Pembelajaran Mesin](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML untuk pemula - Teknik Pembelajaran Mesin")

> 🎥 Klik imej di atas untuk video ringkas yang menerangkan pelajaran ini.

## Pengenalan

Pada tahap tinggi, kemahiran mencipta proses pembelajaran mesin (ML) terdiri daripada beberapa langkah:

1. **Tentukan soalan**. Kebanyakan proses ML bermula dengan mengajukan soalan yang tidak dapat dijawab oleh program kondisional mudah atau enjin berdasarkan peraturan. Soalan-soalan ini sering berkisar pada ramalan berdasarkan koleksi data.
2. **Kumpul dan sediakan data**. Untuk dapat menjawab soalan anda, anda memerlukan data. Kualiti dan, kadang-kala, kuantiti data anda akan menentukan sejauh mana anda dapat menjawab soalan awal anda. Memvisualisasikan data adalah aspek penting dalam fasa ini. Fasa ini juga termasuk membahagikan data kepada kumpulan latihan dan ujian untuk membina model.
3. **Pilih kaedah latihan**. Bergantung pada soalan anda dan sifat data anda, anda perlu memilih cara anda ingin melatih model agar dapat mencerminkan data anda dengan baik dan membuat ramalan yang tepat terhadapnya. Bahagian proses ML anda ini memerlukan kepakaran khusus dan, sering kali, sejumlah besar percubaan.
4. **Latih model**. Menggunakan data latihan anda, anda akan menggunakan pelbagai algoritma untuk melatih model mengenal pasti corak dalam data tersebut. Model mungkin menggunakan berat dalaman yang boleh diselaraskan untuk mengutamakan bahagian data tertentu bagi membina model yang lebih baik.
5. **Nilai model**. Anda menggunakan data yang belum pernah dilihat sebelum ini (data ujian anda) dari set data yang dikumpulkan untuk melihat bagaimana prestasi model.
6. **Penghalusan parameter**. Berdasarkan prestasi model anda, anda boleh mengulangi proses menggunakan parameter yang berbeza, atau pembolehubah, yang mengawal tingkah laku algoritma yang digunakan untuk melatih model.
7. **Buat Ramalan**. Gunakan input baru untuk menguji ketepatan model anda.

## Soalan yang Perlu Ditanya

Komputer sangat mahir dalam menemui corak tersembunyi dalam data. Kebolehan ini sangat membantu penyelidik yang mempunyai soalan tentang domain tertentu yang tidak dapat dijawab dengan mudah dengan membina enjin peraturan berasaskan kondisi. Sebagai contoh tugasan aktuari, seorang saintis data mungkin dapat membina peraturan buatan tangan mengenai kematian perokok vs bukan perokok.

Namun, apabila banyak pembolehubah lain dimasukkan ke dalam persamaan, model ML mungkin lebih cekap untuk meramalkan kadar kematian masa depan berdasarkan sejarah kesihatan lalu. Contoh yang lebih ceria mungkin membuat ramalan cuaca untuk bulan April di lokasi tertentu berdasarkan data yang termasuk latitud, longitud, perubahan iklim, kedekatan dengan lautan, corak aliran jet, dan banyak lagi.

✅ Dek slaid ini tentang model cuaca menawarkan perspektif sejarah untuk menggunakan ML dalam analisis cuaca.  

## Tugasan Pra-pembinaan

Sebelum mula membina model anda, terdapat beberapa tugasan yang perlu anda lengkapkan. Untuk menguji soalan anda dan membentuk hipotesis berdasarkan ramalan model, anda perlu mengenal pasti dan mengkonfigurasi beberapa elemen.

### Data

Untuk dapat menjawab soalan anda dengan sedikit kepastian, anda memerlukan jumlah data yang banyak dan jenis yang betul. Ada dua perkara yang perlu anda lakukan pada tahap ini:

- **Kumpul data**. Dengan mengambil kira pelajaran sebelumnya tentang keadilan dalam analisis data, kumpulkan data anda dengan berhati-hati. Sadarilah sumber data ini, sebarang bias yang wujud, dan dokumentasikan asal-usulnya.
- **Sediakan data**. Terdapat beberapa langkah dalam proses penyediaan data. Anda mungkin perlu mengumpul data dan menormalkannya jika ia berasal dari sumber yang berbeza. Anda boleh meningkatkan kualiti dan kuantiti data melalui pelbagai kaedah seperti menukar rentetan kepada nombor (seperti dalam [Pengelompokan](../../5-Clustering/1-Visualize/README.md)). Anda mungkin juga menjana data baru berdasarkan data asal (seperti dalam [Klasifikasi](../../4-Classification/1-Introduction/README.md)). Anda boleh membersihkan dan mengedit data (seperti yang akan kita lakukan sebelum pelajaran [Aplikasi Web](../../3-Web-App/README.md)). Akhir sekali, anda mungkin juga perlu mengacak dan mengaduk data bergantung pada teknik latihan anda.

✅ Setelah mengumpul dan memproses data anda, luangkan masa untuk melihat jika bentuknya membolehkan anda menangani soalan yang dimaksudkan. Mungkin data tersebut tidak akan berfungsi dengan baik dalam tugasan anda, seperti yang kita temui dalam pelajaran [Pengelompokan](../../5-Clustering/1-Visualize/README.md)!

### Ciri dan Sasaran

[Feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) adalah ciri yang boleh diukur dalam data anda. Dalam banyak set data, ia dinyatakan sebagai tajuk lajur seperti 'tarikh', 'saiz' atau 'warna'. Pembolehubah ciri anda, biasanya diwakili sebagai `X` dalam kod, mewakili pembolehubah input yang akan digunakan untuk melatih model.

Sasaran ialah perkara yang anda cuba ramalkan. Sasaran, biasanya diwakili sebagai `y` dalam kod, adalah jawapan kepada soalan yang anda cuba tanyakan pada data anda: pada bulan Disember, labu warna apa yang akan paling murah? di San Francisco, kejiranan mana yang akan mempunyai harga hartanah terbaik? Kadang-kadang sasaran juga dirujuk sebagai atribut label.

### Memilih pembolehubah ciri anda

🎓 **Pemilihan Ciri dan Pengekstrakan Ciri** Bagaimana anda tahu pembolehubah mana untuk dipilih semasa membina model? Anda mungkin akan melalui proses pemilihan ciri atau pengekstrakan ciri untuk memilih pembolehubah yang sesuai bagi model yang paling berprestasi. Namun, kedua-duanya tidak sama: "Pengekstrakan ciri mencipta ciri baru dari fungsi ciri asal, manakala pemilihan ciri mengembalikan subset ciri." ([sumber](https://wikipedia.org/wiki/Feature_selection))

### Visualisasikan data anda

Aspek penting dalam set peralatan saintis data adalah keupayaan untuk memvisualisasikan data menggunakan beberapa perpustakaan cemerlang seperti Seaborn atau MatPlotLib. Mewakili data secara visual mungkin membolehkan anda menemui korelasi tersembunyi yang boleh anda manfaatkan. Visualisasi anda juga mungkin membantu mendedahkan bias atau data yang tidak seimbang (seperti yang kita temui dalam [Klasifikasi](../../4-Classification/2-Classifiers-1/README.md)).

### Bahagikan set data anda

Sebelum latihan, anda perlu membahagikan set data anda kepada dua atau lebih bahagian saiz tidak sama yang masih mewakili data dengan baik.

- **Latihan**. Bahagian dataset ini digunakan untuk melatih model anda. Set ini mewakili sebahagian besar dataset asal.
- **Ujian**. Set data ujian adalah kumpulan data bebas, yang sering dikumpulkan dari data asal, yang anda gunakan untuk mengesahkan prestasi model yang dibina.
- **Pengesahan**. Set pengesahan adalah kumpulan kecil contoh bebas yang anda gunakan untuk melaras hiperaparameter model, atau seni bina, untuk memperbaiki model. Bergantung pada saiz data anda dan soalan yang anda ajukan, anda mungkin tidak perlu membina set ketiga ini (seperti yang kita catat dalam [Peramalan Siri Masa](../../7-TimeSeries/1-Introduction/README.md)).

## Membina model

Menggunakan data latihan anda, matlamat anda adalah membina model, atau representasi statistik data anda, menggunakan pelbagai algoritma untuk **melatih**nya. Melatih model mendedahkannya kepada data dan membolehkannya membuat andaian tentang corak yang dikesan, disahkan, dan diterima atau ditolak.

### Tentukan kaedah latihan

Bergantung pada soalan anda dan sifat data anda, anda akan memilih kaedah untuk melatihnya. Dengan melayari dokumentasi [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - yang kami gunakan dalam kursus ini - anda boleh menerokai banyak cara untuk melatih model. Bergantung pada pengalaman anda, anda mungkin perlu mencuba beberapa kaedah berbeza untuk membina model terbaik. Anda mungkin akan melalui proses di mana saintis data menilai prestasi model dengan memberikannya data yang belum dilihat, memeriksa ketepatan, bias, dan isu kualiti lain, serta memilih kaedah latihan yang paling sesuai untuk tugasan tersebut.

### Latih model

Dilengkapi dengan data latihan, anda bersedia 'menyesuaikan' untuk mencipta model. Anda akan perasan bahawa dalam banyak perpustakaan ML anda akan menemui kod 'model.fit' - pada masa ini anda menghantar pembolehubah ciri anda sebagai satu tatasusunan nilai (biasanya 'X') dan pembolehubah sasaran (biasanya 'y').

### Nilai model

Setelah proses latihan selesai (ia boleh mengambil banyak iterasi, atau 'epoch', untuk melatih model besar), anda akan dapat menilai kualiti model dengan menggunakan data ujian untuk mengukur prestasinya. Data ini adalah subset daripada data asal yang model tidak pernah analisis sebelum ini. Anda boleh mencetak jadual metrik mengenai kualiti model anda.

🎓 **Pemasangan model**

Dalam konteks pembelajaran mesin, pemasangan model merujuk kepada ketepatan fungsi asas model ketika berusaha menganalisis data yang tidak dikenali olehnya.

🎓 **Underfitting** dan **overfitting** adalah masalah biasa yang menurunkan kualiti model, kerana model tidak cukup sesuai atau terlalu sesuai. Ini menyebabkan model membuat ramalan yang sama ada terlalu rapat atau terlalu longgar dengan data latihannya. Model overfit meramalkan data latihan dengan baik kerana ia telah mempelajari butiran dan bunyi data terlalu baik. Model underfit tidak tepat kerana ia tidak dapat menganalisis dengan betul data latihannya maupun data yang belum pernah 'dilihat'.

![model overfitting](../../../../translated_images/ms/overfitting.1c132d92bfd93cb6.webp)
> Infografik oleh [Jen Looper](https://twitter.com/jenlooper)

## Penghalusan Parameter

Setelah latihan awal selesai, perhatikan kualiti model dan pertimbangkan untuk memperbaikinya dengan mengubah 'hiperparameter'. Baca lebih lanjut mengenai proses ini [dalam dokumentasi](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Ramalan

Ini adalah saat di mana anda boleh menggunakan data baru sepenuhnya untuk menguji ketepatan model anda. Dalam tetapan ML 'terap', di mana anda membina aset web untuk menggunakan model dalam pengeluaran, proses ini mungkin melibatkan pengumpulan input pengguna (contohnya, tekan butang) untuk menetapkan pembolehubah dan menghantarnya ke model bagi inferens atau penilaian.

Dalam pelajaran-pelajaran ini, anda akan mengetahui bagaimana menggunakan langkah-langkah ini untuk menyediakan, membina, menguji, menilai, dan meramalkan - semua gerak kerja seorang saintis data dan lebih lagi, semasa anda melangkah dalam perjalanan anda untuk menjadi jurutera ML 'full stack'.

---

## 🚀Cabaran

Lukis carta aliran yang mencerminkan langkah-langkah seorang pengamal ML. Di manakah anda melihat diri anda sekarang dalam proses ini? Di mana anda meramalkan anda akan menghadapi kesukaran? Apa yang kelihatan mudah bagi anda?

## [Kuiz pasca-ceramah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Pembelajaran Kendiri

Cari secara dalam talian temubual dengan saintis data yang membincangkan kerja harian mereka. Ini adalah [satu](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Tugasan

[Temubual seorang saintis data](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk ketepatan, sila ambil maklum bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang sahih. Untuk maklumat penting, terjemahan profesional oleh manusia adalah disyorkan. Kami tidak bertanggungjawab bagi sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->