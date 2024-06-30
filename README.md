# Dashboard Sentimen Pasar Keuangan

## Deskripsi Proyek
---------------------
Proyek ini merupakan sebuah dashboard prediksi sentimen pasar keuangan yang bertujuan untuk menganalisis berita keuangan dan memberikan prediksi sentimen (positif, negatif, atau netral) dari berita tersebut. Dashboard ini dibangun menggunakan berbagai teknik Machine Learning dan menyediakan antarmuka yang intuitif untuk pengguna.

## Sumber Data
Data Training:
[Financial Phrase Bank](https://huggingface.co/datasets/financial_phrasebank) (Hugging Face)

Data berita diperoleh dari dua sumber yang berbeda:
1. [DailyFX](https://www.dailyfx.com) (Kategori: Forex)
2. [The Economic Times](https://economictimes.indiatimes.com) (Kategori: Banking, Economy, Market, Forex)


## Hasil Komparasi Algoritma Machine Learning:
----------------------------------------------
| Algoritma    | Akurasi | Precision (avg) | Recall (avg) | F1-Score (avg) |
|--------------|---------|-----------------|--------------|----------------|
| Naive Bayes  | 0.735   | 0.73            | 0.74         | 0.72           |
| SVM          | 0.774   | 0.78            | 0.77         | 0.76           |
| TensorFlow   | 0.756   | 0.75            | 0.76         | 0.75           |


## Penjelasan Hasil Komparasi:

Dalam proyek ini, kami menggunakan tiga algoritma machine learning yang berbeda untuk memprediksi sentimen pasar keuangan berdasarkan berita: Naive Bayes, Support Vector Machine (SVM), dan model TensorFlow. Berikut adalah penjelasan hasil komparasi dari ketiga algoritma tersebut:

1. **Naive Bayes:**
   - **Akurasi:** 0.735
   - **Precision (avg):** 0.73
   - **Recall (avg):** 0.74
   - **F1-Score (avg):** 0.72

Naive Bayes menunjukkan performa yang baik dalam hal akurasi dan recall, namun memiliki f1-score yang sedikit lebih rendah dibandingkan dengan SVM dan TensorFlow. Algoritma ini cocok untuk digunakan pada data dengan distribusi kelas yang seimbang.

2. **SVM (Support Vector Machine):**
   - **Akurasi:** 0.774
   - **Precision (avg):** 0.78
   - **Recall (avg):** 0.77
   - **F1-Score (avg):** 0.76

SVM memberikan hasil terbaik di antara ketiga algoritma dengan akurasi, precision, recall, dan f1-score yang lebih tinggi. Algoritma ini sangat efektif dalam menangani data dengan kelas yang tidak seimbang dan menunjukkan kemampuan generalisasi yang baik.

3. **TensorFlow:**
   - **Akurasi:** 0.756
   - **Precision (avg):** 0.75
   - **Recall (avg):** 0.76
   - **F1-Score (avg):** 0.75

Model TensorFlow menunjukkan performa yang kompetitif dengan SVM dan sedikit lebih baik dibandingkan dengan Naive Bayes. Model ini memberikan keseimbangan yang baik antara precision dan recall, membuatnya cocok untuk aplikasi yang memerlukan keseimbangan antara prediksi positif dan negatif yang benar.

Instalasi
============
### Clone Repositori
-----------
```bash
$ git clone https://github.com/mushabtinumbang/market-dashboard.git
$ cd market-dashboard
```
### Membuat Environment
-----------
Langkah selanjutnya untuk menjalakan program ini adalah untuk membuat environment conda. Hal ini bertujuan untuk memastikan semua dependensi dan library yang dipakai nantinya menggunakan versi yang sama dan tidak menghasilkan sebuah error. Untuk menginstal environment, user dapat menjalankan script ini pada terminal.
```bash
$ make create-env
```


Menjalankan Program
===
### Scrape dan Prediksi Data Berita
-----------
Script di bawah ini digunakan untuk menjalankan program utama. Dengan script ini, pengguna dapat melakukan scraping secara real-time pada laman berita keuangan. Pengguna juga bisa mengatur tanggal berita yang ingin di-scrape.

Setelah scraping, pengguna juga bisa langsung memprediksi sentimen dari berita-berita tersebut. Hasil prediksi kemudian akan diolah dan ditampilkan di antarmuka Streamlit.


| Command         | Deskripsi |
|-----------------|------------|
| DATE            | Menentukan periode data yang ingin diproses. Contoh: 'latest' atau jika ingin rentang tanggal, masukkan seperti ini: "2020-01-02\|2020-01-03" (beberapa tanggal dipisahkan dengan \|). |
| DAILYFX         | Menentukan apakah ingin melakukan scraping dari DailyFX atau tidak. Pilih 'y' untuk ya dan 'n' untuk tidak. |
| ECONTIMES       | Menentukan apakah ingin melakukan scraping dari Economic Times atau tidak. Pilih 'y' untuk ya dan 'n' untuk tidak. |
| SUFFIX          | Menentukan akhiran nama file output. |
| RUN_SCRAPER     | Menentukan apakah ingin menjalankan fungsi scraper atau tidak. Pilih 'y' untuk ya dan 'n' untuk tidak. |
| RUN_PREDICTION  | Menentukan apakah ingin menjalankan prediksi menggunakan model machine learning atau tidak. Pilih 'y' untuk ya dan 'n' untuk tidak. |
| PREPARE_STREAMLIT | Menentukan apakah ingin memproses data lebih lanjut untuk Streamlit atau tidak. Pilih 'y' untuk ya dan 'n' untuk tidak. |

Berdasarkan parameter-parameter yang ada, pengguna dapat mengubah tanggal, mengatur laman berita untuk scraping, hingga mengatur pipeline yang akan dijalankan nantinya. Berikut adalah contoh script yang dapat dijalankan.

```bash
$ export DATE='01-06-2024|30-06-2024' &&
export DAILYFX='y' &&
export ECONTIMES='y' &&
export SUFFIX='' &&
export RUN_SCRAPER='y' &&
export RUN_PREDICTION='y' &&
export PREPARE_STREAMLIT='y' &&
make predict-sentiments
```

### Streamlit
-----------
Untuk menjalankan aplikasi Streamlit, gunakan perintah berikut.
```bash
make run-streamlit
```

## Citation

Proyek ini dibuat oleh Mushab Tinumbang di bawah [Universitas Padjadjaran](https://www.unpad.ac.id/).

