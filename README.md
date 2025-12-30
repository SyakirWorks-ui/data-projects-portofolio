# 🛡️End-to-End Financial Fraud Detection System (Optimized for Big Data)

## 📌 Deskripsi Proyek
Proyek ini mengimplementasikan sistem deteksi kecurangan (Fraud) pada transaksi keuangan menggunakan dataset PaySim. Tantangan utama proyek ini adalah mengolah **6,3 Juta baris data** secara efisien menggunakan perangkat dengan sumber daya terbatas (**RAM 4GB**).

## 🚀 Fitur Utama & Metodologi
- **Memory Management:** Menggunakan teknik *downcasting* dan *balanced sampling* (20.000 data) untuk menjaga penggunaan RAM di bawah 1 GB.
- **Advanced EDA:** Analisis korelasi dan distribusi untuk mengidentifikasi pola transaksi mencurigakan.
- **Feature Engineering:** Membuat fitur kalkulasi `errorBalance` untuk mendeteksi ketidakkonsistenan saldo saat transaksi terjadi.
- **Machine Learning:** Menggunakan algoritma **Random Forest** yang berhasil mendeteksi fraud dengan **Confidence Score hingga 100%**.

## 📁 Struktur Folder
- `data/`: Dataset mentah dan dataset hasil sampling.
- `notebooks/`: Script Python tahap demi tahap (Cleaning -> EDA -> Engineering -> Training -> Testing).
- `models/`: File biner model AI (`fraud_model.pkl`).
- `visualizations/`: Hasil grafik analisis data.

## 📊 Hasil Uji Coba (Inference)
Model mampu membedakan transaksi **Normal** dan **Fraud** secara akurat bahkan pada transaksi bernilai besar (10.000.000).

---
**Created by:** [Syakir_Works
]