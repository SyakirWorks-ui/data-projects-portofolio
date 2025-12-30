import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Setup Path (Membaca data hasil sampling Tahap 1)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'balanced_sample_20k.csv')
SAVE_VIZ = os.path.join(BASE_DIR, 'visualizations')

# Pastikan folder visualisasi tersedia
os.makedirs(SAVE_VIZ, exist_ok=True)

def run_strategic_eda():
    print("Loading balanced dataset for EDA...")
    df = pd.read_csv(DATA_PATH)

    # --- ANALISIS 1: Distribusi Fraud berdasarkan Tipe Transaksi ---
    plt.figure(figsize=(10, 6))
    # Melihat tipe transaksi apa yang paling sering disalahgunakan untuk Fraud
    sns.countplot(data=df, x='type', hue='isFraud', palette='viridis')
    plt.title('Fraud vs Normal Transactions by Type')
    plt.xlabel('Transaction Type')
    plt.ylabel('Count')
    plt.legend(title='Is Fraud?', labels=['Normal', 'Fraud'])
    
    # Simpan grafik
    plt.savefig(os.path.join(SAVE_VIZ, '1_fraud_by_type.png'))
    print("[SUCCESS] Grafik 1 disimpan: fraud_by_type.png")
    plt.show()

    # --- ANALISIS 2: Korelasi Antar Fitur (Heatmap) ---
    plt.figure(figsize=(12, 8))
    # Hanya menghitung korelasi untuk kolom angka
    numeric_df = df.select_dtypes(include=['float32', 'float64', 'int8', 'int16', 'int64'])
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    
    plt.savefig(os.path.join(SAVE_VIZ, '2_correlation_heatmap.png'))
    print("[SUCCESS] Grafik 2 disimpan: correlation_heatmap.png")
    plt.show()

    # --- ANALISIS 3: Boxplot Jumlah Transaksi (Amount) ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='isFraud', y='amount', palette='Set2')
    plt.yscale('log') # Gunakan skala log karena rentang angka 'amount' sangat besar
    plt.title('Distribution of Transaction Amount (Log Scale)')
    plt.xticks([0, 1], ['Normal', 'Fraud'])
    
    plt.savefig(os.path.join(SAVE_VIZ, '3_amount_distribution.png'))
    print("[SUCCESS] Grafik 3 disimpan: amount_distribution.png")
    plt.show()

    # Insight Ringkas
    print("\n" + "="*30)
    print("STRATEGIC INSIGHTS:")
    print("1. Cek Grafik 1: Biasanya Fraud hanya terjadi di tipe 'TRANSFER' dan 'CASH_OUT'.")
    print("2. Cek Grafik 2: Perhatikan korelasi antara oldbalanceOrg dan amount.")
    print("3. Cek Grafik 3: Apakah transaksi Fraud cenderung memiliki nominal lebih besar?")
    print("="*30)

if __name__ == "__main__":
    run_strategic_eda()