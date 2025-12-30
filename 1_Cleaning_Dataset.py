import pandas as pd
import numpy as np
import gc
import os

# 1. Setup Path yang dinamis (Menyesuaikan otomatis dengan lokasi folder proyek)
# Script ini akan mencari folder 'data/raw' dari folder utama proyek
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'PS_20174392719_1491204439457_log.csv')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# 2. Tipe data hemat RAM
param_dtypes = {
    'step': 'int16',
    'type': 'category',
    'amount': 'float32',
    'oldbalanceOrg': 'float32',
    'newbalanceOrig': 'float32',
    'oldbalanceDest': 'float32',
    'newbalanceDest': 'float32',
    'isFraud': 'int8'
}

cols_to_use = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
               'oldbalanceDest', 'newbalanceDest', 'isFraud']

def start_phase_1(n_samples=20000):
    print(f"Checking file at: {RAW_DATA_PATH}")
    
    if not os.path.exists(RAW_DATA_PATH):
        print("[ERROR] File CSV tetap tidak ditemukan. Pastikan nama file benar!")
        return

    try:
        print("Reading large dataset (Optimized)...")
        # Load data dengan efisiensi RAM tinggi
        full_df = pd.read_csv(RAW_DATA_PATH, dtype=param_dtypes, usecols=cols_to_use)
        
        print(f"Total rows found: {len(full_df):,}")
        print("Creating a balanced sample (10k Fraud & 10k Normal)...")
        
        # Mengambil semua data Fraud (~8,213 baris)
        fraud_df = full_df[full_df['isFraud'] == 1]
        
        # Mengambil sisa sampel dari data Normal secara acak
        needed_normal = n_samples - len(fraud_df)
        normal_df = full_df[full_df['isFraud'] == 0].sample(n=needed_normal, random_state=42)
        
        # Gabungkan dan acak urutannya
        balanced_df = pd.concat([fraud_df, normal_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Bersihkan RAM segera
        del full_df, fraud_df, normal_df
        gc.collect()
        
        # Simpan hasil ke folder 'processed' agar tahap selanjutnya ringan (hanya 1-2 MB)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        save_path = os.path.join(PROCESSED_DATA_DIR, 'balanced_sample_20k.csv')
        balanced_df.to_csv(save_path, index=False)
        
        print("-" * 30)
        print(f"[SUCCESS] Phase 1 Completed!")
        print(f"Sample saved to: {save_path}")
        print(f"Final RAM usage for this sample: {balanced_df.memory_usage().sum() / 1024**2:.2f} MB")
        print("-" * 30)
        
        print("\nPreview of Balanced Data:")
        print(balanced_df.head())
        
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    start_phase_1(20000)