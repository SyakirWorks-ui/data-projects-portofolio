import pandas as pd
import joblib
import os

# 1. Setup Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fraud_model.pkl')
# Kita gunakan data test yang sudah ada untuk simulasi
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'final_features_20k.csv')

def run_prediction_test():
    # Cek apakah model sudah ada
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model AI tidak ditemukan! Jalankan Tahap 4 terlebih dahulu.")
        return

    print("Memuat model AI dan data simulasi...")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH).sample(10, random_state=7) # Ambil 10 sampel acak

    # 2. Siapkan data untuk prediksi (sama dengan fitur di Tahap 4)
    # Kita hanya mengambil kolom fitur, tanpa kolom target 'isFraud'
    X_test = df.select_dtypes(include=['number']).drop(columns=['isFraud'], errors='ignore')
    
    # 3. Melakukan Prediksi
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1] # Kemungkinan Fraud dalam persen

    # 4. Tampilkan Hasil
    results = pd.DataFrame({
        'Amount': df['amount'],
        'Actual_Status': df['isFraud'].apply(lambda x: 'FRAUD' if x == 1 else 'NORMAL'),
        'AI_Prediction': ['FRAUD' if p == 1 else 'NORMAL' for p in predictions],
        'Confidence_Score': [f"{prob*100:.2f}%" for prob in probabilities]
    })

    print("\n" + "="*50)
    print("HASIL SIMULASI DETEKSI FRAUD OLEH AI")
    print("="*50)
    print(results)
    print("="*50)
    print("\nKeterangan: Confidence Score menunjukkan seberapa yakin AI bahwa itu adalah Fraud.")

if __name__ == "__main__":
    run_prediction_test()