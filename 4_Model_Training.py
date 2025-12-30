import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 1. Setup Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'final_features_20k.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'fraud_model.pkl')

def run_model_training():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] File tidak ditemukan di: {DATA_PATH}")
        return

    print("Loading final features...")
    df = pd.read_csv(DATA_PATH)

    # 2. Perbaikan Logika Pemilihan Fitur (X)
    # Kita hanya mengambil kolom yang bertipe angka dan bukan kolom target 'isFraud'
    # Cara ini jauh lebih aman dari error 'Column Not Found'
    X = df.select_dtypes(include=['number']).drop(columns=['isFraud'], errors='ignore')
    y = df['isFraud']

    print(f"Fitur yang digunakan: {list(X.columns)}")

    # 3. Split Data (80% Latihan, 20% Ujian)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Sedang melatih AI dengan {len(X_train)} data transaksi...")
    
    # 4. Melatih Model (Random Forest)
    # n_estimators=50 agar lebih cepat dan hemat RAM di laptop 4GB
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 5. Evaluasi Hasil Ujian AI
    y_pred = model.predict(X_test)
    
    print("\n" + "="*35)
    print("HASIL EVALUASI KECERDASAN AI")
    print("="*35)
    print(f"Akurasi: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("\nConfusion Matrix (Benar vs Salah Tebak):")
    print(confusion_matrix(y_test, y_pred))
    print("\nLaporan Detail:")
    print(classification_report(y_test, y_pred))
    print("="*35)

    # 6. Simpan "Otak" AI ke folder models
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\n[SUCCESS] Model AI berhasil disimpan di: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    run_model_training()