import pandas as pd
import os

# 1. Setup Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'balanced_sample_20k.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'final_features_20k.csv')

def run_feature_engineering():
    print("Loading data for feature engineering...")
    df = pd.read_csv(INPUT_PATH)

    # 2. Membuat Fitur 'errorBalanceOrig'
    # Detecting if the transaction amount matches the balance change
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']

    # 3. Membuat Fitur 'errorBalanceDest'
    # Detecting if the recipient's balance increased correctly
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

    # 4. One-Hot Encoding for 'type'
    # Converting text categories into numbers for the AI model
    df = pd.get_dummies(df, columns=['type'], prefix='type')

    # 5. Save the final dataset
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    
    print("-" * 30)
    print(f"[SUCCESS] Feature Engineering Finished!")
    print(f"Dataset stored at: {OUTPUT_PATH}")
    print("-" * 30)

if __name__ == "__main__":
    run_feature_engineering()