import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# --- 1. SETTING HALAMAN & TEMA (DESAIN MODERN) ---
st.set_page_config(page_title="Financial Fraud Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS untuk tampilan profesional dan mirip contoh Google
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #262626; /* Dark background */
        color: #E0E0E0; /* Light text */
    }
    /* Header and subheader styling */
    h1, h2, h3, h4, h5, h6 {
        color: #00D1B2; /* Neon green for titles */
    }
    /* Metric cards styling */
    .stMetric {
        background-color: #333333; /* Slightly lighter dark for cards */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3); /* Softer shadow */
        color: #E0E0E0;
    }
    .stMetric label {
        color: #A0A0A0; /* Label color */
    }
    .stMetric .big-font {
        font-size: 2.5em !important;
        font-weight: bold;
        color: #00D1B2; /* Neon green for main metric value */
    }
    /* Dataframe styling */
    .stDataFrame {
        background-color: #333333;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .stPlotlyChart {
        background-color: #333333;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        padding: 10px;
    }
    .css-1d391kg { /* sidebar background */
        background-color: #262626 !important;
    }
    /* Adjust Streamlit specific elements */
    .st-emotion-cache-nahz7x { /* Adjust padding for column alignment */
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIKA PATH OTOMATIS (Penting agar tidak error) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data', 'processed', 'final_features_20k.csv')
model_path = os.path.join(current_dir, 'models', 'fraud_model.pkl')

# --- 3. LOAD DATA & MODEL (Menggunakan cache untuk performa) ---
@st.cache_data
def load_assets():
    try:
        df = pd.read_csv(data_path)
        model = joblib.load(model_path)
        return df, model
    except FileNotFoundError as e:
        st.error(f"Error loading file: {e}. Pastikan `app.py` ada di folder utama proyek dan jalur file sudah benar.")
        st.stop() # Hentikan eksekusi jika file tidak ditemukan

df, model = load_assets()

# --- 4. PREDIKSI UNTUK DATA PADA DASHBOARD ---
# Ini penting agar dashboard memiliki kolom 'prediction' dan 'probability'
X_data = df.select_dtypes(include=['number']).drop(columns=['isFraud'], errors='ignore')
df['prediction'] = model.predict(X_data)
df['probability'] = model.predict_proba(X_data)[:, 1] # Probability of being fraud

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Fraud Detection Settings")
    st.markdown("---")
    st.subheader("Filter Data")
    
    # Filter untuk menampilkan hanya transaksi fraud atau semua
    filter_type = st.radio(
        "Show Transactions:",
        ("All", "Fraudulent Only"),
        index=1 # Default: Fraudulent Only
    )
    
    if filter_type == "Fraudulent Only":
        filtered_df = df[df['prediction'] == 1].copy()
    else:
        filtered_df = df.copy()

    st.markdown("---")
    st.info("Dashboard ini menampilkan deteksi fraud menggunakan model Random Forest.")

# --- 6. HEADER & KPI (Sesuai Layout Google) ---
st.title("Financial Fraud Analysis Dashboard")
st.markdown("<p style='color:#A0A0A0;'>Real-time insights into suspicious financial transactions.</p>", unsafe_allow_html=True)

st.write("") # Spacer

total_transactions = len(filtered_df)
total_fraud = filtered_df[filtered_df['prediction'] == 1].shape[0]
percentage_fraud = (total_fraud / total_transactions) * 100 if total_transactions > 0 else 0
total_fraud_amount = filtered_df[filtered_df['prediction'] == 1]['amount'].sum()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='stMetric'><label>Total Transactions</label><p class='big-font'>{total_transactions:,}</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='stMetric'><label>Fraudulent Cases</label><p class='big-font'>{total_fraud:,}</p></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='stMetric'><label>Total Fraud Amount</label><p class='big-font'>${total_fraud_amount:,.0f}</p></div>", unsafe_allow_html=True)

st.write("") # Spacer

# --- 7. VISUALISASI UTAMA (Meniru Layout Google) ---
st.subheader("Transaction Activity Overview")
row_viz1_col1, row_viz1_col2 = st.columns(2)

with row_viz1_col1:
    st.markdown("#### Fraud by Transaction Type")
    # Membuat kolom 'type' kembali dari one-hot encoding
    type_columns = [col for col in df.columns if col.startswith('type_')]
    df_temp = filtered_df.copy()
    df_temp['transaction_type'] = 'Unknown'
    for col in type_columns:
        df_temp.loc[df_temp[col] == 1, 'transaction_type'] = col.replace('type_', '')
    
    fraud_by_type = df_temp[df_temp['prediction'] == 1].groupby('transaction_type').size().reset_index(name='count')
    fig_type = px.bar(fraud_by_type, y='transaction_type', x='count', 
                      orientation='h', title='Fraudulent Transaction Types',
                      color_discrete_sequence=['#00D1B2'], template="plotly_dark")
    st.plotly_chart(fig_type, use_container_width=True)

with row_viz1_col2:
    st.markdown("#### Fraudulent Trend Over Steps (Hours)")
    fraud_trend = filtered_df[filtered_df['prediction'] == 1].groupby('step')['amount'].sum().reset_index()
    fig_trend = px.line(fraud_trend, x='step', y='amount', title='Total Fraud Amount by Step',
                        color_discrete_sequence=['#FFC107'], template="plotly_dark") # Amber color
    st.plotly_chart(fig_trend, use_container_width=True)

st.write("") # Spacer

row_viz2_col1, row_viz2_col2 = st.columns(2)
with row_viz2_col1:
    st.markdown("#### Distribution of Balance Errors in Fraud Cases")
    # Menggunakan errorBalanceOrig dan errorBalanceDest
    # Menunjukkan seberapa besar manipulasi saldo yang terjadi
    fraud_errors = filtered_df[filtered_df['prediction'] == 1]
    fig_error_orig = px.histogram(fraud_errors, x='errorBalanceOrig', nbins=50, 
                                  title='Error in Sender Balance (Original)',
                                  color_discrete_sequence=['#E91E63'], template="plotly_dark") # Pink color
    st.plotly_chart(fig_error_orig, use_container_width=True)
    
with row_viz2_col2:
    st.markdown("#### Fraud Probability Distribution")
    fig_prob = px.histogram(filtered_df[filtered_df['prediction'] == 1], x='probability', nbins=20,
                            title='AI Confidence in Fraud Detection',
                            color_discrete_sequence=['#1E88E5'], template="plotly_dark") # Blue color
    st.plotly_chart(fig_prob, use_container_width=True)


st.write("") # Spacer

# --- 8. TABEL DETAIL TRANSAKSI (Mirip Google) ---
st.subheader("Recent Suspicious Transactions")
st.markdown("<p style='color:#A0A0A0;'>Details of transactions flagged by the AI model.</p>", unsafe_allow_html=True)

# Menampilkan kolom-kolom yang relevan
display_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                'oldbalanceDest', 'newbalanceDest', 'errorBalanceOrig', 'errorBalanceDest', 
                'prediction', 'probability']

# Menyesuaikan 'type' agar muncul nama aslinya, bukan dummy
df_for_display = filtered_df.copy()
type_map = {
    'type_CASH_IN': 'CASH_IN', 'type_CASH_OUT': 'CASH_OUT',
    'type_DEBIT': 'DEBIT', 'type_PAYMENT': 'PAYMENT', 'type_TRANSFER': 'TRANSFER'
}
# Ini akan membuat kolom 'type' yang bisa dibaca manusia
df_for_display['type'] = df_for_display[type_columns].idxmax(axis=1).map(type_map).fillna('UNKNOWN')

st.dataframe(df_for_display[display_cols].sort_values(by='probability', ascending=False).head(20), use_container_width=True)

st.markdown("---")
st.caption("AI Model developed by [Your Name] for Financial Fraud Detection.")