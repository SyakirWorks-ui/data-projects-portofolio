import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# --- 1. CONFIG HALAMAN ---
st.set_page_config(page_title="Financial Fraud Dashboard", layout="wide")

# --- 2. ULTIMATE CSS (100% VISUAL MATCH) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Background Utama */
    .stApp {
        background-color: #0E1117; 
    }
    
    /* Font Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Styling Kartu KPI (Gaya Glassmorphism) */
    [data-testid="stMetric"] {
        background: rgba(217, 217, 217, 0.9) !important; /* Abu terang sesuai gambar */
        border-radius: 12px !important;
        padding: 15px 20px !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4) !important;
        border: none !important;
    }

    /* Label KPI */
    [data-testid="stMetricLabel"] {
        color: #2C3E50 !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        text-transform: uppercase;
    }

    /* Angka KPI (Neon Glow) */
    [data-testid="stMetricValue"] {
        color: #00A67E !important; 
        font-size: 38px !important;
        font-weight: 800 !important;
        text-shadow: 0 0 5px rgba(0, 166, 126, 0.2);
    }

    /* Header & Subheader */
    h1 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: -1px;
    }
    
    p {
        color: #A0A0A0 !important;
    }

    /* Mengurangi Jarak Antar Elemen agar Padat */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD DATA & MODEL ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data', 'processed', 'final_features_20k.csv')
model_path = os.path.join(current_dir, 'models', 'fraud_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    # Re-map transaction type
    type_cols = [c for c in df.columns if 'type_' in c]
    df['Category'] = df[type_cols].idxmax(axis=1).str.replace('type_', '')
    return df, model

df, model = load_data()

# --- 4. PREDIKSI ---
X = df.select_dtypes(include=['number']).drop(columns=['isFraud'], errors='ignore')
df['is_fraud_pred'] = model.predict(X)
df['prob'] = model.predict_proba(X)[:, 1]

# Ambil hanya data yang terdeteksi fraud untuk dashboard
fraud_df = df[df['is_fraud_pred'] == 1].copy()

# --- 5. HEADER ---
st.title("Dashboard for real time credit card fraud detection")
st.write("This automated AI system assists in monitoring and preventing fraudulent financial activities effectively.")

# --- 6. BARIS 1: KPI METRICS (3 Kolom) ---
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Fraudulent transactions", f"{len(fraud_df)}")
with m2:
    perc = (len(fraud_df)/len(df))*100
    st.metric("% Fraudulent transactions", f"{perc:.3f}%")
with m3:
    total_val = fraud_df['amount'].sum()
    st.metric("Total fraud amount", f"${total_val/1000:,.1f}K")

st.write("") # Spacer

# --- 7. BARIS 2: VISUALISASI UTAMA ---
c1, c2 = st.columns([1.3, 2])

with c1:
    st.markdown("### Fraudulent transactions by location")
    # Ganti Map dengan Heatmap yang estetik
    fig_map = px.density_heatmap(fraud_df, x="step", y="amount", 
                                 color_continuous_scale='Viridis', template="plotly_dark")
    fig_map.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_map, use_container_width=True)

with c2:
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("### Fraud by category")
        cat_data = fraud_df.groupby('Category').size().reset_index(name='count')
        fig_bar = px.bar(cat_data, x='count', y='Category', orientation='h', 
                         color_discrete_sequence=['#00D1B2'], template="plotly_dark")
        fig_bar.update_layout(margin=dict(l=0,r=10,t=20,b=0), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with r2c2:
        st.markdown("### Average fraud trend")
        trend = fraud_df.groupby('step').size().reset_index(name='val')
        fig_line = px.line(trend, x='step', y='val', color_discrete_sequence=['#00D1B2'], template="plotly_dark")
        fig_line.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig_line, use_container_width=True)

# --- 8. BARIS 3: RISK & TABLE ---
st.write("")
c3_1, c3_2 = st.columns([1, 1])

with c3_1:
    st.markdown("### Fraud percentage by risk")
    fraud_df['risk'] = pd.cut(fraud_df['prob'], bins=[0, 0.4, 0.7, 1.0], labels=['Low', 'Medium', 'High'])
    risk_plot = fraud_df.groupby('risk').size().reset_index(name='count')
    fig_risk = px.bar(risk_plot, x='risk', y='count', color='risk',
                      color_discrete_map={'Low':'#2ECC71', 'Medium':'#F1C40F', 'High':'#E74C3C'}, template="plotly_dark")
    fig_risk.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=300, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_risk, use_container_width=True)

with c3_2:
    st.markdown("### Fraudulent transactions list")
    table_df = fraud_df[['Category', 'amount', 'step']].head(8)
    table_df.columns = ['Type', 'Amount ($)', 'Time Step']
    st.dataframe(table_df, use_container_width=True, height=260)

st.markdown("---")
st.caption("AI Model developed for Financial Fraud Detection Portofolio")