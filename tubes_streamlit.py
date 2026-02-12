import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import folium
from streamlit_folium import st_folium

import os

import tensorflow as tf
st.write(tf.__version__)

# --- TAMBAHKAN BAGIAN INI DI PALING ATAS (BARIS 1) ---
# Mengatasi error mutex/lock pada macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# -----------------------------------------------------

# ==========================================
# 1. KONFIGURASI HALAMAN & JUDUL
# ==========================================
st.set_page_config(
    page_title="Beijing Air Quality Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #4CAF50; text-align: center; font-weight: bold;}
    .sub-header {font-size: 1.5rem; color: #333; margin-top: 20px;}
    .insight-box {background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🌍 Beijing Air Quality Analysis & Forecasting</div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center;">
    <b>Kelompok IF1 - 10124010</b><br>
    Rafi Asshiddiqie Tanujaya (10124010) | Muhammad Fauzi (10124017) | Khairul Indra S (10124006)
</div>
<hr>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================

# Koordinat Stasiun (Geo-Analysis Data)
STATION_COORDS = {
    "Aotizhongxin": [39.982, 116.397],
    "Changping": [40.217, 116.230],
    "Dingling": [40.292, 116.220],
    "Dongsi": [39.929, 116.417],
    "Guanyuan": [39.929, 116.339],
    "Gucheng": [39.914, 116.184],
    "Huairou": [40.328, 116.628],
    "Nongzhanguan": [39.937, 116.461],
    "Shunyi": [40.127, 116.655],
    "Tiantan": [39.886, 116.407],
    "Wanliu": [39.987, 116.287],
    "Wanshouxigong": [39.878, 116.352]
}

@st.cache_data
def load_data():
    """Memuat semua file CSV PRSA Data dari direktori saat ini."""
    # Mencari semua file csv yang sesuai pola
    csv_files = []

    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        st.error("Dataset tidak ditemukan! Pastikan file CSV (PRSA_Data_...) berada di folder yang sama.")
        return pd.DataFrame()

    df_list = []
    for filename in csv_files:
        temp_df = pd.read_csv(filename)
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Data Cleaning & Datetime
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    
    # Handling Missing Values (Forward Fill sederhana untuk dashboard)
    cols_to_fill = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    df[cols_to_fill] = df[cols_to_fill].fillna(method='ffill')
    
    return df

try:
    with st.spinner('Memuat dataset...'):
        df = load_data()
        if df.empty:
            st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
menu = st.sidebar.radio("Navigasi", ["Geo-Analysis 🗺️", "Exploratory Data Analysis 📊", "PM2.5 Prediction (LSTM) 🤖", "Kesimpulan 📝"])

# Filter Tahun di Sidebar (Global)
year_list = sorted(df['year'].unique())
selected_year = st.sidebar.selectbox("Pilih Tahun (untuk visualisasi)", year_list)

# Filter Data berdasarkan tahun
df_filtered = df[df['year'] == selected_year]

# ==========================================
# 4. HALAMAN: GEO-ANALYSIS
# ==========================================
if menu == "Geo-Analysis 🗺️":
    st.subheader(f"🗺️ Peta Persebaran Polusi Udara - Tahun {selected_year}")
    
    st.markdown("""
    Halaman ini menampilkan lokasi stasiun pemantauan kualitas udara di Beijing. 
    Warna dan ukuran lingkaran merepresentasikan tingkat rata-rata PM2.5 di stasiun tersebut.
    """)
    
    # Hitung rata-rata per stasiun untuk tahun terpilih
    station_stats = df_filtered.groupby('station')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean().reset_index()
    
    # Tambahkan koordinat
    station_stats['lat'] = station_stats['station'].map(lambda x: STATION_COORDS.get(x, [0,0])[0])
    station_stats['lon'] = station_stats['station'].map(lambda x: STATION_COORDS.get(x, [0,0])[1])
    
    # Peta Folium
    m = folium.Map(location=[40.0, 116.4], zoom_start=9)
    
    # Tambahkan Marker
    for _, row in station_stats.iterrows():
        # Tentukan warna berdasarkan PM2.5 (Indeks Kualitas Udara sederhana)
        pm25 = row['PM2.5']
        color = 'green' if pm25 <= 50 else 'yellow' if pm25 <= 100 else 'orange' if pm25 <= 150 else 'red'
        
        popup_text = f"""
        <b>{row['station']}</b><br>
        PM2.5: {pm25:.2f}<br>
        PM10: {row['PM10']:.2f}<br>
        SO2: {row['SO2']:.2f}
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=10 + (pm25/10), # Ukuran berdasarkan polusi
            popup=popup_text,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(m)

    st_folium(m, width=1000, height=500)
    
    st.markdown("""
    <div class="insight-box">
    <b>Insight Geo-Location:</b><br>
    Stasiun di wilayah selatan dan pusat kota cenderung memiliki tingkat polusi lebih tinggi dibandingkan wilayah utara (seperti Dingling dan Huairou) yang lebih dekat dengan pegunungan dan jauh dari pusat industri/trafik padat.
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. HALAMAN: EDA
# ==========================================
elif menu == "Exploratory Data Analysis 📊":
    st.subheader("📊 Analisis Eksplorasi Data (EDA)")
    
    tab1, tab2, tab3 = st.tabs(["Tren Waktu", "Korelasi", "Perbandingan Stasiun"])
    
    with tab1:
        st.write("### Tren Polutan Bulanan")
        # Agregasi Bulanan
        monthly_trend = df_filtered.groupby('month')[['PM2.5', 'PM10', 'SO2', 'NO2', 'O3']].mean()
        
        # Plot menggunakan Plotly
        fig_trend = px.line(monthly_trend, x=monthly_trend.index, y=['PM2.5', 'PM10', 'SO2', 'NO2', 'O3'],
                            title=f"Rata-rata Polutan Bulanan Tahun {selected_year}",
                            markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.info("Pola Musiman: Terlihat peningkatan PM2.5 yang signifikan di bulan-bulan musim dingin (Desember-Februari) akibat pemanas ruangan dan inversi suhu.")

    with tab2:
        st.write("### Heatmap Korelasi Antar Variabel")
        cols_corr = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        corr_matrix = df_filtered[cols_corr].corr()
        
        fig_corr = plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(fig_corr)
        
        st.markdown("""
        **Analisis Korelasi:**
        - **PM2.5 & PM10:** Korelasi positif sangat kuat (partikel debu).
        - **PM2.5 & TEMP:** Korelasi negatif (suhu dingin cenderung meningkatkan PM2.5).
        - **PM2.5 & WSPM:** Korelasi negatif (angin kencang membantu menyebarkan polutan).
        """)

    with tab3:
        st.write("### Distribusi PM2.5 per Stasiun")
        fig_box = px.box(df_filtered, x='station', y='PM2.5', color='station',
                         title=f"Distribusi PM2.5 per Stasiun ({selected_year})")
        st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# 6. HALAMAN: PREDIKSI (LSTM)
# ==========================================
elif menu == "PM2.5 Prediction (LSTM) 🤖":
    st.subheader("🤖 Prediksi PM2.5 Menggunakan LSTM")
    
    st.markdown("""
    Model Deep Learning (LSTM) digunakan untuk memprediksi konsentrasi PM2.5 satu jam ke depan berdasarkan data historis 24 jam terakhir.
    """)
    
    # Load Model
    model_path = "./model/pm25_lstm_model.keras"
    
    if not os.path.exists(model_path):
        st.warning(f"File model '{model_path}' tidak ditemukan. Silakan upload file model .keras Anda.")
    else:
        try:
            model = load_model(model_path)
            
            # Persiapan Data Input untuk Demo
            st.write("#### Simulasi Prediksi")
            st.write("Pilih stasiun dan waktu untuk mengambil 24 jam data sebelumnya sebagai input model.")
            
            col1, col2 = st.columns(2)
            with col1:
                pred_station = st.selectbox("Pilih Stasiun", df['station'].unique())
            with col2:
                # Batasi tanggal agar aman (tidak di awal dataset)
                min_date = df['datetime'].min() + pd.Timedelta(days=2)
                max_date = df['datetime'].max()
                pred_date = st.date_input("Pilih Tanggal", value=max_date, min_value=min_date, max_value=max_date)
            
            pred_hour = st.slider("Pilih Jam", 0, 23, 12)
            
            # Ambil Data
            target_time = pd.to_datetime(f"{pred_date} {pred_hour}:00:00")
            
            # Filter data stasiun
            df_station = df[df['station'] == pred_station].sort_values('datetime')
            
            # Cari indeks waktu target
            mask = df_station['datetime'] == target_time
            
            if mask.sum() > 0:
                idx = df_station.index[df_station['datetime'] == target_time][0]
                # Ambil 24 jam sebelumnya (LSTM butuh sequence)
                # Pastikan fiturnya sesuai dengan saat training (Analyst (3).ipynb)
                # Biasanya fitur numerik
                feature_cols = ['PM2.5', 'PM10','SO2','NO2','CO','O3']
                
                # Kita perlu mengambil 24 data sebelumnya
                # Locating by position in filtered df might be safer
                pos = df_station.index.get_loc(idx)
                
                if pos >= 24:
                    input_data = df_station.iloc[pos-24:pos][feature_cols]
                    actual_val = df_station.iloc[pos]['PM2.5']
                    
                    st.write("Data Input (24 Jam Terakhir):")
                    st.dataframe(input_data.tail())
                    
                    if st.button("Jalankan Prediksi"):
                        # Preprocessing (Scaling)
                        # PENTING: Scaler harus sama dengan training. Karena file scaler tidak ada,
                        # kita fit scaler baru pada data input (ini pendekatan aproksimasi untuk demo)
                        # atau idealnya fit pada seluruh data latih dulu. 
                        # Di sini kita fit pada sampel data stasiun agar range-nya masuk akal.
                        scaler = MinMaxScaler()
                        scaler.fit(df_station[feature_cols]) # Fit pada sejarah stasiun tsb
                        
                        input_scaled = scaler.transform(input_data)
                        input_reshaped = input_scaled.reshape(1, 24, len(feature_cols))
                        
                        # Prediksi
                        prediction_scaled = model.predict(input_reshaped)
                        
                        # Inverse transform (agak tricky karena scaler fit multi-column)
                        # Kita buat dummy array untuk inverse
                        dummy = np.zeros((1, len(feature_cols)))
                        dummy[0, 0] = prediction_scaled[0, 0] # Asumsi PM2.5 kolom pertama
                        prediction_final = scaler.inverse_transform(dummy)[0, 0]
                        
                        # Tampilkan Hasil
                        st.metric(label="Prediksi PM2.5 (1 Jam ke depan)", value=f"{prediction_final:.2f} µg/m³")
                        st.metric(label="Nilai Aktual", value=f"{actual_val:.2f} µg/m³", delta=f"{prediction_final - actual_val:.2f}")
                        
                        # Plot perbandingan History + Prediksi
                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(y=input_data['PM2.5'].values, x=list(range(-24, 0)), mode='lines+markers', name='History (24h)'))
                        fig_pred.add_trace(go.Scatter(y=[actual_val], x=[0], mode='markers', name='Actual', marker=dict(color='green', size=10)))
                        fig_pred.add_trace(go.Scatter(y=[prediction_final], x=[0], mode='markers', name='Prediction', marker=dict(color='red', symbol='x', size=10)))
                        
                        fig_pred.update_layout(title="Visualisasi Prediksi", xaxis_title="Jam (t-x)", yaxis_title="PM2.5")
                        st.plotly_chart(fig_pred)
                        
                else:
                    st.error("Data historis tidak cukup untuk membuat prediksi (kurang dari 24 jam).")
            else:
                st.error("Data untuk waktu yang dipilih tidak ditemukan.")
                
        except Exception as e:
            st.error(f"Terjadi error pada model: {e}")

# ==========================================
# 7. HALAMAN: KESIMPULAN
# ==========================================
elif menu == "Kesimpulan 📝":
    st.subheader("📝 Kesimpulan & Rekomendasi")
    
    st.markdown("""
    ### Summary
    Dataset multi-station menunjukkan bahwa kualitas udara di Beijing memiliki pola yang sangat dipengaruhi oleh **faktor musiman** dan **lokasi geografis**.
    
    ### Poin Utama:
    1.  **Pola Musiman:** * **Musim Dingin:** Polusi ekstrem (PM2.5 tinggi) karena penggunaan pemanas berbasis batu bara dan kondisi meteorologi yang statis.
        * **Musim Panas:** Risiko Ozon (O3) lebih tinggi akibat reaksi fotokimia sinar matahari yang kuat.
    
    2.  **Geo-Analysis:**
        * Stasiun di pusat kota dan selatan mengalami polusi lebih berat dibanding wilayah utara (Changping, Dingling) yang lebih asri.
    
    3.  **Korelasi:**
        * Suhu (TEMP) dan Kecepatan Angin (WSPM) memiliki korelasi negatif dengan PM2.5. Artinya, angin kencang dan suhu hangat (di musim panas, kecuali ozon) cenderung membantu menurunkan kadar debu halus.
    
    ### Rekomendasi Strategis:
    * **Intervensi Berbasis Musim:** Pengetatan emisi pembakaran harus difokuskan pada musim dingin.
    * **Intervensi Berbasis Lokasi:** Wilayah selatan membutuhkan zona rendah emisi (Low Emission Zone) yang lebih ketat dibanding utara.
    """)

# Footer
st.markdown("---")
st.markdown("© 2024 Proyek Data Science - Kelompok IF1")