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

# --- Mengatasi error mutex/lock pada macOS ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
    .insight-box {background-color: #3aeb34; padding: 20px; border-radius: 10px; border-left: 5px solid #198754; color: #0f513; }
    .pollutant-card {padding: 15px; border-radius: 8px; margin-bottom: 12px;}
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

    # Handling Missing Values
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
menu = st.sidebar.radio("Navigasi", [
    "Informasi Polusi Udara 📚",
    "Geo-Analysis 🗺️",
    "Exploratory Data Analysis 📊",
    "PM2.5 Prediction (LSTM) 🤖",
    "Kesimpulan 📝"
])

# Filter Tahun di Sidebar (Global)
year_list = sorted(df['year'].unique())
selected_year = st.sidebar.selectbox("Pilih Tahun (untuk visualisasi)", year_list)

# Filter Data berdasarkan tahun
df_filtered = df[df['year'] == selected_year]

# ==========================================
# 4. HALAMAN: INFORMASI POLUSI UDARA (BARU)
# ==========================================
if menu == "Informasi Polusi Udara 📚":
    st.subheader("📚 Apa Itu Polusi Udara?")

    st.markdown("""
    <div style="background-color:#f0f8ff; padding:20px; border-radius:10px; border-left:5px solid #2196F3; margin-bottom:20px; color:#1a1a1a;">
    <b style="font-size:1.1rem; color:#1a1a1a;">🌬️ Definisi</b><br>
    Polusi udara adalah kontaminasi udara di dalam maupun luar ruangan oleh bahan kimia, fisik, atau biologis 
    yang mengubah karakteristik alami atmosfer. Sumber utamanya berasal dari kendaraan bermotor, industri, 
    pembangkit listrik, dan aktivitas rumah tangga seperti pembakaran bahan bakar.
    </div>
    """, unsafe_allow_html=True)

    # ---- Tab navigasi internal ----
    info_tab1, info_tab2, info_tab3, info_tab4 = st.tabs([
        "🔬 Jenis Polutan", "⚠️ Dampak Kesehatan", "📊 Indeks Kualitas Udara (AQI)", "📈 Data dalam Dataset Ini"
    ])

    # ===================== TAB 1: JENIS POLUTAN =====================
    with info_tab1:
        st.write("### 🔬 Jenis-Jenis Polutan Udara")
        st.markdown("Berikut adalah **6 polutan utama** yang diukur dalam dataset Beijing ini beserta penjelasan lengkapnya:")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style="background:#fff3e0; padding:15px; border-radius:8px; margin-bottom:12px; border-left:4px solid #FF9800; color:#1a1a1a;">
            <b style="color:#1a1a1a;">💨 PM2.5 (Particulate Matter 2.5)</b><br>
            Partikel debu sangat halus berdiameter ≤ 2.5 mikrometer. Partikel ini sangat berbahaya karena mampu 
            menembus jauh ke dalam paru-paru bahkan masuk ke aliran darah. Sumber utama: kendaraan bermotor, 
            industri, pembakaran biomassa dan batu bara.<br><br>
            <small style="color:#795548;">📏 Satuan: µg/m³ &nbsp;|&nbsp; ✅ Ambang Aman WHO: &lt;15 µg/m³ (tahunan)</small>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background:#fff3e0; padding:15px; border-radius:8px; margin-bottom:12px; border-left:4px solid #FF9800; color:#1a1a1a;">
            <b style="color:#1a1a1a;">🌫️ PM10 (Particulate Matter 10)</b><br>
            Partikel debu kasar berdiameter ≤ 10 mikrometer. Umumnya berasal dari konstruksi bangunan, 
            jalan berdebu, pertanian, dan industri semen. Dapat menyebabkan iritasi saluran pernapasan 
            bagian atas dan mata.<br><br>
            <small style="color:#795548;">📏 Satuan: µg/m³ &nbsp;|&nbsp; ✅ Ambang Aman WHO: &lt;45 µg/m³ (tahunan)</small>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background:#e8f5e9; padding:15px; border-radius:8px; margin-bottom:12px; border-left:4px solid #4CAF50; color:#1a1a1a;">
            <b style="color:#1a1a1a;">🏭 SO2 (Sulfur Dioksida)</b><br>
            Gas tidak berwarna dengan bau menyengat. Dihasilkan dari pembakaran bahan bakar fosil yang mengandung 
            belerang, terutama batu bara. Di atmosfer SO2 bereaksi dengan air membentuk H₂SO₄ (hujan asam) 
            yang merusak ekosistem dan bangunan.<br><br>
            <small style="color:#795548;">📏 Satuan: µg/m³ &nbsp;|&nbsp; ✅ Ambang Aman WHO: &lt;40 µg/m³ (24 jam)</small>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background:#e3f2fd; padding:15px; border-radius:8px; margin-bottom:12px; border-left:4px solid #2196F3; color:#1a1a1a;">
            <b style="color:#1a1a1a;">🚗 NO2 (Nitrogen Dioksida)</b><br>
            Gas coklat kemerahan yang dihasilkan dari pembakaran suhu tinggi pada kendaraan bermotor dan 
            pembangkit listrik. Berkontribusi pada pembentukan smog fotokimia dan hujan asam. Kadar NO2 
            sering digunakan sebagai indikator pencemaran lalu lintas.<br><br>
            <small style="color:#795548;">📏 Satuan: µg/m³ &nbsp;|&nbsp; ✅ Ambang Aman WHO: &lt;25 µg/m³ (24 jam)</small>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background:#e3f2fd; padding:15px; border-radius:8px; margin-bottom:12px; border-left:4px solid #2196F3; color:#1a1a1a;">
            <b style="color:#1a1a1a;">🔥 CO (Karbon Monoksida)</b><br>
            Gas tidak berwarna dan tidak berbau, namun sangat beracun. Dihasilkan dari pembakaran tidak sempurna 
            bahan bakar. CO mengikat hemoglobin dalam darah lebih kuat dari oksigen, sehingga menghambat 
            transportasi O₂ ke seluruh tubuh — dapat berakibat fatal dalam ruang tertutup.<br><br>
            <small style="color:#795548;">📏 Satuan: µg/m³ &nbsp;|&nbsp; ✅ Ambang Aman WHO: &lt;4 mg/m³ (24 jam)</small>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background:#f3e5f5; padding:15px; border-radius:8px; margin-bottom:12px; border-left:4px solid #9C27B0; color:#1a1a1a;">
            <b style="color:#1a1a1a;">☀️ O3 (Ozon Troposfer)</b><br>
            Berbeda dari ozon stratosfer yang melindungi bumi dari UV, ozon di lapisan bawah atmosfer justru 
            berbahaya. Terbentuk dari reaksi fotokimia antara NO2 dan sinar matahari. Kadar tertinggi terjadi 
            di siang hingga sore hari pada musim panas.<br><br>
            <small style="color:#795548;">📏 Satuan: µg/m³ &nbsp;|&nbsp; ✅ Ambang Aman WHO: &lt;100 µg/m³ (8 jam)</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.write("### 🌡️ Faktor Meteorologi yang Mempengaruhi Polusi")
        st.markdown("Selain sumber emisi, kondisi cuaca sangat menentukan seberapa tinggi kadar polutan di udara:")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.info("**🌡️ Suhu (TEMP)**\n\nSuhu rendah di musim dingin menyebabkan inversi termal — polutan terjebak di lapisan bawah atmosfer dan tidak bisa naik.")
        with m2:
            st.info("**🔵 Tekanan Udara (PRES)**\n\nTekanan tinggi menghasilkan kondisi atmosfer stabil dan angin lemah, menyebabkan polutan menumpuk di satu area.")
        with m3:
            st.success("**🌬️ Kecepatan Angin (WSPM)**\n\nAngin kencang secara efektif mengencerkan dan menyebarkan polutan ke area lebih luas, menurunkan konsentrasi lokal.")
        with m4:
            st.success("**🌧️ Curah Hujan (RAIN)**\n\nHujan membantu 'mencuci' partikel debu dan gas dari atmosfer melalui proses wet deposition, membersihkan udara.")

    # ===================== TAB 2: DAMPAK KESEHATAN =====================
    with info_tab2:
        st.write("### ⚠️ Dampak Polusi Udara terhadap Kesehatan")

        st.markdown("""
        <div style="background:#ffebee; padding:15px; border-radius:8px; border-left:5px solid #f44336; margin-bottom:20px; color:#1a1a1a;">
        ⚕️ <b style="color:#1a1a1a;">Menurut WHO (2024)</b>, polusi udara adalah penyebab kematian prematur terbesar di dunia akibat 
        faktor lingkungan — menyebabkan sekitar <b>7 juta kematian prematur per tahun</b> secara global. 
        99% populasi dunia menghirup udara yang melampaui batas aman WHO.
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("""
            <div style="background:#fff9c4; padding:15px; border-radius:8px; color:#1a1a1a;">
            <h4 style="text-align:center; color:#1a1a1a;">🫁 Sistem Pernapasan</h4>
            <ul style="font-size:0.9rem; color:#1a1a1a;">
            <li>Asma & bronkitis kronis</li>
            <li>PPOK (Penyakit Paru Obstruktif Kronis)</li>
            <li>Kanker paru-paru</li>
            <li>Infeksi saluran napas akut</li>
            <li>Penurunan fungsi paru jangka panjang pada anak</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("""
            <div style="background:#fce4ec; padding:15px; border-radius:8px; color:#1a1a1a;">
            <h4 style="text-align:center; color:#1a1a1a;">❤️ Sistem Kardiovaskular</h4>
            <ul style="font-size:0.9rem; color:#1a1a1a;">
            <li>Penyakit jantung koroner</li>
            <li>Stroke iskemik & hemoragik</li>
            <li>Hipertensi</li>
            <li>Aterosklerosis (pengerasan arteri)</li>
            <li>Aritmia & gagal jantung</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_c:
            st.markdown("""
            <div style="background:#e8eaf6; padding:15px; border-radius:8px; color:#1a1a1a;">
            <h4 style="text-align:center; color:#1a1a1a;">🧠 Sistem Saraf & Lainnya</h4>
            <ul style="font-size:0.9rem; color:#1a1a1a;">
            <li>Gangguan kognitif & demensia</li>
            <li>Depresi & kecemasan</li>
            <li>Gangguan perkembangan otak anak</li>
            <li>Diabetes tipe 2</li>
            <li>Berat lahir rendah & kelahiran prematur</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.write("### 👥 Kelompok Paling Rentan terhadap Polusi Udara")

        vuln_data = {
            "Kelompok": ["👶 Anak-anak (0–14 th)", "👴 Lansia (>65 th)", "🤰 Ibu Hamil", "😷 Penderita Asma/PPOK", "👷 Pekerja Outdoor"],
            "Alasan Rentan": [
                "Paru-paru masih berkembang; frekuensi napas lebih tinggi sehingga terpapar lebih banyak polutan per berat badan",
                "Sistem imun dan paru melemah; komorbiditas (jantung, diabetes) memperparah dampak paparan",
                "Paparan PM2.5 berkaitan erat dengan kelahiran prematur, berat lahir rendah, dan preeklamsia",
                "Penyakit yang sudah ada membuat bronkus lebih reaktif; serangan bisa dicetuskan AQI >100",
                "Durasi paparan udara luar lebih panjang setiap harinya dibanding rata-rata populasi"
            ],
            "Tindakan Pencegahan": [
                "Hindari aktivitas luar saat AQI > 100; gunakan masker N95/KN95 saat wajib keluar",
                "Pantau AQI harian; pasang air purifier HEPA di rumah; konsultasi dokter rutin",
                "Konsultasi OB-GYN jika tinggal di zona polusi tinggi; hindari paparan asap kendaraan",
                "Selalu bawa inhaler/obat darurat; buat rencana darurat bersama dokter",
                "Gunakan masker respirator P100; beristirahat di dalam ruangan di jam-jam puncak polusi"
            ]
        }

        vuln_df = pd.DataFrame(vuln_data)
        st.dataframe(vuln_df, use_container_width=True, hide_index=True)

        st.write("")
        st.write("### 🌍 Paparan PM2.5 di Beijing vs Standar WHO")

        # Visualisasi perbandingan rata-rata tahunan per stasiun vs standar WHO
        station_annual = df.groupby('station')['PM2.5'].mean().reset_index()
        station_annual.columns = ['Stasiun', 'Rata-rata PM2.5']
        station_annual = station_annual.sort_values('Rata-rata PM2.5', ascending=True)

        fig_who = go.Figure()
        fig_who.add_trace(go.Bar(
            x=station_annual['Rata-rata PM2.5'],
            y=station_annual['Stasiun'],
            orientation='h',
            marker_color=['#f44336' if v > 15 else '#4CAF50' for v in station_annual['Rata-rata PM2.5']],
            name='Rata-rata PM2.5'
        ))
        fig_who.add_vline(x=15, line_dash="dash", line_color="blue",
                          annotation_text="Batas WHO (15 µg/m³)", annotation_position="top right")
        fig_who.update_layout(
            title="Rata-rata PM2.5 per Stasiun vs Batas Aman WHO",
            xaxis_title="PM2.5 (µg/m³)",
            height=420
        )
        st.plotly_chart(fig_who, use_container_width=True)

    # ===================== TAB 3: AQI =====================
    with info_tab3:
        st.write("### 📊 Indeks Kualitas Udara (Air Quality Index / AQI)")

        st.markdown("""
        **AQI** adalah skala standar yang digunakan pemerintah dan lembaga lingkungan untuk mengkomunikasikan 
        tingkat polusi udara kepada masyarakat umum secara mudah dipahami. Semakin tinggi nilainya, 
        semakin berbahaya kualitas udaranya.
        """)

        aqi_data = {
            "Kategori": ["Baik", "Sedang", "Tidak Sehat (Kel. Sensitif)", "Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"],
            "Rentang AQI": ["0–50", "51–100", "101–150", "151–200", "201–300", "301–500"],
            "PM2.5 (µg/m³)": ["0–12.0", "12.1–35.4", "35.5–55.4", "55.5–150.4", "150.5–250.4", "≥250.5"],
            "Indikator": ["🟢", "🟡", "🟠", "🔴", "🟣", "🟤"],
            "Rekomendasi Aktivitas": [
                "Aman beraktivitas di luar ruangan untuk semua orang",
                "Dapat diterima; kelompok sensitif sebaiknya perhatikan kondisi",
                "Kelompok sensitif kurangi aktivitas luar; orang sehat masih aman",
                "Semua orang mulai merasakan dampak; kelompok sensitif hindari keluar",
                "Peringatan darurat; semua orang hindari aktivitas luar yang lama",
                "Darurat kesehatan; semua orang terdampak serius — tetap di dalam!"
            ]
        }

        aqi_df = pd.DataFrame(aqi_data)
        st.dataframe(aqi_df, use_container_width=True, hide_index=True)

        st.write("")
        st.write("### 🗺️ Berapa Parah Polusi Beijing? — Analisis dari Dataset Nyata")

        def classify_aqi(pm25):
            if pd.isna(pm25): return "Tidak Diketahui"
            if pm25 <= 12: return "🟢 Baik"
            elif pm25 <= 35.4: return "🟡 Sedang"
            elif pm25 <= 55.4: return "🟠 Tidak Sehat (Sensitif)"
            elif pm25 <= 150.4: return "🔴 Tidak Sehat"
            elif pm25 <= 250.4: return "🟣 Sangat Tidak Sehat"
            else: return "🟤 Berbahaya"

        df_aqi = df.dropna(subset=['PM2.5']).copy()
        df_aqi['Kategori AQI'] = df_aqi['PM2.5'].apply(classify_aqi)
        aqi_counts = df_aqi['Kategori AQI'].value_counts().reset_index()
        aqi_counts.columns = ['Kategori', 'Jumlah Jam']
        aqi_counts['Persentase (%)'] = (aqi_counts['Jumlah Jam'] / aqi_counts['Jumlah Jam'].sum() * 100).round(1)
        aqi_counts = aqi_counts[aqi_counts['Kategori'] != 'Tidak Diketahui']

        color_map = {
            "🟢 Baik": "#4CAF50",
            "🟡 Sedang": "#FFC107",
            "🟠 Tidak Sehat (Sensitif)": "#FF9800",
            "🔴 Tidak Sehat": "#f44336",
            "🟣 Sangat Tidak Sehat": "#9C27B0",
            "🟤 Berbahaya": "#795548"
        }

        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig_aqi = px.pie(
                aqi_counts, names='Kategori', values='Jumlah Jam',
                title="Distribusi Kategori AQI PM2.5 — Semua Stasiun (2013–2017)",
                color='Kategori',
                color_discrete_map=color_map
            )
            fig_aqi.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_aqi, use_container_width=True)

        with c2:
            st.write("#### Ringkasan Distribusi AQI")
            st.dataframe(aqi_counts, use_container_width=True, hide_index=True)

            pct_unhealthy = aqi_counts[aqi_counts['Kategori'].isin(
                ["🔴 Tidak Sehat", "🟣 Sangat Tidak Sehat", "🟤 Berbahaya"]
            )]['Persentase (%)'].sum()

            pct_good = aqi_counts[aqi_counts['Kategori'].isin(
                ["🟢 Baik", "🟡 Sedang"]
            )]['Persentase (%)'].sum()

            st.error(f"🔴 **{pct_unhealthy:.1f}%** jam dalam kategori *Tidak Sehat* hingga *Berbahaya*")
            st.success(f"🟢 Hanya **{pct_good:.1f}%** jam dalam kategori *Baik* atau *Sedang*")

        # Tren AQI per tahun
        st.write("#### 📅 Tren Kategori AQI per Tahun")
        df_aqi['year'] = df_aqi['year'].astype(int)
        aqi_yearly = df_aqi.groupby(['year', 'Kategori AQI']).size().reset_index(name='Jumlah Jam')
        aqi_yearly_pct = aqi_yearly.copy()
        total_per_year = aqi_yearly.groupby('year')['Jumlah Jam'].transform('sum')
        aqi_yearly_pct['Persentase (%)'] = (aqi_yearly['Jumlah Jam'] / total_per_year * 100).round(1)

        fig_yearly = px.bar(
            aqi_yearly_pct, x='year', y='Persentase (%)',
            color='Kategori AQI', color_discrete_map=color_map,
            title="Distribusi Kategori AQI per Tahun (%)",
            labels={'year': 'Tahun'},
            barmode='stack'
        )
        st.plotly_chart(fig_yearly, use_container_width=True)

    # ===================== TAB 4: DATA DATASET =====================
    with info_tab4:
        st.write("### 📈 Mengenal Dataset Beijing PRSA")

        st.markdown("""
        Dataset **PRSA (Public Research Station of Air Quality) Beijing** adalah salah satu dataset udara terbuka 
        paling komprehensif di dunia untuk studi polusi perkotaan.
        """)

        # Kartu info dataset
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("🏢 Jumlah Stasiun", "12 Stasiun")
        with d2:
            st.metric("📅 Periode Data", "2013–2017 (4 Tahun)")
        with d3:
            st.metric("⏱️ Resolusi", "Per Jam (Hourly)")
        with d4:
            st.metric("📊 Total Baris Data", f"{len(df):,}")

        st.write("#### 📋 Statistik Deskriptif Polutan (Semua Stasiun, 2013–2017)")

        summary_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        summary_data = []
        for col in summary_cols:
            summary_data.append({
                "Polutan": col,
                "Rata-rata": f"{df[col].mean():.2f}",
                "Minimum": f"{df[col].min():.2f}",
                "Median (Q2)": f"{df[col].median():.2f}",
                "Q3 (75%)": f"{df[col].quantile(0.75):.2f}",
                "Maksimum": f"{df[col].max():.2f}",
                "Data Tersedia (%)": f"{df[col].notna().sum() / len(df) * 100:.1f}%",
                "Satuan": "µg/m³"
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.write("#### 🏭 Profil Rata-rata Polutan per Stasiun")

        station_profile = df.groupby('station')[summary_cols].mean().round(2).reset_index()
        station_profile = station_profile.rename(columns={'station': 'Stasiun'})
        station_profile = station_profile.sort_values('PM2.5', ascending=False)
        st.dataframe(station_profile, use_container_width=True, hide_index=True)

        # Heatmap polutan per stasiun
        st.write("#### 🌡️ Heatmap Intensitas Polutan per Stasiun")
        heatmap_data = station_profile.set_index('Stasiun')[summary_cols]

        # Normalisasi untuk heatmap agar skala sebanding
        heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

        fig_heat, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(heatmap_norm, annot=heatmap_data.values, fmt='.0f',
                    cmap='YlOrRd', ax=ax, linewidths=0.5,
                    cbar_kws={'label': 'Intensitas Relatif (Dinormalisasi)'})
        ax.set_title("Intensitas Polutan per Stasiun (nilai aktual ditampilkan, warna = relatif)")
        ax.set_xlabel("Polutan")
        ax.set_ylabel("Stasiun")
        plt.tight_layout()
        st.pyplot(fig_heat)

        st.markdown("""
        <div class="insight-box">
        <b>💡 Fakta Menarik dari Dataset Ini:</b><br>
        • Rata-rata PM2.5 di Beijing (~83 µg/m³) adalah <b>5.5× di atas batas aman WHO</b> (15 µg/m³).<br>
        • Nilai PM2.5 tertinggi mencapai <b>898 µg/m³</b> — hampir 60× di atas standar WHO!<br>
        • Stasiun Wanshouxigong (pusat kota) secara konsisten mencatat polusi tertinggi.<br>
        • Stasiun Dingling dan Huairou (dekat pegunungan utara) memiliki kualitas udara relatif terbaik.
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 5. HALAMAN: GEO-ANALYSIS
# ==========================================
elif menu == "Geo-Analysis 🗺️":
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
            radius=10 + (pm25/10),
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
# 6. HALAMAN: EDA
# ==========================================
elif menu == "Exploratory Data Analysis 📊":
    st.subheader("📊 Analisis Eksplorasi Data (EDA)")

    tab1, tab2, tab3 = st.tabs(["Tren Waktu", "Korelasi", "Perbandingan Stasiun"])

    with tab1:
        st.write("### Tren Polutan Bulanan")
        monthly_trend = df_filtered.groupby('month')[['PM2.5', 'PM10', 'SO2', 'NO2', 'O3']].mean()

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
# 7. HALAMAN: PREDIKSI (LSTM)
# ==========================================
elif menu == "PM2.5 Prediction (LSTM) 🤖":
    st.subheader("🤖 Prediksi PM2.5 Menggunakan LSTM")

    st.markdown("""
    Model Deep Learning (LSTM) digunakan untuk memprediksi konsentrasi PM2.5 satu jam ke depan berdasarkan data historis 24 jam terakhir.
    """)

    # Inisialisasi session_state untuk menyimpan hasil prediksi
    # agar tidak hilang saat Streamlit re-render halaman
    if 'pred_result' not in st.session_state:
        st.session_state.pred_result = None

    model_path = "./model/pm25_lstm_model.keras"

    if not os.path.exists(model_path):
        st.warning(f"File model '{model_path}' tidak ditemukan. Silakan upload file model .keras Anda.")
    else:
        try:
            model = load_model(model_path)

            st.write("#### Simulasi Prediksi")
            st.write("Pilih stasiun dan waktu untuk mengambil 24 jam data sebelumnya sebagai input model.")

            col1, col2 = st.columns(2)
            with col1:
                pred_station = st.selectbox("Pilih Stasiun", df['station'].unique())
            with col2:
                min_date = df['datetime'].min() + pd.Timedelta(days=2)
                max_date = df['datetime'].max()
                pred_date = st.date_input("Pilih Tanggal", value=max_date, min_value=min_date, max_value=max_date)

            pred_hour = st.slider("Pilih Jam", 0, 23, 12)

            # Jika parameter input berubah, hapus hasil lama agar tidak membingungkan
            current_key = f"{pred_station}_{pred_date}_{pred_hour}"
            if st.session_state.pred_result is not None:
                if st.session_state.pred_result.get('key') != current_key:
                    st.session_state.pred_result = None

            target_time = pd.to_datetime(f"{pred_date} {pred_hour}:00:00")
            df_station = df[df['station'] == pred_station].sort_values('datetime')
            mask = df_station['datetime'] == target_time

            if mask.sum() > 0:
                idx = df_station.index[df_station['datetime'] == target_time][0]
                feature_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
                pos = df_station.index.get_loc(idx)

                if pos >= 24:
                    input_data = df_station.iloc[pos-24:pos][feature_cols]
                    actual_val = df_station.iloc[pos]['PM2.5']

                    st.write("Data Input (24 Jam Terakhir):")
                    st.dataframe(input_data.tail())

                    if st.button("🔍 Jalankan Prediksi"):
                        with st.spinner("Menjalankan model LSTM..."):
                            scaler = MinMaxScaler()
                            scaler.fit(df_station[feature_cols])

                            input_scaled = scaler.transform(input_data)
                            input_reshaped = input_scaled.reshape(1, 24, len(feature_cols))

                            prediction_scaled = model.predict(input_reshaped)

                            dummy = np.zeros((1, len(feature_cols)))
                            dummy[0, 0] = prediction_scaled[0, 0]
                            prediction_final = float(scaler.inverse_transform(dummy)[0, 0])

                        # Simpan semua hasil ke session_state — tidak akan hilang saat re-render
                        st.session_state.pred_result = {
                            'key': current_key,
                            'prediction_final': prediction_final,
                            'actual_val': float(actual_val),
                            'history_pm25': input_data['PM2.5'].tolist(),
                            'station': pred_station,
                        }

                    # Render hasil dari session_state (persisten lintas re-render)
                    if st.session_state.pred_result is not None:
                        res = st.session_state.pred_result
                        prediction_final = res['prediction_final']
                        actual_val = res['actual_val']
                        history_pm25 = res['history_pm25']
                        station_name = res['station']

                        st.markdown("---")
                        st.write("#### 📊 Hasil Prediksi")

                        m1, m2, m3 = st.columns(3)
                        m1.metric(
                            label="🤖 Prediksi PM2.5 (1 Jam ke depan)",
                            value=f"{prediction_final:.2f} µg/m³"
                        )
                        m2.metric(
                            label="✅ Nilai Aktual",
                            value=f"{actual_val:.2f} µg/m³",
                            delta=f"{prediction_final - actual_val:.2f}"
                        )
                        m3.metric(
                            label="📏 Error Absolut",
                            value=f"{abs(prediction_final - actual_val):.2f} µg/m³"
                        )

                        # Plot prediksi
                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(
                            y=history_pm25, x=list(range(-24, 0)),
                            mode='lines+markers', name='History (24h)',
                            line=dict(color='#2196F3')
                        ))
                        fig_pred.add_trace(go.Scatter(
                            y=[actual_val], x=[0],
                            mode='markers', name='Aktual',
                            marker=dict(color='green', size=12, symbol='circle')
                        ))
                        fig_pred.add_trace(go.Scatter(
                            y=[prediction_final], x=[0],
                            mode='markers', name='Prediksi',
                            marker=dict(color='red', size=12, symbol='x')
                        ))
                        fig_pred.update_layout(
                            title=f"Visualisasi Prediksi PM2.5 — Stasiun {station_name}",
                            xaxis_title="Jam Relatif (0 = waktu target)",
                            yaxis_title="PM2.5 (µg/m³)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)

                        # Peta lokasi stasiun
                        st.write("#### 📍 Lokasi Stasiun Pemantauan")
                        st.markdown(f"Berikut adalah posisi stasiun **{station_name}** pada peta Beijing:")

                        station_coords = STATION_COORDS.get(station_name, [40.0, 116.4])
                        pred_map = folium.Map(location=station_coords, zoom_start=11)

                        # Semua stasiun lain sebagai titik abu-abu
                        for st_name, coords in STATION_COORDS.items():
                            if st_name != station_name:
                                folium.CircleMarker(
                                    location=coords,
                                    radius=6,
                                    color='#888888',
                                    fill=True,
                                    fill_color='#888888',
                                    fill_opacity=0.5,
                                    popup=folium.Popup(st_name, parse_html=True)
                                ).add_to(pred_map)

                        # Stasiun terpilih: titik kuning besar
                        folium.CircleMarker(
                            location=station_coords,
                            radius=16,
                            color='#B8860B',
                            fill=True,
                            fill_color='#FFD700',
                            fill_opacity=0.95,
                            popup=folium.Popup(
                                f"<b>📍 {station_name}</b><br>Lat: {station_coords[0]}<br>Lon: {station_coords[1]}",
                                parse_html=True
                            ),
                            tooltip=f"⭐ {station_name} (dipilih)"
                        ).add_to(pred_map)

                        folium.Marker(
                            location=station_coords,
                            icon=folium.DivIcon(
                                html=f'<div style="font-size:11px; font-weight:bold; color:#1a1a1a; '
                                     f'background:rgba(255,215,0,0.85); padding:3px 7px; border-radius:5px; '
                                     f'border:1.5px solid #B8860B; white-space:nowrap; margin-top:18px;">'
                                     f'📍 {station_name}</div>',
                                icon_size=(160, 30),
                                icon_anchor=(80, 0)
                            )
                        ).add_to(pred_map)

                        st_folium(pred_map, width=900, height=400)

                else:
                    st.error("Data historis tidak cukup untuk membuat prediksi (kurang dari 24 jam).")
            else:
                st.error("Data untuk waktu yang dipilih tidak ditemukan.")

        except Exception as e:
            st.error(f"Terjadi error pada model: {e}")

# ==========================================
# 8. HALAMAN: KESIMPULAN
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