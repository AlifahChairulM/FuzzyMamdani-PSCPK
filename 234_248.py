import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="SPK Laptop", page_icon="💻", layout="wide")

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #ffffff; border-radius: 5px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD & PREPROCESS DATA ---
@st.cache_data
def load_and_clean_data():
    try:
        # Mencari path absolut agar tidak Error File Not Found
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'leptop.csv')
        
        # Jika path tidak ditemukan (misal jalan di environment berbeda), fallback ke nama file langsung
        if not os.path.exists(file_path):
            file_path = 'leptop.csv'
            
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # Konversi RAM ke Integer
        df['Ram'] = df['Ram'].astype(str).str.replace('GB', '', case=False).astype(int)
        
        # Konversi ROM (SSD/HDD) ke GB
        df['ROM'] = df['ROM'].astype(str).str.upper().str.replace('SSD', '').str.replace('HDD', '').str.strip()
        df['ROM'] = df['ROM'].apply(lambda x: int(x.replace('TB', '')) * 1024 if 'TB' in x else int(x.replace('GB', '')))
        
        # Konversi Harga ke IDR
        df['Price_IDR'] = (df['price'] * 60).astype(int)
        
        # Ambil 300 data acak tetap
        df_final = df.sample(n=300, random_state=42).reset_index(drop=True)
        df_final['No'] = df_final.index + 1
        return df_final
    except Exception as e:
        st.error(f"Error pada Dataset: {e}")
        return pd.DataFrame()

# --- 4. STANDAR LOGIKA FUZZY ---
def init_fuzzy():
    harga = ctrl.Antecedent(np.arange(0, 31, 1), 'harga') 
    ram = ctrl.Antecedent(np.arange(0, 66, 1), 'ram')
    rom = ctrl.Antecedent(np.arange(0, 2049, 1), 'rom') 
    rekomendasi = ctrl.Consequent(np.arange(0, 101, 1), 'rekomendasi')

    harga['murah'] = fuzz.trapmf(harga.universe, [0, 0, 5, 12])
    harga['sedang'] = fuzz.trimf(harga.universe, [8, 15, 22])
    harga['mahal'] = fuzz.trapmf(harga.universe, [18, 25, 30, 30])
    
    ram['kecil'] = fuzz.trimf(ram.universe, [0, 4, 12])
    ram['sedang'] = fuzz.trimf(ram.universe, [8, 16, 32])
    ram['besar'] = fuzz.trapmf(ram.universe, [24, 32, 65, 65])
    
    rom['kecil'] = fuzz.trimf(rom.universe, [0, 0, 512])
    rom['sedang'] = fuzz.trimf(rom.universe, [256, 512, 1024])
    rom['besar'] = fuzz.trapmf(rom.universe, [512, 1024, 2048, 2048])
    
    rekomendasi['rendah'] = fuzz.trimf(rekomendasi.universe, [0, 0, 50])
    rekomendasi['menengah'] = fuzz.trimf(rekomendasi.universe, [30, 50, 70])
    rekomendasi['tinggi'] = fuzz.trapmf(rekomendasi.universe, [60, 80, 100, 100])

    rules = [
        ctrl.Rule(harga['murah'] & ram['besar'] & rom['besar'], rekomendasi['tinggi']),
        ctrl.Rule(harga['murah'] & ram['sedang'], rekomendasi['tinggi']),
        ctrl.Rule(ram['besar'] | rom['besar'], rekomendasi['tinggi']),
        ctrl.Rule(harga['mahal'] & ram['kecil'], rekomendasi['rendah']),
        ctrl.Rule(harga['sedang'] & rom['sedang'], rekomendasi['menengah']),
        ctrl.Rule(harga['mahal'] & ram['besar'], rekomendasi['menengah'])
    ]
    
    sys = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(sys), harga, ram, rom, rekomendasi

# --- 5. EKSEKUSI DATA & FUZZY ---
df_final = load_and_clean_data()
simulasi, h_obj, ra_obj, ro_obj, rek_obj = init_fuzzy()

# Jalankan perhitungan skor otomatis untuk dataset
if not df_final.empty:
    with st.spinner('Menghitung skor rekomendasi dataset...'):
        hasil_skor = []
        for _, r in df_final.iterrows():
            try:
                simulasi.input['harga'] = r['Price_IDR'] / 1000000
                simulasi.input['ram'] = r['Ram']
                simulasi.input['rom'] = r['ROM']
                simulasi.compute()
                hasil_skor.append(simulasi.output['rekomendasi'])
            except:
                hasil_skor.append(20.0) # Default score if calculation fails
        df_final['Skor'] = hasil_skor

# --- 6. SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4442/4442880.png", width=100)
st.sidebar.header("Filter Simulasi")
s_harga = st.sidebar.slider("Harga (Juta)", 0, 30, 15)
s_ram = st.sidebar.select_slider("RAM (GB)", [4, 8, 16, 32, 64], 8)
s_rom = st.sidebar.select_slider("Storage (GB)", [128, 256, 512, 1024, 2048], 512)
btn = st.sidebar.button("Proses Analisis", use_container_width=True)

# --- 7. MAIN CONTENT ---
st.title("Sistem Informasi Madani - Laptop Edition")
st.write("Implementasi Logic Fuzzy Mamdani untuk Decision Support System.")

t1, t2, t3 = st.tabs(["📊 Hasil Perankingan", "🧪 Simulator Pakar", "📈 Grafik Logika"])

with t1:
    if not df_final.empty:
        st.success(f"Berhasil memproses {len(df_final)} data!")
        df_top = df_final.sort_values('Skor', ascending=False).head(10).copy()
        df_top['Ranking'] = range(1, 11)
        
        st.subheader("10 Laptop Terbaik Berdasarkan Sistem")
        st.dataframe(
            df_top[['Ranking', 'brand', 'name', 'Price_IDR', 'Ram', 'ROM', 'Skor']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.error("Gagal memproses data. Cek file leptop.csv")

with t2:
  with t2:
    if btn:
        # Input ke mesin fuzzy
        simulasi.input['harga'] = s_harga
        simulasi.input['ram'] = s_ram
        simulasi.input['rom'] = s_rom
        
        # Hitung
        simulasi.compute()
        out = simulasi.output['rekomendasi']
        
        # Tampilkan Metrik Skor
        c1, c2 = st.columns(2)
        c1.metric("Skor Kelayakan", f"{out:.2f}")
        c2.info(f"Hasil Defuzzifikasi Centroid: {out:.2f}")
        
        # Tampilkan Grafik Hasil (Defuzzifikasi)
        st.write("### Visualisasi Hasil Defuzzifikasi")
        fig_sim, ax_sim = plt.subplots(figsize=(8, 4))
        rek_obj.view(sim=simulasi, ax=ax_sim)
        st.pyplot(fig_sim)
        plt.close(fig_sim) # Tutup biar memori aman
    else:
        st.info("Silakan atur kriteria di sidebar dan klik 'Proses Analisis'.")

with t3:
    st.subheader("Visualisasi Himpunan Fuzzy")
    st.write("Grafik membership function untuk kriteria input dan output.")
    
    col_a, col_b = st.columns(2)
    
    # Fungsi pembantu agar tidak nulis berulang-ulang
    def render_fuzzy_plot(obj, title):
        st.markdown(f"**{title}**")
        fig, ax = plt.subplots(figsize=(8, 4))
        obj.view(ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    with col_a:
        render_fuzzy_plot(h_obj, "Kriteria: Harga")
        render_fuzzy_plot(ra_obj, "Kriteria: RAM")
        
    with col_b:
        render_fuzzy_plot(ro_obj, "Kriteria: Storage (ROM)")
        render_fuzzy_plot(rek_obj, "Output: Skor Rekomendasi")