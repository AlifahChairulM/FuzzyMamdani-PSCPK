import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz 
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="234_248", layout="wide")

# --- 2. LOAD & PREPROCESS DATA ---
@st.cache_data
def load_and_clean_data():
    try:
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'leptop.csv')
        if not os.path.exists(file_path): file_path = 'leptop.csv'
            
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df['Ram'] = df['Ram'].astype(str).str.replace('GB', '', case=False).astype(int)
        df['ROM'] = df['ROM'].astype(str).str.upper().str.replace('SSD', '').str.replace('HDD', '').str.strip()
        df['ROM'] = df['ROM'].apply(lambda x: int(x.replace('TB', '')) * 1024 if 'TB' in x else int(x.replace('GB', '')))
        df['Price_IDR'] = (df['price'] * 60).astype(int)
        
        # Penambahan GPU_Size (Asumsi kriteria ke-5)
        df['GPU_Size'] = df['GPU'].str.extract(r'(\d+)').fillna(0).astype(int)
        
        df_final = df.sample(n=300, random_state=42).reset_index(drop=True)
        df_final.insert(0, 'No', range(1, len(df_final) + 1))
        return df_final
    except Exception as e:
        st.error(f"Error pada Dataset: {e}")
        return pd.DataFrame()

# --- 3. STANDAR LOGIKA FUZZY ---
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

# Fungsi untuk Fuzzifikasi
def fuzzifikasi(harga_val, ram_val, rom_val, h_obj, ra_obj, ro_obj):
    h_murah = fuzz.interp_membership(h_obj.universe, h_obj['murah'].mf, harga_val)
    h_sedang = fuzz.interp_membership(h_obj.universe, h_obj['sedang'].mf, harga_val)
    h_mahal = fuzz.interp_membership(h_obj.universe, h_obj['mahal'].mf, harga_val)
    
    ra_kecil = fuzz.interp_membership(ra_obj.universe, ra_obj['kecil'].mf, ram_val)
    ra_sedang = fuzz.interp_membership(ra_obj.universe, ra_obj['sedang'].mf, ram_val)
    ra_besar = fuzz.interp_membership(ra_obj.universe, ra_obj['besar'].mf, ram_val)
    
    ro_kecil = fuzz.interp_membership(ro_obj.universe, ro_obj['kecil'].mf, rom_val)
    ro_sedang = fuzz.interp_membership(ro_obj.universe, ro_obj['sedang'].mf, rom_val)
    ro_besar = fuzz.interp_membership(ro_obj.universe, ro_obj['besar'].mf, rom_val)
    
    return {
        'harga': {'murah': h_murah, 'sedang': h_sedang, 'mahal': h_mahal},
        'ram': {'kecil': ra_kecil, 'sedang': ra_sedang, 'besar': ra_besar},
        'rom': {'kecil': ro_kecil, 'sedang': ro_sedang, 'besar': ro_besar}
    }

# Fungsi untuk Defuzzifikasi dengan berbagai metode
def defuzzify_output(aggregated, universe, method):
    if method == 'centroid':
        return fuzz.defuzz(universe, aggregated, 'centroid')
    elif method == 'bisector':
        return fuzz.defuzz(universe, aggregated, 'bisector')
    elif method == 'mom':
        return fuzz.defuzz(universe, aggregated, 'mom')
    elif method == 'som':
        return fuzz.defuzz(universe, aggregated, 'som')
    elif method == 'lom':
        return fuzz.defuzz(universe, aggregated, 'lom')
    else:
        return fuzz.defuzz(universe, aggregated, 'centroid')  # default

# Inisialisasi Data
df_final = load_and_clean_data()
simulasi, h_obj, ra_obj, ro_obj, rek_obj = init_fuzzy()

# --- 4. NAVIGASI SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4442/4442880.png", width=80)
    st.title("Menu Navigasi")
    menu = st.radio("Pilih Halaman:", ["🏠 Home", "👥 Profil Anggota", "📊 Dataset", "🧮 Hitung SPK"])

# --- HALAMAN 1: HOME ---
    # --- HALAMAN 1: HOME ---
if menu == "🏠 Home":
    # CSS untuk konsistensi font di halaman Home
    st.markdown("""
    <style>
    /* Membuat semua teks di home konsisten */
    .home-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .home-container h1 {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1E3A5F;
    }
    .home-container h2 {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #2E4A7A;
    }
    .home-container h3 {
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        color: #3A5A8C;
    }
    .home-container p, .home-container li, .home-container td, .home-container th {
        font-size: 1rem;
        line-height: 1.5;
    }
    .home-container table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .home-container th, .home-container td {
        border: 1px solid #ddd;
        padding: 8px 12px;
        text-align: left;
    }
    .home-container th {
        background-color: #f2f2f2;
        font-weight: 600;
    }
    .home-container hr {
        margin: 1.5rem 0;
    }
    </style>
    <div class="home-container">
    """, unsafe_allow_html=True)
    
    # Judul Utama
    st.markdown("# 🎓 Sistem Pendukung Keputusan Pemilihan Laptop")
    st.markdown("### Metode Fuzzy Mamdani")
    st.markdown("---")
    
    # --- PENGERTIAN ---
    st.markdown("## 📖 Pengertian")
    st.markdown("""
    **Sistem Pendukung Keputusan (SPK)** adalah sistem berbasis komputer yang membantu pengambil keputusan 
    dalam menyelesaikan masalah semi-terstruktur dengan memanfaatkan data dan model keputusan.
    
    **Metode Fuzzy Mamdani** adalah salah satu metode dalam logika fuzzy yang menggunakan penalaran 
    IF-THEN (Jika-Maka) untuk memproses informasi yang tidak pasti. Metode ini terdiri dari 4 tahap:
    
    1. **Fuzzifikasi** → Mengubah input tegas (crisp) menjadi variabel fuzzy
    2. **Inferensi** → Menerapkan basis aturan (rule base)
    3. **Komposisi Aturan** → Menggabungkan output semua aturan
    4. **Defuzzifikasi** → Mengubah output fuzzy menjadi nilai tegas (skor)
    """)
    
    st.markdown("---")
    
    # --- STUDI KASUS ---
    st.markdown("## 📝 Studi Kasus")
    
    # 1. Deskripsi Masalah
    st.markdown("### 1. Deskripsi Masalah")
    st.markdown("""
    Seorang mahasiswa Informatika sedang mencari laptop untuk menunjang kebutuhan perkuliahan seperti 
    *coding*, *multitasking*, dan pengolahan data. Mengingat banyaknya opsi di pasar, ia memutuskan 
    untuk membangun sebuah sistem cerdas menggunakan metode **Fuzzy Mamdani**.
    
    Dalam pengambilan keputusan, mahasiswa tersebut menetapkan hirarki prioritas sebagai berikut:
    
    | Prioritas | Kriteria | Alasan |
    |:---:|:---|:---|
    | **1** | **Harga** | Mengingat keterbatasan anggaran (Kriteria Utama) |
    | **2** | **Kapasitas RAM** | Untuk kelancaran pengerjaan tugas berat |
    | **3** | **Penyimpanan/ROM** | Untuk manajemen file proyek |
    """)
    
    # 2. Definisi Variabel
    st.markdown("### 2. Definisi Variabel & Himpunan Fuzzy")
    st.markdown("Sistem ini memodelkan kriteria ke dalam variabel input dan output menggunakan **Fungsi Keanggotaan Segitiga (Triangular Membership Function)** :")
    
    # Tabel Variabel Input
    st.markdown("#### A. Variabel Input")
    input_table = """
    | Kriteria | Himpunan Fuzzy | Parameter [a, b, c] |
    |:---|:---|:---:|
    | **Harga (Juta Rp)** | Murah | [0, 0, 12] |
    | | Sedang | [8, 15, 22] |
    | | Mahal | [18, 30, 30] |
    | **RAM (GB)** | Kecil | [0, 4, 12] |
    | | Sedang | [8, 16, 32] |
    | | Besar | [24, 64, 64] |
    | **ROM (GB)** | Kecil | [0, 0, 512] |
    | | Sedang | [256, 512, 1024] |
    | | Besar | [512, 2048, 2048] |
    """
    st.markdown(input_table)
    
    # Tabel Variabel Output
    st.markdown("#### B. Variabel Output")
    output_table = """
    | Kriteria | Himpunan Fuzzy | Parameter [a, b, c] |
    |:---|:---|:---:|
    | **Kelayakan (0-100)** | Rendah | [0, 0, 50] |
    | | Menengah | [30, 50, 70] |
    | | Tinggi | [50, 100, 100] |
    """
    st.markdown(output_table)
    
    # 3. Basis Aturan
    st.markdown("### 3. Basis Aturan (Knowledge Base)")
    st.markdown("Penyusunan aturan didasarkan pada prioritas mahasiswa dengan menempatkan aspek ekonomi sebagai faktor dominan:")
    
    rules_table = """
    | Aturan | IF (Kondisi) | THEN (Output) |
    |:---:|:---|:---:|
    | **R1** | Harga Murah AND RAM Besar AND ROM Besar | **Tinggi** |
    | **R2** | Harga Murah AND RAM Sedang | **Tinggi** |
    | **R3** | RAM Besar OR ROM Besar | **Tinggi** |
    | **R4** | Harga Mahal AND RAM Kecil | **Rendah** |
    | **R5** | Harga Sedang AND ROM Sedang | **Menengah** |
    | **R6** | Harga Mahal AND RAM Besar | **Menengah** |
    """
    st.markdown(rules_table)
    
    # 4. Instruksi Tugas
    st.markdown("### 4. Instruksi Tugas")
    st.markdown("""
    Berdasarkan parameter sistem di atas, lakukanlah analisis sebagai berikut:
    
    1. **Analisis Fuzzifikasi** → Tentukan nilai derajat keanggotaan (μ) tiap kriteria
    2. **Evaluasi Inferensi** → Tentukan nilai α-predikat untuk setiap aturan yang aktif
    3. **Defuzzifikasi** → Hitung nilai tegas menggunakan metode **Centroid**
    4. **Ranking** → Urutkan 10 alternatif laptop terbaik dari 300 data sampel
    """)
    
    # --- LANGKAH-LANGKAH ---
    st.markdown("---")
    st.markdown("## Langkah-langkah Perhitungan")
    
    st.markdown("""
    | Langkah | Keterangan |
    |:---:|:---|
    | **1** | **Fuzzifikasi** - Mengubah input tegas (angka) menjadi variabel fuzzy (linguistik) |
    | **2** | **Inferensi** - Menerapkan basis aturan (Rule Base) IF-THEN |
    | **3** | **Komposisi Aturan** - Menggabungkan output semua aturan menggunakan operator MAX |
    | **4** | **Defuzzifikasi** - Mengubah hasil fuzzy menjadi nilai tegas (Skor 0-100) |
    """)
    
    # Informasi tambahan tentang metode
    st.info("💡 **Informasi:** Metode ini sangat efektif untuk menangani ketidakpastian dalam pemilihan laptop berdasarkan preferensi subjektif user.")
    
    # Tutup div container
    st.markdown("</div>", unsafe_allow_html=True)


# --- HALAMAN 2: PROFIL ANGGOTA ---
elif menu == "👥 Profil Anggota":
    st.title("Kelompok 5 - IF-I")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Anggota 1**")
        st.write("- Nama: ALIFAH CHAIRUL MUNAWAR")
        st.write("- NIM: 123240234")
    with col2:
        st.info("**Anggota 2**")
        st.write("- Nama: RENO ")
        st.write("- NIM: 123240248")

# --- HALAMAN 3: DATASET ---
elif menu == "📊 Dataset":
    st.title("Dataset Laptop")
    st.write("Menampilkan data yang telah dibersihkan dengan kriteria: Harga, RAM, ROM, Skor Rating, dan GPU Size.")
    st.dataframe(df_final[['No', 'brand', 'name', 'Price_IDR', 'Ram', 'ROM', 'spec_rating', 'GPU_Size']], 
                 use_container_width=True, hide_index=True)

# --- HALAMAN 4: HITUNG SPK ---
elif menu == "🧮 Hitung SPK":
    st.title("Proses Perhitungan SPK Fuzzy Mamdani")
    
    # Tampilkan ringkasan alur perhitungan
    with st.expander("Alur Perhitungan SPK", expanded=True):
        st.markdown("""
        **Tahapan Perhitungan Sistem Pendukung Keputusan (SPK) dengan Metode Fuzzy Mamdani:**
        
        1. **Input Nilai Crisp** → Masukkan nilai harga, RAM, dan ROM
        2. **Fuzzifikasi** → Mengubah nilai crisp menjadi derajat keanggotaan fuzzy
        3. **Inferensi (Rule Evaluation)** → Menerapkan basis aturan IF-THEN
        4. **Komposisi Aturan (Aggregation)** → Menggabungkan output semua aturan
        5. **Defuzzifikasi** → Mengubah output fuzzy menjadi nilai crisp (skor)
        6. **Ranking Data** → Menghitung skor untuk semua laptop dan menentukan peringkat
        """)
    
    # --- STEP 1: PREFERENSI USER (INPUT CRISP) ---
    st.header("📌 Langkah 1: Input Nilai Crisp")
    st.markdown("Pengguna memasukkan kriteria kebutuhan laptop dalam bentuk angka tegas (crisp):")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1: 
        s_harga = st.slider("Budget Harga (Juta Rp)", 0, 30, 15, 
                           help="Masukkan budget maksimal laptop dalam Juta Rupiah")
        st.caption(f"Nilai crisp input: **{s_harga} Juta Rp**")
    with col_in2: 
        s_ram = st.select_slider("Minimal RAM (GB)", [4, 8, 16, 32, 64], 8,
                                help="Masukkan kebutuhan RAM minimal dalam GB")
        st.caption(f"Nilai crisp input: **{s_ram} GB**")
    with col_in3: 
        s_rom = st.select_slider("Minimal ROM (GB)", [128, 256, 512, 1024, 2048], 512,
                                help="Masukkan kebutuhan penyimpanan minimal dalam GB")
        st.caption(f"Nilai crisp input: **{s_rom} GB**")

    # --- STEP 2: FUZZIFIKASI ---
    st.header("📌 Langkah 2: Fuzzifikasi")
    st.markdown("Mengubah nilai crisp menjadi derajat keanggotaan fuzzy menggunakan fungsi keanggotaan segitiga/trapesium.")
    
    fuzz_vals = fuzzifikasi(s_harga, s_ram, s_rom, h_obj, ra_obj, ro_obj)
    
    # Tampilkan perhitungan fuzzifikasi untuk setiap variabel
    st.subheader("2.1 Perhitungan Derajat Keanggotaan Harga")
    st.markdown(f"""
    **Nilai Harga: {s_harga} Juta Rp**
    
    **Fungsi Keanggotaan Harga:**
    - Murah: μ = max(0, min((x-0)/(12-0), (30-x)/(30-0))) = **{fuzz_vals['harga']['murah']:.4f}**
    - Sedang: μ = max(0, min((x-8)/(15-8), (22-x)/(22-15))) = **{fuzz_vals['harga']['sedang']:.4f}**
    - Mahal: μ = max(0, min((x-18)/(30-18), (30-x)/(30-18))) = **{fuzz_vals['harga']['mahal']:.4f}**
    """)
    
    st.subheader("2.2 Perhitungan Derajat Keanggotaan RAM")
    st.markdown(f"""
    **Nilai RAM: {s_ram} GB**
    
    **Fungsi Keanggotaan RAM:**
    - Kecil: μ = max(0, min((x-0)/(12-0), (30-x)/(30-0))) = **{fuzz_vals['ram']['kecil']:.4f}**
    - Sedang: μ = max(0, min((x-8)/(16-8), (32-x)/(32-16))) = **{fuzz_vals['ram']['sedang']:.4f}**
    - Besar: μ = max(0, min((x-24)/(64-24), (65-x)/(65-24))) = **{fuzz_vals['ram']['besar']:.4f}**
    """)
    
    st.subheader("2.3 Perhitungan Derajat Keanggotaan ROM")
    st.markdown(f"""
    **Nilai ROM: {s_rom} GB**
    
    **Fungsi Keanggotaan ROM:**
    - Kecil: μ = max(0, min((x-0)/(512-0), (2048-x)/(2048-0))) = **{fuzz_vals['rom']['kecil']:.4f}**
    - Sedang: μ = max(0, min((x-256)/(512-256), (1024-x)/(1024-512))) = **{fuzz_vals['rom']['sedang']:.4f}**
    - Besar: μ = max(0, min((x-512)/(2048-512), (2048-x)/(2048-512))) = **{fuzz_vals['rom']['besar']:.4f}**
    """)

    # Tabel Ringkasan Fuzzifikasi
    st.subheader("📊 Ringkasan Hasil Fuzzifikasi")
    fuzz_summary = pd.DataFrame({
        'Variabel': ['Harga', 'RAM', 'ROM'],
        'Nilai Input': [f"{s_harga} Juta", f"{s_ram} GB", f"{s_rom} GB"],
        'μ Kecil/Rendah': [f"{fuzz_vals['harga']['murah']:.3f}", 
                          f"{fuzz_vals['ram']['kecil']:.3f}", 
                          f"{fuzz_vals['rom']['kecil']:.3f}"],
        'μ Sedang/Menengah': [f"{fuzz_vals['harga']['sedang']:.3f}", 
                             f"{fuzz_vals['ram']['sedang']:.3f}", 
                             f"{fuzz_vals['rom']['sedang']:.3f}"],
        'μ Besar/Tinggi': [f"{fuzz_vals['harga']['mahal']:.3f}", 
                          f"{fuzz_vals['ram']['besar']:.3f}", 
                          f"{fuzz_vals['rom']['besar']:.3f}"]
    })
    st.dataframe(fuzz_summary, hide_index=True, use_container_width=True)

    # Visualisasi Fungsi Keanggotaan
    st.subheader("2.4 Visualisasi Fungsi Keanggotaan")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        fig_h, ax_h = plt.subplots(figsize=(5, 3))
        h_obj.view(ax=ax_h)
        ax_h.axvline(x=s_harga, color='r', linestyle='--', linewidth=2, label=f'Input: {s_harga}')
        ax_h.set_title('Variabel Harga')
        ax_h.legend()
        st.pyplot(fig_h)
        plt.close(fig_h)
    with col_d2:
        fig_ra, ax_ra = plt.subplots(figsize=(5, 3))
        ra_obj.view(ax=ax_ra)
        ax_ra.axvline(x=s_ram, color='r', linestyle='--', linewidth=2, label=f'Input: {s_ram}')
        ax_ra.set_title('Variabel RAM')
        ax_ra.legend()
        st.pyplot(fig_ra)
        plt.close(fig_ra)
    with col_d3:
        fig_ro, ax_ro = plt.subplots(figsize=(5, 3))
        ro_obj.view(ax=ax_ro)
        ax_ro.axvline(x=s_rom, color='r', linestyle='--', linewidth=2, label=f'Input: {s_rom}')
        ax_ro.set_title('Variabel ROM')
        ax_ro.legend()
        st.pyplot(fig_ro)
        plt.close(fig_ro)

    # --- STEP 3: INFERENSI (RULE EVALUATION) ---
    st.header("📌 Langkah 3: Inferensi (Evaluasi Aturan)")
    st.markdown("Menerapkan basis aturan IF-THEN menggunakan operator AND (min) dan OR (max).")
    
    # Hitung aktivasi aturan secara manual
    h_murah = fuzz_vals['harga']['murah']
    h_sedang = fuzz_vals['harga']['sedang']
    h_mahal = fuzz_vals['harga']['mahal']
    ra_kecil = fuzz_vals['ram']['kecil']
    ra_sedang = fuzz_vals['ram']['sedang']
    ra_besar = fuzz_vals['ram']['besar']
    ro_kecil = fuzz_vals['rom']['kecil']
    ro_sedang = fuzz_vals['rom']['sedang']
    ro_besar = fuzz_vals['rom']['besar']
    
    # Rule activations
    rule1 = min(h_murah, ra_besar, ro_besar)  # tinggi
    rule2 = min(h_murah, ra_sedang)  # tinggi
    rule3 = max(ra_besar, ro_besar)  # tinggi
    rule4 = min(h_mahal, ra_kecil)  # rendah
    rule5 = min(h_sedang, ro_sedang)  # menengah
    rule6 = min(h_mahal, ra_besar)  # menengah
    
    # Tampilkan tabel aturan
    st.subheader("3.1 Basis Aturan dan Nilai α-predikat")
    rules_data = {
        'Aturan': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6'],
        'IF (Kondisi)': [
            'Harga Murah AND RAM Besar AND ROM Besar',
            'Harga Murah AND RAM Sedang',
            'RAM Besar OR ROM Besar',
            'Harga Mahal AND RAM Kecil',
            'Harga Sedang AND ROM Sedang',
            'Harga Mahal AND RAM Besar'
        ],
        'THEN (Output)': ['Tinggi', 'Tinggi', 'Tinggi', 'Rendah', 'Menengah', 'Menengah'],
        'Nilai Keanggotaan': [
            f"min({h_murah:.3f}, {ra_besar:.3f}, {ro_besar:.3f}) = {rule1:.4f}",
            f"min({h_murah:.3f}, {ra_sedang:.3f}) = {rule2:.4f}",
            f"max({ra_besar:.3f}, {ro_besar:.3f}) = {rule3:.4f}",
            f"min({h_mahal:.3f}, {ra_kecil:.3f}) = {rule4:.4f}",
            f"min({h_sedang:.3f}, {ro_sedang:.3f}) = {rule5:.4f}",
            f"min({h_mahal:.3f}, {ra_besar:.3f}) = {rule6:.4f}"
        ],
        'α-predikat': [rule1, rule2, rule3, rule4, rule5, rule6]
    }
    df_rules = pd.DataFrame(rules_data)
    st.dataframe(df_rules, hide_index=True, use_container_width=True)
    
    # --- STEP 4: KOMPOSISI ATURAN (AGGREGATION) ---
    st.header("📌 Langkah 4: Komposisi Aturan (Aggregation)")
    st.markdown("Menggabungkan output dari semua aturan menggunakan operator MAX (penggabungan himpunan fuzzy).")
    
    # Aggregated output
    aggregated = np.fmax(
        np.fmax(np.fmin(rule1, rek_obj['tinggi'].mf), np.fmin(rule2, rek_obj['tinggi'].mf)),
        np.fmax(np.fmin(rule3, rek_obj['tinggi'].mf), 
        np.fmax(np.fmin(rule4, rek_obj['rendah'].mf), 
        np.fmax(np.fmin(rule5, rek_obj['menengah'].mf), np.fmin(rule6, rek_obj['menengah'].mf))))
    )
    
    st.markdown("""
    **Proses Agregasi:**
    
    Output = (R1 ∪ R2 ∪ R3 ∪ R4 ∪ R5 ∪ R6)
    
    Dimana:
    - R1 (Tinggi) dengan α = {:.4f}
    - R2 (Tinggi) dengan α = {:.4f}
    - R3 (Tinggi) dengan α = {:.4f}
    - R4 (Rendah) dengan α = {:.4f}
    - R5 (Menengah) dengan α = {:.4f}
    - R6 (Menengah) dengan α = {:.4f}  
    """.format(rule1, rule2, rule3, rule4, rule5, rule6))
    
    # Visualisasi hasil agregasi
    st.subheader("4.1 Visualisasi Hasil Agregasi")
    fig_agg, ax_agg = plt.subplots(figsize=(10, 5))
    ax_agg.fill_between(rek_obj.universe, 0, aggregated, alpha=0.3, color='blue', label='Area Hasil Agregasi')
    ax_agg.plot(rek_obj.universe, aggregated, 'b-', linewidth=2, label='Aggregated Output')
    ax_agg.set_xlabel('Skor Rekomendasi')
    ax_agg.set_ylabel('Derajat Keanggotaan')
    ax_agg.set_title('Hasil Komposisi Aturan (Aggregated Output)')
    ax_agg.grid(True, alpha=0.3)
    ax_agg.legend()
    st.pyplot(fig_agg)
    plt.close(fig_agg)

        # --- STEP 5: DEFUZZIFIKASI ---
    st.header("📌 Langkah 5: Defuzzifikasi (Keputusan Final)")
    st.markdown("""
    **Defuzzifikasi** adalah tahap mengubah himpunan fuzzy hasil agregasi menjadi **nilai tegas (crisp value)** 
    yang merupakan **KEPUTUSAN FINAL** sistem. Nilai ini bisa langsung di-ranking dan dibandingkan antar laptop.
    """)
    
    st.subheader("5.1 Perhitungan dengan Berbagai Metode Defuzzifikasi")
    
    methods = ['centroid', 'bisector', 'mom', 'som', 'lom']
    method_names = {
        'centroid': 'Centroid (Center of Area)',
        'bisector': 'Bisector (Garis Bagi Area)',
        'mom': 'MOM (Mean of Maximum)',
        'som': 'SOM (Smallest of Maximum)',
        'lom': 'LOM (Largest of Maximum)'
    }
    
    method_formulas = {
        'centroid': 'z* = ∫ μ(z)·z dz / ∫ μ(z) dz',
        'bisector': '∫_{a}^{z*} μ(z) dz = ∫_{z*}^{b} μ(z) dz',
        'mom': 'z* = (z₁ + z₂ + ... + zₙ) / n',
        'som': 'z* = min{z | μ(z) = max(μ)}',
        'lom': 'z* = max{z | μ(z) = max(μ)}'
    }
    
    scores = {}
    
    # Tampilkan setiap metode dalam baris yang terorganisir
    for method in methods:
        st.markdown(f"### 📈 {method_names[method]}")
        
        # Buat 2 kolom: kiri untuk grafik, kanan untuk rumus dan hasil
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            # Hitung defuzzifikasi
            score = defuzzify_output(aggregated, rek_obj.universe, method)
            scores[method] = score
            
            # Buat grafik
            fig_def, ax_def = plt.subplots(figsize=(8, 4))
            
            # Plot area hasil agregasi
            ax_def.fill_between(rek_obj.universe, 0, aggregated, alpha=0.3, color='blue', label='Area Hasil Agregasi')
            ax_def.plot(rek_obj.universe, aggregated, 'b-', linewidth=2, label='Aggregated Output')
            
            # Plot garis vertikal hasil defuzzifikasi
            ax_def.axvline(x=score, color='red', linestyle='--', linewidth=2, 
                          label=f'Hasil Defuzzifikasi: {score:.2f}')
            
            # Isi area di bawah garis (opsional)
            ax_def.fill_betweenx([0, max(aggregated)], score, score, alpha=0.5, color='red')
            
            ax_def.set_xlabel('Skor Rekomendasi (0-100)', fontsize=10)
            ax_def.set_ylabel('Derajat Keanggotaan (μ)', fontsize=10)
            ax_def.set_title(f'{method_names[method]}', fontsize=12, fontweight='bold')
            ax_def.grid(True, alpha=0.3)
            ax_def.legend(loc='upper left', fontsize=9)
            
            st.pyplot(fig_def)
            plt.close(fig_def)
        
        with col_right:
            # Tampilkan rumus
            st.markdown("**Rumus:**")
            st.latex(method_formulas[method])
            
            # Tampilkan hasil perhitungan
            st.markdown("**Hasil Perhitungan:**")
            
            if method == 'centroid':
                # Perhitungan detail centroid
                numerator = np.sum(aggregated * rek_obj.universe)
                denominator = np.sum(aggregated)
                st.latex(f"z* = \\frac{{{numerator:.2f}}}{{{denominator:.2f}}} = {score:.4f}")
                st.caption(f"∑(μ(z)×z) = {numerator:.2f}")
                st.caption(f"∑ μ(z) = {denominator:.2f}")
            
            elif method == 'bisector':
                st.latex(f"z* = {score:.4f}")
                st.caption("Nilai z* yang membagi area menjadi dua bagian sama besar")
            
            elif method == 'mom':
                # Cari nilai z dengan derajat keanggotaan maksimum
                max_mu = np.max(aggregated)
                max_indices = np.where(aggregated >= max_mu - 0.0001)[0]
                mean_max = np.mean(rek_obj.universe[max_indices])
                st.latex(f"z* = \\frac{{({mean_max:.2f})}}{{1}} = {score:.4f}")
                st.caption(f"μ_max = {max_mu:.4f}")
                st.caption(f"Domain dengan μ_max: {rek_obj.universe[max_indices[0]]} - {rek_obj.universe[max_indices[-1]]}")
            
            elif method == 'som':
                max_mu = np.max(aggregated)
                som_value = rek_obj.universe[np.where(aggregated >= max_mu - 0.0001)[0]][0]
                st.latex(f"z* = {som_value:.2f}")
                st.caption(f"Nilai terkecil dengan μ_max = {max_mu:.4f}")
            
            elif method == 'lom':
                max_mu = np.max(aggregated)
                lom_value = rek_obj.universe[np.where(aggregated >= max_mu - 0.0001)[0]][-1]
                st.latex(f"z* = {lom_value:.2f}")
                st.caption(f"Nilai terbesar dengan μ_max = {max_mu:.4f}")
            
            # Tampilkan hasil dalam metric
            st.metric("Nilai Keputusan Final", f"{score:.2f}", 
                     delta="dari 0-100" if score >= 50 else None,
                     help="Ini adalah KEPUTUSAN FINAL dari sistem")
            
            # Interpretasi keputusan
            if score >= 70:
                st.success("**KEPUTUSAN: SANGAT DIREKOMENDASIKAN**")
            elif score >= 50:
                st.info("**KEPUTUSAN: CUKUP DIREKOMENDASIKAN**")
            elif score >= 30:
                st.warning("**KEPUTUSAN: KURANG DIREKOMENDASIKAN**")
            else:
                st.error("**KEPUTUSAN: TIDAK DIREKOMENDASIKAN**")
        
        st.divider()  # Pembatas antar metode
    
    # --- TABEL PERBANDINGAN SEMUA METODE ---
    st.subheader("5.2 Tabel Perbandingan Semua Metode Defuzzifikasi")
    
    comparison_data = []
    for method in methods:
        comparison_data.append({
            'Metode': method_names[method],
            'Rumus': method_formulas[method],
            'Hasil (Skor)': f"{scores[method]:.4f}",
            'Interpretasi': 'Sangat Direkomendasikan' if scores[method] >= 70 else 
                           'Cukup Direkomendasikan' if scores[method] >= 50 else
                           'Kurang Direkomendasikan' if scores[method] >= 30 else
                           'Tidak Direkomendasikan'
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, hide_index=True, use_container_width=True)
    
    # --- PERHITUNGAN DETAIL METODE CENTROID (METODE DEFAULT) ---
    st.subheader("5.3 Perhitungan Detail Metode Centroid (Metode Default)")
    
    centroid_score = scores['centroid']
    
    # Buat tabel perhitungan numerik
    st.markdown("**Perhitungan Numerik dengan Data Diskrit:**")
    
    # Ambil sample data untuk ditampilkan (setiap 20 poin)
    sample_indices = np.arange(0, len(rek_obj.universe), 10)
    sample_z = rek_obj.universe[sample_indices]
    sample_mu = aggregated[sample_indices]
    
    calc_data = []
    for i, (z, mu) in enumerate(zip(sample_z, sample_mu)):
        calc_data.append({
            'i': i,
            'z (Skor)': f"{z:.0f}",
            'μ(z)': f"{mu:.4f}",
            'μ(z) × z': f"{mu * z:.2f}"
        })
    
    st.dataframe(pd.DataFrame(calc_data), hide_index=True, use_container_width=True)
    
    # --- STEP 6: RANKING DATA ---
    st.header("📌 Langkah 6: Ranking Data")
    st.markdown("""
    Pada tahap ini, sistem akan menghitung skor rekomendasi dari 300 data laptop
    menggunakan metode Fuzzy Mamdani yang sama, kemudian mengurutkannya dari skor tertinggi ke terendah.
    """)
    
    with st.spinner('Menghitung skor untuk 300 data laptop... Mohon tunggu'):
        skor_total = []
        
        # Progress bar untuk perhitungan
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Loop untuk setiap baris di df_final
        for idx, row in df_final.iterrows():
            # Update progress setiap 30 data
            if idx % 30 == 0:
                progress_bar.progress((idx + 1) / len(df_final))
                status_text.text(f"Memproses data ke-{idx + 1} dari {len(df_final)}...")
            
            try:
                # KONVERSI: Harga dari Rupiah ke Juta (karena domain fuzzy 0-30 Juta)
                harga_juta = row['Price_IDR'] / 1000000
                
                # MASUKAN KE SISTEM FUZZY
                simulasi.input['harga'] = harga_juta
                simulasi.input['ram'] = row['Ram']
                simulasi.input['rom'] = row['ROM']
                simulasi.compute()
                
                # AMBIL SKOR REKOMENDASI
                skor = simulasi.output['rekomendasi']
                skor_total.append(skor)
                
            except Exception as e:
                # Jika ada error (misal nilai di luar domain), beri skor rendah
                skor_total.append(20.0)
        
        # Bersihkan progress bar
        progress_bar.empty()
        status_text.empty()
        
        # TAMBAHKAN SKOR KE DATAFRAME
        df_final['Skor_Akhir'] = skor_total
        df_final['Skor_Akhir'] = df_final['Skor_Akhir'].round(2)
        
        # BUAT DATAFRAME RANKING (urut dari skor tertinggi)
        df_rank = df_final.copy()
        df_rank = df_rank.sort_values('Skor_Akhir', ascending=False).reset_index(drop=True)
        df_rank.insert(0, 'Peringkat', range(1, len(df_rank) + 1))
        
        # Tambahkan status rekomendasi
        def get_status(skor):
            if skor >= 70:
                return "🟢 Sangat Direkomendasikan"
            elif skor >= 50:
                return "🔵 Direkomendasikan"
            elif skor >= 30:
                return "🟡 Cukup Direkomendasikan"
            else:
                return "🔴 Tidak Direkomendasikan"
        
        df_rank['Status'] = df_rank['Skor_Akhir'].apply(get_status)

    # --- TAMPILAN HASIL RANKING ---
    st.subheader("TOP 10 LAPTOP DENGAN SKOR TERTINGGI")
    st.markdown("Berikut adalah 10 laptop terbaik berdasarkan hasil perhitungan Fuzzy Mamdani:")
    
    # Ambil top 10
    top10 = df_rank.head(10).copy()
    
    # Tampilkan dengan kolom yang jelas
    st.dataframe(
        top10[['Peringkat', 'No', 'brand', 'name', 'Price_IDR', 'Ram', 'ROM', 'Skor_Akhir', 'Status']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Peringkat": st.column_config.NumberColumn("Rank", format="%d"),
            "No": "Data Ke-",
            "brand": "Merek",
            "name": "Nama Laptop",
            "Price_IDR": st.column_config.NumberColumn("Harga (Rp)", format="Rp %d"),
            "Ram": "RAM (GB)",
            "ROM": "ROM (GB)",
            "Skor_Akhir": st.column_config.NumberColumn("Skor", format="%.2f"),
            "Status": "Rekomendasi"
        }
    )
    
    # Tampilkan detail 3 besar
    st.subheader("🥇 Detail 3 Laptop Terbaik")
    
    col1, col2, col3 = st.columns(3)
    
    for i, col in enumerate([col1, col2, col3]):
        if i < len(top10):
            laptop = top10.iloc[i]
            medals = ["🥇", "🥈", "🥉"]
            colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
            
            with col:
                st.markdown(f"""
                <div style='background-color: {colors[i]}; padding: 15px; border-radius: 10px; text-align: center;'>
                    <h2>{medals[i]} Peringkat {i+1}</h2>
                    <h3>{laptop['brand']}</h3>
                    <p><b>{laptop['name'][:30]}...</b></p>
                    <hr>
                    <p> Harga: Rp {laptop['Price_IDR']:,.0f}</p>
                    <p> RAM: {laptop['Ram']} GB | ROM: {laptop['ROM']} GB</p>
                    <p> Skor: <b>{laptop['Skor_Akhir']:.2f}</b>/100</p>
                    <p>{laptop['Status']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # --- TABEL RANKING 11-20 ---
    with st.expander("📋 Lihat Peringkat 11 - 20"):
        rank_11_20 = df_rank.iloc[10:20].copy()
        st.dataframe(
            rank_11_20[['Peringkat', 'No', 'brand', 'name', 'Price_IDR', 'Ram', 'ROM', 'Skor_Akhir', 'Status']],
            use_container_width=True,
            hide_index=True
        )
    
    # --- TABEL RANKING 21-50 ---
    with st.expander("📋 Lihat Peringkat 21 - 50"):
        rank_21_50 = df_rank.iloc[20:50].copy()
        st.dataframe(
            rank_21_50[['Peringkat', 'No', 'brand', 'name', 'Price_IDR', 'Ram', 'ROM', 'Skor_Akhir', 'Status']],
            use_container_width=True,
            hide_index=True
        )
    
    