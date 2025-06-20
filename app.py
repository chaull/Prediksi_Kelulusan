import streamlit as st
import pandas as pd
import numpy as np
import pickle # For saving and loading models (if used)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # Needed for dummy model training

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide", # Menggunakan layout lebar untuk aplikasi
    initial_sidebar_state="auto"
)

# --- Custom CSS for Aesthetics and Hiding Number Input Spinners ---
# This CSS makes the app look nicer and removes the default +/- buttons from number inputs.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #333;
    }

    .main {
        background-color: #f0f2f6; /* Light gray background for the main content area */
        padding: 20px;
        border-radius: 10px;
    }

    .stApp {
        background: linear-gradient(to right, #ece9e6, #ffffff); /* Subtle gradient for the overall app background */
    }

    .stContainer {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Soft shadow for containers */
        margin-bottom: 20px;
        border: 1px solid #e0e0e0; /* Light border */
    }

    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 25px;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05); /* Subtle text shadow for title */
    }

    h2, h3, h4 {
        color: #34495e;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .stButton>button {
        background-color: #4CAF50; /* Green button */
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease; /* Smooth transition for hover effect */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        width: 100%; /* Make button full width */
    }

    .stButton>button:hover {
        background-color: #45a049; /* Darker green on hover */
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15); /* Enhanced shadow on hover */
        transform: translateY(-2px); /* Slight lift on hover */
    }

    .stSuccess {
        background-color: #e6ffe6; /* Light green for success messages */
        color: #28a745; /* Darker green text */
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #28a745;
        font-weight: 600;
        text-align: center;
        margin-top: 20px;
    }

    .stError {
        background-color: #ffe6e6; /* Light red for error messages */
        color: #dc3545; /* Darker red text */
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dc3545;
        font-weight: 600;
        text-align: center;
        margin-top: 20px;
    }

    .stInfo {
        background-color: #e6f7ff; /* Light blue for info messages */
        color: #007bff; /* Darker blue text */
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #007bff;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* Custom progress bar style */
    .progress-container {
        width: 100%;
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 20px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1); /* Inner shadow for depth */
    }

    .progress-bar {
        height: 30px;
        line-height: 30px; /* Vertically center text */
        color: white;
        text-align: center;
        border-radius: 10px;
        transition: width 0.5s ease-in-out; /* Smooth animation for width change */
        background: linear-gradient(to right, #28a745, #8bc34a); /* Green gradient */
    }

    .progress-bar.red {
        background: linear-gradient(to right, #dc3545, #ff5722); /* Red gradient for "Tidak Lulus" */
    }

    .stMarkdown p {
        line-height: 1.6; /* Better readability for paragraphs */
    }

    /* CSS to hide the increment/decrement buttons (spinners) for number inputs */
    input[type="number"]::-webkit-outer-spin-button,
    input[type="number"]::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }

    input[type="number"] {
        -moz-appearance: textfield; /* Firefox specific */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to load and train the model (or load an existing one)
# @st.cache_resource ensures this function runs only once for performance
@st.cache_resource
def load_and_train_model():
    # --- PENTING: GANTI BAGIAN INI DENGAN KODE ASLI PELATIHAN MODEL ANDA ---
    # Jika Anda sudah memiliki model dan encoder yang tersimpan (misal file .pkl),
    # Anda sangat disarankan untuk memuatnya di sini daripada melatih ulang.
    # Contoh pemuatan model:
    # try:
    #     with open('nama_model_anda.pkl', 'rb') as f:
    #         model = pickle.load(f)
    #     with open('nama_le_gender.pkl', 'rb') as f:
    #         le_gender = pickle.load(f)
    #     with open('nama_le_status_mhs.pkl', 'rb') as f:
    #         le_status_mahasiswa = pickle.load(f)
    #     with open('nama_le_status_nikah.pkl', 'rb') as f:
    #         le_status_nikah = pickle.load(f)
    #     # Pastikan daftar fitur ini sesuai dengan yang digunakan saat melatih model asli
    #     model_features = ['Usia', 'IPK', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'IPS6', 'IPS7', 'IPS8',
    #                       'Gender_encoded', 'Status_Mahasiswa_encoded', 'Status_Nikah_encoded']
    #     st.success("Model dan encoder berhasil dimuat dari file. (Ini hanya pesan di development)")
    #     return model, le_gender, le_status_mahasiswa, le_status_nikah, model_features
    # except FileNotFoundError:
    #     st.warning("File model atau encoder tidak ditemukan. Melatih ulang model dummy untuk demo...")

    # Data dummy yang diperluas untuk contoh pelatihan
    # Data ini sengaja dibuat agar mencakup semua kategori yang ada di radio button
    data = {
        'Gender': ['Pria', 'Wanita'] * 10,
        'Status_Mahasiswa': ['Bekerja', 'Tidak Bekerja'] * 10,
        'Status_Nikah': ['Belum Menikah', 'Menikah'] * 10,
        'Usia': [20, 21, 22, 20, 23, 21, 24, 20, 25, 22, 19, 20, 21, 22, 23, 20, 24, 21, 25, 22],
        'IPS1': [3.5, 3.8, 2.5, 3.9, 3.0, 3.7, 3.1, 3.8, 2.0, 3.6, 3.7, 3.5, 3.8, 2.6, 3.9, 3.0, 3.2, 3.7, 2.1, 3.6],
        'IPS2': [3.6, 3.7, 2.6, 3.8, 3.1, 3.6, 3.2, 3.7, 2.1, 3.5, 3.8, 3.6, 3.7, 2.7, 3.8, 3.1, 3.3, 3.6, 2.2, 3.5],
        'IPS3': [3.4, 3.9, 2.4, 4.0, 2.9, 3.8, 3.0, 3.9, 1.9, 3.7, 3.6, 3.4, 3.9, 2.5, 4.0, 2.9, 3.1, 3.8, 2.0, 3.7],
        'IPS4': [3.5, 3.8, 2.5, 3.9, 3.0, 3.7, 3.1, 3.8, 2.0, 3.6, 3.7, 3.5, 3.8, 2.6, 3.9, 3.0, 3.2, 3.7, 2.1, 3.6],
        'IPS5': [3.6, 3.7, 2.6, 3.8, 3.1, 3.6, 3.2, 3.7, 2.1, 3.5, 3.8, 3.6, 3.7, 2.7, 3.8, 3.1, 3.3, 3.6, 2.2, 3.5],
        'IPS6': [3.4, 3.9, 2.4, 4.0, 2.9, 3.8, 3.0, 3.9, 1.9, 3.7, 3.6, 3.4, 3.9, 2.5, 4.0, 2.9, 3.1, 3.8, 2.0, 3.7],
        'IPS7': [3.5, 3.8, 2.5, 3.9, 3.0, 3.7, 3.1, 3.8, 2.0, 3.6, 3.7, 3.5, 3.8, 2.6, 3.9, 3.0, 3.2, 3.7, 2.1, 3.6],
        'IPS8': [3.6, 3.7, 2.6, 3.8, 3.1, 3.6, 3.2, 3.7, 2.1, 3.5, 3.8, 3.6, 3.7, 2.7, 3.8, 3.1, 3.3, 3.6, 2.2, 3.5],
        'Kelulusan': [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1] # 1 for Lulus, 0 for Tidak Lulus
    }
    df_dummy = pd.DataFrame(data)

    # Hitung IPK sebagai rata-rata dari IPS1-IPS8 untuk data dummy
    ips_cols = [f'IPS{i}' for i in range(1, 9)]
    df_dummy['IPK'] = df_dummy[ips_cols].mean(axis=1)

    # Inisialisasi LabelEncoder
    le_gender = LabelEncoder()
    le_status_mahasiswa = LabelEncoder()
    le_status_nikah = LabelEncoder()

    # Secara eksplisit fit LabelEncoder dengan SEMUA kemungkinan kategori
    # Ini sangat penting untuk mencegah error "unseen label" saat transform
    le_gender.fit(['Pria', 'Wanita'])
    le_status_mahasiswa.fit(['Bekerja', 'Tidak Bekerja'])
    le_status_nikah.fit(['Belum Menikah', 'Menikah'])

    # Transform kolom kategorikal di data dummy
    df_dummy['Gender_encoded'] = le_gender.transform(df_dummy['Gender'])
    df_dummy['Status_Mahasiswa_encoded'] = le_status_mahasiswa.transform(df_dummy['Status_Mahasiswa'])
    df_dummy['Status_Nikah_encoded'] = le_status_nikah.transform(df_dummy['Status_Nikah'])

    # Definisikan fitur (X) dan target (y)
    # Pastikan urutan kolom sesuai dengan yang diharapkan model Anda saat pelatihan
    model_features = [
        'Usia', 'IPK', # IPK is now calculated
        'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'IPS6', 'IPS7', 'IPS8',
        'Gender_encoded', 'Status_Mahasiswa_encoded', 'Status_Nikah_encoded'
    ]
    X_dummy = df_dummy[model_features]
    y_dummy = df_dummy['Kelulusan']

    # Latih model RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_dummy, y_dummy)

    # Mengembalikan model, encoder, dan daftar fitur agar konsisten
    return model, le_gender, le_status_mahasiswa, le_status_nikah, model_features

# Muat model dan encoders (akan dilatih sekali saat aplikasi dimulai atau dimuat dari cache)
model, le_gender, le_status_mahasiswa, le_status_nikah, model_features = load_and_train_model()

# --- Judul Aplikasi Streamlit ---
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🎓 Aplikasi Prediksi Kelulusan Mahasiswa</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1em;'>Isi form di bawah ini untuk memprediksi probabilitas kelulusan mahasiswa berdasarkan berbagai kriteria.</p>", unsafe_allow_html=True)

# --- Bagian Form Input ---
st.markdown("---") # Separator visual
st.subheader('Data Diri Mahasiswa')

# Menggunakan kolom untuk tata letak yang lebih rapi
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        # Menggunakan st.radio untuk Jenis Kelamin
        gender = st.radio('Jenis Kelamin', ['Pria', 'Wanita'], index=0, help="Pilih jenis kelamin mahasiswa.", key='gender_radio')
        # Menggunakan st.radio untuk Status Mahasiswa
        status_mahasiswa = st.radio('Status Mahasiswa', ['Bekerja', 'Tidak Bekerja'], index=1, help="Pilih status pekerjaan mahasiswa.", key='status_mhs_radio')
    with col2:
        usia = st.number_input('Usia', min_value=17, max_value=70, value=20, help="Masukkan usia mahasiswa.")
        # Menggunakan st.radio untuk Status Pernikahan
        status_nikah = st.radio('Status Pernikahan', ['Belum Menikah', 'Menikah'], index=0, help="Pilih status pernikahan mahasiswa.", key='status_nikah_radio')

st.markdown("---") # Separator visual
st.subheader('Nilai Akademik')

with st.container():
    ips_values = {}
    num_ips_cols = 4 # Jumlah kolom untuk input IPS
    cols_ips = st.columns(num_ips_cols)

    for i in range(1, 9):
        with cols_ips[(i - 1) % num_ips_cols]:
            # st.number_input tanpa tombol +/- karena CSS kustom di atas
            ips_values[f'IPS{i}'] = st.number_input(f'IPS Semester {i}', min_value=0.0, max_value=4.0, value=3.0, step=0.01, key=f'ips_{i}', help=f"Masukkan Indeks Prestasi Semester {i} (0.0 - 4.0).")

# Hitung IPK dari nilai IPS yang diinput
if ips_values:
    calculated_ipk = np.mean(list(ips_values.values()))
else:
    calculated_ipk = 0.0 # Default value if no IPS inputs (shouldn't happen with fixed loops)

# Tampilkan IPK yang dihitung sebelum tombol prediksi
st.info(f"<b>IPK yang Dihitung Otomatis: <span style='font-size:1.2em;'>{calculated_ipk:.2f}</span></b>", unsafe_allow_html=True)

st.markdown("---") # Separator visual
st.subheader('Prediksi Kelulusan')

# --- Tombol Prediksi ---
if st.button('Prediksi Kelulusan'):
    try:
        # Preprocessing input untuk fitur kategorikal menggunakan encoder yang sudah dilatih
        # Gunakan .transform() langsung karena encoder sudah di-fit dengan semua kelas yang mungkin
        gender_encoded = le_gender.transform([gender])[0]
        status_mahasiswa_encoded = le_status_mahasiswa.transform([status_mahasiswa])[0]
        status_nikah_encoded = le_status_nikah.transform([status_nikah])[0]

        # Siapkan data input dalam format dictionary
        input_dict = {
            'Usia': usia,
            'IPK': calculated_ipk, # Menggunakan IPK yang sudah dihitung
            'Gender_encoded': gender_encoded,
            'Status_Mahasiswa_encoded': status_mahasiswa_encoded,
            'Status_Nikah_encoded': status_nikah_encoded,
        }
        for i in range(1, 9): # Tambahkan semua nilai IPS ke dictionary
            input_dict[f'IPS{i}'] = ips_values[f'IPS{i}']

        # Buat DataFrame dari input pengguna, pastikan urutan kolom sesuai dengan model_features
        input_data_df = pd.DataFrame([input_dict])[model_features]

        # Lakukan Prediksi
        prediction = model.predict(input_data_df)
        prediction_proba = model.predict_proba(input_data_df) # Probabilitas untuk kedua kelas (0 dan 1)

        st.markdown("---") # Separator visual
        st.subheader('Hasil Prediksi & Skala Probabilitas')

        pass_proba = prediction_proba[0][1] * 100 # Probabilitas Lulus (kelas 1)
        fail_proba = prediction_proba[0][0] * 100 # Probabilitas Tidak Lulus (kelas 0)

        # Tampilkan pesan sukses/error
        if prediction[0] == 1:
            st.success(f'Mahasiswa diprediksi **LULUS**! 🎉 Probabilitas: **{pass_proba:.2f}%**')
        else:
            st.error(f'Mahasiswa diprediksi **TIDAK LULUS**! 😔 Probabilitas: **{fail_proba:.2f}%**')

        # Visualisasi skala probabilitas kustom (mirip progress bar)
        progress_bar_color_class = "progress-bar" # Default hijau untuk "Lulus"
        # Ubah warna menjadi merah jika probabilitas lulus kurang dari 50%
        if pass_proba < 50:
            progress_bar_color_class = "progress-bar red"

        st.markdown(f"""
            <div class="progress-container">
                <div class="{progress_bar_color_class}" style="width: {pass_proba:.2f}%;">
                    {pass_proba:.2f}% Lulus
                </div>
            </div>
        """, unsafe_allow_html=True)


    except Exception as e:
        # Menangkap error tak terduga dan menampilkannya
        st.error(f"Terjadi kesalahan: {e}. Mohon coba lagi. Jika masalah berlanjut, pastikan semua input valid dan model Anda dilatih dengan benar. Debugging Info: {e}")

st.markdown('---')
st.caption('Aplikasi ini dibuat menggunakan Streamlit dan scikit-learn. Model prediksi menggunakan data dummy; untuk hasil yang akurat, gantilah dengan model yang telah dilatih pada dataset asli Anda.')
