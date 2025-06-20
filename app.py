import streamlit as st
import pandas as pd
import numpy as np
import pickle # Untuk menyimpan dan memuat model
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # Diperlukan untuk pelatihan model dummy

# Fungsi untuk memuat dan melatih model (atau memuat model yang sudah ada)
@st.cache_resource # Cache resource untuk menghindari pelatihan ulang setiap kali aplikasi berjalan
def load_and_train_model():
    # --- PENTING: GANTI BAGIAN INI DENGAN KODE ASLI PELATIHAN MODEL ANDA ---
    # Jika Anda sudah memiliki model dan encoder yang tersimpan (misal file .pkl),
    # Anda bisa memuatnya di sini. Contoh:
    # try:
    #     with open('model.pkl', 'rb') as f:
    #         model = pickle.load(f)
    #     with open('le_gender.pkl', 'rb') as f:
    #         le_gender = pickle.load(f)
    #     with open('le_status_mhs.pkl', 'rb') as f:
    #         le_status_mhs = pickle.load(f)
    #     with open('le_status_nikah.pkl', 'rb') as f:
    #         le_status_nikah = pickle.load(f)
    #     # Definisikan urutan fitur yang tepat yang diharapkan model Anda
    #     # IPK akan dihitung, jadi pastikan model Anda dilatih dengan IPK sebagai fitur
    #     model_features = ['Usia', 'IPK', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'IPS6', 'IPS7', 'IPS8',
    #                       'Gender_encoded', 'Status_Mahasiswa_encoded', 'Status_Nikah_encoded']
    #     st.success("Model dan encoder berhasil dimuat dari file.")
    #     return model, le_gender, le_status_mhs, le_status_nikah, model_features
    # except FileNotFoundError:
    #     st.warning("File model atau encoder tidak ditemukan. Melatih ulang model dummy...")

    # Data dummy yang diperluas untuk contoh pelatihan
    # Di aplikasi nyata, Anda akan menggunakan data dari file 'Kelulusan Train.csv' Anda.
    data = {
        'Gender': ['Pria', 'Wanita', 'Pria', 'Wanita', 'Pria', 'Wanita', 'Pria', 'Wanita', 'Pria', 'Wanita',
                   'Pria', 'Wanita', 'Pria', 'Wanita', 'Pria', 'Wanita', 'Pria', 'Wanita', 'Pria', 'Wanita'],
        'Status_Mahasiswa': ['Bekerja', 'Tidak Bekerja', 'Bekerja', 'Tidak Bekerja', 'Bekerja', 'Tidak Bekerja', 'Bekerja', 'Tidak Bekerja', 'Bekerja', 'Tidak Bekerja',
                             'Bekerja', 'Tidak Bekerja', 'Bekerja', 'Tidak Bekerja', 'Bekerja', 'Tidak Bekerja', 'Bekerja', 'Tidak Bekerja', 'Bekerja', 'Tidak Bekerja'],
        'Usia': [20, 21, 22, 20, 23, 21, 24, 20, 25, 22, 19, 20, 21, 22, 23, 20, 24, 21, 25, 22],
        'Status_Nikah': ['Belum Menikah', 'Menikah', 'Belum Menikah', 'Belum Menikah', 'Menikah', 'Belum Menikah', 'Menikah', 'Belum Menikah', 'Menikah', 'Belum Menikah',
                         'Belum Menikah', 'Belum Menikah', 'Menikah', 'Belum Menikah', 'Menikah', 'Belum Menikah', 'Belum Menikah', 'Menikah', 'Menikah', 'Belum Menikah'],
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

    # Calculate IPK as average of IPS1-IPS8 for dummy data
    ips_cols = [f'IPS{i}' for i in range(1, 9)]
    df_dummy['IPK'] = df_dummy[ips_cols].mean(axis=1)

    # Inisialisasi LabelEncoder untuk setiap kolom kategorikal
    le_gender = LabelEncoder()
    le_status_mahasiswa = LabelEncoder()
    le_status_nikah = LabelEncoder()

    # Fit dan transform kolom kategorikal
    df_dummy['Gender_encoded'] = le_gender.fit_transform(df_dummy['Gender'])
    df_dummy['Status_Mahasiswa_encoded'] = le_status_mahasiswa.fit_transform(df_dummy['Status_Mahasiswa'])
    df_dummy['Status_Nikah_encoded'] = le_status_nikah.fit_transform(df_dummy['Status_Nikah'])

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

# Muat model dan encoders (akan dilatih ulang sekali atau dimuat jika file ada)
model, le_gender, le_status_mahasiswa, le_status_nikah, model_features = load_and_train_model()

# --- Judul Aplikasi Streamlit ---
st.title('Aplikasi Prediksi Kelulusan Mahasiswa')
st.write('Isi form di bawah ini untuk memprediksi probabilitas kelulusan mahasiswa berdasarkan berbagai kriteria.')

# --- Bagian Form Input ---
st.header('Data Diri Mahasiswa')

# Menggunakan kolom untuk tata letak yang lebih rapi
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox('Jenis Kelamin', ['Pria', 'Wanita'])
    # Diperbarui: Opsi Status Mahasiswa
    status_mahasiswa = st.selectbox('Status Mahasiswa', ['Bekerja', 'Tidak Bekerja'])
with col2:
    usia = st.number_input('Usia', min_value=17, max_value=70, value=20, help="Masukkan usia mahasiswa.")
    status_nikah = st.selectbox('Status Pernikahan', ['Belum Menikah', 'Menikah', 'Cerai', 'Janda/Duda'])

st.header('Nilai Akademik')
ips_values = {}
# Menggunakan st.columns untuk tata letak input IPS yang lebih teratur
num_ips_cols = 4 # Jumlah kolom untuk input IPS
cols_ips = st.columns(num_ips_cols)

for i in range(1, 9):
    with cols_ips[(i - 1) % num_ips_cols]: # Mendistribusikan input ke kolom
        ips_values[f'IPS{i}'] = st.number_input(f'IPS Semester {i}', min_value=0.0, max_value=4.0, value=3.0, step=0.01, key=f'ips_{i}', help=f"Masukkan Indeks Prestasi Semester {i} (0.0 - 4.0).")

# IPK sekarang dihitung otomatis, tidak lagi menjadi input langsung
# st.number_input('IPK (Indeks Prestasi Kumulatif)', min_value=0.0, max_value=4.0, value=3.0, step=0.01, help="Masukkan Indeks Prestasi Kumulatif (0.0 - 4.0).")

st.header('Prediksi Kelulusan')
# --- Tombol Prediksi ---
if st.button('Prediksi Kelulusan'):
    try:
        # Preprocessing input untuk fitur kategorikal menggunakan encoder yang sudah dilatih
        gender_encoded = le_gender.transform([gender])[0]
        status_mahasiswa_encoded = le_status_mahasiswa.transform([status_mahasiswa])[0]
        status_nikah_encoded = le_status_nikah.transform([status_nikah])[0]

        # Hitung IPK dari rata-rata IPS yang diinput
        # Pastikan list IPS tidak kosong sebelum menghitung rata-rata
        if ips_values:
            calculated_ipk = np.mean(list(ips_values.values()))
        else:
            calculated_ipk = 0.0 # Default value if no IPS inputs (though this shouldn't happen with fixed loops)
        st.info(f"IPK yang Dihitung Otomatis: {calculated_ipk:.2f}") # Tampilkan IPK yang dihitung

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
        # Ini penting agar model menerima input dalam urutan yang benar
        input_data_df = pd.DataFrame([input_dict])[model_features]

        # Lakukan Prediksi
        prediction = model.predict(input_data_df)
        prediction_proba = model.predict_proba(input_data_df) # Probabilitas untuk kedua kelas (0 dan 1)

        st.subheader('Hasil Prediksi:')
        if prediction[0] == 1:
            st.success('Mahasiswa diprediksi **LULUS**! 🎉')
        else:
            st.error('Mahasiswa diprediksi **TIDAK LULUS**! 😔')

        # Menampilkan probabilitas dengan format yang mudah dibaca
        st.write(f"Probabilitas Lulus: **{prediction_proba[0][1]*100:.2f}%**")
        st.write(f"Probabilitas Tidak Lulus: **{prediction_proba[0][0]*100:.2f}%**")

    except ValueError as ve:
        # Menangkap error jika nilai kategorikal tidak valid untuk encoder
        st.error(f"Kesalahan pada input kategori: {ve}. Pastikan semua pilihan kategori valid dan sesuai dengan opsi yang tersedia.")
    except Exception as e:
        # Menangkap error umum lainnya
        st.error(f"Terjadi kesalahan umum: {e}. Mohon coba lagi atau hubungi dukungan teknis.")

st.markdown('---')
st.caption('Aplikasi ini dibuat menggunakan Streamlit dan scikit-learn. Model prediksi menggunakan data dummy; untuk hasil yang akurat, gantilah dengan model yang telah dilatih pada dataset asli Anda.')
