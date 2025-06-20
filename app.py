import streamlit as st
import pandas as pd
import numpy as np
import io # Untuk membaca CSV dari string data
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide", # Menggunakan layout lebar
    initial_sidebar_state="auto"
)

# Fungsi untuk memuat data CSV dari konten yang disediakan dan melatih model.
# Fungsi ini di-cache (@st.cache_resource) agar model hanya dilatih sekali saat aplikasi
# pertama kali dijalankan atau saat kode/konten CSV berubah.
@st.cache_resource
def load_and_train_model(csv_content):
    """
    Memuat data dari konten CSV yang disediakan, melakukan preprocessing,
    melatih model RandomForestClassifier, dan mengembalikan model yang sudah dilatih
    beserta LabelEncoders yang sudah di-fit dan daftar fitur.
    Fungsi ini di-cache untuk mencegah pelatihan ulang yang tidak perlu.
    """
    try:
        # Membaca data CSV dari string konten yang disediakan, menggunakan semicolon sebagai delimiter
        df = pd.read_csv(io.StringIO(csv_content), delimiter=';')

        # --- Pembersihan Data dan Preprocessing ---
        # 1. Mengubah nama kolom agar konsisten dengan logika aplikasi
        df = df.rename(columns={
            'JENIS KELAMIN': 'Gender',
            'STATUS MAHASISWA': 'Status_Mahasiswa',
            'UMUR': 'Usia',
            'STATUS NIKAH': 'Status_Nikah',
            'IPS 1': 'IPS1',
            'IPS 2': 'IPS2',
            'IPS 3': 'IPS3',
            'IPS 4': 'IPS4',
            'IPS 5': 'IPS5',
            'IPS 6': 'IPS6',
            'IPS 7': 'IPS7',
            'IPS 8': 'IPS8',
            'STATUS KELULUSAN': 'Kelulusan'
        })

        # 2. Mengkonversi kolom IPS ke numerik, menangani koma dan nilai yang hilang
        ips_cols = [f'IPS{i}' for i in range(1, 9)]
        data_cols_to_numeric = ips_cols

        for col in data_cols_to_numeric:
            if col in df.columns:
                # Mengganti koma dengan titik untuk konversi desimal dan mengubah ke numerik
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce') # Konversi ke numerik, ubah kesalahan menjadi NaN
                # Mengisi nilai NaN di kolom IPS dengan 0.0 (asumsi 0.0 untuk IPS yang tidak tercatat/hilang)
                df[col] = df[col].fillna(0.0)
            else:
                st.warning(f"Kolom '{col}' tidak ditemukan dalam data CSV. Pastikan nama kolom sudah benar.")
                return None, None, None, None, None # Menunjukkan kegagalan

        # 3. Menghitung ulang IPK dari nilai IPS
        df['IPK'] = df[ips_cols].mean(axis=1)

        # 4. Inisialisasi dan fit LabelEncoders untuk fitur kategorikal
        le_gender = LabelEncoder()
        le_status_mahasiswa = LabelEncoder()
        le_status_nikah = LabelEncoder()

        # Mendefinisikan semua kategori yang mungkin untuk fitting yang kuat, sesuai dengan opsi input aplikasi
        gender_mapping_dict = {
            'LAKI - LAKI': 'Pria',
            'PEREMPUAN': 'Wanita'
        }
        status_mhs_mapping_dict = {
            'MAHASISWA': 'Tidak Bekerja' # Asumsi 'MAHASISWA' di CSV sesuai dengan 'Tidak Bekerja'
        }
        status_nikah_mapping_dict = {
            'BELUM MENIKAH': 'Belum Menikah',
            'MENIKAH': 'Menikah'
        }

        # Menerapkan pemetaan sebelum encoding
        df['Gender_mapped'] = df['Gender'].map(gender_mapping_dict).fillna(df['Gender'])
        df['Status_Mahasiswa_mapped'] = df['Status_Mahasiswa'].map(status_mhs_mapping_dict).fillna(df['Status_Mahasiswa'])
        df['Status_Nikah_mapped'] = df['Status_Nikah'].map(status_nikah_mapping_dict).fillna(df['Status_Nikah'])

        # Fit encoder dengan semua kategori *yang diharapkan* dari tombol radio aplikasi
        le_gender.fit(['Pria', 'Wanita'])
        le_status_mahasiswa.fit(['Bekerja', 'Tidak Bekerja'])
        le_status_nikah.fit(['Belum Menikah', 'Menikah'])

        # Transformasi kolom kategorikal
        df['Gender_encoded'] = le_gender.transform(df['Gender_mapped'])
        df['Status_Mahasiswa_encoded'] = le_status_mahasiswa.transform(df['Status_Mahasiswa_mapped'])
        df['Status_Nikah_encoded'] = le_status_nikah.transform(df['Status_Nikah_mapped'])

        # 5. Menyiapkan variabel target (Kelulusan)
        # Memetakan 'TEPAT' ke 1 (Lulus Tepat Waktu), nilai lain ke 0 (Terlambat Lulus)
        df['Kelulusan_encoded'] = df['Kelulusan'].apply(lambda x: 1 if x == 'TEPAT' else 0)

        # Mendefinisikan fitur (X) dan target (y) untuk model
        model_features = [
            'Usia', 'IPK',
            'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'IPS6', 'IPS7', 'IPS8',
            'Gender_encoded', 'Status_Mahasiswa_encoded', 'Status_Nikah_encoded'
        ]

        # Memastikan semua fitur model ada di DataFrame sebelum pelatihan
        X = df[model_features]
        y = df['Kelulusan_encoded']

        # Melatih model RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        st.success("Model berhasil dilatih menggunakan data 'Kelulusan Test.csv' yang Anda berikan!")
        return model, le_gender, le_status_mahasiswa, le_status_nikah, model_features

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat atau melatih model dari data CSV: {e}")
        st.warning("Menggunakan model dummy sementara karena kesalahan. Harap periksa format file CSV Anda.")
        # Fallback ke model dummy jika pemuatan/pelatihan CSV gagal
        data = {
            'Gender': ['Pria', 'Wanita'] * 10,
            'Status_Mahasiswa': ['Bekerja', 'Tidak Bekerja'] * 10,
            'Status_Nikah': ['Belum Menikah', 'Menikah'] * 10,
            'Usia': [20 + i for i in range(20)],
            'IPS1': [3.5]*20, 'IPS2': [3.0]*20, 'IPS3': [3.2]*20, 'IPS4': [3.1]*20,
            'IPS5': [3.3]*20, 'IPS6': [2.9]*20, 'IPS7': [2.8]*20, 'IPS8': [3.0]*20,
            'Kelulusan': [1 if i % 2 == 0 else 0 for i in range(20)]
        }
        df_dummy = pd.DataFrame(data)
        ips_cols_dummy = [f'IPS{i}' for i in range(1, 9)]
        df_dummy['IPK'] = df_dummy[ips_cols_dummy].mean(axis=1)

        le_gender = LabelEncoder()
        le_status_mahasiswa = LabelEncoder()
        le_status_nikah = LabelEncoder()

        le_gender.fit(['Pria', 'Wanita'])
        le_status_mahasiswa.fit(['Bekerja', 'Tidak Bekerja'])
        le_status_nikah.fit(['Belum Menikah', 'Menikah'])

        df_dummy['Gender_encoded'] = le_gender.transform(df_dummy['Gender'])
        df_dummy['Status_Mahasiswa_encoded'] = le_status_mahasiswa.transform(df_dummy['Status_Mahasiswa'])
        df_dummy['Status_Nikah_encoded'] = le_status_nikah.transform(df_dummy['Status_Nikah'])

        model_features = [
            'Usia', 'IPK',
            'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'IPS6', 'IPS7', 'IPS8',
            'Gender_encoded', 'Status_Mahasiswa_encoded', 'Status_Nikah_encoded'
        ]
        X_dummy = df_dummy[model_features]
        y_dummy = df_dummy['Kelulusan']

        model = RandomForestClassifier(random_state=42)
        model.fit(X_dummy, y_dummy)
        return model, le_gender, le_status_mahasiswa, le_status_nikah, model_features

# --- Mengambil konten CSV menggunakan tool ---
# Catatan: Baris ini akan dieksekusi secara otomatis oleh sistem untuk mendapatkan konten file.
# Anda tidak perlu melihat definisi tool di sini, cukup panggil `content_fetcher.fetch`.
try:
    # Memanggil tool untuk mengambil konten file "Kelulusan Test.csv"
    # Konten ini kemudian akan diteruskan ke fungsi load_and_train_model.
    csv_file_content = content_fetcher.fetch(query="Kelulusan Test.csv", source_references=[{"id": "uploaded:Kelulusan Test.csv", "type": "text/csv"}])
except Exception as e:
    st.error(f"Gagal mengambil konten file 'Kelulusan Test.csv': {e}. Pastikan file sudah diunggah.")
    csv_file_content = "" # Set kosong agar load_and_train_model bisa menggunakan fallback dummy

# Memuat model dan encoder menggunakan konten CSV yang diambil.
# Karena @st.cache_resource, ini hanya akan dijalankan sekali di awal.
model, le_gender, le_status_mahasiswa, le_status_nikah, model_features = load_and_train_model(csv_file_content)

# Menangani kasus di mana pemuatan/pelatihan model gagal secara kritis
if model is None:
    st.stop() # Menghentikan aplikasi jika komponen penting tidak dimuat

# --- Judul Aplikasi Streamlit ---
st.title('Aplikasi Prediksi Kelulusan Mahasiswa')
st.write('Isi formulir di bawah ini untuk memprediksi probabilitas kelulusan mahasiswa berdasarkan berbagai kriteria.')

# --- Bagian Formulir Input ---
st.header('Data Diri Mahasiswa')

col1, col2 = st.columns(2)
with col1:
    gender = st.radio('Jenis Kelamin', ['Pria', 'Wanita'], index=0)
    status_mahasiswa = st.radio('Status Mahasiswa', ['Bekerja', 'Tidak Bekerja'], index=1)
with col2:
    usia = st.number_input('Usia', min_value=17, max_value=70, value=20, help="Masukkan usia mahasiswa.")
    status_nikah = st.radio('Status Pernikahan', ['Belum Menikah', 'Menikah'], index=0)

st.header('Nilai Akademik')
ips_values = {}
num_ips_cols = 4
cols_ips = st.columns(num_ips_cols)

for i in range(1, 9):
    with cols_ips[(i - 1) % num_ips_cols]:
        ips_values[f'IPS{i}'] = st.number_input(f'IPS Semester {i}', min_value=0.0, max_value=4.0, value=3.0, step=0.01, key=f'ips_{i}', help=f"Masukkan Indeks Prestasi Semester {i} (0.0 - 4.0).")

# Menghitung IPK dari nilai IPS yang diinput
if ips_values:
    calculated_ipk = np.mean(list(ips_values.values()))
else:
    calculated_ipk = 0.0

# Menampilkan IPK yang dihitung sebelum tombol prediksi
st.info(f"IPK yang Dihitung Otomatis: **{calculated_ipk:.2f}**")

st.header('Prediksi Kelulusan')
# --- Tombol Prediksi ---
if st.button('Prediksi Kelulusan'):
    try:
        # Preprocessing input untuk fitur kategorikal menggunakan encoder yang sudah dilatih
        gender_encoded = le_gender.transform([gender])[0]
        status_mahasiswa_encoded = le_status_mahasiswa.transform([status_mahasiswa])[0]
        status_nikah_encoded = le_status_nikah.transform([status_nikah])[0]

        # Menyiapkan data input dalam format dictionary
        input_dict = {
            'Usia': usia,
            'IPK': calculated_ipk,
            'Gender_encoded': gender_encoded,
            'Status_Mahasiswa_encoded': status_mahasiswa_encoded,
            'Status_Nikah_encoded': status_nikah_encoded,
        }
        for i in range(1, 9):
            input_dict[f'IPS{i}'] = ips_values[f'IPS{i}']

        # Membuat DataFrame dari input pengguna, memastikan urutan kolom sesuai dengan model_features
        input_data_df = pd.DataFrame([input_dict])[model_features]

        # Melakukan Prediksi
        prediction = model.predict(input_data_df)
        prediction_proba = model.predict_proba(input_data_df)

        st.subheader('Hasil Prediksi & Skala Probabilitas')

        pass_proba = prediction_proba[0][1] * 100
        fail_proba = prediction_proba[0][0] * 100

        if prediction[0] == 1:
            st.success(f'Mahasiswa diprediksi **LULUS TEPAT WAKTU**! 🎉 Probabilitas: **{pass_proba:.2f}%**')
        else:
            st.error(f'Mahasiswa diprediksi **TERLAMBAT LULUS**! 😔 Probabilitas: **{fail_proba:.2f}%**')

        st.write("Skala Probabilitas Lulus Tepat Waktu:")
        st.progress(pass_proba / 100)


    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}. Mohon coba lagi. "
                 f"Jika masalah berlanjut, pastikan semua input valid.")
        st.write("Detail kesalahan (untuk debugging):", e)

st.markdown('---')
st.caption('Aplikasi ini dibuat menggunakan Streamlit dan scikit-learn. Model prediksi dilatih pada data "Kelulusan Test.csv" yang Anda berikan. Karena data CSV yang diberikan hanya memiliki satu kategori utama untuk "STATUS MAHASISWA" (`MAHASISWA`) dan "STATUS KELULUSAN" (`TEPAT`), model mungkin memiliki keterbatasan dalam memprediksi variasi yang tidak ada dalam data latih. `st.cache_resource` digunakan agar model hanya dilatih sekali di awal dan kemudian siap digunakan.')
