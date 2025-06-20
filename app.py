import streamlit as st
import pandas as pd
import numpy as np
import pickle # Untuk menyimpan dan memuat model
from sklearn.preprocessing import LabelEncoder
# Import model yang Anda gunakan, contoh:
from sklearn.ensemble import RandomForestClassifier

# --- Bagian Simulasi Pra-pelatihan dan Penyimpanan Model (Anda bisa mengganti ini dengan memuat model yang sudah ada) ---
# Sebaiknya, Anda melatih model di luar aplikasi Streamlit dan menyimpannya (misal dengan pickle),
# lalu memuatnya di aplikasi Streamlit.

# Data dummy untuk contoh LabelEncoder dan pelatihan model
# Di aplikasi nyata, Anda akan memuat model yang sudah dilatih
@st.cache_resource # Cache resource untuk menghindari pelatihan ulang setiap kali aplikasi berjalan
def load_and_train_model():
    # Ini hanya contoh. Di aplikasi nyata, Anda mungkin memuat model dari file .pkl
    # atau melatihnya jika datasetnya kecil dan training cepat.
    data = {
        'Gender': ['Pria', 'Wanita', 'Pria', 'Wanita', 'Pria', 'Wanita'],
        'Usia': [20, 21, 22, 20, 23, 21],
        'IPK': [3.5, 3.8, 3.0, 3.9, 3.2, 3.7],
        'Kelulusan': [1, 1, 0, 1, 0, 1] # 1 untuk Lulus, 0 untuk Tidak Lulus
    }
    df_dummy = pd.DataFrame(data)

    le_gender = LabelEncoder()
    df_dummy['Gender_encoded'] = le_gender.fit_transform(df_dummy['Gender'])

    X_dummy = df_dummy[['Usia', 'IPK', 'Gender_encoded']]
    y_dummy = df_dummy['Kelulusan']

    model = RandomForestClassifier(random_state=42)
    model.fit(X_dummy, y_dummy)

    return model, le_gender

model, le_gender = load_and_train_model()

# --- Judul Aplikasi ---
st.title('Aplikasi Prediksi Kelulusan Mahasiswa')
st.write('Isi form di bawah untuk memprediksi kelulusan mahasiswa.')

# --- Form Input ---
st.header('Data Mahasiswa')

gender = st.selectbox('Jenis Kelamin', ['Pria', 'Wanita'])
usia = st.number_input('Usia', min_value=18, max_value=60, value=20)
ipk = st.number_input('IPK (Indeks Prestasi Kumulatif)', min_value=0.0, max_value=4.0, value=3.0, step=0.01)

# --- Tombol Prediksi ---
if st.button('Prediksi Kelulusan'):
    try:
        # Preprocessing input
        gender_encoded = le_gender.transform([gender])[0] # Menggunakan encoder yang sama saat melatih model

        # Buat DataFrame dari input pengguna
        input_data = pd.DataFrame([[usia, ipk, gender_encoded]],
                                  columns=['Usia', 'IPK', 'Gender_encoded'])

        # Prediksi
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.subheader('Hasil Prediksi:')
        if prediction[0] == 1:
            st.success('Mahasiswa diprediksi **LULUS**! 🎉')
        else:
            st.error('Mahasiswa diprediksi **TIDAK LULUS**! 😔')

        st.write(f"Probabilitas Lulus: {prediction_proba[0][1]*100:.2f}%")
        st.write(f"Probabilitas Tidak Lulus: {prediction_proba[0][0]*100:.2f}%")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}. Pastikan semua input valid.")

st.markdown('---')
st.caption('Aplikasi ini dibuat menggunakan Streamlit dan scikit-learn.')
