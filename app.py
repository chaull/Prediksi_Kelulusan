import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan encoder
model = joblib.load('model_kelulusan.pkl')
encoders = joblib.load('encoders.pkl')
fitur_model = joblib.load('fitur_model.pkl')

# Set halaman
st.set_page_config(page_title="Prediksi Kelulusan", page_icon="🎓", layout="centered")
st.markdown("<h2 style='text-align: center; color: #4B8BBE;'>🎓 Prediksi Kelulusan Mahasiswa</h2>", unsafe_allow_html=True)
st.markdown("---")

with st.form("form_prediksi"):
    st.markdown("### 🧾 Informasi Mahasiswa")
    col1, col2 = st.columns(2)
    with col1:
        jenis_kelamin = st.selectbox("👤 Jenis Kelamin", encoders['JENIS KELAMIN'].classes_)
        status_nikah = st.selectbox("💍 Status Nikah", encoders['STATUS NIKAH'].classes_)
    with col2:
        status_mahasiswa = st.selectbox("🎓 Status Mahasiswa", encoders['STATUS MAHASISWA'].classes_)
        umur = st.number_input("📅 Umur", min_value=15, max_value=100, step=1)

    st.markdown("---")
    st.markdown("### 📚 Nilai IPS Semester")

    # Buat layout IPS jadi 4 baris x 2 kolom
    ips = []
    for row in range(4):
        col1, col2 = st.columns(2)
        with col1:
            ips_val1 = st.text_input(f"IPS {row*2 + 1}", min_value=0.0, max_value=4.0, step=0.01, key=f"ips_{row*2 + 1}")
            ips.append(ips_val1)
        with col2:
            ips_val2 = st.text_input(f"IPS {row*2 + 2}", min_value=0.0, max_value=4.0, step=0.01, key=f"ips_{row*2 + 2}")
            ips.append(ips_val2)

    # Hitung IPK
    ipk = round(np.mean(ips), 2)
    st.success(f"📈 IPK Otomatis: {ipk}")

    st.markdown("---")
    submit = st.form_submit_button("🔍 Prediksi Kelulusan")

# Proses prediksi
if submit:
    input_data = {
        'JENIS KELAMIN': encoders['JENIS KELAMIN'].transform([jenis_kelamin])[0],
        'STATUS MAHASISWA': encoders['STATUS MAHASISWA'].transform([status_mahasiswa])[0],
        'UMUR': umur,
        'STATUS NIKAH': encoders['STATUS NIKAH'].transform([status_nikah])[0],
        'IPS 1': ips[0],
        'IPS 2': ips[1],
        'IPS 3': ips[2],
        'IPS 4': ips[3],
        'IPS 5': ips[4],
        'IPS 6': ips[5],
        'IPS 7': ips[6],
        'IPS 8': ips[7],
        'IPK': ipk
    }

    df_input = pd.DataFrame([input_data])
    df_input.columns = df_input.columns.str.strip()
    fitur_model = [col.strip() for col in fitur_model]

    try:
        df_input = df_input[fitur_model]
        input_array = df_input.to_numpy()

        pred = model.predict(input_array)[0]
        prob = model.predict_proba(input_array)[0]
        hasil = encoders['STATUS KELULUSAN'].inverse_transform([pred])[0]

        st.balloons()
        st.success(f"🎯 Hasil Prediksi: Mahasiswa diperkirakan akan **{hasil.upper()}**")
        st.markdown("### 📊 Rincian Probabilitas")
        for i, label in enumerate(encoders['STATUS KELULUSAN'].classes_):
            st.write(f"- **{label}**: {round(prob[i]*100, 2)}%")

    except KeyError as e:
        st.error(f"❌ Kolom input tidak cocok dengan model: {e}")
        st.write("Kolom input:", df_input.columns.tolist())
        st.write("Kolom yang diminta model:", fitur_model) 

