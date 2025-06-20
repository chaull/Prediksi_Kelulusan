import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide", # Menggunakan layout lebar
    initial_sidebar_state="auto"
)

# --- Custom CSS for enhanced aesthetics ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #333;
    }

    .main {
        background-color: #f0f2f6; /* Light gray background */
        padding: 20px;
        border-radius: 10px;
    }

    .stApp {
        background: linear-gradient(to right, #ece9e6, #ffffff); /* Subtle gradient */
    }

    .stContainer {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }

    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 25px;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }

    h2, h3, h4 {
        color: #34495e;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        width: 100%; /* Make button full width */
    }

    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }

    .stSuccess {
        background-color: #e6ffe6; /* Light green */
        color: #28a745; /* Darker green text */
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #28a745;
        font-weight: 600;
        text-align: center;
        margin-top: 20px;
    }

    .stError {
        background-color: #ffe6e6; /* Light red */
        color: #dc3545; /* Darker red text */
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dc3545;
        font-weight: 600;
        text-align: center;
        margin-top: 20px;
    }

    .stInfo {
        background-color: #e6f7ff; /* Light blue */
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
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }

    .progress-bar {
        height: 30px;
        line-height: 30px;
        color: white;
        text-align: center;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
        background: linear-gradient(to right, #28a745, #8bc34a); /* Green gradient */
    }

    .progress-bar.red {
        background: linear-gradient(to right, #dc3545, #ff5722); /* Red gradient */
    }

    .stMarkdown p {
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to load and train the model (or load an existing one)
@st.cache_resource # Cache resource to avoid retraining every time the app runs
def load_and_train_model():
    # --- IMPORTANT: REPLACE THIS SECTION WITH YOUR ACTUAL MODEL TRAINING CODE ---
    # If you have a saved model and encoders (e.g., .pkl files),
    # you can load them here. Example:
    # try:
    #     with open('model.pkl', 'rb') as f:
    #         model = pickle.load(f)
    #     with open('le_gender.pkl', 'rb') as f:
    #         le_gender = pickle.load(f)
    #     with open('le_status_mhs.pkl', 'rb') as f:
    #         le_status_mhs = pickle.load(f)
    #     with open('le_status_nikah.pkl', 'rb') as f:
    #         le_status_nikah = pickle.load(f)
    #     # Define the exact feature order your model expects
    #     # IPK will be calculated, so ensure your model was trained with IPK as a feature
    #     model_features = ['Usia', 'IPK', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'IPS6', 'IPS7', 'IPS8',
    #                       'Gender_encoded', 'Status_Mahasiswa_encoded', 'Status_Nikah_encoded']
    #     st.success("Model dan encoder berhasil dimuat dari file.")
    #     return model, le_gender, le_status_mhs, le_status_nikah, model_features
    # except FileNotFoundError:
    #     st.warning("File model atau encoder tidak ditemukan. Melatih ulang model dummy...")

    # Extended dummy data for training example
    # In a real application, you would use data from your 'Kelulusan Train.csv' file.
    data = {
        'Gender': ['Pria', 'Wanita'] * 10, # 20 entries
        'Status_Mahasiswa': ['Bekerja', 'Tidak Bekerja'] * 10, # 20 entries
        'Status_Nikah': ['Belum Menikah', 'Menikah'] * 10, # 20 entries
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

    # Calculate IPK as average of IPS1-IPS8 for dummy data
    ips_cols = [f'IPS{i}' for i in range(1, 9)]
    df_dummy['IPK'] = df_dummy[ips_cols].mean(axis=1)

    # Initialize LabelEncoders
    le_gender = LabelEncoder()
    le_status_mahasiswa = LabelEncoder()
    le_status_nikah = LabelEncoder()

    # Explicitly fit LabelEncoders with ALL possible categories to prevent unseen label errors
    le_gender.fit(['Pria', 'Wanita'])
    le_status_mahasiswa.fit(['Bekerja', 'Tidak Bekerja'])
    le_status_nikah.fit(['Belum Menikah', 'Menikah'])

    # Transform categorical columns in dummy data
    df_dummy['Gender_encoded'] = le_gender.transform(df_dummy['Gender'])
    df_dummy['Status_Mahasiswa_encoded'] = le_status_mahasiswa.transform(df_dummy['Status_Mahasiswa'])
    df_dummy['Status_Nikah_encoded'] = le_status_nikah.transform(df_dummy['Status_Nikah'])

    # Define features (X) and target (y)
    # Ensure column order matches what your model expects during training
    model_features = [
        'Usia', 'IPK', # IPK is now calculated
        'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'IPS6', 'IPS7', 'IPS8',
        'Gender_encoded', 'Status_Mahasiswa_encoded', 'Status_Nikah_encoded'
    ]
    X_dummy = df_dummy[model_features]
    y_dummy = df_dummy['Kelulusan']

    # Train RandomForestClassifier model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_dummy, y_dummy)

    # Return model, encoders, and feature list for consistency
    return model, le_gender, le_status_mahasiswa, le_status_nikah, model_features

# Load model and encoders (will be trained once or loaded if files exist)
model, le_gender, le_status_mahasiswa, le_status_nikah, model_features = load_and_train_model()

# --- Streamlit Application Title ---
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🎓 Aplikasi Prediksi Kelulusan Mahasiswa</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1em;'>Isi form di bawah ini untuk memprediksi probabilitas kelulusan mahasiswa berdasarkan berbagai kriteria.</p>", unsafe_allow_html=True)

# --- Input Form Section ---
st.markdown("---")
st.subheader('Data Diri Mahasiswa')

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Jenis Kelamin', ['Pria', 'Wanita'])
        status_mahasiswa = st.selectbox('Status Mahasiswa', ['Bekerja', 'Tidak Bekerja'])
    with col2:
        usia = st.number_input('Usia', min_value=17, max_value=70, value=20, help="Masukkan usia mahasiswa.")
        status_nikah = st.selectbox('Status Pernikahan', ['Belum Menikah', 'Menikah'])

st.markdown("---")
st.subheader('Nilai Akademik')

with st.container():
    ips_values = {}
    num_ips_cols = 4
    cols_ips = st.columns(num_ips_cols)

    for i in range(1, 9):
        with cols_ips[(i - 1) % num_ips_cols]:
            ips_values[f'IPS{i}'] = st.number_input(f'IPS Semester {i}', min_value=0.0, max_value=4.0, value=3.0, step=0.01, key=f'ips_{i}', help=f"Masukkan Indeks Prestasi Semester {i} (0.0 - 4.0).")

# Calculate IPK from the input IPS values
if ips_values:
    calculated_ipk = np.mean(list(ips_values.values()))
else:
    calculated_ipk = 0.0

# Display the calculated IPK before the prediction button
st.info(f"<b>IPK yang Dihitung Otomatis: <span style='font-size:1.2em;'>{calculated_ipk:.2f}</span></b>", unsafe_allow_html=True)

st.markdown("---")
st.subheader('Prediksi Kelulusan')

# --- Prediction Button ---
if st.button('Prediksi Kelulusan'):
    try:
        # Preprocessing input for categorical features using trained encoders
        gender_encoded = le_gender.transform([gender])[0]
        status_mahasiswa_encoded = le_status_mahasiswa.transform([status_mahasiswa])[0]
        status_nikah_encoded = le_status_nikah.transform([status_nikah])[0]

        # Prepare input data in dictionary format
        input_dict = {
            'Usia': usia,
            'IPK': calculated_ipk,
            'Gender_encoded': gender_encoded,
            'Status_Mahasiswa_encoded': status_mahasiswa_encoded,
            'Status_Nikah_encoded': status_nikah_encoded,
        }
        for i in range(1, 9):
            input_dict[f'IPS{i}'] = ips_values[f'IPS{i}']

        # Create DataFrame from user input, ensure column order matches model_features
        input_data_df = pd.DataFrame([input_dict])[model_features]

        # Perform Prediction
        prediction = model.predict(input_data_df)
        prediction_proba = model.predict_proba(input_data_df)

        st.markdown("---")
        st.subheader('Hasil Prediksi & Skala Probabilitas')

        # Display results and probability scale
        col_res, col_scale = st.columns([1, 2]) # Adjust column width for result and scale

        with col_res:
            if prediction[0] == 1:
                st.success('Mahasiswa diprediksi **LULUS**! 🎉')
            else:
                st.error('Mahasiswa diprediksi **TIDAK LULUS**! 😔')

        with col_scale:
            pass_proba = prediction_proba[0][1] * 100
            fail_proba = prediction_proba[0][0] * 100

            st.write(f"Probabilitas Lulus: **{pass_proba:.2f}%**")
            st.write(f"Probabilitas Tidak Lulus: **{fail_proba:.2f}%**")

            # Custom visual scale (progress bar like)
            progress_bar_color_class = "progress-bar" if pass_proba >= 50 else "progress-bar red"
            st.markdown(f"""
                <div class="progress-container">
                    <div class="{progress_bar_color_class}" style="width: {pass_proba:.2f}%;">
                        {pass_proba:.2f}% Lulus
                    </div>
                </div>
            """, unsafe_allow_html=True)


    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}. Mohon coba lagi. Jika masalah berlanjut, pastikan semua input valid dan model Anda dilatih dengan benar.")

st.markdown('---')
st.caption('Aplikasi ini dibuat menggunakan Streamlit dan scikit-learn. Model prediksi menggunakan data dummy; untuk hasil yang akurat, gantilah dengan model yang telah dilatih pada dataset asli Anda.')
