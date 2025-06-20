
# Create DataFrame from user input, ensure column order matches model_features
input_data_df = pd.DataFrame([input_dict])[model_features]

# Perform Prediction
prediction = model.predict(input_data_df)
prediction_proba = model.predict_proba(input_data_df) # Probabilities for both classes (0 and 1)

st.subheader('Hasil Prediksi:')
if prediction[0] == 1:
st.success('Mahasiswa diprediksi **LULUS**! �')
else:
st.error('Mahasiswa diprediksi **TIDAK LULUS**! 😔')

# Display probabilities in a readable format
st.write(f"Probabilitas Lulus: **{prediction_proba[0][1]*100:.2f}%**")
st.write(f"Probabilitas Tidak Lulus: **{prediction_proba[0][0]*100:.2f}%**")

except Exception as e:
# Catch any unexpected errors
st.error(f"Terjadi kesalahan: {e}. Mohon coba lagi. Jika masalah berlanjut, pastikan semua input valid dan model Anda dilatih dengan benar.")

st.markdown('---')
st.caption('Aplikasi ini dibuat menggunakan Streamlit dan scikit-learn. Model prediksi menggunakan data dummy; untuk hasil yang akurat, gantilah dengan model yang telah dilatih pada dataset asli Anda.')
�
