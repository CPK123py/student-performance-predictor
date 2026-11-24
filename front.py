import streamlit as st
import pickle
import numpy as np

with open('Xmodel\perfomate.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Student Performance Predictor")
inputs=[st.radio("Gender", [0, 1]),
          st.radio("Part-time job", [0, 1]),
          st.slider("Absence days", 0, 20, 2),
          st.radio("Extracurricular", [0, 1]),
          st.slider("Self-study hours", 0, 20, 10),
          st.selectbox("Career Aspiration", list(range(8)))]

if st.button("Predict Grade"):
    result = model.predict([inputs])
    st.success(f"Predicted Grade: {result[0]:.2f}")
