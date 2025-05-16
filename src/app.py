import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px

import warnings

# Cargar modelo y escalador
model = joblib.load("models/knn_model.pkl")
scaler = joblib.load("models/scaler.pkl")



# Cargar dataset para obtener rangos de los sliders
url = "https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/refs/heads/main/winequality-red.csv"
df = pd.read_csv(url, sep=";")

st.title("Predicción de Calidad de Vino 🍷")

# Crear sliders para cada feature
features = []
columns = df.drop(columns=["quality"]).columns

for col in columns:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    val = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2, step=0.01)
    features.append(val)

# Botón de predicción
if st.button("Predecir calidad"):
    input_scaled = scaler.transform([features])
    prediction = model.predict(input_scaled)[0]

    # Mapear resultado
    calidad = {0: "Baja", 1: "Media", 2: "Alta"}
    st.success(f"✅ Este vino probablemente sea de **calidad {calidad[prediction]}**.")
