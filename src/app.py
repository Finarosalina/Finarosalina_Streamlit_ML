import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px


# ** RENDER Finarosalina_Streamlit_ML **
# https://finarosalina-streamlit-ml.onrender.com


st.set_page_config(page_title="Predicción de Vino", layout="centered")
st.title("🍷 Predicción de Calidad de Vino")

#  cargar modelo y escalador
try:
    model = joblib.load("src/models/knn_model.pkl")
    scaler = joblib.load("src/models/scaler.pkl")
except FileNotFoundError:
    st.error("❌ Error: No se encontró el archivo del modelo o el escalador.")
    st.stop()

# Cargar dataset desde URL
url = "https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/refs/heads/main/winequality-red.csv"

try:
    df = pd.read_csv(url, sep=";")
except Exception as e:
    st.error(f"❌ Error al cargar el dataset: {e}")
    st.stop()

# Crear sliders para entrada de usuario
st.subheader("🔧 Ajusta las características del vino")
features = []
columns = df.drop(columns=["quality"]).columns

for col in columns:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    val = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2, step=0.01)
    features.append(val)

# Mapeo de clases
calidad = {0: "Baja", 1: "Media", 2: "Alta"}

# Predicción
if st.button("🔍 Predecir calidad"):
    input_scaled = scaler.transform([features])
    prediction = model.predict(input_scaled)[0]

    st.success(f"✅ Este vino probablemente sea de **calidad {calidad[prediction]}**.")

