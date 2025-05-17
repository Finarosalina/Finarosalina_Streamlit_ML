import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px

import warnings

# Cargar modelo y escalador
model = joblib.load("src/models/knn_model.pkl")
scaler = joblib.load("src/models/scaler.pkl")



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

try:
    model = joblib.load("models/knn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except FileNotFoundError:
    st.error("Error: No se encontró el archivo del modelo o el escalador.")
    st.stop()

try:
    df = pd.read_csv(url, sep=";")
except Exception as e:
    st.error(f"Error al cargar el dataset: {e}")
    st.stop()

# Mostrar gráfico de la calidad predicha
fig = px.bar(
    x=["Baja", "Media", "Alta"],
    y=[1 if prediction == i else 0 for i in range(3)],
    labels={"x": "Calidad", "y": "Probabilidad"},
    title="Predicción de Calidad del Vino",
    color=["red", "orange", "green"],
)
st.plotly_chart(fig)
