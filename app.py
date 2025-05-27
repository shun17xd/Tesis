
import streamlit as st
import joblib
import numpy as np
import os

st.title("Predicción de Riesgo de Diabetes")

modelo_path = os.path.join(os.path.dirname(__file__), "modelo_diabetes.pkl")

if not os.path.exists(modelo_path):
    st.error("⚠️ No se encontró el archivo modelo_diabetes.pkl. Por favor, asegúrate de haber ejecutado modelo_diabetes.py.")
    st.stop()

model = joblib.load(modelo_path)

BMI = st.slider("BMI", 15, 50, 25)
Age = st.slider("Edad", 18, 80, 35)
HighBP = st.selectbox("¿Tiene presión alta?", [0, 1])
HighChol = st.selectbox("¿Tiene colesterol alto?", [0, 1])
GenHlth = st.slider("Salud general (1=Excelente, 5=Mala)", 1, 5, 3)
PhysHlth = st.slider("Días de malestar físico (últimos 30 días)", 0, 30, 5)
MentHlth = st.slider("Días de malestar mental (últimos 30 días)", 0, 30, 5)
DiffWalk = st.selectbox("¿Dificultad para caminar?", [0, 1])
Income = st.slider("Nivel de ingresos (1=bajo, 8=alto)", 1, 8, 4)
Education = st.slider("Nivel educativo (1=bajo, 6=alto)", 1, 6, 3)

if st.button("Predecir"):
    input_data = np.array([[BMI, Age, HighBP, HighChol, GenHlth,
                            PhysHlth, MentHlth, DiffWalk, Income, Education]])
    proba = model.predict_proba(input_data)[0][1]
    st.success(f"✅ Probabilidad estimada de tener diabetes: {proba * 100:.1f}%")
