
README – Aplicativo Predictor de Diabetes con Entorno Virtual
=============================================================

Este paquete contiene:

- modelo_diabetes.py: script para entrenar el modelo y guardar modelo_diabetes.pkl
- app.py: app web en Streamlit para predecir el riesgo de diabetes
- requirements.txt: lista de librerías necesarias

PASOS PARA USAR EL APLICATIVO
-----------------------------

1. Abre PowerShell o CMD y navega a la carpeta del proyecto:

    cd "C:\Users\ssoy_\OneDrive\Desktop\TESIS"

2. Crea un entorno virtual (solo una vez):

    python -m venv venv

3. Activa el entorno virtual:

    .\venv\Scripts\activate

4. Instala las librerías necesarias:

    pip install -r requirements.txt

5. Asegúrate de tener en la carpeta el archivo CSV:

    diabetes_012_health_indicators_BRFSS2015_VR V2.csv

6. Ejecuta el script para entrenar el modelo:

    python modelo_diabetes.py

   Esto generará el archivo modelo_diabetes.pkl

7. Ejecuta la app web:

    streamlit run app.py

8. Cuando termines, puedes salir del entorno virtual con:

    deactivate
