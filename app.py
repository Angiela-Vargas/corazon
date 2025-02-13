import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import requests
from PIL import Image

# Cargar el modelo y el escalador
scaler = joblib.load('escalador.bin')
model = joblib.load('modelo_knn.bin')

# Título y Descripción de la aplicación
st.title("Asistente IA para cardiólogos")
st.write("""
    Esta aplicación está diseñada para predecir si una persona tiene o no problemas cardíacos 
    utilizando un modelo de IA entrenado con datos de edad y colesterol. 
    Puedes ingresar los valores para edad y colesterol, y el sistema te dirá si la persona 
    tiene problemas cardíacos o no.
    
    **Realizado por Alfredo Díaz**
""")

# Crear un tab de instrucciones
with st.expander("Instrucciones"):
    st.write("""
        1. Ingresa la edad de la persona en el rango de 18 a 80 años.
        2. Ingresa el valor de colesterol entre 50 y 600.
        3. Dirígete a la pestaña 'Predicción' para ver el diagnóstico.
    """)

# Crear un selector para elegir el tab entre "Captura de datos" y "Predicción"
tab = st.radio("Selecciona una opción", ("Captura de datos", "Predicción"))

# **Tab de Captura de datos**
if tab == "Captura de datos":
    st.header("Ingrese los datos de la persona")
    edad = st.slider("Edad", 18, 80, 30)
    colesterol = st.slider("Colesterol", 50, 600, 200)
    st.session_state.edad = edad  # Guardamos los valores de sesión
    st.session_state.colesterol = colesterol

# **Tab de Predicción**
elif tab == "Predicción":
    if hasattr(st.session_state, 'edad') and hasattr(st.session_state, 'colesterol'):
        # Recuperamos los valores de los datos ingresados en la sesión
        edad = st.session_state.edad
        colesterol = st.session_state.colesterol
        
        # Crear un DataFrame con los datos de entrada y asegurarnos de que tengan los mismos nombres de columna
        input_data = pd.DataFrame({
            'edad': [edad],
            'colesterol': [colesterol]
        })
        
        # Normalizamos los datos de entrada con el escalador cargado
        input_data_normalized = scaler.transform(input_data)
        
        # Realizamos la predicción con el modelo KNN
        prediccion = model.predict(input_data_normalized)
        
        # Mostrar la predicción
        if prediccion == 1:
            st.write("**La persona tiene problemas cardíacos.**")
            # Mostrar la imagen correspondiente
            image_url = "https://www.clikisalud.net/wp-content/uploads/2018/09/problemas-cardiacos-jovenes.jpg"
        else:
            st.write("**La persona no tiene problemas cardíacos.**")
            # Mostrar la imagen correspondiente
            image_url = "https://s28461.pcdn.co/wp-content/uploads/2017/07/Tu-corazo%CC%81n-consejos-para-mantenerlo-sano-y-fuerte.jpg"
        
        # Mostrar la imagen
        image = Image.open(requests.get(image_url, stream=True).raw)
        st.image(image, caption="Resultado de la predicción", use_column_width=True)
    else:
        st.warning("Por favor, ingresa los datos primero en la pestaña 'Captura de datos'.")
