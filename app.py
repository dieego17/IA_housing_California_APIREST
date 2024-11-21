from fastapi import FastAPI
from pandas.core.interchange.dataframe_protocol import DataFrame
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Cargar modelo desde el archivo .h5
model = tf.keras.models.load_model("akinator.keras")

# Cargar transformador
with open('col_transformer.pkl', 'rb') as f:
    ct = pickle.load(f)

# Inicializar FastAPI
app = FastAPI()

# Definir la estructura de entrada con Pydantic
class HouseFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: object

# Ruta para verificar que el servicio está activo
@app.get("/")
def home():
    return {"message": "API para predicción de precios de casas en California"}

# Ruta para realizar predicciones
@app.post("/predict/")
def predict(features: HouseFeatures):
    # Convertir los datos de entrada a formato numpy array
    input_data = np.array([[
        features.longitude,
        features.latitude,
        features.housing_median_age,
        features.total_rooms,
        features.total_bedrooms,
        features.population,
        features.households,
        features.median_income,
        features.ocean_proximity
    ]])

    # Adaptar el array a un DataFrame
    input_df = pd.DataFrame(input_data, columns=[
        'longitude', 'latitude', 'housing_median_age',
        'total_rooms', 'total_bedrooms', 'population',
        'households', 'median_income', 'ocean_proximity'
    ])

    # Normalizar el DataFrame
    normalized_data = ct.transform(input_df)

    # Realizar la predicción con los datos normalizados
    prediction = model.predict(normalized_data)

    # Devolver la predicción como JSON
    return {"predicted_price": float(prediction[0][0])}
