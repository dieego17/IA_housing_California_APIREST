# Predicción de Precios de Viviendas en California

Este proyecto utiliza un modelo de aprendizaje profundo entrenado con TensorFlow para predecir el valor de las viviendas en California en función de características como la ubicación, el tamaño de la vivienda, la edad de la vivienda, y más.

## Tecnologías Utilizadas

- **TensorFlow**: Para la creación y entrenamiento del modelo de predicción.
- **Pandas**: Para la manipulación de datos.
- **Scikit-learn**: Para el preprocesamiento de datos.
- **FastAPI**: Para crear una API que permita realizar predicciones en tiempo real.
- **Matplotlib**: Para visualizar el proceso de entrenamiento del modelo.

## Descripción del Proyecto

Este proyecto incluye los siguientes componentes:

1. **Entrenamiento del Modelo**: 
   - El modelo se entrena utilizando un conjunto de datos de viviendas en California.
   - El preprocesamiento de datos incluye normalización y codificación de variables categóricas.
   
2. **Modelo de Predicción**: 
   - El modelo de red neuronal entrenado se guarda en formato `.keras`.
   
3. **API con FastAPI**: 
   - Se proporciona una API que permite enviar datos de características de una vivienda y recibir una predicción del valor de la vivienda.
   - La API recibe datos en formato JSON y devuelve la predicción del precio de la vivienda.

## Estructura de Archivos

- `housing.csv`: Conjunto de datos de las viviendas en California.
- `col_transformer.pkl`: Transformador utilizado para normalizar los datos de entrada.
- `akinator.keras`: Modelo entrenado de predicción de precios de viviendas.
- `app.py`: Código para la API en FastAPI.
- `README.md`: Este archivo.

## Requisitos

- Python 3.7 o superior
- TensorFlow 2.x
- FastAPI
- Pandas
- Scikit-learn
- Uvicorn (para ejecutar el servidor FastAPI)

## Instalación

1. Clona este repositorio:

  ```bash
  git clone https://github.com/tu-usuario/tu-repositorio.git
  cd tu-repositorio
  ```

2. Crea un entorno virtual e instálalo:
  ```bash
  python -m venv env
  source env/bin/activate  # En Windows usa 'env\Scripts\activate'
  pip install -r requirements.txt
  ```

## Uso

### Entrenamiento del Modelo

El modelo de predicción puede ser entrenado utilizando el archivo `train_model.py`. Solo necesitas ejecutar el siguiente comando:

  ```bash
  python train_model.py
```

Este script cargará el conjunto de datos, preprocesará los datos, entrenará el modelo y guardará el modelo entrenado en el archivo akinator.keras.

### API para Predicción
Para ejecutar la API en FastAPI, usa el siguiente comando:
```bash
uvicorn app:app --reload
```

Esto iniciará el servidor de desarrollo en http://127.0.0.1:8000.

Ruta de Prueba
Para comprobar si la API está funcionando correctamente, accede a:
```bash
GET http://127.0.0.1:8000/
```

Respuesta esperada:
```bash
{
  "message": "API para predicción de precios de casas en California"
}
```

Ruta para Realizar Predicciones
Envía una solicitud POST a:
```bash
POST http://127.0.0.1:8000/predict/
```

Con el siguiente cuerpo JSON:
```bash
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41.0,
  "total_rooms": 880.0,
  "total_bedrooms": 129.0,
  "population": 322.0,
  "households": 126.0,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"
}
```

La respuesta será:
```bash
{
  "predicted_price": 350000.0
}
```

### Contribuciones
Si deseas contribuir a este proyecto, por favor realiza un fork y envía un pull request con tus mejoras.
