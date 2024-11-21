import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import train_test_split

df = pd.read_csv('housing.csv')
print(df.head())

print(df.shape,df.info())

df['total_bedrooms'].fillna(df['total_bedrooms'].mean(),inplace=True)

print(df.info())

print(df['ocean_proximity'].value_counts(),df['ocean_proximity'])

X = df.drop(['median_house_value'],axis=1)
y = df['median_house_value']

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Crear un column transformer que nos ayude a normalizar/preprocesar nuestros datos
ct = make_column_transformer(
    (MinMaxScaler(), ['longitude', 'latitude', 'housing_median_age',
                      'total_rooms', 'total_bedrooms', 'population',
                      'households', 'median_income']),
    (OneHotEncoder(), ['ocean_proximity'])
)

ct.fit(X)
X_normal = ct.transform(X)

# Guardar el transformador de columnas para normalizar los datos que le lleguen del API
import pickle
with open('col_transformer.pkl', 'wb') as f:
    pickle.dump(ct, f)

# Separar datos de entrenamiento y de test
X_train_normal,X_test_normal,y_train,y_test = train_test_split(X_normal,y,test_size=0.2,random_state=1)

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8192, activation='relu'),

    tf.keras.layers.Dense(1)
])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=200)

model.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='mae',
              metrics=['mae'])

# model.fit(X_train_normal,y_train,epochs=100, callbacks=[callback])
# plt.plot(model.history.history['loss'])
# model.evaluate(X_test_normal, y_test)

model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1)
])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=200)

model2.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mae',
              metrics=['mae'])

# model2.fit(X_train_normal,y_train,epochs=100, callbacks=[callback])

# plt.plot(model2.history.history['loss'])

# model.evaluate(X_test_normal,y_test)

"""MODELO 3"""

model3 = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=200)

model3.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mae',
              metrics=['mae'])

model3.fit(X_train_normal,y_train,epochs=100, callbacks=[callback])

plt.plot(model3.history.history['loss'])

model3.evaluate(X_test_normal,y_test)


print(X_test_normal[0].shape)

tf.expand_dims(X_test_normal[0],axis=0)

print("Target:",y_test.iloc[300],"Prediction:", model3.predict(tf.expand_dims(X_test_normal[300],axis=0)))

# print("Datos normalizados\n",X_normal[300], "\nDatos sin normalizar\n", X.iloc[300])

model3.save('akinator.keras')