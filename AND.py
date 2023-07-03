#Importamos las librerias necesarias
import tensorflow as tf
import numpy as np

# Datos de entrada y salida
entrada = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
salida_esperada = np.array([[0], [0], [0], [1]])

# Definición de la arquitectura de la red neuronal
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4, activation='sigmoid', input_shape=(2,)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compilación del modelo
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
modelo.fit(entrada, salida_esperada, epochs=5000, verbose=0)

# Evaluación del modelo
puntuacion = modelo.evaluate(entrada, salida_esperada, verbose=0)

# Predicción del modelo
prediccion = modelo.predict(entrada)

# Predicción del modelo sobre nuevas entradas
nueva_entrada = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
nueva_prediccion = modelo.predict(nueva_entrada)

# Imprimir los datos de entrada y la predicción correspondiente
for i in range(len(entrada)):
    print(f"Entrada: {entrada[i]} - Predicción: {int(round(prediccion[i][0]))}")