# -*- coding: utf-8 -*-
"""
**Reconocimiento de dígitos escritos a mano mediante una red neuronal**

En este notebook, implementamos una red neuronal simple de dos capas y la entrenamos utilizando el conjunto de datos MNIST, ampliamente utilizado en el reconocimiento de dígitos escritos a mano. Este conjunto de datos proporciona una colección de imágenes de dígitos, junto con las etiquetas correspondientes que indican a qué número se refiere cada imagen.

Los desafíos asociados a este conjunto de datos incluyen la variabilidad en la escritura a mano y la calidad de las imágenes, ya que las personas pueden representar los dígitos de diversas formas y tamaños. Por esta razón, nuestro objetivo principal es entrenar y evaluar un algoritmo de reconocimiento de dígitos escritos a mano en el contexto de la clasificación de imágenes, con el fin de lograr la mejor precisión posible en la identificación de los dígitos.

Nuestra red neuronal tendrá una arquitectura sencilla de dos capas:

Capa de entrada A[0]: Compuesta de 784 unidades correspondientes a los 784 píxeles de cada imagen de entrada de 28x28.
Capa oculta A[1]: Compuesta por 10 unidades con activación ReLU.
Capa de salida A[2]: Compuesta por 10 unidades correspondientes a las clases de diez dígitos con activación softmax.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

data = pd.read_csv('train.csv')

# ANALISIS EDA

data.shape

data.dtypes

# Visualizar las primeras filas del conjunto de datos
data.head()

# Visualizar las últimas filas del conjunto de datos
data.tail()

data.describe()

data.info()

data.isnull().sum()

# Contar y visualizar la distribución de etiquetas (dígitos)
sns.countplot(data['label'])
plt.title('Distribución de Etiquetas')
plt.show()

# Visualizar algunas imágenes de dígitos escritos a mano
# Podes ajustar el número de imágenes a mostrar cambiando el valor de "imagenes_para_mostrar"
imagenes_para_mostrar = 10

plt.figure(figsize=(15, 7))
for i in range(imagenes_para_mostrar):
    plt.subplot(1, imagenes_para_mostrar, i + 1)
    image_data = data.iloc[i, 1:].values
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f'Etiqueta: {data.iloc[i, 0]}')
    plt.axis('off')

plt.show()

"""## Conclusión EDA

> La columna "label" contiene las etiquetas que identifican el número correspondiente a cada imagen, mientras que las 784 columnas restantes, denominadas "pixel0" a "pixel783", almacenan los valores de píxeles que representan imágenes en escala de grises de 28x28 píxeles. El conjunto de datos consiste en 42,000 imágenes, cada una representando un solo dígito, variando del 0 al 9 y escritas a mano. Estas etiquetas proporcionan información sobre el número que se ha escrito en cada imagen.

>  Todas las columnas en el conjunto de datos son de tipo int64, lo que significa que contienen valores enteros. Es importante destacar que no se encontraron valores nulos en el conjunto de datos, lo que indica que está limpio y bien estructurado, sin datos faltantes.

> La columna "label" se utilizará como la etiqueta de clase para el entrenamiento de modelos de reconocimiento de dígitos escritos a mano, mientras que las columnas "pixel0" a "pixel783" contienen los valores de píxeles que representan las imágenes de los dígitos. La distribución de etiquetas es equilibrada, con un número similar de instancias para cada uno de los dígitos del 0 al 9, lo que es beneficioso para el desarrollo de modelos de clasificación sin sesgos.

# DIVISIÓN DE DATOS
"""

data = np.array(data) # Convertir la variable data en un array NumPy
m, n = data.shape # Obtener las dimensiones del conjunto de datos (m=42000, n=785)
np.random.shuffle(data) # Shuffle (mezclar) antes de dividir en conjuntos de desarrollo y entrenamiento

# Dividimos las etiquetas y características de los conjuntos de desarrollo y entrenamiento:
data_dev = data[0:1000].T # Ajustamos la orientación de los datos
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255. # Normalizar los pixeles (rango [0, 1])

data_train = data[1000:m].T # Ajustamos la orientación de los datos
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255. # Normalizar los pixeles (rango [0, 1])
_,m_train = X_train.shape # Obtiene el número de ejemplos de entrenamiento

Y_train

"""Lo primero y fundamental a comprobar es la frecuencia de las clases en el conjunto de datos, ya que un conjunto de datos equilibrado siempre es una buena opción para comenzar. Sin embargo, esto no es aplicable en todos los casos, ya que existen varias tareas de aprendizaje supervisado en las que las clases no están equilibradas. Además, en el caso de detección de anomalías, a menudo se encuentra una gran disparidad entre la clase positiva y la negativa."""

# Función para generar gráficos de frecuencia por dígito
def plot_digit_frequency(data, title):
    labels, counts = np.unique(data, return_counts=True)
    colors = plt.get_cmap('Set3')(np.linspace(0, 1, len(labels)))
    plt.bar(labels, counts, align='center', color=colors)
    plt.xlabel('Dígito')
    plt.ylabel('Frecuencia')
    plt.title(title)
    plt.show()

# Gráfico de frecuencia para el conjunto de desarrollo
plot_digit_frequency(Y_dev, 'Frecuencia de Dígitos en Conjunto de Desarrollo')

# Gráfico de frecuencia para el conjunto de entrenamiento
plot_digit_frequency(Y_train, 'Frecuencia de Dígitos en Conjunto de Entrenamiento')

"""Observamos que tanto el conjunto de desarrollo como el de entrenamiento tienen un buen equilibrio entre las distintas etiquetas, así que podemos seguir adelante.

# RED NEURONAL

## Funciones del ciclo de entrenamiento:
"""

#  Inicializa los parámetros de la red neuronal, incluyendo las matrices de pesos (W1 y W2) y los vectores de sesgo (b1 y b2), utilizando valores aleatorios entre -0.5 y 0.5.
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2
# Función de activación Rectified Linear Unit (ReLU) en la matriz Z, que aplica la operación de umbral en cero. Devuelve una matriz con los mismos valores que Z donde los valores negativos se vuelven cero y los valores no negativos se mantienen iguales.
def ReLU(Z):
    return np.maximum(Z, 0)

#  Función de activación softmax en la matriz Z, que se utiliza para calcular las probabilidades de pertenencia de las clases.
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# Realiza la propagación hacia adelante en la red neuronal.
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Calcula la derivada de la función de activación ReLU (valores positivos se establecen en 1 y los valores negativos en 0).
def ReLU_deriv(Z):
    return Z > 0

# Convierte un vector de etiquetas Y en una representación one-hot, donde cada etiqueta se convierte en un vector binario con un 1 en la posición correspondiente a la etiqueta y 0 en las demás posiciones.
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Realiza la retropropagación en la red neuronal para calcular los gradientes de los parámetros.
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

#Actualiza los parámetros de la red neuronal utilizando los gradientes calculados (dW1, db1, dW2, db2) y una tasa de aprendizaje (alpha).
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

"""## Funciones del proceso de entrenamiento:"""

# Toma las salidas y devuelve el índice del elemento con el valor máximo a lo largo del eje 0 (predice la clase para cada ejemplo al encontrar la neurona de salida con la probabilidad más alta).
def get_predictions(A2):
    return np.argmax(A2, 0)

# Calcula la precisión de las predicciones comparando las predicciones (obtenidas anteriormente) con las etiquetas reales.
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size # Devuelve la fracción de predicciones correctas en comparación con el total de ejemplos en el conjunto de datos.

# Lleva a cabo el proceso de entrenamiento de la red neuronal utilizando el algoritmo de gradiente descendiente.
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params() #  Inicializamos los parámetros de la red.
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X) # Calculamos las activaciones.
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y) # Retropropagación para calcular los gradientes.
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha) #  Los parámetros de la red se actualizan utilizando el gradiente descendiente.
        if i % 10 == 0:
            print("Iteracion: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y)) # Se calcula y muestra la precisión de las predicciones cada diez iteraciones.
    return W1, b1, W2, b2

"""El ciclo de entrenamiento se repite varias veces (según el número de iteraciones) para ajustar gradualmente los parámetros de la red y mejorar su rendimiento.

## Ejecutamos el proceso de entrenamiento:
"""

# Utilizando una tasa de aprendizaje de 0.10 y realizando 500 iteraciones para actualizar los parámetros.
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

"""Obtuvimos *85.61*% de precisión en el conjunto de entrenamiento.

## Funciones para realizar predicciones y verificar cómo se comporta:
"""

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

"""## Evaluación del rendimiento del modelo:"""

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)

"""Tiene una precisión del 86.60%, por lo que nuestro modelo generalizó bastante bien a partir de los datos de entrenamiento."""

# Calculamos la matriz de confusión
confusion_lr = confusion_matrix(Y_dev, dev_predictions)

print(confusion_lr)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_lr, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión")
plt.xlabel("Valores Predichos")
plt.ylabel("Valores Reales")
plt.show()

"""# CONCLUSIÓN:

El modelo logró un buen rendimiento en la tarea de reconocimiento de dígitos escritos a mano, pero aún existe margen para mejorar su precisión y robustez. A pesar de contar solo con tres capas, los resultados son bastante buenos y confiables. Por supuesto, podríamos considerar las siguientes áreas para mejorar aún más el rendimiento del modelo:

Posibles áreas de mejora:

1- Aumentar la complejidad de la red neuronal, por ejemplo, agregando más capas ocultas o incrementando el número de unidades en la capa oculta, lo que permitiría a la red capturar patrones más complejos.

2- Ajustar los hiperparámetros, como la tasa de aprendizaje, el número de iteraciones de entrenamiento y la inicialización de los parámetros, para encontrar la configuración óptima.

3- Explorar técnicas de regularización con el fin de prevenir el sobreajuste y mejorar la generalización del modelo.

4- Realizar una evaluación más detallada del modelo utilizando la matriz de confusión para inspeccionar los errores comunes y obtener una comprensión más profunda del desempeño y las áreas que requieren atención.

El modelo demostró ser efectivo, pero la optimización de hiperparámetros y la exploración de técnicas adicionales pueden ayudar a perfeccionarlo y hacerlo aún más confiable.
"""