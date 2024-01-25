from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import lit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import pandas as pd
import time

# Iniciar una sesión de Spark
spark = SparkSession.builder.appName("Python Spark SQL basic example").getOrCreate()

# Marca el tiempo de inicio
start_time = time.time() * 1000

# Leer el archivo CSV
data1 = spark.read.csv('data1.csv', header=True, inferSchema=True)
data2 = spark.read.csv('data2.csv', header=True, inferSchema=True)

# Crear una lista con los nombres de las columnas que quieres mantener en todas las tablas
columnas_comunes = ['hora', 'dia_semana']

# Crear una lista para almacenar las nuevas tablas
tablas = []

# Para cada columna de p_0 a p_6
for i in range(7):
    # Crear el nombre de la columna
    columna_p = f'p_{i}'
    
    # Crear una nueva tabla con las columnas comunes y la columna p_i
    tabla = data1.select(columnas_comunes + [columna_p])
    
    # Agregar la nueva tabla a la lista de tablas
    tablas.append(tabla)

# Crear una lista para almacenar los modelos entrenados
modelos = []


# Para cada tabla
for i, tabla in enumerate(tablas):
    # Obtener el nombre de la última columna
    ultima_columna = tabla.columns[-1]
    print(f'Entrenando modelo para la columna {ultima_columna}')

    # Definir las variables de entrada (X) como las columnas 'hora' y 'dia_semana'
    assembler = VectorAssembler(inputCols=columnas_comunes, outputCol="features")
    output = assembler.transform(tabla)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    train_data, test_data = output.randomSplit([0.8, 0.2])

    # Crear y entrenar el modelo de árbol de decisión
    dtc = DecisionTreeClassifier(labelCol=ultima_columna, featuresCol="features")
    dtc_model = dtc.fit(train_data)

    # Agregar el modelo entrenado a la lista de modelos
    modelos.append(dtc_model)

    # Evaluar el modelo
    predictions = dtc_model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol=ultima_columna, predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f'Precisión del modelo {ultima_columna}: {accuracy}')

# Supongamos que tienes un conjunto de datos de prueba en un archivo CSV
X_test = spark.read.csv('X_test.csv', header=True, inferSchema=True)

# Transformar los datos de prueba
assembler = VectorAssembler(inputCols=columnas_comunes, outputCol="features")
X_test = assembler.transform(X_test)

# Supongamos que tienes un conjunto de datos de prueba X_test
predicciones = []

for i, modelo in enumerate(modelos):
    prediction = modelo.transform(X_test)
    num_unos = prediction.filter(prediction['prediction'] == lit(1)).count()
    predicciones.append(num_unos)

# Guardar num_unos en un archivo CSV
pd.DataFrame(predicciones, columns=['num_unos']).to_csv('num_unos.csv', index=False)

# Crear un gráfico de barras con estos números
plt.bar(range(len(predicciones)),predicciones)
plt.xlabel('Espacios de parqueo de la cámara 1')
plt.ylabel('Uso predecido por el modelo 1')
plt.title('Uso de espacios de parqueo de la cámara 1')

# Guardar la gráfica en un archivo
plt.savefig('grafica.png')

plt.close()

# Crear una lista para almacenar las nuevas tablas
tablas2 = []

# Para cada columna de p_0 a p_9
for i in range(10):
    # Crear el nombre de la columna
    columna_p = f'p_{i}'
    
    # Crear una nueva tabla con las columnas comunes y la columna p_i
    tabla = data2.select(columnas_comunes + [columna_p])
    
    # Agregar la nueva tabla a la lista de tablas
    tablas2.append(tabla)

# Crear una lista para almacenar los modelos entrenados
modelos2 = []


# Para cada tabla
for i, tabla in enumerate(tablas2):
    # Obtener el nombre de la última columna
    ultima_columna = tabla.columns[-1]
    print(f'Entrenando modelo para la columna {ultima_columna}')

    # Definir las variables de entrada (X) como las columnas 'hora' y 'dia_semana'
    assembler = VectorAssembler(inputCols=columnas_comunes, outputCol="features")
    output = assembler.transform(tabla)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    train_data, test_data = output.randomSplit([0.8, 0.2])

    # Crear y entrenar el modelo de árbol de decisión
    dtc = DecisionTreeClassifier(labelCol=ultima_columna, featuresCol="features")
    dtc_model = dtc.fit(train_data)

    # Agregar el modelo entrenado a la lista de modelos
    modelos2.append(dtc_model)

    # Evaluar el modelo
    predictions = dtc_model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol=ultima_columna, predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f'Precisión del modelo {ultima_columna}: {accuracy}')

# Supongamos que tienes un conjunto de datos de prueba en un archivo CSV
X_test = spark.read.csv('X_test.csv', header=True, inferSchema=True)

# Transformar los datos de prueba
assembler = VectorAssembler(inputCols=columnas_comunes, outputCol="features")
X_test = assembler.transform(X_test)

# Supongamos que tienes un conjunto de datos de prueba X_test
predicciones2 = []

for i, modelo in enumerate(modelos2):
    prediction = modelo.transform(X_test)
    num_unos = prediction.filter(prediction['prediction'] == lit(1)).count()
    predicciones2.append(num_unos)

# Guardar num_unos en un archivo CSV
pd.DataFrame(predicciones2, columns=['num_unos']).to_csv('num_unos.csv', index=False)

# Crear un gráfico de barras con estos números
plt.bar(range(len(predicciones2)),predicciones2)
plt.xlabel('Espacios de parqueo de la cámara 2')
plt.ylabel('Uso predecido por el modelo 2')
plt.title('Uso de espacios de parqueo de la cámara 2')

# Guardar la gráfica en un archivo
plt.savefig('grafica2.png')

plt.close()

# Calcular el tiempo de ejecución en milisegundos
execution_time = time.time() * 1000 - start_time
print(f"El tiempo de ejecución fue de {execution_time} milisegundos.")