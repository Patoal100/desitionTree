import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import split, to_date, date_format, hour, minute

# Crear una sesión de Spark
spark = SparkSession.builder.appName('DecisionTree').getOrCreate()

# Marca el tiempo de inicio
start_time = time.time()
# Lee los archivos CSV
data1 = spark.read.csv('cam_1_df.csv', header=True, inferSchema=True)
data2 = spark.read.csv('cam_2_df.csv', header=True, inferSchema=True)

# Elimina la primera fila
data1 = data1.filter(col('index') != 0)
data2 = data2.filter(col('index') != 0)

# Reemplaza los valores NaN en la columna 'p_0' con 1
data2 = data2.na.fill({'p_0': 1})

# Convierte la columna 'p_0' a enteros
data2 = data2.withColumn('p_0', col('p_0').cast('int'))

# Estas son las formas correctas de hacerlo en PySpark
data1 = data1.withColumn('fecha', split(data1['fecha'], '_').getItem(0))
data1 = data1.withColumn('hora', split(data1['fecha'], '_').getItem(1))

data2 = data2.withColumn('fecha', split(data2['fecha'], '_').getItem(0))
data2 = data2.withColumn('hora', split(data2['fecha'], '_').getItem(1))


# Separar la columna fecha en dos columnas fecha y hora 
data1 = data1.withColumn('fecha', split(data1['fecha'], '_').getItem(0))
data1 = data1.withColumn('hora', split(data1['fecha'], '_').getItem(1))

data2 = data2.withColumn('fecha', split(data2['fecha'], '_').getItem(0))
data2 = data2.withColumn('hora', split(data2['fecha'], '_').getItem(1))

# Convertir la columna de fecha en un objeto de fecha y hora
data1 = data1.withColumn('fecha', to_date(data1['fecha'], 'yyyy-MM-dd'))
data2 = data2.withColumn('fecha', to_date(data2['fecha'], 'yyyy-MM-dd'))

# Obtener el día de la semana donde lunes es 0 y el domingo es 6
data1 = data1.withColumn('dia_semana', date_format(data1['fecha'], 'u').cast('int') - 1)
data2 = data2.withColumn('dia_semana', date_format(data2['fecha'], 'u').cast('int') - 1)

# Eliminar la columna 'fecha'
data1 = data1.drop('fecha')
data2 = data2.drop('fecha')

# Ordenar el DataFrame por la columna 'libres' de menor a mayor
data1 = data1.orderBy('libres')
data2 = data2.orderBy('libres')

# Convertir la hora a minutos pasados desde la medianoche
data1 = data1.withColumn('hora', hour(data1['hora']) * 60 + minute(data1['hora']))
data2 = data2.withColumn('hora', hour(data2['hora']) * 60 + minute(data2['hora']))

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
for tabla in tablas:
    # Obtener el nombre de la última columna
    ultima_columna = tabla.columns[-1]
    print(f'Entrenando modelo para la columna {ultima_columna}')

    # Definir las variables de entrada (X) como las columnas 'hora' y 'dia_semana'
    assembler = VectorAssembler(inputCols=columnas_comunes, outputCol="features")
    tabla = assembler.transform(tabla)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    train, test = tabla.randomSplit([0.8, 0.2], seed=42)
    
    # Crear y entrenar el modelo de árbol de decisión
    modelo = DecisionTreeClassifier(labelCol=ultima_columna, featuresCol="features")
    modelo = modelo.fit(train)
    
    # Agregar el modelo entrenado a la lista de modelos
    modelos.append(modelo)

    # Evaluar el modelo
    predictions = modelo.transform(test)
    evaluator = MulticlassClassificationEvaluator(labelCol=ultima_columna, predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f'Precisión del modelo {ultima_columna}: {accuracy}')

# Guardar el DataFrame 'predictions' como un archivo CSV
predictions.write.csv('predictions.csv')

# Convertir el DataFrame 'predictions' a un DataFrame de pandas
predictions_pandas = predictions.toPandas()

# Crear un gráfico de barras con estos números
predictions_pandas.plot(kind='bar')
# Contar el número de ocurrencias de cada valor en la columna 'prediction'
num_predictions = predictions_pandas['prediction'].value_counts()

# Crear un gráfico de barras con estos números
plt.bar(num_predictions.index, num_predictions.values)
plt.xlabel('Espacios de parqueo de la cámara 2')
plt.ylabel('Uso predecido por el modelo 2')
plt.title('Uso de espacios de parqueo de la cámara 2')
# Guardar la gráfica en un archivo
plt.savefig('grafica.png')

plt.close()
