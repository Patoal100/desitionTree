import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Marca el tiempo de inicio
start_time = time.time()
# Leer el archivo CSV
data1 = pd.read_csv('cam_1_df.csv')
data2 = pd.read_csv('cam_2_df.csv')

# Eliminar la fila de prueba
data1 = data1.drop(0)
data2 = data2.drop(0)

# Seleccionar las filas que contienen al menos un valor NaN
# filas_con_nan = data2[data2.isna().any(axis=1)]

# Imprimir las filas con valores NaN
# print(filas_con_nan)

# Imprimir el contenido de la fila 8174 para revisar la columna 'p_0'
# print(data2.iloc[8173])  # Python usa indexación basada en 0, por lo que debes restar 1 a 8174

# Reemplazar los valores NaN en la columna 'p_0' con 1
data2['p_0'] = data2['p_0'].fillna(1)

# Convertir la columna 'p_0' a enteros
data2['p_0'] = data2['p_0'].astype(int)


# Separar la columna fecha en dos columnas fecha y hora 
data1[['fecha', 'hora']] = data1['fecha'].str.split('_', expand=True)
data2[['fecha', 'hora']] = data2['fecha'].str.split('_', expand=True)

# Convertir la columna de fecha en un objeto de fecha y hora
data1['fecha'] = pd.to_datetime(data1['fecha'])
data2['fecha'] = pd.to_datetime(data2['fecha'])

# Obtener el día de la semana donde lunes es 0 y el domingo es 6
data1['dia_semana'] = data1['fecha'].dt.dayofweek
data2['dia_semana'] = data2['fecha'].dt.dayofweek

# Eliminar la columna 'fecha'
data1 = data1.drop('fecha', axis=1)
data2 = data2.drop('fecha', axis=1)

# Ordenar el DataFrame por la columna 'libres' de menor a mayor
data1 = data1.sort_values('libres')
data2 = data2.sort_values('libres')

# Convertir la columna 'hora' a formato datetime
data1['hora'] = pd.to_datetime(data1['hora'], format='%H:%M:%S')
data2['hora'] = pd.to_datetime(data2['hora'], format='%H:%M:%S')

# Convertir la hora a minutos pasados desde la medianoche
data1['hora'] = data1['hora'].dt.hour * 60 + data1['hora'].dt.minute
data2['hora'] = data2['hora'].dt.hour * 60 + data2['hora'].dt.minute

data1.to_csv('data1.csv', index=False)
data2.to_csv('data2.csv', index=False)
# Imprimir el contenido del archivo CSV
# print(data1)
# print(data2)

# Crear una lista con los nombres de las columnas que quieres mantener en todas las tablas
columnas_comunes = ['hora', 'dia_semana']

# Crear una lista para almacenar las nuevas tablas
tablas = []

# Para cada columna de p_0 a p_6
for i in range(7):
    # Crear el nombre de la columna
    columna_p = f'p_{i}'
    
    # Crear una nueva tabla con las columnas comunes y la columna p_i
    tabla = data1[columnas_comunes + [columna_p]]
    
    # Agregar la nueva tabla a la lista de tablas
    tablas.append(tabla)
    #print(tabla)

# Crear una lista para almacenar los modelos entrenados
modelos = []

# Para cada tabla
for tabla in tablas:
    # Obtener el nombre de la última columna
    ultima_columna = tabla.columns[-1]
    print(f'Entrenando modelo para la columna {ultima_columna}')

    # Definir las variables de entrada (X) como las columnas 'hora' y 'dia_semana'
    X = tabla[columnas_comunes]
    y = tabla[ultima_columna]
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear y entrenar el modelo de árbol de decisión
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    
    # Agregar el modelo entrenado a la lista de modelos
    modelos.append(modelo)
    print(f'Precisión del modelo {ultima_columna}: {modelo.score(X_test, y_test)}')



# Supongamos que tienes un conjunto de datos de prueba X_test
predicciones = pd.DataFrame()

for i, modelo in enumerate(modelos):
    predicciones[f'p_{i}'] = modelo.predict(X_test)

# Ahora, 'predicciones' es un DataFrame que contiene las predicciones de cada modelo


# Calcular el número de '1' que cada modelo predijo
num_unos = predicciones.apply(lambda x: (x == 1).sum())

# Guardar num_unos en un archivo CSV
num_unos.to_csv('num_unos.csv', index=False)

# Crear un gráfico de barras con estos números
plt.bar(num_unos.index, num_unos.values)
plt.xlabel('Espacios de parqueo de la cámara 1')
plt.ylabel('Uso predecido por el modelo 1')
plt.title('Uso de espacios de parqueo de la cámara 1')
# plt.show()

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
    tabla = data2[columnas_comunes + [columna_p]]
    
    # Agregar la nueva tabla a la lista de tablas
    tablas2.append(tabla)
    #print(tabla)

# Crear una lista para almacenar los modelos entrenados
modelos2 = []

# Para cada tabla
for tabla in tablas2:
    # Obtener el nombre de la última columna
    ultima_columna = tabla.columns[-1]
    print(f'Entrenando modelo para la columna {ultima_columna}')

    # Definir las variables de entrada (X) como las columnas 'hora' y 'dia_semana'
    X = tabla[columnas_comunes]
    y = tabla[ultima_columna]
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear y entrenar el modelo de árbol de decisión
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    
    # Agregar el modelo entrenado a la lista de modelos
    modelos2.append(modelo)
    print(f'Precisión del modelo {ultima_columna}: {modelo.score(X_test, y_test)}')



# Supongamos que tienes un conjunto de datos de prueba X_test
predicciones2 = pd.DataFrame()

for i, modelo in enumerate(modelos2):
    predicciones2[f'p_{i}'] = modelo.predict(X_test)

# Ahora, 'predicciones' es un DataFrame que contiene las predicciones de cada modelo


# Calcular el número de '1' que cada modelo predijo
num_unos = predicciones2.apply(lambda x: (x == 1).sum())

# Guardar num_unos en un archivo CSV
num_unos.to_csv('num_unos2.csv', index=False)

# Marca el tiempo de finalización
end_time = time.time()

# Calcula e imprime el tiempo de ejecución en milisegundos
execution_time = (end_time - start_time) * 1000
print(f"El tiempo de ejecución del método es: {execution_time} milisegundos")

# Crear un gráfico de barras con estos números
plt.bar(num_unos.index, num_unos.values)
plt.xlabel('Espacios de parqueo de la cámara 2')
plt.ylabel('Uso predecido por el modelo 2')
plt.title('Uso de espacios de parqueo de la cámara 2')
# plt.show()
# Guardar la gráfica en un archivo
plt.savefig('grafica2.png')

plt.close()
