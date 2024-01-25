import pandas as pd

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

# Guradar los datos procesados en un archivo CSV
data1.to_csv('data1.csv', index=False)
data2.to_csv('data2.csv', index=False)