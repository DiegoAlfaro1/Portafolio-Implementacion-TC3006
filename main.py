import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Lectura de dataset
dataset = "Coffe_sales.csv"
df = pd.read_csv(dataset)


# Transformacion de datos

'''
1. Eliminar la columna de Monthsort ya que la columna Month_name tiene los nombres de los meses
2. Cambiar el nombre de la columna money a Price para que sea mas descriptiva
3. Eliminar la columna hour_of_day ya que tenemos la columna Time que es mas exacta
4. Eliminar la columna weekday_sort ya que solo nos da el numero del dia de la semana, ya tenemos una columna con el dia de la semana
5. Cambiar el nombre de la columna cash_type a payment_type para que sea mas descriptiva
6. Cambiar los nombres de las columnas para que tengan la primera letra mayusculas todas
7. Agregar columna con booleano para saber si la venta de ese momento fue premium o no, fue premium si es mas de 33
8. Partir el dataframe en 2 para tener train y test, sin usar scklear
'''

# Eliminacion de columnas

df = df.drop(columns=['Monthsort','hour_of_day','Weekdaysort'])

# Cambiar nombres de columnas

df.rename(columns={'cash_type':'Payment_type', 'money':'Price', 'coffee_name':'Coffe_name', 'Time_of_Day': 'Time_of_day'}, inplace=True)

# Agregar columna de venta premium
df['Premium_sale'] = (df["Price"] > 33).astype(bool)

# Partir el dataset para tener train y test

df_train = df.sample(frac=0.7, random_state=42)
df_test = df.drop(df_train.index)

print(df_train.head(10))
print(df_test.head(10))