import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Lectura de dataset
dataset = "Coffe_sales.csv"
df = pd.read_csv(dataset)


# Transformacion de datos

'''
1. Eliminar la columna de Month_name ya que la columna Monthsort tiene los nombres de los meses
2. Cambiar el nombre de la columna money a Price para que sea mas descriptiva
3. Eliminar la columna Time ya que tenemos la columna hour_of_day que nos da la hora necesaria
4. Eliminar la columna Weekday ya que tenemos weekdaysort que nos da el numero de dia
5. Cambiar el nombre de la columna cash_type a payment_type para que sea mas descriptiva
6. Cambiar los nombres de las columnas para que tengan la primera letra mayusculas todas
7. Agregar columna con booleano para saber si la venta de ese momento fue premium o no, fue premium si es mas de 33
8. Partir el dataframe en 2 para tener train y test, sin usar scklear
'''


# Eliminacion de columnas

df = df.drop(columns=['Month_name','Time','Weekday'])

# Cambiar nombres de columnas

df.rename(columns={'cash_type':'Payment_type', 'money':'Price', 'coffee_name':'Coffe_name', 'Time_of_Day': 'Time_of_day', 'hour_of_day': 'Hour_of_day'}, inplace=True)

# Agregar columna de venta premium
df['Premium_sale'] = (df["Price"] > 33).astype(int)

# Partir el dataset para tener train y test

df_train = df.sample(frac=0.7, random_state=42)
df_test = df.drop(df_train.index)

# Matriz de features

x_train = df_train[['Hour_of_day', "Weekdaysort", "Monthsort"]].values.astype(float)
x_test = df_test[['Hour_of_day', "Weekdaysort", "Monthsort"]].values.astype(float)

# Target vector

y_train = df_train['Premium_sale'].values
y_test = df_test['Premium_sale'].values


# Normalizacion de X usando min-max

X_min = x_train.min(axis=0)
X_max = x_train.max(axis=0)

x_train_scaled = (x_train - X_min) / (X_max - X_min)
x_test_scaled = (x_test - X_min) / (X_max - X_min)

# Agregar columna de bias, columna de 1
x_train_with_bias = np.column_stack([np.ones(x_train_scaled.shape[0]), x_train_scaled])
x_test_with_bias = np.column_stack([np.ones(x_test_scaled.shape[0]), x_test_scaled])

# Creacion de vector de pesos

w = np.random.normal(0,0.01,4)

# df.to_excel("transformed_coffee_sales.xlsx")

# Funcion sigmoide

def sigmoide(z):

    # Clip para prevenir overflow
    z = np.clip(z,-500,500)

    return 1 / (1 + np.exp(-z))

def probabilidad(X, w):
    z = X @ w

    probabilidades = sigmoide(z)

    return probabilidades

test_samples = x_train_with_bias[:5]  # First 5 samples
test_predictions = probabilidad(test_samples, w)

print("Sample features (first 5 training samples):")
print(test_samples)
print(f"\nCurrent weights: {w}")
print(f"Predicted probabilities: {test_predictions}")
print(f"Actual labels: {y_train[:5]}")

# Test shapes
print(f"\nShape check:")
print(f"Input X shape: {test_samples.shape}")
print(f"Weights shape: {w.shape}")
print(f"Output predictions shape: {test_predictions.shape}")

