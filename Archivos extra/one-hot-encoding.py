import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''

Este es un codigo parecido a main.py, la diferencia es que aqui se utiliza one-hot encoding para tener los tipos cafes y que se intente ser mas preciso el modelo, 
a pesar de que con esta solucion si se hizo mas accurate el modelo, mostraba senales de overfitting.

Razones de no usarse, a pesar de que esto demuestra un problema mas alla de la implementacion si no que del problema planteado
donde en realidad estamos prediciendo el precio con el tipo y esto hace que el modelo trabaje de manera circular

Solucion, cambiar a otro problema, coo el de cuando las personas compran el cafe caro vs el barato, donde veriamos los patrones de demanda de los consumidores

'''

# Configuración de visualizaciones
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("="*60)
print("    REGRESION LOGISTICA DESDE CERO")
print("    Prediccion de Ventas Premium de Cafe")
print("="*60)

# Lectura de dataset
dataset = "Coffe_sales.csv"
df = pd.read_csv(dataset)

print(f"\nDATOS CARGADOS:")
print(f"   • Total de transacciones: {len(df):,}")
print(f"   • Columnas originales: {len(df.columns)}")

# PARTE 1: TRANSFORMACION DE DATOS
print(f"\n FASE 1: PREPARACION DE DATOS")

# Eliminacion de columnas redundantes
df = df.drop(columns=['Month_name','Time','Weekday'])

# Cambiar nombres de columnas
df.rename(columns={
    'cash_type':'Payment_type', 
    'money':'Price', 
    'coffee_name':'Coffe_name', 
    'Time_of_Day': 'Time_of_day', 
    'hour_of_day': 'Hour_of_day'
}, inplace=True)

# Crear variable objetivo: Venta Premium (>$33)
df['Premium_sale'] = (df["Price"] > 33).astype(int)

premium_count = df['Premium_sale'].sum()
premium_pct = (premium_count / len(df)) * 100

print(f"   • Variable objetivo creada: Premium_sale")
print(f"   • Ventas premium (>$33): {premium_count:,} ({premium_pct:.1f}%)")
print(f"   • Ventas estándar (≤$33): {len(df)-premium_count:,} ({100-premium_pct:.1f}%)")

# VISUALIZACION 1: Análisis exploratorio de datos
print(f"\n GENERANDO VISUALIZACIONES EXPLORATORIAS...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Análisis Exploratorio de Ventas de Café', fontsize=16, fontweight='bold')

# Distribución de ventas premium vs estándar
sns.countplot(data=df, x='Premium_sale', ax=axes[0,0])
axes[0,0].set_title('Distribución de Ventas Premium vs Estándar')
axes[0,0].set_xlabel('Tipo de Venta (0=Estándar, 1=Premium)')
axes[0,0].set_ylabel('Cantidad de Transacciones')

# Ventas premium por hora del día
hourly_premium = df.groupby('Hour_of_day')['Premium_sale'].agg(['count', 'sum', 'mean']).reset_index()
sns.lineplot(data=hourly_premium, x='Hour_of_day', y='mean', marker='o', ax=axes[0,1])
axes[0,1].set_title('Tasa de Ventas Premium por Hora del Día')
axes[0,1].set_xlabel('Hora del Día')
axes[0,1].set_ylabel('Proporción de Ventas Premium')
axes[0,1].set_ylim(0, 1)

# Ventas premium por día de semana
weekday_premium = df.groupby('Weekdaysort')['Premium_sale'].agg(['count', 'sum', 'mean']).reset_index()
sns.barplot(data=weekday_premium, x='Weekdaysort', y='mean', ax=axes[1,0])
axes[1,0].set_title('Tasa de Ventas Premium por Día de Semana')
axes[1,0].set_xlabel('Día de Semana (1=Lun, 7=Dom)')
axes[1,0].set_ylabel('Proporción de Ventas Premium')

# Ventas premium por mes
monthly_premium = df.groupby('Monthsort')['Premium_sale'].agg(['count', 'sum', 'mean']).reset_index()
sns.barplot(data=monthly_premium, x='Monthsort', y='mean', ax=axes[1,1])
axes[1,1].set_title('Tasa de Ventas Premium por Mes')
axes[1,1].set_xlabel('Mes (1=Ene, 12=Dec)')
axes[1,1].set_ylabel('Proporción de Ventas Premium')

plt.tight_layout()
plt.show()

# PASO 2: AGREGAR CARACTERISTICAS DEL TIPO DE CAFE
print(f"   • Agregando codificacion de tipo de cafe...")

# One-hot encoding para tipos de cafe
coffee_dummies = pd.get_dummies(df['Coffe_name'], prefix='Coffee')
df = pd.concat([df, coffee_dummies], axis=1)

print(f"   • Tipos de cafe codificados: {len(coffee_dummies.columns)} caracteristicas")
print(f"   • Nuevas columnas: {list(coffee_dummies.columns)}")

# Actualizar lista de caracteristicas para el modelo
coffee_columns = [col for col in df.columns if col.startswith('Coffee_')]
feature_columns = ['Hour_of_day', 'Weekdaysort', 'Monthsort'] + coffee_columns

print(f"   • Total de caracteristicas para el modelo: {len(feature_columns)}")
print(f"   • Caracteristicas: {feature_columns}")

# Division train/test
df_train = df.sample(frac=0.7, random_state=42)
df_test = df.drop(df_train.index)

print(f"   • Conjunto entrenamiento: {len(df_train):,} muestras")
print(f"   • Conjunto prueba: {len(df_test):,} muestras")

# Matriz de características - ACTUALIZADA CON TIPOS DE CAFE
x_train = df_train[feature_columns].values.astype(float)
x_test = df_test[feature_columns].values.astype(float)

print(f"   • Nueva forma de matriz entrenamiento: {x_train.shape}")
print(f"   • Nueva forma de matriz prueba: {x_test.shape}")

# Vector objetivo
y_train = df_train['Premium_sale'].values
y_test = df_test['Premium_sale'].values

# Normalizacion Min-Max
X_min = x_train.min(axis=0)
X_max = x_train.max(axis=0)

x_train_scaled = (x_train - X_min) / (X_max - X_min)
x_test_scaled = (x_test - X_min) / (X_max - X_min)

print(f"   • Características normalizadas (0-1)")
print(f"   • Rangos originales - Hora: {X_min[0]:.0f}-{X_max[0]:.0f}, Día: {X_min[1]:.0f}-{X_max[1]:.0f}, Mes: {X_min[2]:.0f}-{X_max[2]:.0f}")

# Agregar columna de sesgo (bias)
x_train_with_bias = np.column_stack([np.ones(x_train_scaled.shape[0]), x_train_scaled])
x_test_with_bias = np.column_stack([np.ones(x_test_scaled.shape[0]), x_test_scaled])

print(f"   • Columna de sesgo agregada - Forma final: {x_train_with_bias.shape}")

# PARTE 2: IMPLEMENTACION DE ALGORITMOS
print(f"\n FASE 2: ALGORITMOS DE APRENDIZAJE")

def sigmoide(z):
    """Funcion sigmoide: convierte valores reales a probabilidades (0-1)"""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def probabilidad(X, w):
    """Predice probabilidades usando regresion logistica"""
    z = X @ w
    return sigmoide(z)

def cost(y_true, y_pred):
    """Calcula el costo usando perdida logaritmica (log loss)"""
    epsilon = 1e-15
    y_pred_clip = np.clip(y_pred, epsilon, 1-epsilon)
    return -np.mean(y_true * np.log(y_pred_clip) + (1-y_true) * np.log(1-y_pred_clip))

def gradiente(X, y_true, y_pred):
    """Calcula gradientes para actualizacion de pesos"""
    m = X.shape[0]
    error = y_pred - y_true
    return (1/m) * X.T @ error

def gradient_descent(w, gradientes, learning_rate):
    """Actualiza pesos usando descenso por gradiente"""
    return w - learning_rate * gradientes

print(f"   • Funciones implementadas: Sigmoide, Prediccion, Costo, Gradientes")

# PARTE 3: ENTRENAMIENTO DEL MODELO
print(f"\n FASE 3: ENTRENAMIENTO DEL MODELO")

def entrenar_con_convergencia(X, y, learning_rate=0.005, max_epochs=2000, umbral=1e-6):
    """Entrena el modelo hasta convergencia"""
    num_features = X.shape[1]  # Ahora incluye tipos de cafe
    pesos = np.random.normal(0, 0.01, num_features)
    historial_costos = []
    
    for epoch in range(max_epochs):
        predicciones = probabilidad(X, pesos)
        costo = cost(y, predicciones)
        historial_costos.append(costo)
        
        gradientes = gradiente(X, y, predicciones)
        pesos = gradient_descent(pesos, gradientes, learning_rate)
        
        if epoch > 10:
            cambio_costo = historial_costos[-2] - historial_costos[-1]
            if cambio_costo < umbral:
                break
                
    return pesos, historial_costos, epoch

# Entrenar el modelo
print(f"   • Iniciando entrenamiento...")
final_weights, cost_history, final_epoch = entrenar_con_convergencia(
    X=x_train_with_bias,
    y=y_train,
    learning_rate=0.005,
    max_epochs=6000,
    umbral=1e-6
)

mejora_costo = cost_history[0] - cost_history[-1]
print(f"   • Entrenamiento completado en {final_epoch} epocas")
print(f"   • Costo inicial: {cost_history[0]:.6f}")
print(f"   • Costo final: {cost_history[-1]:.6f}")
print(f"   • Mejora total: {mejora_costo:.6f}")

# VISUALIZACION 2: Curva de aprendizaje
print(f"\n GENERANDO VISUALIZACION DE ENTRENAMIENTO...")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cost_history, linewidth=2, color='darkblue')
plt.title('Curva de Aprendizaje - Evolución del Costo', fontweight='bold')
plt.xlabel('Epoca')
plt.ylabel('Costo (Log Loss)')
plt.grid(True, alpha=0.3)

# Zoom a las últimas épocas
plt.subplot(1, 2, 2)
start_epoch = max(0, final_epoch - 500)
plt.plot(range(start_epoch, len(cost_history)), cost_history[start_epoch:], 
         linewidth=2, color='darkred')
plt.title(f'Convergencia - Últimas {min(500, final_epoch)} Épocas', fontweight='bold')
plt.xlabel('Epoca')
plt.ylabel('Costo (Log Loss)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# PARTE 4: EVALUACION DEL MODELO
print(f"\n FASE 4: EVALUACION DEL MODELO")

def evaluar_modelo(pesos, x_train, y_train, x_test, y_test):
    """Evalua el rendimiento del modelo entrenado"""
    train_prob = probabilidad(x_train, pesos)
    train_pred = (train_prob >= 0.5).astype(int)
    
    test_prob = probabilidad(x_test, pesos)
    test_pred = (test_prob >= 0.5).astype(int)
    
    return {
        'train_accuracy': np.mean(train_pred == y_train),
        'test_accuracy': np.mean(test_pred == y_test),
        'train_cost': cost(y_train, train_prob),
        'test_cost': cost(y_test, test_prob),
        'test_proba': test_prob,
        'test_pred': test_pred
    }

resultados = evaluar_modelo(final_weights, x_train_with_bias, y_train, x_test_with_bias, y_test)

print(f"\n RENDIMIENTO DEL MODELO:")
print(f"   • Precision entrenamiento: {resultados['train_accuracy']:.1%}")
print(f"   • Precision prueba: {resultados['test_accuracy']:.1%}")
print(f"   • Mejora vs. azar: +{resultados['test_accuracy']-0.5:.1%}")

# VISUALIZACION 3: Análisis de resultados del modelo
print(f"\n GENERANDO VISUALIZACIONES DE RESULTADOS...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Análisis de Resultados del Modelo', fontsize=16, fontweight='bold')

# Distribución de probabilidades predichas
axes[0,0].hist(resultados['test_proba'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Umbral de decisión')
axes[0,0].set_title('Distribución de Probabilidades Predichas')
axes[0,0].set_xlabel('Probabilidad Predicha')
axes[0,0].set_ylabel('Frecuencia')
axes[0,0].legend()

# Matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, resultados['test_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
axes[0,1].set_title('Matriz de Confusión')
axes[0,1].set_xlabel('Predicción')
axes[0,1].set_ylabel('Valor Real')

# Comparación de probabilidades por clase real
df_resultados = pd.DataFrame({
    'Probabilidad': resultados['test_proba'],
    'Clase_Real': ['Estándar' if y == 0 else 'Premium' for y in y_test]
})
sns.boxplot(data=df_resultados, x='Clase_Real', y='Probabilidad', ax=axes[1,0])
axes[1,0].set_title('Distribución de Probabilidades por Clase Real')
axes[1,0].set_ylabel('Probabilidad Predicha')

# Importancia de características (pesos del modelo)
caracteristicas = ['Sesgo', 'Hora del día', 'Día de semana', 'Mes']
pesos_df = pd.DataFrame({
    'Caracteristica': caracteristicas,
    'Peso': final_weights,
    'Peso_Abs': np.abs(final_weights)
})
sns.barplot(data=pesos_df, x='Peso', y='Caracteristica', palette='viridis', ax=axes[1,1])
axes[1,1].set_title('Importancia de Características (Pesos del Modelo)')
axes[1,1].set_xlabel('Peso')
axes[1,1].axvline(x=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()

# VISUALIZACION 4: Análisis de patrones temporales con predicciones
print(f"\n GENERANDO ANALISIS DE PATRONES TEMPORALES...")

# Crear DataFrame para análisis temporal
df_test_analysis = df_test.copy()
df_test_analysis['Probabilidad_Predicha'] = resultados['test_proba']
df_test_analysis['Prediccion_Correcta'] = (resultados['test_pred'] == y_test)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Rendimiento del Modelo por Patrones Temporales', fontsize=16, fontweight='bold')

# Precisión por hora
hourly_performance = df_test_analysis.groupby('Hour_of_day').agg({
    'Prediccion_Correcta': 'mean',
    'Premium_sale': 'mean',
    'Probabilidad_Predicha': 'mean'
}).reset_index()

sns.lineplot(data=hourly_performance, x='Hour_of_day', y='Prediccion_Correcta', 
             marker='o', label='Precisión del Modelo', ax=axes[0])
sns.lineplot(data=hourly_performance, x='Hour_of_day', y='Premium_sale', 
             marker='s', label='Tasa Real Premium', ax=axes[0])
axes[0].set_title('Rendimiento por Hora del Día')
axes[0].set_xlabel('Hora del Día')
axes[0].set_ylabel('Proporción')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Precisión por día de semana
weekly_performance = df_test_analysis.groupby('Weekdaysort').agg({
    'Prediccion_Correcta': 'mean',
    'Premium_sale': 'mean'
}).reset_index()

x_pos = np.arange(len(weekly_performance))
width = 0.35
axes[1].bar(x_pos - width/2, weekly_performance['Prediccion_Correcta'], 
           width, label='Precisión del Modelo', alpha=0.8)
axes[1].bar(x_pos + width/2, weekly_performance['Premium_sale'], 
           width, label='Tasa Real Premium', alpha=0.8)
axes[1].set_title('Rendimiento por Día de Semana')
axes[1].set_xlabel('Día de Semana (1=Lun, 7=Dom)')
axes[1].set_ylabel('Proporción')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(weekly_performance['Weekdaysort'])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Precisión por mes
monthly_performance = df_test_analysis.groupby('Monthsort').agg({
    'Prediccion_Correcta': 'mean',
    'Premium_sale': 'mean'
}).reset_index()

sns.lineplot(data=monthly_performance, x='Monthsort', y='Prediccion_Correcta', 
             marker='o', label='Precisión del Modelo', ax=axes[2])
sns.lineplot(data=monthly_performance, x='Monthsort', y='Premium_sale', 
             marker='s', label='Tasa Real Premium', ax=axes[2])
axes[2].set_title('Rendimiento por Mes')
axes[2].set_xlabel('Mes (1=Ene, 12=Dic)')
axes[2].set_ylabel('Proporción')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n PESOS APRENDIDOS (INTERPRETACION):")
# Mostrar todos los pesos pero destacar los mas importantes
pesos_ordenados = pesos_df.sort_values('Peso_Abs', ascending=False)
print("Top 10 caracteristicas mas importantes:")
for i, (_, row) in enumerate(pesos_ordenados.head(10).iterrows()):
    peso = row['Peso']
    caracteristica = row['Caracteristica']
    direccion = "favorece" if peso > 0 else "reduce"
    print(f"   • {caracteristica}: {peso:+.3f} ({direccion} ventas premium)")

print(f"\n EJEMPLOS DE PREDICCION:")
print("Predicha | Real | Probabilidad | Estado")
print("-" * 40)
correctas = 0
for i in range(15):
    pred = 1 if resultados['test_proba'][i] >= 0.5 else 0
    real = y_test[i]
    prob = resultados['test_proba'][i]
    estado = "Correcta" if pred == real else "Incorrecta"
    if pred == real:
        correctas += 1
    print(f"   {pred}      |  {real}   |   {prob:.3f}    | {estado}")

print(f"\n RESUMEN FINAL:")
print(f"   • Modelo: Regresion Logistica (implementacion desde cero)")
print(f"   • Objetivo: Predecir ventas premium de cafe (>$33)")
print(f"   • Caracteristicas: {len(feature_columns)} total (temporales + tipos de cafe)")
print(f"   • Precision final: {resultados['test_accuracy']:.1%}")
print(f"   • Ejemplos correctos: {correctas}/15 ({correctas/15:.1%})")

print(f"\n" + "="*60)
print(f"    ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
print(f"="*60)