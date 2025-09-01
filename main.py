import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

dataset = "Housing.csv"
df = pd.read_csv(dataset)

def prepare_housing_features(df):
    """ Funcion para procesar los datos """
    df_processed = df.copy()
    
    # Cambiar las columnas que sean si o no a 1 y 0
    binary_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for feature in binary_features:
        df_processed[f'{feature}_num'] = (df_processed[feature] == 'yes').astype(int)
    
    # Procesar si la casa esta amueblada, semi amueblada o no amueblada
    df_processed['furnished'] = (df_processed['furnishingstatus'] == 'furnished').astype(int)
    df_processed['semi_furnished'] = (df_processed['furnishingstatus'] == 'semi-furnished').astype(int)

    # Procesar los cuartos, teniendo en cuenta los banos como un cuarto mas, tambien agregando cuantos banos por cuarto hay
    df_processed['total_rooms'] = df_processed['bedrooms'] + df_processed['bathrooms']
    df_processed['area_per_bedroom'] = df_processed['area'] / (df_processed['bedrooms'] + 1e-6)
    df_processed['bathrooms_per_bedroom'] = df_processed['bathrooms'] / (df_processed['bedrooms'] + 1e-6)
    
    return df_processed

df_housing = prepare_housing_features(df)

available_features = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
    'mainroad_num', 'prefarea_num', 'guestroom_num', 'basement_num', 
    'hotwaterheating_num', 'airconditioning_num', 'furnished', 'semi_furnished',
    'total_rooms', 'area_per_bedroom', 'bathrooms_per_bedroom'
]

def predict(X, weight):
    """ Funcion para hacer las predicciones """
    return X @ weight

def cost_with_l2(X, y, weight, lambda_reg=0.0):
    """ Funcion de costos usando penalizacion L2"""
    predictions = predict(X, weight)
    # funcion mean square error
    mse_cost = (1/2) * np.mean((predictions - y) ** 2)
    # Nos saltamos el bias weight usamos L2 para poder hacer los pesos mas pequenos y evitar overfitting
    l2_penalty = (lambda_reg / 2) * np.sum(weight[1:] ** 2)
    # Regresamos las sumas de MSE y L2
    return mse_cost + l2_penalty

def gradient_with_l2(X, y, weight, lambda_reg=0.0):
    """ Funcion para obtener el gradiente usando regularizacion L2 """

    # Obtenemos las predicciones usando la funcion predict
    predictions = predict(X, weight)
    # Obtenemos el error restando la verdad de as predicciones
    error = predictions - y
    # Obtenemos el gradiente  multiplicalndo (1/la cantidad de muestras) y multiplicamos eso por X transpuesta y multiplicando esa matri por el error 
    grad = (1/len(y)) * X.T @ error
    
    # Hacemos un array de 0 igual al weight
    l2_grad = np.zeros_like(weight)
    l2_grad[1:] = lambda_reg * weight[1:]
    grad += l2_grad
    
    return grad

def train_model_with_tracking(X_train, y_train, X_test, y_test, lambda_reg=0.0, learning_rate=0.01, max_epochs=1000, tolerance=1e-6):
    """ Funcion para entrenar al modelo"""

    # cantidad de muestras que tenemos
    n_features = X_train.shape[1]
    # Inicializamos los pesos como random, del tamano de als muestras
    weight = np.random.normal(0, 0.01, n_features)
    # Historial de costos
    train_cost_history = []
    test_cost_history = []
    train_r2_history = []
    test_r2_history = []
    
    for epoch in range(max_epochs):
        # Optenemos el costo actual
        train_cost = cost_with_l2(X_train, y_train, weight, lambda_reg)
        test_cost = cost_with_l2(X_test, y_test, weight, lambda_reg)
        
        train_cost_history.append(train_cost)
        test_cost_history.append(test_cost)
        
        # Calculate R2 scores
        train_pred = predict(X_train, weight)
        test_pred = predict(X_test, weight)
        train_r2 = calculate_r2(y_train, train_pred)
        test_r2 = calculate_r2(y_test, test_pred)
        
        train_r2_history.append(train_r2)
        test_r2_history.append(test_r2)
        
        # Obtenemos el gradiente
        grads = gradient_with_l2(X_train, y_train, weight, lambda_reg)
        # Actualizamos los pesos con el gradiente y el learning rate
        weight = weight - learning_rate * grads
        
        if len(train_cost_history) > 1:
            if abs(train_cost_history[-2] - train_cost_history[-1]) < tolerance:
                break
    
    return weight, train_cost_history, test_cost_history, train_r2_history, test_r2_history

def calculate_r2(y_true, y_pred):
    """ Funcion para calcular coeficiente de determinacion (R2) """

    # Obtenemos el mean de y
    y_mean = np.mean(y_true)
    # Suma total de los cuadrados
    ss_total = np.sum((y_true - y_mean) ** 2)
    # Suma residual de los cuadrados
    ss_residual = np.sum((y_pred - y_true) ** 2)
    return 1 - (ss_residual / ss_total)

def k_fold_cross_validation(X, y, k=10, lambda_values=[0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]):
    """ 
    Funcion para implementar K-Fold cross-validation 
    
    k = numero de folds en el que se dividira el dataset
    lambda_values = lista de valores de regularizacion a probar
    
    Devuelve el mejor lambda y un diccionario con los resultados de todos los folds.
    """

    n_samples = len(X)                # numero total de muestras
    fold_size = n_samples // k        # tamano de cada fold
    
    best_lambda = 0.0                 # lambda con mejor performance
    best_score = -np.inf              # mejor R2 promedio
    results = {}                      # diccionario para almacenar resultados
    
    # Itera sobre cada valor de lambda para evaluarlo
    for lambda_reg in lambda_values:
        fold_scores = []  # almacenara R2 de cada fold para este lambda
        
        # Divide el dataset en k folds
        for fold in range(k):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k-1 else n_samples  # ultimo fold puede ser mas grande
            
            # Selecciona el fold actual como conjunto de validacion
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]
            
            # El resto se usa como conjunto de entrenamiento
            X_train_fold = np.concatenate([X[:val_start], X[val_end:]])
            y_train_fold = np.concatenate([y[:val_start], y[val_end:]])
            
            # Estandarizacion: transforma features para tener mean 0 y std 1
            X_mean = X_train_fold.mean(axis=0)
            X_std = X_train_fold.std(axis=0)
            
            # Escala los datos
            X_train_scaled = (X_train_fold - X_mean) / X_std
            X_val_scaled = (X_val - X_mean) / X_std
            
            # Agrega columna de bias (intercept) de 1s
            X_train_bias = np.column_stack([np.ones(X_train_scaled.shape[0]), X_train_scaled])
            X_val_bias = np.column_stack([np.ones(X_val_scaled.shape[0]), X_val_scaled])
            
            # Entrena el modelo usando el lambda actual
            weights, _, _, _, _ = train_model_simple(X_train_bias, y_train_fold, lambda_reg)
            
            # Hace predicciones sobre el conjunto de validacion
            val_predictions = predict(X_val_bias, weights)
            
            # Calcula R2 para este fold y lo guarda
            fold_r2 = calculate_r2(y_val, val_predictions)
            fold_scores.append(fold_r2)
        
        # Calcula R2 promedio y std para este lambda
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        results[lambda_reg] = {'mean': mean_score, 'std': std_score, 'scores': fold_scores}
        
        # Actualiza mejor lambda si el promedio R2 es mayor
        if mean_score > best_score:
            best_score = mean_score
            best_lambda = lambda_reg
    
    return best_lambda, results

def train_model_simple(X_train, y_train, lambda_reg=0.0, learning_rate=0.01, max_epochs=1000, tolerance=1e-6):
    """ Funcion para entrenar el modelo sin trackear los costos """
    # Cantidad de muestras
    n_features = X_train.shape[1]
    # Inicializar los pesos como random
    weight = np.random.normal(0, 0.01, n_features)
    #Historial de costos
    cost_history = []
    
    for epoch in range(max_epochs):
        # Optener el costo actual
        current_cost = cost_with_l2(X_train, y_train, weight, lambda_reg)
        cost_history.append(current_cost)
        
        # Obtener los gradientes
        grads = gradient_with_l2(X_train, y_train, weight, lambda_reg)
        weight = weight - learning_rate * grads
        
        if len(cost_history) > 1:
            if abs(cost_history[-2] - cost_history[-1]) < tolerance:
                break
    
    return weight, cost_history, None, None, None

# Preparacion de los datos usando los features disponbles
X = df_housing[available_features].values.astype(float)
# Target del modelo, en este caso el precio
y = df_housing['price'].values.astype(float)

# Cross-validation
print("Finding best regularization...")
best_lambda, cv_results = k_fold_cross_validation(X, y, k=5)
print(f"Best lambda: {best_lambda}")

# Split final de los datos
train_size = int(len(df_housing) * 0.8)
df_shuffled = df_housing.sample(frac=1, random_state=42).reset_index(drop=True)

df_train = df_shuffled.iloc[:train_size].copy()
df_test = df_shuffled.iloc[train_size:].copy()

x_train = df_train[available_features].values.astype(float)
x_test = df_test[available_features].values.astype(float)
y_train = df_train['price'].values.astype(float)
y_test = df_test['price'].values.astype(float)

# Estandarizacion usando Z-Score
X_mean = x_train.mean(axis=0)
X_std = x_train.std(axis=0)

#Escalamiento de datos
x_train_scaled = (x_train - X_mean) / X_std
x_test_scaled = (x_test - X_mean) / X_std

# Agregar columna de bias
x_train_with_bias = np.column_stack([np.ones(x_train_scaled.shape[0]), x_train_scaled])
x_test_with_bias = np.column_stack([np.ones(x_test_scaled.shape[0]), x_test_scaled])

print(f"Training: {len(x_train)} houses | Testing: {len(x_test)} houses")


# Entrenar el modelo usando la funcion para trackear
final_weights, train_costs, test_costs, train_r2s, test_r2s = train_model_with_tracking(
    x_train_with_bias, y_train, x_test_with_bias, y_test, 
    best_lambda, learning_rate=0.01, max_epochs=2000
)

# Predicciones finales
test_predictions = predict(x_test_with_bias, final_weights)
train_predictions = predict(x_train_with_bias, final_weights)

# Obtener metricas finales como el R2 y el MAPE (Mean absolute percentage error)
test_r2 = calculate_r2(y_test, test_predictions)
train_r2 = calculate_r2(y_train, train_predictions)
# Metrica de que tan off estan las predicciones como porcentaje de los valores actuales
test_mape = np.mean(np.abs((test_predictions - y_test) / y_test)) * 100
train_mape = np.mean(np.abs((train_predictions - y_train) / y_train)) * 100

print(f"\nResultados:")
print(f"Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}")
print(f"Train MAPE: {train_mape:.1f}% | Test MAPE: {test_mape:.1f}%")

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

axes[0,0].plot(train_costs, label='Training Cost', color='blue', linewidth=2)
axes[0,0].plot(test_costs, label='Test Cost', color='red', linewidth=2)
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Cost')
axes[0,0].set_title('Training vs Test Cost')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(train_r2s, label='Training R2', color='blue', linewidth=2)
axes[0,1].plot(test_r2s, label='Test R2', color='red', linewidth=2)
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('R2 Score')
axes[0,1].set_title('Training vs Test Accuracy')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

axes[0,2].scatter(y_test, test_predictions, alpha=0.6, color='blue', s=30)
axes[0,2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0,2].set_xlabel('Actual Price ($)')
axes[0,2].set_ylabel('Predicted Price ($)')
axes[0,2].set_title(f'Predictions vs Actual (R2={test_r2:.3f})')
axes[0,2].grid(True, alpha=0.3)

residuals = test_predictions - y_test
axes[1,0].hist(residuals, bins=25, alpha=0.7, color='green', edgecolor='black')
axes[1,0].set_xlabel('Residuals ($)')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title(f'Residuals Distribution')
axes[1,0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
axes[1,0].grid(True, alpha=0.3)

percentage_errors = np.abs((test_predictions - y_test) / y_test) * 100
axes[1,1].hist(percentage_errors, bins=25, alpha=0.7, color='orange', edgecolor='black')
axes[1,1].set_xlabel('Absolute Percentage Error (%)')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title(f'Error Distribution (MAPE={test_mape:.1f}%)')
axes[1,1].axvline(x=test_mape, color='red', linestyle='--', alpha=0.7, label=f'Mean: {test_mape:.1f}%')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

price_bins = np.percentile(y_test, [0, 33, 67, 100])
bin_labels = ['Low', 'Mid', 'High']
bin_errors = []
for i in range(len(price_bins)-1):
    mask = (y_test >= price_bins[i]) & (y_test <= price_bins[i+1])
    if np.sum(mask) > 0:
        bin_mape = np.mean(np.abs((test_predictions[mask] - y_test[mask]) / y_test[mask])) * 100
        bin_errors.append(bin_mape)
    else:
        bin_errors.append(0)

axes[1,2].bar(bin_labels, bin_errors, alpha=0.7, color=['blue', 'green', 'red'])
axes[1,2].set_xlabel('Price Range')
axes[1,2].set_ylabel('MAPE (%)')
axes[1,2].set_title('Error by Price Tier')
axes[1,2].grid(True, alpha=0.3)

# Plot 7: Feature importance
feature_impacts = []
for i, (feature, weight) in enumerate(zip(available_features, final_weights[1:]), 0):
    original_impact = weight / X_std[i]
    feature_impacts.append((feature, original_impact, abs(original_impact)))

feature_impacts.sort(key=lambda x: x[2], reverse=True)
top_features = feature_impacts[:8]
feature_names_clean = [f.replace('_num', '').replace('_', ' ').title() for f, _, _ in top_features]
impacts = [abs(imp) for _, imp, _ in top_features]
colors = ['blue' if feature_impacts[i][1] > 0 else 'red' for i in range(8)]

axes[2,0].barh(range(len(top_features)), impacts, color=colors, alpha=0.7)
axes[2,0].set_yticks(range(len(top_features)))
axes[2,0].set_yticklabels(feature_names_clean)
axes[2,0].set_xlabel('Impact on Price ($)')
axes[2,0].set_title('Feature Importance')
axes[2,0].grid(True, alpha=0.3)

axes[2,1].scatter(df_test['area'], y_test, alpha=0.6, color='blue', label='Actual', s=30)
axes[2,1].scatter(df_test['area'], test_predictions, alpha=0.6, color='red', label='Predicted', s=30)
axes[2,1].set_xlabel('Area (sq ft)')
axes[2,1].set_ylabel('Price ($)')
axes[2,1].set_title('Price vs Area')
axes[2,1].legend()
axes[2,1].grid(True, alpha=0.3)

lambda_values = list(cv_results.keys())
mean_scores = [cv_results[l]['mean'] for l in lambda_values]
std_scores = [cv_results[l]['std'] for l in lambda_values]
axes[2,2].errorbar(lambda_values, mean_scores, yerr=std_scores, marker='o', capsize=5)
axes[2,2].axvline(x=best_lambda, color='red', linestyle='--', alpha=0.7)
axes[2,2].set_xscale('log')
axes[2,2].set_xlabel('Regularization Parameter')
axes[2,2].set_ylabel('CV R2')
axes[2,2].set_title('Cross-Validation Results')
axes[2,2].grid(True, alpha=0.3)

plt.tight_layout(pad=6)
plt.show()

# Ejemplos de predicciones que hizo el modelo
print(f"\nSample Predictions:")
sample_indices = np.random.choice(len(y_test), 3, replace=False)
for i, idx in enumerate(sample_indices, 1):
    actual = y_test[idx]
    predicted = test_predictions[idx]
    error_pct = abs((predicted - actual) / actual) * 100
    house = df_test.iloc[idx]
    print(f"House {i}: {house['area']:,} sqft | Actual: ${actual:,.0f} | Predicted: ${predicted:,.0f} | Error: {error_pct:.1f}%")