import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

dataset = "Housing.csv"
df = pd.read_csv(dataset)

print(f"Dataset: {len(df)} houses, Price range: ${df['price'].min():,} - ${df['price'].max():,}")

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

def train_model(X_train, y_train, lambda_reg=0.0, learning_rate=0.01, max_epochs=1000, tolerance=1e-6):
    """ Funcion para entrenar al modelo"""

    # cantidad de muestras que tenemos
    n_features = X_train.shape[1]
    # Inicializamos los pesos como random, del tamano de als muestras
    weight = np.random.normal(0, 0.01, n_features)
    # Historial de costos
    cost_history = []
    
    for epoch in range(max_epochs):
        # Optenemos el costo actual
        current_cost = cost_with_l2(X_train, y_train, weight, lambda_reg)
        cost_history.append(current_cost)
        
        # Obtenemos el gradiente
        grads = gradient_with_l2(X_train, y_train, weight, lambda_reg)
        # Actualizamos los pesos con el gradiente y el learning rate
        weight = weight - learning_rate * grads
        
        if len(cost_history) > 1:
            if abs(cost_history[-2] - cost_history[-1]) < tolerance:
                break
    
    return weight, cost_history

def calculate_r2(y_true, y_pred):
    """ Funcion para calcular coeficiente de determinacion (R2) """

    # Obtenemos el mean de y
    y_mean = np.mean(y_true)
    # Suma total de los cuadrados
    ss_total = np.sum((y_true - y_mean) ** 2)
    # Suma residual de los cuadrados
    ss_residual = np.sum((y_pred - y_true) ** 2)
    return 1 - (ss_residual / ss_total)

def k_fold_cross_validation(X, y, k=5, lambda_values=[0.0, 0.001, 0.01, 0.1, 1.0]):
    """ 
    Funcion para implementar K-Fold cross-validation 
    
    k = numero de folds en el que se dividira el dataset
    lambda_values = lista de valores de regularizacion a probar
    
    Devuelve el mejor lambda y un diccionario con los resultados de todos los folds.
    """

    n_samples = len(X)                # numero total de muestras
    fold_size = n_samples // k        # tamaño de cada fold
    
    best_lambda = 0.0                 # lambda con mejor performance
    best_score = -np.inf              # mejor R2 promedio
    results = {}                      # diccionario para almacenar resultados
    
    print("\nK-Fold Cross-Validation:")
    
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
            weights, _ = train_model(X_train_bias, y_train_fold, lambda_reg, learning_rate=0.001, max_epochs=3000)
            
            # Hace predicciones sobre el conjunto de validacion
            val_predictions = predict(X_val_bias, weights)
            
            # Calcula R2 para este fold y lo guarda
            fold_r2 = calculate_r2(y_val, val_predictions)
            fold_scores.append(fold_r2)
        
        # Calcula R2 promedio y std para este lambda
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        results[lambda_reg] = {'mean': mean_score, 'std': std_score, 'scores': fold_scores}
        
        print(f"Regularization {lambda_reg:6.3f}: R2 = {mean_score:.4f} ± {std_score:.4f}")
        
        # Actualiza mejor lambda si el promedio R2 es mayor
        if mean_score > best_score:
            best_score = mean_score
            best_lambda = lambda_reg
    
    print(f"Best regularization: {best_lambda} with R2 = {best_score:.4f}")
    return best_lambda, results

X = df_housing[available_features].values.astype(float)
y = df_housing['price'].values.astype(float)

best_lambda, cv_results = k_fold_cross_validation(X, y, k=5)

train_size = int(len(df_housing) * 0.8)
df_shuffled = df_housing.sample(frac=1, random_state=42).reset_index(drop=True)

df_train = df_shuffled.iloc[:train_size].copy()
df_test = df_shuffled.iloc[train_size:].copy()

x_train = df_train[available_features].values.astype(float)
x_test = df_test[available_features].values.astype(float)
y_train = df_train['price'].values.astype(float)
y_test = df_test['price'].values.astype(float)

X_mean = x_train.mean(axis=0)
X_std = x_train.std(axis=0)
x_train_scaled = (x_train - X_mean) / X_std
x_test_scaled = (x_test - X_mean) / X_std

x_train_with_bias = np.column_stack([np.ones(x_train_scaled.shape[0]), x_train_scaled])
x_test_with_bias = np.column_stack([np.ones(x_test_scaled.shape[0]), x_test_scaled])

print(f"\nFinal Training: {len(x_train)} train | {len(x_test)} test houses")

final_weights, cost_history = train_model(x_train_with_bias, y_train, best_lambda, learning_rate=0.001, max_epochs=4000)

test_predictions = predict(x_test_with_bias, final_weights)
train_predictions = predict(x_train_with_bias, final_weights)

def evaluate_model(y_true, y_pred, dataset_name=""):
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    percentage_errors = np.abs(errors / y_true) * 100
    
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(percentage_errors)
    r2 = calculate_r2(y_true, y_pred)
    
    print(f"\n{dataset_name} Performance:")
    print(f"R2: {r2:.4f} | MAE: ${mae:,.0f} | RMSE: ${rmse:,.0f} | MAPE: {mape:.1f}%")
    
    within_10 = np.mean(percentage_errors <= 10) * 100
    within_20 = np.mean(percentage_errors <= 20) * 100
    print(f"Within 10% error: {within_10:.1f}% | Within 20% error: {within_20:.1f}%")
    
    return {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape}

train_metrics = evaluate_model(y_train, train_predictions, "Training")
test_metrics = evaluate_model(y_test, test_predictions, "Test")

overfitting_gap = train_metrics['r2'] - test_metrics['r2']
print(f"\nOverfitting check: {overfitting_gap:.3f} (good if < 0.10)")

mean_baseline_mae = np.mean(np.abs(np.full(len(y_test), np.mean(y_test)) - y_test))
improvement = ((mean_baseline_mae - test_metrics['mae']) / mean_baseline_mae) * 100
print(f"Improvement over baseline: {improvement:.1f}%")

def interpret_features(weights, feature_names, X_std):
    print(f"\nFeature Impact Analysis:")
    
    feature_impacts = []
    for i, (feature, weight) in enumerate(zip(feature_names[1:], weights[1:]), 0):
        original_impact = weight / X_std[i]
        feature_impacts.append((feature, original_impact, abs(original_impact)))
    
    feature_impacts.sort(key=lambda x: x[2], reverse=True)
    
    print("Top 8 features:")
    for i, (feature, impact, abs_impact) in enumerate(feature_impacts[:8], 1):
        direction = "+" if impact > 0 else "-"
        
        if 'area' in feature.lower():
            print(f"{i}. {feature:20s}: {direction}${abs(impact):6.0f} per sq ft")
        elif feature in ['bedrooms', 'bathrooms', 'stories']:
            print(f"{i}. {feature:20s}: {direction}${abs(impact):8,.0f} per unit")
        elif '_num' in feature:
            clean_name = feature.replace('_num', '')
            print(f"{i}. {feature:20s}: {direction}${abs(impact):8,.0f} if present")
        else:
            print(f"{i}. {feature:20s}: {direction}${abs(impact):8,.0f} per unit")
    
    return feature_impacts

feature_impacts = interpret_features(final_weights, ['Bias'] + available_features, X_std)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
lambda_values = list(cv_results.keys())
mean_scores = [cv_results[l]['mean'] for l in lambda_values]
std_scores = [cv_results[l]['std'] for l in lambda_values]
plt.errorbar(lambda_values, mean_scores, yerr=std_scores, marker='o', capsize=5)
plt.axvline(x=best_lambda, color='red', linestyle='--', alpha=0.7)
plt.xscale('log')
plt.xlabel('Regularization Parameter')
plt.ylabel('Cross-Validation R2')
plt.title('Hyperparameter Tuning')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.scatter(y_test, test_predictions, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'Predicted vs Actual (R2 = {test_metrics["r2"]:.3f})')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
residuals = test_predictions - y_test
plt.scatter(test_predictions, residuals, alpha=0.6, color='red')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Predicted Price ($)')
plt.ylabel('Residual ($)')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
plt.plot(cost_history, color='purple', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title(f'Training Convergence')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
top_features = feature_impacts[:8]
feature_names_clean = [f.replace('_num', '').replace('_', ' ').title() for f, _, _ in top_features]
impacts = [abs(imp) for _, imp, _ in top_features]
colors = ['blue' if feature_impacts[i][1] > 0 else 'red' for i in range(8)]
plt.barh(range(len(top_features)), impacts, color=colors, alpha=0.7)
plt.yticks(range(len(top_features)), feature_names_clean)
plt.xlabel('Impact on Price ($)')
plt.title('Feature Importance')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
percentage_errors = np.abs((test_predictions - y_test) / y_test) * 100
plt.hist(percentage_errors, bins=20, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Absolute Percentage Error (%)')
plt.ylabel('Frequency')
plt.title(f'Error Distribution (MAPE = {test_metrics["mape"]:.1f}%)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nSample Predictions:")
sample_indices = np.random.choice(len(y_test), 3, replace=False)

for i, idx in enumerate(sample_indices, 1):
    actual = y_test[idx]
    predicted = test_predictions[idx]
    error_pct = abs((predicted - actual) / actual) * 100
    house = df_test.iloc[idx]
    
    print(f"House {i}: {house['area']:,} sq ft, {house['bedrooms']} bed, {house['bathrooms']} bath")
    print(f"  Actual: ${actual:,.0f} | Predicted: ${predicted:,.0f} | Error: {error_pct:.1f}%")

print(f"\nFINAL MODEL SUMMARY")
print(f"Best Regularization: {best_lambda}")
print(f"Test R2: {test_metrics['r2']:.4f} | Test MAPE: {test_metrics['mape']:.1f}%")
print(f"Baseline Improvement: {improvement:.1f}%")
print(f"Training Iterations: {len(cost_history)}")
print(f"Top Price Driver: {feature_impacts[0][0].replace('_num', '').title()}")
print(f"Predictions within 20%: {np.mean(percentage_errors <= 20):.0%}")