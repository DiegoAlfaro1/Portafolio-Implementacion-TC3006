import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

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

def k_fold_cross_validation_sklearn(X, y, k=10, lambda_values=[0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]):
    """ 
    Funcion para implementar K-Fold cross-validation usando sklearn
    
    k = numero de folds en el que se dividira el dataset
    lambda_values = lista de valores de regularizacion a probar
    
    Devuelve el mejor lambda y un diccionario con los resultados de todos los folds.
    """
    
    results = {}
    best_lambda = 0.0
    best_score = -np.inf
    
    # Crear KFold object
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    for lambda_reg in lambda_values:
        fold_scores = []
        
        for train_idx, val_idx in kfold.split(X):
            # Dividir datos
            X_train_fold, X_val = X[train_idx], X[val_idx]
            y_train_fold, y_val = y[train_idx], y[val_idx]
            
            # Estandarizacion
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val)
            
            # Entrenar modelo Ridge con el lambda actual
            model = Ridge(alpha=lambda_reg, random_state=42, max_iter=1000)
            model.fit(X_train_scaled, y_train_fold)
            
            # Hacer predicciones
            val_predictions = model.predict(X_val_scaled)
            
            # Calcular R2
            fold_r2 = r2_score(y_val, val_predictions)
            fold_scores.append(fold_r2)
        
        # Calcular estadisticas
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        results[lambda_reg] = {'mean': mean_score, 'std': std_score, 'scores': fold_scores}
        
        # Actualizar mejor lambda
        if mean_score > best_score:
            best_score = mean_score
            best_lambda = lambda_reg
    
    return best_lambda, results

def train_model_with_tracking_sklearn(X_train, y_train, X_test, y_test, lambda_reg=0.0, max_epochs=2000):
    """
    Funcion para entrenar el modelo usando sklearn con tracking de metricas
    Simula el entrenamiento iterativo para mantener compatibilidad con los graficos
    """
    
    # Estandarizar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Listas para historial
    train_cost_history = []
    test_cost_history = []
    train_r2_history = []
    test_r2_history = []
    
    # Para simular el entrenamiento progresivo, usaremos diferentes alphas decrecientes
    # y entrenaremos multiples modelos para simular las epochs
    alphas = np.logspace(2, np.log10(lambda_reg) if lambda_reg > 0 else -4, max_epochs)
    
    for i, current_alpha in enumerate(alphas):
        # Entrenar modelo con el alpha actual
        model = Ridge(alpha=current_alpha, random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Predicciones
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Calcular costos (MSE + regularizacion)
        train_mse = mean_squared_error(y_train, train_pred) / 2
        test_mse = mean_squared_error(y_test, test_pred) / 2
        
        # Agregar termino de regularizacion L2
        l2_penalty = (current_alpha / 2) * np.sum(model.coef_ ** 2)
        train_cost = train_mse + l2_penalty
        test_cost = test_mse + l2_penalty
        
        train_cost_history.append(train_cost)
        test_cost_history.append(test_cost)
        
        # Calcular R2 scores
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        train_r2_history.append(train_r2)
        test_r2_history.append(test_r2)
        
        # Early stopping simulado
        if i > 10 and abs(train_cost_history[-2] - train_cost_history[-1]) < 1e-6:
            break
    
    # Entrenar modelo final con el lambda objetivo
    final_model = Ridge(alpha=lambda_reg, random_state=42, max_iter=1000)
    final_model.fit(X_train_scaled, y_train)
    
    # Retornar el modelo, scaler y historiales
    return final_model, scaler, train_cost_history, test_cost_history, train_r2_history, test_r2_history

# Preparacion de los datos usando los features disponbles
X = df_housing[available_features].values.astype(float)
# Target del modelo, en este caso el precio
y = df_housing['price'].values.astype(float)

# Cross-validation
print("Finding best regularization...")
best_lambda, cv_results = k_fold_cross_validation_sklearn(X, y, k=5)
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

print(f"Training: {len(x_train)} houses | Testing: {len(x_test)} houses")

# Entrenar el modelo usando sklearn
final_model, scaler, train_costs, test_costs, train_r2s, test_r2s = train_model_with_tracking_sklearn(
    x_train, y_train, x_test, y_test, 
    best_lambda, max_epochs=2000
)

# Escalar datos para predicciones finales
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Predicciones finales
test_predictions = final_model.predict(x_test_scaled)
train_predictions = final_model.predict(x_train_scaled)

# Obtener metricas finales como el R2 y el MAPE (Mean absolute percentage error)
test_r2 = r2_score(y_test, test_predictions)
train_r2 = r2_score(y_train, train_predictions)
# Metrica de que tan off estan las predicciones como porcentaje de los valores actuales
test_mape = mean_absolute_percentage_error(y_test, test_predictions) * 100
train_mape = mean_absolute_percentage_error(y_train, train_predictions) * 100

print(f"\nResultados:")
print(f"Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}")
print(f"Train MAPE: {train_mape:.1f}% | Test MAPE: {test_mape:.1f}%")

# ============= VISUALIZACIONES (IDENTICAS AL ORIGINAL) =============
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
# Obtener los coeficientes del modelo y ajustarlos por la escala
X_std = scaler.scale_  # Desviacion estandar usada por el scaler
feature_impacts = []
for i, (feature, coef) in enumerate(zip(available_features, final_model.coef_)):
    # Ajustar el impacto por la escala original
    original_impact = coef / X_std[i]
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