import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

np.random.seed(42)

# =======================
# 1. Cargar dataset
# =======================
dataset = "Housing.csv"
df = pd.read_csv(dataset)

# =======================
# 2. Preprocesamiento
# =======================
def prepare_housing_features(df):
    df_processed = df.copy()
    binary_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for feature in binary_features:
        df_processed[f'{feature}_num'] = (df_processed[feature] == 'yes').astype(int)
    
    df_processed['furnished'] = (df_processed['furnishingstatus'] == 'furnished').astype(int)
    df_processed['semi_furnished'] = (df_processed['furnishingstatus'] == 'semi-furnished').astype(int)

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

X = df_housing[available_features].values.astype(float)
y = df_housing['price'].values.astype(float)

# =======================
# 3. Split Train / Val / Test
# =======================
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42)

print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}\n")

# =======================
# 4. Función para entrenar y evaluar Ridge
# =======================
def train_ridge(X_train, y_train, X_val, y_val, alpha=0.0):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model = Ridge(alpha=alpha, random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'val_r2': r2_score(y_val, y_val_pred),
        'train_mape': mean_absolute_percentage_error(y_train, y_train_pred) * 100,
        'val_mape': mean_absolute_percentage_error(y_val, y_val_pred) * 100,
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred))
    }
    
    return model, scaler, y_train_pred, y_val_pred, metrics

# =======================
# 5. Evaluación sin regularización
# =======================
print("Evaluando modelo sin regularización...")
alpha_no_reg = 0.0
model_no_reg, scaler_no_reg, y_train_pred_nr, y_val_pred_nr, metrics_nr = train_ridge(
    X_train, y_train, X_val, y_val, alpha=alpha_no_reg
)
print("Train R2:", metrics_nr['train_r2'], "| Validation R2:", metrics_nr['val_r2'])
print("Train RMSE:", metrics_nr['train_rmse'], "| Validation RMSE:", metrics_nr['val_rmse'])
print("Train MAPE:", metrics_nr['train_mape'], "| Validation MAPE:", metrics_nr['val_mape'], "\n")

# =======================
# 6. K-Fold CV para encontrar mejor alpha
# =======================
def k_fold_cv_ridge(X, y, k=5, alphas=[0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = {}
    for alpha in alphas:
        fold_r2 = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            _, _, _, _, metrics = train_ridge(X_tr, y_tr, X_va, y_va, alpha=alpha)
            fold_r2.append(metrics['val_r2'])
        results[alpha] = {'mean_r2': np.mean(fold_r2), 'std_r2': np.std(fold_r2)}
    best_alpha = max(results, key=lambda a: results[a]['mean_r2'])
    return best_alpha, results

alpha_values = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
best_alpha, cv_results = k_fold_cv_ridge(X_train_val, y_train_val, k=5, alphas=alpha_values)
print(f"Best alpha (regularization): {best_alpha}\n")

# =======================
# 7. Entrenar modelo final con mejor alpha
# =======================
print("Entrenando modelo con regularización...")
model_reg, scaler_reg, y_train_pred_r, y_val_pred_r, metrics_r = train_ridge(
    X_train, y_train, X_val, y_val, alpha=best_alpha
)
print("Train R2:", metrics_r['train_r2'], "| Validation R2:", metrics_r['val_r2'])
print("Train RMSE:", metrics_r['train_rmse'], "| Validation RMSE:", metrics_r['val_rmse'])
print("Train MAPE:", metrics_r['train_mape'], "| Validation MAPE:", metrics_r['val_mape'], "\n")

# =======================
# 8. Evaluación final en Test
# =======================
X_test_scaled = scaler_reg.transform(X_test)
y_test_pred = model_reg.predict(X_test_scaled)

test_metrics = {
    'test_r2': r2_score(y_test, y_test_pred),
    'test_mape': mean_absolute_percentage_error(y_test, y_test_pred) * 100,
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred))
}

print("Métricas Test Set:")
print(f"R2: {test_metrics['test_r2']}")
print(f"RMSE: {test_metrics['test_rmse']}")
print(f"MAPE: {test_metrics['test_mape']}\n")

# =======================
# 9. Visualizaciones y guardado
# =======================
plt.figure(figsize=(10,6))
plt.bar(['Train R2', 'Val R2'], [metrics_nr['train_r2'], metrics_nr['val_r2']], color=['blue','orange'])
plt.title('Bias/Variance sin regularización (R2)')
plt.savefig("bias_variance_no_reg.png")
plt.close()

plt.figure(figsize=(10,6))
plt.bar(['Train R2', 'Val R2'], [metrics_r['train_r2'], metrics_r['val_r2']], color=['blue','orange'])
plt.title(f'Bias/Variance con regularización (alpha={best_alpha})')
plt.savefig("bias_variance_reg.png")
plt.close()

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predictions vs Actual (Test)')
plt.savefig("pred_vs_actual.png")
plt.close()

residuals = y_test_pred - y_test
plt.figure(figsize=(10,6))
plt.hist(residuals, bins=25, color='green', alpha=0.7)
plt.axvline(0, color='red', linestyle='--')
plt.title('Residuals Distribution (Test)')
plt.savefig("residuals.png")
plt.close()

percentage_errors = np.abs((y_test_pred - y_test) / y_test) * 100
plt.figure(figsize=(10,6))
plt.hist(percentage_errors, bins=25, color='orange', alpha=0.7)
plt.axvline(np.mean(percentage_errors), color='red', linestyle='--', label=f'Mean: {np.mean(percentage_errors):.1f}%')
plt.legend()
plt.title('Absolute Percentage Error (Test)')
plt.savefig("percentage_error.png")
plt.close()

# Feature importance
X_std = scaler_reg.scale_
feature_impacts = [(f, coef / X_std[i]) for i, (f, coef) in enumerate(zip(available_features, model_reg.coef_))]
feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
top_features = feature_impacts[:8]
plt.figure(figsize=(10,6))
plt.barh([f[0] for f in top_features], [abs(f[1]) for f in top_features], color='skyblue')
plt.title('Top 8 Feature Importance')
plt.savefig("feature_importance.png")
plt.close()

# Cross-validation plot
alpha_vals = list(cv_results.keys())
mean_r2s = [cv_results[a]['mean_r2'] for a in alpha_vals]
std_r2s = [cv_results[a]['std_r2'] for a in alpha_vals]
plt.figure(figsize=(10,6))
plt.errorbar(alpha_vals, mean_r2s, yerr=std_r2s, marker='o', capsize=5)
plt.axvline(best_alpha, color='red', linestyle='--')
plt.xscale('log')
plt.title('Cross-Validation R2 vs Alpha')
plt.savefig("cv_r2_alpha.png")
plt.close()

print("Todas las gráficas se han guardado en archivos PNG para el reporte.")
