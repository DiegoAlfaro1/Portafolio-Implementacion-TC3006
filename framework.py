from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Housing.csv")
np.random.seed(42)

"""
Preparar los datos para empezar a implementar el modelo de Random Forest con scikit

Se usa la misma preparacion de datos que en la implementacion sin framework
"""

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

train, test = train_test_split(df_housing,test_size=0.20,random_state=42, shuffle=True)

x_train = train[available_features]
y_train = train['price']

x_test = test[available_features]
y_test = test['price']

"""
Hiper parametros
"""

n_estimators = 300
max_depth = 12
min_samples_split = 5
random_state = 42

"""
Implementacion de random forest
"""

# parametros de prueba para encontrar cuales serian los mejores parametros para el modelo
param_grid = {
    'n_estimators': [ 250, 500, 700],
    'max_depth': [30, 50, None], 
    'min_samples_split': [210, 40, 80],
    'min_samples_leaf': [2, 4, 7, 10]
}

rf_base = RandomForestRegressor(random_state=42)


"""
GridSearch es una tecnica de validacion cruzada

Se ejecuta a traves de los diferentes params en el grid y extrae los mejores valores
"""

grid_search = GridSearchCV(
    rf_base,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

"""
Asi es como obtengo los mejores parametros, pero se utilizan los mejores hardcodeados mas adelante para hacer que le programa no tarde en correr
"""
# Esta linea tarda bastante en correr
# grid_search.fit(train[available_features], train['price'])
# best_params = grid_search.best_params_
# best_scores = grid_search.best_score_



# Primer intento - Overfit
# max depth: 30
# min_smaples_leaf: 2
# min_samples_split: 10
# n_estimators: 1000

# Segundo intento - Overfit - menos R2
# max_depth=5,
# min_samples_leaf=2,
# min_samples_split=20,
# n_estimators=500,
# random_state=42

# Tercer intento 
# n_estimators=500,
# max_depth=15,
# min_samples_split=20,
# min_samples_leaf=10,
# max_features='sqrt',
# random_state=42

# Usando los parametros que dio el GridSearch sin necesidad de correr el fit para que no tarde tanto
final_rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42
)

final_rf.fit(x_train,y_train)

"""
Metricas del modelo para poder hacer diagnosticos
"""

# Predicciones

y_train_pred = final_rf.predict(x_train)
y_test_pred = final_rf.predict(x_test)

# Metricas

train_r2 = r2_score(y_train,y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("Train R²:", train_r2)
print("Test R²:", test_r2)
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Train MAE:", train_mae)
print("Test MAE:", test_mae)

train_rmse_list = [train_rmse]
test_rmse_list = [test_rmse]
train_r2_list = [train_r2]
test_r2_list = [test_r2]

fig, axs = plt.subplots(3, 2, figsize=(16, 15))
axs = axs.ravel()

# 1. Train vs Test RMSE
axs[0].bar(['Train RMSE', 'Test RMSE'], [train_rmse, test_rmse], color=['blue','green'])
axs[0].set_title("Train vs Test RMSE")

# 2. Train vs Test R²
axs[1].bar(['Train R²', 'Test R²'], [train_r2, test_r2], color=['blue','green'])
axs[1].set_ylim(0, 1)
axs[1].set_title("Train vs Test R²")

# 3. Predicted vs Actual (Test)
axs[2].scatter(y_test, y_test_pred, alpha=0.5, color="purple")
axs[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
axs[2].set_xlabel("Actual Prices")
axs[2].set_ylabel("Predicted Prices")
axs[2].set_title("Predicted vs Actual (Test)")

# 4. Residuals distribution
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred
axs[3].hist(residuals_train, bins=20, alpha=0.6, color='blue', label='Train')
axs[3].hist(residuals_test, bins=20, alpha=0.6, color='green', label='Test')
axs[3].set_title("Residuals Distribution")
axs[3].legend()

# 5. Absolute error distribution
abs_error_train = np.abs(residuals_train)
abs_error_test = np.abs(residuals_test)
axs[4].hist(abs_error_train, bins=20, alpha=0.6, color='blue', label='Train')
axs[4].hist(abs_error_test, bins=20, alpha=0.6, color='green', label='Test')
axs[4].set_title("Absolute Error Distribution")
axs[4].legend()

# 6. Feature importance
importances = final_rf.feature_importances_
indices = np.argsort(importances)[::-1]
axs[5].bar(range(len(importances)), importances[indices], align="center")
axs[5].set_xticks(range(len(importances)))
axs[5].set_xticklabels([available_features[i] for i in indices], rotation=45, ha="right")
axs[5].set_title("Feature Importance")

plt.tight_layout(pad=10)
plt.show()
