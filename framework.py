from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
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

print(train.shape)
print(test.shape)

"""
Hiper parametros
"""

n_estimators = 300
max_depth = 12
min_samples_split = 5
random_state = 42

"""
Implementacion
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

grid_search.fit(train[available_features], train['price'])

best_params = grid_search.best_params_
best_scores = grid_search.best_score_

"""
GridSearchCV nos da que los mejores parametros son

max depth: 30
min_smaples_leaf: 2
min_samples_split: 10
n_estimators: 700

Con un score de 0.608

"""


print(best_params)
print(best_scores)

# final_rf = RandomForestRegressor(**best_params, random_state=42)