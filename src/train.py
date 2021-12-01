import os
import pandas as pd
import numpy as np

from utils.funciones import *

import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle           
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print("\nComienza el script train.py\n")

# Constantes
# ROOT_PATH = os.path.abspath('.')
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = ROOT_PATH + '\\data\\raw\\'
PROC_DATA_PATH = ROOT_PATH + '\\data\\proc\\'

# Get dataframe
measure = 'measure1'
print('Cargando datos')
df = get_df(RAW_DATA_PATH, measure)
print('Datos cargados\n')

# Save dataframe 
print("Se empiezan a procesar los datos")
df.to_excel(PROC_DATA_PATH + measure + '_wifi_smartphone.xlsx')
print("Datos procesados")
print("Se guardan los datos procesados\n")

# Split labels and features
X = df.iloc[:,1:-2]
y = df.iloc[:,-2:]

# Suffle data
X, y = shuffle(X, y, random_state=44)

# Split train test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=44)

# Build pipeline
pipeline = Pipeline(steps=[
                            ("scaler",StandardScaler()),
                            ('knn', KNeighborsRegressor(n_neighbors=3, weights='distance'))
                            ])

# Train model
print("Se empieza a entrenar el modelo\n")
pipeline.fit(X_train, y_train)

with open(ROOT_PATH + '\\model\\selected_models\\finished_model.model', "wb") as archivo_salida:
    pickle.dump(pipeline, archivo_salida)

predictions = pipeline.predict(X_test)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print("Resultados:")
print("MSE:  %.7f " % mean_squared_error(y_test, predictions))
print("RMSE: %.5f " % np.sqrt(mean_squared_error(y_test, predictions)))
print("MAE:  %.5f " % mean_absolute_error(y_test, predictions))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

# # scores_df = scores....
# # scores_df.to_csv('data\\metrics\\scores.csv')

print('\nFin de train.py\n')