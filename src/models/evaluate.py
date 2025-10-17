import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from pathlib import Path
import pickle
import json


model_path = Path("models/modele_entraine.pkl")

with open(model_path,"rb") as f:
    rfr = pickle.load(f)


data_dir = Path("data/processed_data/X_test_scaled.csv")
X = pd.read_csv(data_dir)

data_dir = Path("data/processed_data/y_test.csv")
y = pd.read_csv(data_dir)


y_pred = rfr.predict(X)

MAE = mean_absolute_error(np.ravel(y),y_pred)
MSE = mean_squared_error(np.ravel(y),y_pred)
R2 = r2_score(np.ravel(y),y_pred)
RMSE = root_mean_squared_error(np.ravel(y),y_pred)

data_processed_dir = Path("data/processed_data/y_pred.csv")
data_processed_dir.parent.mkdir(parents=True, exist_ok=True)
np.savetxt(data_processed_dir, y_pred, delimiter=',')

dic = {"MAE":MAE,"MSE":MSE,"R2":R2,"RMSE":RMSE}

scores_path = Path("metrics/scores.json")
with open(scores_path,"w") as f:
    json.dump(dic,f)