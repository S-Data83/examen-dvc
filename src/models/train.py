import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import pickle

model_path = Path("models/params/meilleurs_params.pkl")

with open(model_path,"rb") as f:
    rfr = pickle.load(f)

data_dir = Path("data/processed_data/normalized/X_train_scaled.csv")
X = pd.read_csv(data_dir)

data_dir = Path("data/processed_data/separated/y_train.csv")
y = pd.read_csv(data_dir)


rfr.fit(X,np.ravel(y))

model_path = Path("models/entraine/modele_entraine.pkl")

with open(model_path,"wb") as f:
    pickle.dump(rfr,f)