import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import pickle


data_dir = Path("data/processed_data/normalized/X_train_scaled.csv")
X = pd.read_csv(data_dir)

data_dir = Path("data/processed_data/separated/y_train.csv")
y = pd.read_csv(data_dir)

parameters = {'n_estimators':[10,50,100,200,500],
    'criterion':["squared_error", "absolute_error", "friedman_mse", "poisson"],
    'min_samples_split':[2,4,10,20]}

rfr = RandomForestRegressor()

clf = GridSearchCV(rfr,parameters)

clf.fit(X,np.ravel(y))

best_estimator = clf.best_estimator_

model_path = Path("models/meilleurs_params.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)
with open(model_path,"wb") as f:
    pickle.dump(best_estimator,f)

    