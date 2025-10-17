import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path



data_dir = Path("data/processed_data/separated/X_train.csv")
X_train = pd.read_csv(data_dir)

data_dir = Path("data/processed_data/separated/X_test.csv")
X_test = pd.read_csv(data_dir)



X_train.drop(["date"],axis = 1,inplace = True)
X_test.drop(["date"],axis = 1,inplace = True)


X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()


scaler = MinMaxScaler()

X_train_scaled[X_train_scaled.columns] = scaler.fit_transform(X_train)
X_test_scaled[X_test_scaled.columns] = scaler.fit_transform(X_test)


data_processed_dir = Path("data/processed_data/normalized/X_train_scaled.csv")
data_processed_dir.parent.mkdir(parents=True, exist_ok=True)
X_train_scaled.to_csv(data_processed_dir,index = False)

data_processed_dir = Path("data/processed_data/normalized/X_test_scaled.csv")
data_processed_dir.parent.mkdir(parents=True, exist_ok=True)
X_test_scaled.to_csv(data_processed_dir,index = False)
