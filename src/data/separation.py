import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

data_dir = Path("data/raw_data/raw.csv")

df = pd.read_csv(data_dir)

X = df.drop(["silica_concentrate"],axis = 1)
y = df[["silica_concentrate"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111)

data_processed_dir = Path("data/processed_data/separated/X_train.csv")
data_processed_dir.parent.mkdir(parents=True, exist_ok=True)
X_train.to_csv(data_processed_dir,index = False)

data_processed_dir = Path("data/processed_data/separated/X_test.csv")
data_processed_dir.parent.mkdir(parents=True, exist_ok=True)
X_test.to_csv(data_processed_dir,index = False)

data_processed_dir = Path("data/processed_data/separated/y_train.csv")
data_processed_dir.parent.mkdir(parents=True, exist_ok=True)
y_train.to_csv(data_processed_dir,index = False)

data_processed_dir = Path("data/processed_data/separated/y_test.csv")
data_processed_dir.parent.mkdir(parents=True, exist_ok=True)
y_test.to_csv(data_processed_dir,index = False)



