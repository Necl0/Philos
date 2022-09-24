from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy
import pandas as pd
import pickle

# read in dataset
df = pd.read_csv(r"C:\Users\josmo\Downloads\GroundStateEnergyDataset\roboBohr.csv")

df.pop("pubchem_id")
df.pop("Unnamed: 0")

# initialize feature scaler
scaler = MinMaxScaler(feature_range=(-1, 1))

for column in df.columns:
    scaler.fit(df[column].values.reshape(-1, 1))
    df[column] = scaler.transform(df[column].values.reshape(-1, 1))

# shuffle dataset
df = df.sample(frac=1)

y = df.pop('Eat')

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=25)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f'Accuracy on test set: {acc*100}')

with open('model.pkl', 'wb') as files:
    pickle.dump(model, files)
