#%%
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

row_limit = 10000

# Columns
time_columns = ['FIRSTBAG', 'LASTBAG', 'BLOCKTIME',
                'RUNWAYTIME', 'SCHEDULEDBLOCKTIME']
category_columns = ['RACETRACK', 'AIRLINE',
                    'GATE', 'ACGRP', 'HANDLER_ID', 'FLIGHT_TYPE', 'DESTINATION', 'WEEKDAY', 'HOUR']
id_columns = ['ID', 'FLIGHT_NO']

# Load a limited amount
data = pd.read_csv('flight_arrival.csv', sep=',',
                   parse_dates=time_columns, nrows=row_limit)

# Calculate time differences
data['FIRSTBAGDELTA'] = (
    data['FIRSTBAG'] - data['BLOCKTIME']).dt.total_seconds()

data['LASTBAGDELTA'] = (
    data['LASTBAG'] - data['BLOCKTIME']).dt.total_seconds()

# Add weekday and hour
data['WEEKDAY'] = data['BLOCKTIME'].dt.weekday
data['HOUR'] = data['BLOCKTIME'].dt.hour

# One hot encode
data = pd.get_dummies(data, columns=category_columns)

data = data.drop(time_columns + id_columns, axis=1)
#%%
# Split into data and target variable.
X = data.drop('FIRSTBAGDELTA', axis=1)
y = data['FIRSTBAGDELTA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Prediction model
lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

# Visualize it
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')

print('Score:', model.score(X_test, y_test))

#%%
data.to_csv('flight_arrival_cleaned.csv')
