import pandas as pd
import numpy as np
from pytorch_forecasting import GroupNormalizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import datetime
from keras import optimizers
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

#Point to the LSTM Electricity dataset
df = pd.read_csv(r'LSTMDatasets\LSTMElectricity.csv')


weather_features = ['power_usage', 'days_from_start',  
                     "temperature", "apparent_temperature","precipitation","wind_speed"]

social_features = ['power_usage', 'days_from_start', 
                   "residents", "hometype_flat", "hometype_house", "build_era","income"]

calendar_features = ['power_usage', 'days_from_start',
             "day_4", "day_5", "day_6", "day_0", "day_1", "day_2", "day_3",
             "month_1", "month_2", "month_3", "month_4", "month_5", "month_6", "month_7", "month_8", "month_9", "month_10", "month_11",
             "month_12"]

calendar_social_features = ['power_usage', 'days_from_start', 
             "day_4", "day_5", "day_6", "day_0", "day_1", "day_2", "day_3",
             "month_1", "month_2", "month_3", "month_4", "month_5", "month_6", "month_7", "month_8", "month_9", "month_10", "month_11",
             "month_12", "residents", "hometype_flat", "hometype_house", "build_era","income"]

calendar_weather_features = ['power_usage', 'days_from_start', 
             "day_4", "day_5", "day_6", "day_0", "day_1", "day_2", "day_3",
             "month_1", "month_2", "month_3", "month_4", "month_5", "month_6", "month_7", "month_8", "month_9", "month_10", "month_11",
             "month_12", "temperature", "apparent_temperature","precipitation","wind_speed"]

calendar_weather_social_features = ['power_usage', 'days_from_start', 
             "day_4", "day_5", "day_6", "day_0", "day_1", "day_2", "day_3",
             "month_1", "month_2", "month_3", "month_4", "month_5", "month_6", "month_7", "month_8", "month_9", "month_10", "month_11",
             "month_12", "temperature", "apparent_temperature","precipitation","wind_speed", 
             "residents", "hometype_flat", "hometype_house", "build_era", "income"]
weather_social_features = ['power_usage', 'days_from_start', 
                     "temperature", "apparent_temperature","precipitation","wind_speed", 
                     "residents", "hometype_flat", "hometype_house", "build_era","income"]

no_features=['power_usage']


#Replace the following variable with which set of features you will like to train the model with
features = calendar_weather_social_features

temp_features = features + ['id']
df_temp = df[temp_features]
df_temp.fillna(method='ffill', inplace=True)
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_temp[features])
df_scaled = pd.DataFrame(df_scaled, columns=features)
df_scaled['id'] = df['id'].values
unique_ids = df_scaled['id'].unique()
test_dfs = []
train_dfs = []

for id in unique_ids:
    data = df_scaled[df['id']==id]
    single_train, single_test = train_test_split(data, test_size=0.2, shuffle=False)
    train_dfs.append(single_train)
    test_dfs.append(single_test)

train_df = pd.concat(train_dfs).reset_index(drop=True)
test_df = pd.concat(test_dfs).reset_index(drop=True)

def create_sequences_grouped_by_home_id(df_scaled, n_input, feature_columns, target_column):
    X, y = [], []
    for home_id in df_scaled['id'].unique():
         house_data = df_scaled[df_scaled['id'] == home_id]
         for i in range(len(house_data) - n_input):
             end_ix = i + n_input
             seq_x = house_data.iloc[i:end_ix][feature_columns].values
             seq_y = house_data.iloc[end_ix][target_column]
             X.append(seq_x)
             y.append(seq_y)
    return np.array(X), np.array(y)


feature_columns = features  
target_column = 'power_usage'
n_input = 50

# Create sequences
X_train, y_train = create_sequences_grouped_by_home_id(train_df, n_input=n_input, feature_columns=feature_columns, target_column=target_column)
X_test, y_test = create_sequences_grouped_by_home_id(test_df,  n_input=n_input, feature_columns=feature_columns, target_column=target_column)



n_features = len(features) 
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#Save to a location of your choice
checkpoint_filepath = 'ElectrictyTest.keras'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_best_only=True)

model = Sequential()
model.add(LSTM(200, return_sequences=True ,input_shape=(n_input, n_features)))
model.add(LSTM(200))
model.add(Dropout(0.1))
model.add(Dense(1))
gd = optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=gd, loss='mse')

#Comment the next line if you want to test a saved model
model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=[model_checkpoint_callback, early_stopping], validation_data=(X_test, y_test))

#If you would like to test a previous model, change the path and uncomment the code below
#model = keras.models.load_model(r"C:\Users\sukhy\Documents\Thesis\Models\LSTM\Electricity\Calendar+Weather+Social\calendar+weather+social_one.keras")

predictions = model.predict(X_test)


dummy_array_for_inverse = np.zeros((predictions.shape[0], len(features))) 
dummy_array_for_inverse[:, 0] = predictions.flatten()
predictions_rescaled = scaler.inverse_transform(dummy_array_for_inverse)[:, 0]
dummy_array_for_y_test = np.zeros((y_test.shape[0], len(features)))  
dummy_array_for_y_test[:, 0] = y_test.flatten()  

y_test_rescaled = scaler.inverse_transform(dummy_array_for_y_test)[:, 0]
mse = mean_squared_error(y_test_rescaled, predictions_rescaled, squared=False)
mape = mean_absolute_percentage_error(y_test_rescaled, predictions_rescaled)

print(f"Mean Absolute Percentage Error: {mape}")
print(f"Root Mean Squared Error: {mse}")



