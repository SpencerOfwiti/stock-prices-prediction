# importing libraries
from data_processing import df, data, pd, np, plt, scaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# creating a dataframe
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

# setting date as index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

# splitting data into training and testing sets
dataset = new_data.values

train = dataset[0:987, :]
test = dataset[987:, :]

# scaling the data
scaled_data = scaler.fit_transform(dataset)

# converting data to x_train and y_train
x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i-60: i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# creating the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# compiling the model
model.compile(loss='mean_squared_error', optimizer='adam')

# fitting the model
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# predicting 246 values using past 60 from the train data
inputs = new_data[len(new_data) - len(test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

x_test = []
for i in range(60, inputs.shape[0]):
    x_test.append(inputs[i-60: i, 0])
x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
closing_price = model.predict(x_test)
closing_price = scaler.inverse_transform(closing_price)

# RMSE
rms = np.sqrt(np.mean(np.power((test - closing_price), 2)))
print(rms)

# plotting
train = new_data[:987]
test = new_data[987:]
test['Predictions'] = closing_price
plt.plot(train['Close'])
plt.plot(test[['Close', 'Predictions']])
plt.show()
