# importing libraries
from data_processing import df, data, pd, np, plt
from fbprophet import Prophet

# creating a dataframe
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

new_data['Date'] = pd.to_datetime(new_data.Date, format='%Y-%m-%d')
new_data.index = new_data['Date']

# preparing data
new_data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

# splitting into training and testing data
train = new_data[:987]
test = new_data[987:]

# Creating and fitting the model
model = Prophet()
model.fit(train)

# predictions
close_prices = model.make_future_dataframe(periods=len(test))
forecast = model.predict(close_prices)

# RMSE
forecast_test = forecast['yhat'][987:]
rms = np.sqrt(np.mean(np.power((np.array(test['y']) - np.array(forecast_test)), 2)))
print(rms)

# plot
test['Predictions'] = forecast_test.values
plt.plot(train['y'])
plt.plot(test[['y', 'Predictions']])
plt.show()
