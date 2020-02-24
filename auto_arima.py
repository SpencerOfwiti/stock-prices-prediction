# importing libraries
from data_processing import data, pd, np, plt
from pyramid.arima import auto_arima

# splitting into training and testing data
train = data[:987]
test = data[987:]

training = train['Close']
testing = test['Close']

# creating the model
model = auto_arima(training, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1,
                   trace=True, error_action='ignore', suppress_warnings=True)

# fitting the model
model.fit(training)

# using the model to make predictions
forecast = model.predict(n_periods=248)
forecast = pd.DataFrame(forecast, index=test.index, columns=['Prediction'])

# RMSE
rms = np.sqrt(np.mean(np.power((np.array(test['Close']) - np.array(forecast['Prediction'])), 2)))
print(rms)

# plot
plt.plot(train['Close'])
plt.plot(test['Close'])
plt.plot(forecast['Prediction'])
plt.show()
