# importing libraries
from data_processing import new_data, train, test, np, plt, x_train, x_test, y_train, y_test
from sklearn.linear_model import LinearRegression

# implement linear regression
regr = LinearRegression()
regr.fit(x_train, y_train)

# make predictions and find RMSE
preds = regr.predict(x_test)
rms = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)), 2)))
print(rms)

# plot
test['Predictions'] = preds

test.index = new_data[987:].index
train.index = new_data[:987].index

plt.plot(train['Close'])
plt.plot(test[['Close', 'Predictions']])
plt.show()
