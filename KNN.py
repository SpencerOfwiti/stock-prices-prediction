# importing libraries
from data_processing import scaler, train, test, x_train, x_test, y_train, y_test, pd, np, plt
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV

# scaling data
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)

# using gridsearch to find the best parameter
params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

# fit the model and make predictions
model.fit(x_train, y_train)
preds = model.predict(x_test)

# RMSE
rms = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(preds)), 2)))
print(rms)

# plot
test['Predictions'] = preds
plt.plot(test[['Close', 'Predictions']])
plt.plot(train['Close'])
plt.show()
