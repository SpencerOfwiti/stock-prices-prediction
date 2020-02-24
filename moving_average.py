# importing data_processing
from data_processing import train, test, np, plt

# moving average
# create predictions for test set
# check RMSE using the actual values
preds = []
for i in range(0, test.shape[0]):
    a = train['Close'][len(train) - 248 + i:].sum() + sum(preds)
    b = a / 248
    preds.append(b)

# checking RMSE value
rms = np.sqrt(np.mean(np.power((np.array(test['Close']) - preds), 2)))
print('\n RMSE value on test set:')
print(rms)

# plot
test['Predictions'] = preds
plt.plot(train['Close'])
plt.plot(test[['Close', 'Predictions']])
plt.show()
