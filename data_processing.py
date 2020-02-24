# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from fastai.tabular.transform import add_datepart

# setting figure size
rcParams['figure.figsize'] = 20, 10

# setting plotting styles
style.use('fivethirtyeight')

# normalizing data
scaler = MinMaxScaler(feature_range=(0, 1))

# read the file
df = pd.read_csv('NSE-TATAGLOBAL11.csv')

# setting index as date
df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']

# shape of data
print('\n Shape of the data:')
print(df.shape)

# first five rows
print(df.head())

# plot
plt.figure(figsize=(16, 8))
plt.plot(df['Close'], label='Close Price History')
plt.show()

# sort dataset in ascending order
data = df.sort_index(ascending=True, axis=0)

# creating dataframe with date and the target variable
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

# create date features
add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis=1, inplace=True)  # elapsed will be the time stamp

# distinguishing mondays and fridays from other dates
new_data['mon_fri'] = 0
for i in range(0, len(new_data)):
    if new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4:
        new_data['mon_fri'][i] = 1
    else:
        new_data['mon_fri'][i] = 0

# splitting data into train and test data
train = new_data[:987]
test = new_data[987:]

# shape of training data
print('\n Shape of training data:')
print(train.shape)

# shape of testing data
print('\n Shape of testing data:')
print(test.shape)

# splitting data for normalization
x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_test = test.drop('Close', axis=1)
y_test = test['Close']

