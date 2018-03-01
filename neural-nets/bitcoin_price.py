import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../data/kaggle/bitcoin_price.csv',parse_dates=['Date'])
df = df[['Date', 'Close']].sort_values(['Date'], ascending=True)

#create historical data
def create_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


x, y = create_dataset(df['Close'])
print (x.shape)
print (y.shape)
print(x)

size = int(len(x) * 0.70)

x_train, x_test = x[0:size], x[size:len(x)]
y_train, y_test = y[0:size], y[size:len(x)]


print ('x_train',x_train.shape, x_train[:5],y_train[:5])
print ('x_test',x_test.shape, x_test[:5],y_test[:5])
