# Modelling Volatile Time Series with LSTM Networks

Here is an illustration of how a long-short term memory network (LSTM) can be used to model a volatile time series.

Yearly rainfall data can be quite volatile. Unlike temperature, which typically demonstrates a clear trend through the seasons, rainfall as a time series can be quite volatile. In Ireland, it is not uncommon for summer months to see as much rain as that of winter months.

Here is a graphical illustration of rainfall patterns from November 1959 for Newport, Ireland:

(1)

Using a standard ARIMA model on volatile data such as this is typically not sufficient, as the inherent volatility in the series leads to wide confidence intervals in the forecast.

(2)

However, as a sequential neural network, LSTM models can prove superior in accounting for the volatility in a time series.

## Data Manipulation and Model Configuration

The dataset in question comprises of 722 months of rainfall data.

712 data points are selected for training and validation purposes, i.e. to build the LSTM model. Then, the last 10 months of data are used as test data to compare with the predictions from the LSTM model.

Here is a snippet of the dataset:

(3)

A dataset matrix is then formed in order to regress the time series against past values:

```
# Form dataset matrix
def create_dataset(df, previous=1):
    dataX, dataY = [], []
    for i in range(len(df)-previous-1):
        a = df[i:(i+previous), 0]
        dataX.append(a)
        dataY.append(df[i + previous, 0])
    return np.array(dataX), np.array(dataY)
```

The data is then normalized with MinMaxScaler:

(4)

With the *previous* parameter set to 120, the training and validation datasets are created. For reference, *previous = 120* means that the model is using past values from *t - 120* down to *t - 1* to predict the rainfall value at time *t*.

The choice of the *previous* parameter is subject to trial and error, but 120 time periods were chosen to ensure capture of the volatility or extreme values demonstrated by the time series.

```
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

# Training and Validation data partition
train_size = int(len(df) * 0.8)
val_size = len(df) - train_size
train, val = df[0:train_size,:], df[train_size:len(df),:]

# Number of previous
previous = 120
X_train, Y_train = create_dataset(train, previous)
X_val, Y_val = create_dataset(val, previous)
```

The inputs are then reshaped to be in the format of *samples, time steps, features*.

```
# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

# Generate LSTM network
model = tf.keras.Sequential()
model.add(LSTM(4, input_shape=(1, previous)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history=model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=448, verbose=2)


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
```

The model is trained across 100 epochs, and a batch size of 448 (equal to the number of data points in the training set) is specified.

