# Modelling Volatile Time Series with LSTM Networks

Here is an illustration of how a long-short term memory network (LSTM) can be used to model a volatile time series.

Yearly rainfall data can be quite volatile. Unlike temperature, which typically demonstrates a clear trend through the seasons, rainfall as a time series can be quite volatile. In Ireland, it is not uncommon for summer months to see as much rain as that of winter months.

Here is a graphical illustration of rainfall patterns from November 1959 for Newport, Ireland:

(rainfall)

Using a standard ARIMA model on volatile data such as this is typically not sufficient, as the inherent volatility in the series leads to wide confidence intervals in the forecast.

(arima)

However, as a sequential neural network, LSTM models can prove superior in accounting for the volatility in a time series.

## Data Manipulation and Model Configuration

x
