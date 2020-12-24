# Time Step configuration on Long Short-Term Memory Model for Taiwanese Stock Forecast

This code is the implementation of the paper **Time Step configuration on Long Short-Term Memory Model for Taiwanese Stock Forecast**

## Installation

Python version 3.6 and install the dependencies:

```
pip install -r requirements.txt
```

## Usage

We implemented Random Forest, Support Vector Regressor, Autoregression, and LSTM for forecasting the stock price.

```
python stock.py -b 128 -e 200 -m lstm -y 9 -t 1
```

The explanation of the parameters is explained as follow:

```
-b: the number of batch size, default value is 128
-e: the number of epochs, default value is 200
-m: model type, default is *lstm* - LSTM, the others are: *rf* - Random Forest, *svm* - Support Vector Regressor, and *autoreg* - Autoregression
-y: training years, the default value is 9, our erexperiments inspected on 9, 10, and 11.
-t: timesteps, the number of timestep for LSTM, the default value is 1.
```

## Authors
