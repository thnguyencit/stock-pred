# Time Step configuration on Long Short-Term Memory Model for Taiwanese Stock Forecast

This code is the implementation of the paper **An Effective Way for Taiwanese Stock Price prediction: Boosting the performance with Machine Learning Techniques**

Please cite this articale, if the code can help you.

Nguyen HT, Tran TB, Bui PHD. An effective way for Taiwanese stock price prediction: Boosting the performance with machine learning techniques. Concurrency Computat Pract Exper. 2021;e6437. https://doi.org/10.1002/cpe.6437

or 
```
@article{Nguyen2021,
  doi = {10.1002/cpe.6437},
  url = {https://doi.org/10.1002/cpe.6437},
  year = {2021},
  month = jun,
  publisher = {Wiley},
  author = {Hai T. Nguyen and Toan B. Tran and Phuong H. D. Bui},
  title = {An effective way for Taiwanese stock price prediction: Boosting the performance with machine learning techniques},
  journal = {Concurrency and Computation: Practice and Experience}
}
```
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

![](https://komarev.com/ghpvc/?username=thnguyencit)

