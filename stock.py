
from optparse import OptionParser
import configparser
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv, read_excel
import math
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
import glob
import os
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import pandas as pd
from keras_sequential_ascii import keras2ascii
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from statsmodels.tsa.ar_model import AutoReg

from visualize import plot_stock_price
from bokeh.plotting import figure, ColumnDataSource
from bokeh.io import export_png

os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)

# convert an array of values into a dataset matrix
def create_dataset(dataset, timesteps=1):
	dataX, dataY = [], []
	r = len(dataset)-timesteps-1
	for i in range(r):
		a = dataset[i:(i+timesteps), 0]
		dataX.append(a)
		dataY.append(dataset[i + timesteps, 0])
	X = np.array(dataX)
	return X, np.array(dataY)
	
# fix random seed for reproducibility
np.random.seed(7)


def create_model(timesteps):
	model = Sequential()
	model.add(LSTM(options.units, return_sequences = True, input_shape=(1, timesteps)))
	model.add(Dropout(0.1))
	model.add(LSTM(units = 50, return_sequences = True))
	model.add(LSTM(units = 50))
	model.add(Dense(1))
	model.summary()
	model.compile(loss='mean_squared_error', optimizer='adam', metrics = [tf.keras.metrics.RootMeanSquaredError()])
	keras2ascii(model) 
	return model

def create_rf(rf_n_estimators=500,rf_max_depth=-1,rf_min_samples_split=3 , rf_random_state = None, rf_max_features=5):
	model = RandomForestRegressor(n_estimators=rf_n_estimators, min_samples_split=rf_min_samples_split, random_state= rf_random_state, max_features='auto')
	return model

def create_svm():
	model = SVR(kernel='rbf', C=1, gamma=2)
	return model

def create_autoreg(train):
	model = AutoReg(train, lags=29)
	return model

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))

save_path = os.path.join(os.getcwd(), "stock", "results")

if not os.path.isdir(save_path):
	os.mkdir(save_path)

def run(model_type, training_years=9, timesteps=1):
	prefix_dataset = ['TAIEX']
	for prefix in prefix_dataset:
		print("Creating data from {}".format(prefix))
		_prefix = os.path.join(os.getcwd(), "stock", "data",'{}*'.format(prefix))
		file_names = glob.glob(_prefix)
		# create dataset
		stock_prices = np.array([])
		dates = np.array([])
		vis_stock = np.array([])
		real_test = np.array([])
		real_date = np.array([])
		train_size = 0
		for index,file_name in enumerate(file_names):
			dataset = read_excel(file_name, header=None, usecols=[1])
			date = read_excel(file_name, header=None, usecols=[0])
			date = date.astype('str').values.tolist()
			date = [x[0] for x in date]
			_stock_prices = dataset.astype('float32').to_numpy()
			stock_prices = np.append(stock_prices, _stock_prices)
			if train_size !=0:
				real_test = np.append(real_test, _stock_prices)
				real_date = np.append(real_date, date)
			if index == training_years:
				train_size = len(stock_prices)
			dates = np.append(dates, date)
			
		test_size = len(stock_prices) - train_size
		stock_prices = stock_prices.reshape(-1,1)
		stock_prices = scaler.fit_transform(stock_prices)
		train, test = stock_prices[0:train_size,:], stock_prices[train_size:len(stock_prices),:]

		trainX, trainY = create_dataset(train, timesteps)
		testX, testY = create_dataset(test, timesteps)
		if model_type == 'lstm':
			# reshape input to be [samples, features, time steps]
			trainX = trainX.reshape(trainX.shape[0], 1, timesteps)
			testX = testX.reshape(testX.shape[0], 1, timesteps)
			model = create_model(timesteps)
			
			history_callback = model.fit(trainX, trainY, batch_size=options.batchsize, 
			verbose=2, epochs = options.epoch,
			validation_data=(testX, testY))
			
			trainPredict = model.predict(trainX)
			testPredict = model.predict(testX)
			# invert predictions
			trainPredict = scaler.inverse_transform(trainPredict)
			trainY = scaler.inverse_transform([trainY])
			testPredict = scaler.inverse_transform(testPredict)
			testY = scaler.inverse_transform([testY])
			# calculate root mean squared error
			trainScore_mae = mean_absolute_error(trainY[0], trainPredict[:,0])
			trainScore_rmse = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
			print("LSTM on {} training years".format(training_years))
			print('Train Score: RMSE= ' + str (trainScore_rmse) + ' mae='+ str (trainScore_mae))
			testScore_rmse = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
			testScore_mae = mean_absolute_error(testY[0], testPredict[:,0])
			print('Test Score: RMSE= '+ str (testScore_rmse) + ' mae='+str (testScore_mae))
			
			train_loss = history_callback.history['loss']
			val_loss = history_callback.history['val_loss']

			plt.plot(train_loss)
			plt.plot(val_loss)
			plt.title('Long Short-term Memory - Model loss')
			plt.ylabel('Loss')
			plt.xlabel('Epoch')
			plt.legend(['Training loss', 'Validation loss'], loc='upper right')
			img_path = os.path.join(save_path,'{}_training_years_val_loss_'.format(training_years) + '_{}_{}'.format(prefix, index) + '_{}_'.format('LSTM') + '_testing_' + '.png')
			plt.savefig(img_path)
			plt.figure().clear()
		else:
			trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
			testX = np.reshape(testX, (testX.shape[0], testX.shape[1]))
			trainY = trainY.ravel()
			testY = testY.ravel()
			
			if model_type == 'rf':
				model = create_rf()
			elif model_type == 'svm':
				model = create_svm()
			elif model_type == 'autoreg':
				trainX = np.squeeze(trainX)
				testX = np.squeeze(testX)
				model = create_autoreg(trainX)
			else:
				raise("Model exeption")

			if model_type in ['rf', 'svm']:
				model.fit(trainX, trainY)
				trainPredict = model.predict(trainX)
				testPredict = model.predict(testX)
			if model_type == 'autoreg':
				model = model.fit()
				coef = model.params
				# walk forward over time steps in test
				window = 29
				history = train[len(trainX)-window:]
				history = [history[i] for i in range(len(history))]
				testPredict = []
				trainPredict = []
				for t in range(len(testX)):
					length = len(history)
					lag = [history[i] for i in range(length-window,length)]
					yhat = coef[0]
					for d in range(window):
						yhat += coef[d+1] * lag[window-d-1]
					obs = testX[t]
					yhat = np.squeeze(yhat)
					yhat = np.asscalar(yhat)
					testPredict.append(yhat)
					history.append(obs)
				for t in range(len(trainX)):
					length = len(history)
					lag = [history[i] for i in range(length-window,length)]
					yhat = coef[0]
					for d in range(window):
						yhat += coef[d+1] * lag[window-d-1]
					obs = trainX[t]
					yhat = np.squeeze(yhat)
					yhat = np.asscalar(yhat)
					trainPredict.append(yhat)
					history.append(obs)

			# invert predictions
			trainPredict = np.reshape(trainPredict, (-1,1))
			testPredict = np.reshape(testPredict, (-1,1))
			trainY = trainY.reshape((-1,1))
			testY = testY.reshape((-1,1))
			trainPredict = scaler.inverse_transform(trainPredict)
			trainY = scaler.inverse_transform(trainY)
			testPredict = scaler.inverse_transform(testPredict)
			testY = scaler.inverse_transform(testY)
			# calculate root mean squared error
			trainScore_mae = mean_absolute_error(trainY, trainPredict)
			trainScore_rmse = math.sqrt(mean_squared_error(trainY, trainPredict))

			if model_type == 'rf':
				print("RANDOM FOREST on {} training years".format(training_years))
			elif model_type == 'svm':
				print("SVR on {} training years".format(training_years))
			elif model_type == 'autoreg':
				print("AutoReg on {} training years".format(training_years))

			print('Train Score: RMSE= ' + str (trainScore_rmse) + ' mae='+ str (trainScore_mae))
			testScore_rmse = math.sqrt(mean_squared_error(testY, testPredict))
			testScore_mae = mean_absolute_error(testY, testPredict)
			print('Test Score: RMSE= '+ str (testScore_rmse) + ' mae='+str (testScore_mae))

		print("saving log to file")

		from time import gmtime, strftime
		res_sum = np.c_[ (
			trainScore_mae,
			trainScore_rmse,
			testScore_mae,
			testScore_rmse,
		)]

		if model_type == 'lstm':
			model_name = 'lstm_timestep_{}'.format(timesteps)
		else:
			model_name = model_type + '_{}'.format(timesteps)

		time_text = str(strftime("%Y%m%d_%H%M%S", gmtime()))
		txt = os.path.join(save_path,'{}_training_years_'.format(training_years) + '_{}_{}'.format(prefix, index) + '_{}_'.format(model_name) + '.txt')
		f=open(txt ,'w')
		np.savetxt(f,(options,args), fmt="%s", delimiter="\t")
		title_cols = np.array([['trainScore_mae','trainScore','testScore_mae','testScore']])
		np.savetxt(f,title_cols, fmt="%s",delimiter="\t")
		np.savetxt(f,res_sum,delimiter='\t', fmt='%s')
		f.close()
		if model_type == 'lstm':
			model_path = os.path.join(save_path, "model_{}_{}.h5".format(model_name, prefix))
			print("save to {}".format(model_path))
			model.save(model_path)
		elif model_type == 'rf' or model_type == 'svm':
			filename = os.path.join(save_path, "model_{}_{}.sav".format(model_name, prefix))
			pickle.dump(model, open(filename, 'wb'))

		trainPredictPlot = np.empty_like(stock_prices)
		trainPredictPlot[:, :] = np.nan
		trainPredictPlot[timesteps:len(trainPredict)+timesteps, :] = trainPredict
		# shift test predictions for plotting
		testPredictPlot = np.empty_like(stock_prices)
		testPredictPlot[:, :] = np.nan
		testPredictPlot[len(trainPredict)+(timesteps*2)+1:len(stock_prices)-1, :] = testPredict

		img_path = os.path.join(save_path,'{}_training_years_'.format(training_years) + '_{}_{}'.format(prefix, index) + '_{}_'.format(model_name) + '.png')
		plt.savefig(img_path)
		plt.figure().clear()
		if model_type == 'lstm' or model_type == 'autoreg':
			testX = testX.reshape(-1,1)
			testPredict = testPredict.reshape(-1,1)
		test_price = np.squeeze(scaler.inverse_transform(testX))
		predicted_price = np.squeeze(testPredict)
		real_date = np.delete(real_date, 0, 0)
		real_date = np.delete(real_date, -1, 0)

		df_stock = np.squeeze(np.dstack((real_date, np.squeeze(test_price), np.squeeze(predicted_price))))
		df_stock = pd.DataFrame(df_stock, columns=['Date', 'Price', 'Predicted'])
		stock = ColumnDataSource(data=dict(Date=[], Price=[], Predicted=[]))
		stock.data = stock.from_df(df_stock)
		p = plot_stock_price(stock)
		print("")


if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option('-b','--batchsize', type="int",default = 128, help='specify the batch size')
	parser.add_option('-e','--epoch', type="int",default = 200, help='specific number of epoch')
	parser.add_option('-u','--units', type="int",default = 50, help='units for lstm')
	parser.add_option('-m','--model_type', default = 'lstm', help='model type, support: lstm, svm, rf, autoreg')
	parser.add_option('-y','--training_years',type="int", default = 9, help='number of training years, 9, 10, and 11')
	parser.add_option('-t','--timesteps',type="int", default = 1, help='timestep for lstm, default = 1')

	(options, args) = parser.parse_args()
	run(model_type=options.model_type, training_years=options.training_years, timesteps=options.timesteps)