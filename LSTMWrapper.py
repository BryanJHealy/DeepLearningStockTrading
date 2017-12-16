from LSTM_K import LSTM_1
import sys

class LSTMWrapper():
	def __init__(self,weights=None):
		num_cells = [30,60,90]	
		window_sizes = [30,60,90]
		epochs = [10,1,100]
		dropouts = [0.3,0.15,0.5]
		activation_functions = ['relu','tanh','sigmoid']
		optimizers = ['adam','rmsprop','sgd']
								
		index = 3
		epoch = 5
		self.window = 30
		n_cells = 60
		drop = 0.3
		opt = optimizers[0]
		activation = activation_functions[0]
		sys.stdout = open('data{}.txt'.format(index), 'w')
		print('Number of LSTM cells: {}\nSequence length: {}\nEpochs per fold: {}\nDropout: {}\nActivation function: {}\nOptimizer: {}\n'.format(n_cells,self.window,epoch,drop,activation,opt))
		self.model = LSTM_1(2,num_cells=n_cells,sequence_length=self.window,dropout=drop,epochs_per_fold=epoch,activation_function=activation,optimizer=opt)
		if weights == None:
			self.model.train()
		else:
			self.model.model.load_weights(weights)
		index +=1

		# index = 0
		# for opt in optimizers:
			# for activation in activation_functions:
				# for drop in dropouts:
					# for epoch in epochs:
						# for window in window_sizes:
							# for n_cells in num_cells:
								# sys.stdout = open('data{}.txt'.format(index), 'w')
								# print('Number of LSTM cells: {}\nSequence length: {}\nEpochs per fold: {}\nDropout: {}\nActivation function: {}\nOptimizer: {}\n'.format(n_cells,window,epoch,drop,activation,opt))
								# model = LSTM_1(2,num_cells=n_cells,sequence_length=window,dropout=drop,epochs_per_fold=epoch,activation_function=activation,optimizer=opt)
								# model.train()
								# index +=1
		self.purchase_value = 0
	
	
# returns:
    #   0: Strong buy
    #   1: Buy
    #   2: Hold
    #   3: Sell
    #   4: Strong sell
	def classify(self,index):
		historic = self.model.test_data[index]
		predicted_max = self.model.predictions[index][1]
		sell_cost = 0.0075
		purchase_cost = 0.0075
		current = historic[-1]
		mean = historic.mean()
		std_dev = historic.std()
		min_train = historic.min()
		max_train = historic.max()
		max_pred = predicted_max
		max_profit_margin = max_pred/current
		minimum_sell = current/(1-sell_cost)*(1-purchase_cost)

		if (min_train <= mean - 2*std_dev) and (current <= mean-1.8*std_dev) and (current > min_train) and (max_pred > minimum_sell):
			return 0
		elif max_pred > minimum_sell:
			return 1
		elif (current >= minimum_sell) and (max_pred < current):
			if (max_train >= mean + 2*std_dev) and (current >= mean+1.8*std_dev) and (current <= max_train):
				return 4
			else:
				return 3
		else:
			return 2
				
	def get_average_profit(self):
		self.model.test()
		assets = 1000 #investing with $1000
		profits = []
		invested = False
		for index in range(len(self.model.test_data)):
			classification = self.classify(index)
			#if invested and (classification == 4) and (classification == 3):
			if invested and (classification == 4):
				invested = False
				profit = (self.model.test_data[index][-1] - self.purchase_value)*assets
				assets += profit
				profits.append(profit)
				print("sold for ${} profit".format(profit))
			#elif (not invested) and (classification == 0):
			elif (not invested) and ((classification == 0) or (classification == 1)):
				self.purchase_value = self.model.test_data[index][-1]
				invested = True
				print("Invested ${}".format(assets))
		print("Profits: {}".format(profits))
		print("Total profits = ${}".format(sum(profits)))
		
if __name__ == '__main__':
	#wrapper = LSTMWrapper(weights='model_weights')
	wrapper = LSTMWrapper()
	wrapper.get_average_profit()