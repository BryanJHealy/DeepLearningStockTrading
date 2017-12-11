from LSTM_K import LSTM_1
import sys

# class LSTMWrapper():
	# def __init__(self):
		# self.num_inputs = 4
		# self.sequence_length = 30
		# self.num_outputs = 2
		# self.model = LSTM_K.LSTM_1(num_inputs=4,sequence_length=30,num_outputs=2)
		# self.data = DataCleaner.DataCleaner()
		

num_cells = [30,60,90]	
window_sizes = [30,60,90]
epochs = [10,1,100]
dropouts = [0.3,0.15,0.0]
activation_functions = ['relu','tanh','sigmoid']
optimizers = ['adam','rmsprop','sgd']

index = 0
for opt in optimizers:
	for activation in activation_functions:
		for drop in dropouts:
			for epoch in epochs:
				for window in window_sizes:
					for n_cells in num_cells:
						sys.stdout = open('data{}.txt'.format(index), 'w')
						print('Number of LSTM cells: {}\nSequence length: {}\nEpochs per fold: {}\nDropout: {}\nActivation function: {}\nOptimizer: {}\n'.format(n_cells,window,epoch,drop,activation,opt))
						model = LSTM_1(2,num_cells=n_cells,sequence_length=window,dropout=drop,epochs_per_fold=epoch,activation_function=activation,optimizer=opt)
						model.train()
						index +=1