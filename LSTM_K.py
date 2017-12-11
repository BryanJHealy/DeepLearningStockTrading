import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

import DataCleaner

class LSTM_1():
    def __init__(self,num_outputs,num_cells=42,sequence_length=30,dropout=0.0,epochs_per_fold=10,activation_function='relu',optimizer='adam'):
        self.num_outputs = num_outputs
        self.cleaner = DataCleaner.DataCleaner()
        self.sequence_length = sequence_length
        self.epochs = epochs_per_fold
        
        self.batch_size = 1
        self.model = Sequential()
        self.model.add(LSTM(num_cells,input_shape=(None,1),activation=activation_function))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(2))
        self.model.compile(optimizer=optimizer,loss='mse')
        
    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)
    
    def train(self):
        for fold in range(1,5):
            print('Cross Validation Fold #', fold)
            training_data,crossv_data = self.cleaner.get_clean_data(fold)
            train_data = []
            for index in range(len(training_data)):
                train_data.append(training_data[index][0])
                train_data.append(training_data[index][1])
            train_data = np.array(train_data).reshape([len(train_data),1])
            trainX, trainY = self.create_dataset(train_data, look_back=self.sequence_length)
            trainY = trainX[self.sequence_length:,:]
            trainX = trainX[:-self.sequence_length,:]
            trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
            trainy = []
            for index in range(len(trainY)):
                trainy.append([trainY[index].min(),trainY[index].max()])
            trainY = np.array(trainy)
            
            cv_data = []
            for index in range(len(crossv_data)):
                cv_data.append(crossv_data[index][0])
                cv_data.append(crossv_data[index][1])
            cv_data = np.array(cv_data).reshape([len(cv_data),1])
            cvX, cvY = self.create_dataset(cv_data, look_back=self.sequence_length)
            cvY = cvX[self.sequence_length:,:]
            cvX = cvX[:-self.sequence_length,:]
            cvX = np.reshape(cvX, (cvX.shape[0], cvX.shape[1], 1))
            cvy = []
            for index in range(len(cvY)):
                cvy.append([cvY[index].min(),cvY[index].max()])
            cvY = np.array(cvy)
            
            self.model.fit(trainX, trainY, epochs=self.epochs, batch_size = self.batch_size, verbose=1, shuffle=False)
            train_score = self.model.evaluate(trainX, trainY, batch_size = self.batch_size, verbose=0)
            print('Train score: ', train_score)
            cv_score = self.model.evaluate(cvX, cvY, batch_size = self.batch_size, verbose=0)
            print('Cross Validation score: ', cv_score)
        
if __name__ == '__main__':
    model = LSTM_1(1,30,2)
    model.train()