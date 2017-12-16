import numpy as np
from scipy.optimize import curve_fit
import DataCleaner

class Baseline():
    def __init__(self, assets):
        self.assets = assets
        self.model = [1,1,1,0]
        self.window_size = 60
        self.x_train = []
        self.y_train = []
        self.x_pred = []
        self.y_pred = []
        self.purchase_cost = 0.0075
        self.sell_cost = 0.0075
        self.investment = self.assets*(1-self.purchase_cost)
        self.minimum_sell_value = self.investment/(1-self.sell_cost)
        self.actual_profit = 0
        self.purchase_value = 0
        
    def func(self,x,a,b,c,d):
        return (a*x*x*x) + (b*x*x) + (c*x) + d
     
    #X: 0-n
    #Y: stock value @ sample n (open0,close0,open1,close1,...,open_n,close_n)
    def train(self,X,Y):
        self.x_train = X
        self.y_train = Y
        self.x_pred = []
        self.y_pred = []
        self.model, self.covariance = curve_fit(self.func,X,Y)
    
    #n: number of 1/2 days (1 open + 1 close per day) since the end of most
    #recent training data to predict stock value for
    def predict(self, n):
        for i in range(len(n)):
            index = self.x_train[-1] + i
            self.x_pred.append(index)
            self.y_pred.append(self.func(index,*self.model))
            
    # returns:
    #   0: Strong buy
    #   1: Buy
    #   2: Hold
    #   3: Sell
    #   4: Strong sell
    def classify(self):
        mean = np.mean(self.y_train)
        std_dev = np.std(self.y_train)
        min_train = min(self.y_train)
        max_train = max(self.y_train)
        max_pred = max(self.y_pred)
        max_profit_margin = max_pred/self.y_train[-1]
        minimum_sell = self.y_train[-1]/(1-self.sell_cost)*(1-self.purchase_cost)
    
        if (min_train <= mean - 2*std_dev) and (self.y_train[-1] <= mean-1.8*std_dev) and (self.y_train[-1] > min_train) and (max_pred > minimum_sell):
            return 0
        elif (max_pred > minimum_sell):
            return 1
        elif (self.y_train[-1] >= minimum_sell) and (max_pred < self.y_train[-1]):
            if (max_train >= mean + 2*std_dev) and (self.y_train[-1] >= mean+1.8*std_dev) and (self.y_train[-1] <= max_train):
                return 4
            else:
                return 3
        else:
            return 2
            
def get_average_profit():
    assets = 1000 #investing with $1000
    base = Baseline(assets)
    cleaner = DataCleaner.DataCleaner()
    profits = []
    invested = False
    num_folds = 4
    window_size = 30
    for k in range(1,num_folds+1):
        training_data,cv_data = cleaner.get_clean_data(k)
        for index in range(0,len(training_data),window_size):
            window_data = training_data[index:index+window_size]
            open_values = []
            close_values = []
            for day in window_data:
                open_values.append(day[0])
                close_values.append(day[1])
            flat_window_data = []
            for index in range(len(open_values)):
                flat_window_data.append(open_values[index])
                flat_window_data.append(close_values[index])
            if len(flat_window_data) != 60:
                continue
            base.train(list(range(index,index+base.window_size)),flat_window_data)
            base.predict(list(range(index+base.window_size,index+(2*base.window_size))))
            if invested and (base.classify() == 4 ):
                invested = False
                profit = (base.y_train[-1] - base.purchase_value)*base.assets
                profits.append(profit)
                print("sold for ${} profit".format(profit))
            elif (not invested) and ((base.classify() == 0) or (base.classify() == 1)):
                base.purchase_value = base.y_train[-1]
                invested = True
                print("Invested")
    print("Profits: {}".format(profits))
    print("Total profits = ${}".format(sum(profits)))
    
if __name__ == "__main__":
    get_average_profit()