import numpy as np
import pandas as pd

class sample_rate(Enum):
	daily = 0
	weekly = 1
	monthly = 2
	annual = 3
	
class Stock():
	def __init__(ticker_symbol):
		print('\n')
	
class Statistics():
	def __init__():
		self.risk_free_return_rate = 0 #approximately

		self.daily_sharpe_constant = np.sqrt(252)
		self.weekly_sharpe_constant = np.sqrt(52)
		self.monthly_sharpe_constant = np.sqrt(12)
		self.annual_sharpe_constant = 1
		
		sample_rate = daily_sharp_constant
		#sharpe_constant * mean([sample_rate]_returns - [sample_rate]_rf) / std([sample_rate]_returns)
		
	def get_cumulative_return(self, start_date, end_date, stock):
		start_value = stock.get_value(start_date)
		end_value = stock.get_value(end_date)
		return (end_value / start_value) - 1
		
	def get_average_daily_return(self, start_date, end_date, stock):
		daily_returns = stock.get_daily_returns(start_date, end_date)
		return
		
	def get_risk(self, start_date, end_date, stock):
		return np.std(stock.get_daily_returns())
		
	def get_sharpe_ratio(self, start_date, end_date, sample_rate, stock):
		return (self.get_average_daily_return(start_date, end_date, stock) - self.risk_free_return_rate) / self.get_risk(start_date, end_date, stock)
		
if __name__ == "__main__":
	print("Provides staitstics on stocks")