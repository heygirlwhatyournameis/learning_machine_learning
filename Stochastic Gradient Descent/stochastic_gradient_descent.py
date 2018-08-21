import csv
import random

class Database:
	def __init__(self):
		self.dataset=list()
	def _load_csv(self,filename):
		with open(filename) as file:
			reader=csv.reader(file,delimiter=';')
			titles=next(reader)
			for row in reader:
				if not row:
					continue
				self.dataset.append(row)

	def _data_str_to_float(self):
		col_len=len(self.dataset[0])
		for row in self.dataset:
			for i in range(col_len):
				row[i]=float(row[i].strip())

	def _dataset_min_max(self):
		col_len=len(self.dataset[0])
		self.minmax=list()
		for i in range(col_len):
			col_values=[row[i] for row in self.dataset]
			col_max=max(col_values)
			col_min=min(col_values)
			self.minmax.append([col_min,col_max])

	def _dataset_normalize(self):
		col_len=len(self.dataset[0])
		for row in self.dataset:
			for i in range(col_len):
				row[i]=(row[i]-self.minmax[i][0])/self.minmax[i][1]

	def split_data(self,percent):
		self.training_data=list()
		self.testing_data=list()
		for row in self.dataset:
			if random.random() >= percent:
				self.testing_data.append(row)
			else:
				self.training_data.append(row)

	def get_data(self,filename,percent):
		self._load_csv(filename)
		self._data_str_to_float()
		self._dataset_min_max()
		self._dataset_normalize()
		self.split_data(percent)
		return self.training_data,self.testing_data

class LinearUnit:
	def __init__(self,activator):
		self.activator=activator

	def __str__(self):
		return repr(self.weights)

	def train(self,training_data,iterator,rate):
		self.weights=[0.0 for _ in range(len(training_data[0]))]
		for i in range(iterator):
			for row in training_data:
				y_hat=self.predict(row)
				self._update_weights(y_hat,row,rate)

	def predict(self,data):
		result=self.weights[0]
		for i in range(len(data) - 1):
			result+=self.weights[i+1]*data[i]
		return self.activator(result)

	def _update_weights(self,prediction,data,rate):
		delta = data[-1]-prediction
		self.weights[0] += rate*delta
		for i in range(len(data) - 1):
			self.weights[i+1] += rate*delta*data[i]

def activator(value):
	return value

if __name__ == '__main__':
	d=Database()
	training_data,testing_data=d.get_data('winequality-white.csv',0.99)
	
	lu=LinearUnit(activator)
	lu.train(training_data,100,0.01)

	print(lu)

	for row in testing_data:
		prediction=lu.predict(row)
		print('expected={0},predict={1}'.format(row[-1],prediction))