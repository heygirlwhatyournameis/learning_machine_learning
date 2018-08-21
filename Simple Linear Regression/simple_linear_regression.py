import matplotlib.pyplot as plt
from csv import reader
from math import sqrt
from random import randrange
from random import seed

#Utilizes
def mean(values):
	return sum(values) / len(values)

def varince(values):
	return sum([(x-mean(values))**2 for x in values])

def covarince(x,y):
	cov=0.0
	mean_x, mean_y=mean(x),mean(y)
	for i in range(len(x)):
		cov+=(x[i]-mean_x)*(y[i]-mean_y)
	return cov

#Algorithm
def confficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean,y_mean=mean(x),mean(y)
	w1= covarince(x,y) / varince(x)
	w0 = y_mean - w1* x_mean
	return w0,w1

def simple_linear_regression(train,test):
	predict=list()
	w0,w1=confficients(train)
	for row in test:
		predict.append(w0+w1*row[0])
	return predict

#Preparation
def evaluate_algorithm(dataset,algorithm,*args):
	#Split data
	train_set,test_set=split_data(dataset,0.6)
	print(train_set)
	print(test_set)
	#predict
	predict = algorithm(train_set,test_set,*args)
	actual = [row[-1] for row in test_set]
	figure(train_set,test_set,predict)
	#rmse
	return rmse_mertic(actual,predict)

def split_data(dataset,percent):
	train=list()
	train_size=percent*len(dataset)
	dataset_copy=list(dataset)
	while len(train)<train_size:
		index=randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train,dataset_copy

def rmse_mertic(actual,predict):
	sum_error=0.0
	for i in range(len(actual)):
		sum_error+= (predict[i]-actual[i])**2
	mean_error = sum_error / len(actual)
	return sqrt(mean_error)

#Load Files
def load_csv(filename):
	dataset=list()
	with open(filename,'r') as file:
		csv_reader=reader(file)
		headings=next(csv_reader)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def str_column_to_float(dataset,column):
	for row in dataset:
		row[column]=float(row[column].strip())

#Draw Figure
def figure(train_set,test_set,predict):
	x = [row[0] for row in test_set]
	y = [row[1] for row in test_set]
	#散点图
	#plt.axis([0,6,0,6])
	plt.plot(x,y,'rs')
	plt.plot(x,predict,'ob')
	plt.show()

seed(2)
#Load data
dataset=load_csv('data.csv')
for col in range(len(dataset[0])):
	str_column_to_float(dataset,col)
rmse=evaluate_algorithm(dataset,simple_linear_regression)
print('RMSE:%.3f'%(rmse))