from random import random
from random import seed
from math import exp

def init_network(n_inputs,n_hidden,n_outputs):
	network=list()
	hidden_layers = [ {'weights':[random() for _ in range(n_inputs+1)]} for _ in range(n_hidden) ]
	ouput_layers = [ {'weights':[random() for _ in range(n_hidden+1)]} for _ in range(n_outputs) ]
	network.append(hidden_layers)
	network.append(ouput_layers)
	return network

def activate(weights,inputs):
	result = weights[-1]
	for i in range(len(weights)-1):
		result +=  weights[i] * inputs[i]
	return result

def transfer(x):
	return 1.0 / ( 1.0 + exp(-x))

def transfer_derivative(output):
	return output * (1 - output)

def forward_propagate(network,row):
	inputs=row
	for layer in network:
		new_inputs=[]
		for neuron in layer:
			activation = activate(neuron['weights'],inputs)
			neuron['output']=transfer(activation)
			new_inputs.append(neuron['output'])
		inputs=new_inputs
	return inputs

def backward_propagate(network,expected):
	for i in reversed(range(len(network))):
		layer=network[i]
		errors=list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i+1]:
					error += neuron['responsibility'] * neuron['weights'][j]
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron=layer[j]
				errors.append(expected[j]-neuron['output'])
		for j in range(len(layer)):
			neuron=layer[j]
			neuron['responsibility'] = errors[j] * transfer_derivative(neuron['output'])

def update_weights(network,row,learning_rate):
	inputs=row[:-1]
	for i in range(len(network)):
		layer=network[i]
		if i!=0:
			inputs=[neuron['output'] for neuron in network[i-1]]
		for neuron in layer:
			for j in range(len(inputs)):
				neuron['weights'][j] += learning_rate * neuron['responsibility'] * inputs[j]
			neuron['weights'][-1] += learning_rate * neuron['responsibility']

def train_network(network,train,learning_rate,n_epoch,n_outputs):
	for epoch in range(n_epoch):
		sum_error=0.0
		for row in train:
			outputs=forward_propagate(network,row)
			expected=[0 for i in range(n_outputs)]
			expected[ row[-1] ] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate(network,expected)
			update_weights(network,row,learning_rate)
		print('>Train:%d, Error=%.3f' % (epoch+1,sum_error))

def predict(network,row):
	outputs=forward_propagate(network,row)
	return outputs.index(max(outputs))

if __name__ == '__main__':
	seed(2)
	dataset=[[1,1,0],[1,0,1],[0,1,1],[0,0,0]]
	n_inputs=len(dataset[0]) - 1
	n_outputs=len(set([row[-1] for row in dataset]))
	network=init_network(n_inputs,2,n_outputs)
	print(network)
	train_network(network,dataset,0.5,2000,n_outputs)
	for layer in network:
		print(layer)
	for row in dataset:
		predicition=predict(network,row)
		print('#Expected=%d,#Output=%d' % (row[-1], predicition))