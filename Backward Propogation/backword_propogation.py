import numpy as np
from math import exp

def init_network():
	layers=list()
	input_layer=np.array([[-1.0,1.0],[1.0,1.0]])
	hidden_layer=np.array([[1.0,1.0],[-1.0,1.0]])
	layers.append(input_layer)
	layers.append(hidden_layer)
	return layers

def forward_prop(network,inputs,activator):
	outputs=list()
	for layer in network:
		temp=[sum(inputs*layer[i]) for i in range(len(layer))]
		inputs=[activator(temp[i]) for i in range(len(temp))]
		outputs.append(inputs)
	return outputs

def backward_prop(network,outputs,exception,act_derivative,rate):
	for i in reversed(range(len(network))):
		layer=network[i]
		if i==len(network)-1:
			d=list()
			for j in range(len(layer)):
				delta=exception[j]-outputs[i][j]
				d_temp=sigmoid_derivative(outputs[i][j])*delta
				d.append(d_temp)
				for k in range(len(layer[j])):
					layer[j][k]+=rate*d_temp*outputs[i][j]
		else:
			d_t=list(d)
			d=list()
			layer=network[i]
			next_layer=network[i+1]
			for j in range(len(layer)):
				delta=0.0
				for k in range(len(next_layer)):
					delta+=next_layer[j][k]*d_t[j]
					d_temp=sigmoid_derivative(outputs[i][j])*delta
					d.append(d_temp)
					for k in range(len(layer[j])):
						layer[j][k]+=rate*d_temp*outputs[i][j]

	return network

def sigmoid(x):
	return 1/(1+exp(-x))

def sigmoid_derivative(value):
	return value*(1-value)

if __name__ == '__main__':
	inputs=np.array([1,-1]).T
	network=init_network()
	outputs=forward_prop(network,inputs,sigmoid)
	print(outputs)

	exception=np.array([1,0]).T
	network=backward_prop(network,outputs,exception,sigmoid_derivative,0.1)
	print(network)