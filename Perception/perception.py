class Perception:
	def __init__(self,input_para_num,func_activator):
		self.w=[0.0 for _ in range(input_para_num)]
		self.activator=func_activator

	def __str__(self):
		return repr(self.w)

	def train(self,dataset,iteration,rate):
		for i in range(iteration):
			for data in dataset:
				prediction=self.predict(data)
				if prediction!=data[-1]:
					self._update_weights(data,prediction,rate)

	def predict(self,data):
		result=0.0
		for i in range(len(self.w)):
			result+=self.w[i]*data[i]
		return func_activator(result)

	def _update_weights(self,data,prediction,rate):
		delta_w=data[-1]-prediction
		for i in range(len(self.w)):
			self.w[i]+=rate*delta_w*data[i]

def func_activator(value):
	return 1.0 if value>0.0 else 0.0

def main():
	dataset=[[-1,0,0,0],[-1,0,1,1],[-1,1,0,1],[-1,1,1,1]]
	p=Perception(3,func_activator)
	p.train(dataset,10,0.1)
	print(p)
	print('0 and 0 = %d' % p.predict([-1,0,0]))
	print('0 and 1 = %d' % p.predict([-1,0,1]))
	print('1 and 0 = %d' % p.predict([-1,1,0]))
	print('1 and 1 = %d' % p.predict([-1,1,1]))

main()
