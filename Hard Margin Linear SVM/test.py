#Use the QP tool in cvxopt to solve 2d hard margin SVM problem and draw out a figure

import numpy as np 
import matplotlib.pyplot as plt
from cvxopt import solvers,matrix

data=np.array([
	[0.,0.,-1.],
	[0.,3.,-1.],
	[2.,2.,-1.],
	[2.1,2.,-1.],
	[2.1,2.1,-1.],
	[2.,2.1,-1.],
	[2.,0.,1.],
	[2.,0.1,1.],
	[2.2,0.1,1.],
	[3.,0.,1.],
	[2.,1.,1.]
])
x=data[:,:2]
y=data[:,2]

n=np.size(x,0)
d=2

x=np.array(x)

#quadratic plan
#covert to standard form

G=np.column_stack((np.ones(n).T,x))

i=0;
while i<n:
	G[i]=y[i]*G[i]
	i+=1

P=matrix([[0,0,0],[0,1.,0],[0,0,1.]])
q=matrix(np.zeros(d+1))
G=matrix(G)
h=matrix(np.ones(n)*-1)

w=solvers.qp(P,q,G,h)
u=(w['x'])
b=-1*u[0]/u[2]
k=-1*u[1]/u[2]

#Draw the points
x1=x[:,0]
x2=x[:,1]
plt.scatter(x1,x2,alpha=0.5)

#Draw the hyperplane
points=np.arange(0,np.max(x1),0.1);
plane=k*points+b
plt.plot(points,plane)

plt.show()
print(x)