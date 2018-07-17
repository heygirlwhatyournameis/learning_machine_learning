#用对偶方法+线性核函数解决二维数据分类问题
#作者：陈中舒
#Make use of dual method and linear kenerl function to solve 2d classifaction problem
#Author: ZHONGSHU CHEN

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

#Kernel Function
def kernel(x1,x2):
	return np.dot(x1.T,x2)

#P
P=np.empty((n,n))

#Define the indexes
i=0
j=0
while i<n:
	j=0
	while j<n:
		P[i][j]=y[i]*y[j]*kernel(x[i],x[j])
		j+=1
	i+=1
P=matrix(P)

#Q,G,H,A,B
q=matrix(-np.ones(n))
G=matrix(-np.eye(n))
h=matrix(np.zeros(n))
A=matrix(y).T
b=matrix(1.)

#Quadratic Plan
print(P)

a_hat=solvers.qp(P,q,G,h,A,b)
a=a_hat['x']

#Solve w,b
w=np.zeros(2)
i=0
while i<n:
	w=w+(a[i]*y[i]*x[i]).T
	i+=1

i=0
while i<n:
	if a[i]>0:
		b=1/y[i]-w.T*x[i]
		break
	i+=1


b=-1*b[0]/w[1]
k=-1*w[0]/w[1]

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