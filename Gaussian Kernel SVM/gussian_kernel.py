#用对偶方法+线性核函数解决二维数据分类问题
#作者：陈中舒
#Make use of dual method and linear kenerl function to solve 2d classifaction problem
#Author: ZHONGSHU CHEN

import numpy as np 
import matplotlib.pyplot as plt
from cvxopt import solvers,matrix
from matplotlib import colors

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
lamb=1

#Kernel Function
def kernel(x1,x2):
	return np.exp(-lamb* np.dot( (x1-x2).T, (x1-x2)))
	
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
B=matrix(1.)

#Quadratic Plan
print(P)

a_hat=solvers.qp(P,q,G,h,A,B)
a=a_hat['x']

#Solve w,b
w=np.zeros(2)
i=0
while i<n:
	w=w+(a[i]*y[i]*x[i]).T
	i+=1

i=0
b=0
while i<n:
	if a[i]>0:
		b=1/y[i]-np.dot(w.T,x[i])
		break
	i+=1

def threshold(x_hat):
	i=0
	t=0
	while i<n:
		t+=a[i]*y[i]*kernel(x_hat,x[i])
		i+=1
	return (1 if t+b>=0 else 0)

def predict(data):
	result=np.empty(int(data.size/2))
	for i in range(int(data.size/2)):
		result[i]=threshold(data[i])
	return result



#Draw the hyperplane
x1, x2 = np.mgrid[0:3:200j, 0:3:200j] 
grid_test = np.stack((x1.flat, x2.flat), axis=1) 
grid_hat = predict(grid_test).reshape(x1.shape)

cmap=colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
pcolors=colors.ListedColormap(['g','r','b'])
plt.pcolormesh(x1, x2, grid_hat, cmap=cmap)

#Draw the points
x_1=x[:,0]
x_2=x[:,1]
plt.scatter(x_1,x_2,c=y,alpha=0.5,cmap=pcolors,edgecolors='k')


plt.show()