import csv
import random
import math
import operator

#Load data
def load_csv(filename):
	with open(filename,'r') as csvfile:
		lines=csv.reader(csvfile)
		dataset=list(lines)
	for x in range(len(dataset)):
		for y in range(4):
			dataset[x][y]=float(dataset[x][y])	
	return dataset

def split_data(dataset,percent):
	training_set=[]
	testing_set=[]
	for data in dataset:
		if random.random() < percent:
			training_set.append(data)
		else:
			testing_set.append(data)
	return training_set,testing_set

def EuclidDist(point1,point2,len):
	distance=0.0
	for i in range(len):
		distance += (point1[i]-point2[i])**2
	return math.sqrt(distance)

def getNeighbours(training_set,testing_instance,k):
	distances=[]
	dimension=len(testing_instance)-1
	for i in range(len(training_set)):
		dist=EuclidDist(training_set[i],testing_instance,dimension)
		distances.append((training_set[i],dist))
	distances.sort(key=operator.itemgetter(1))
	neighbours=[]
	for i in range(k):
		neighbours.append(distances[i][0])
	return neighbours

def getClass(neighbours):
	classVotes={}
	for i in range(len(neighbours)):
		nb_class=neighbours[i][-1]
		if nb_class in classVotes:
			classVotes[nb_class]+=1
		else:
			classVotes[nb_class]=1
	sortedVotes=sorted(classVotes.items(),key=operator.itemgetter(1))
	return sortedVotes[0][0]

def getAccurancy(testing_set,predctions):
	correct=0
	for i in range(len(predctions)):
		print(testing_set[i],predctions[i])
		if testing_set[i][-1] == predctions[i]:
			correct+=1
	return correct/len(predctions)*100


def main():
	k=3
	dataset=load_csv('dataset.csv')
	training_set,testing_set=split_data(dataset,0.7)
	predctions=[]
	for instance in testing_set:
		n=getNeighbours(training_set,instance,k)
		predctions.append(getClass(n))
	print('Accurancy:'+repr(getAccurancy(testing_set,predctions))+'%')

main()
	