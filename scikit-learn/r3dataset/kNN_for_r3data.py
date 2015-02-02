#!/usr/bin/python
'''
Created on 29-Jan-2015

@author: swapniel
'''
import csv
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#Load training data from trainr3.csv
X=[]
y=[]

print 'Fetching data...'
with open('trainr3.csv','rb') as f:
    read = csv.reader(f, delimiter = ',', quoting = csv.QUOTE_NONNUMERIC)
    for row in read:
        if row==[]:
            continue
        X.append(row[:-1])
        y.append(int(row[-1]))

#Load testing data from testr30.csv
X0 = []
y0 = []
with open('testr30.csv','rb') as f0:
    read = csv.reader(f0, delimiter = ',', quoting = csv.QUOTE_NONNUMERIC)
    for row in read:
        if row == []:
            continue
        X0.append(row[:-1])
        y0.append(int(row[-1]))

#Load testing data from testr31.csv
X1 = []
y1 = []
with open('testr31.csv','rb') as f1:
    read = csv.reader(f1, delimiter = ',', quoting = csv.QUOTE_NONNUMERIC)
    for row in read:
        if row==[]:
            continue
        X1.append(row[:-1])
        y1.append(int(row[-1]))

#Generate plot data
print 'Generating plots...'
ks = range(1,21)
acc0 = []
acc1 = []
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    total0 = len(X0)
    correctlyclassified0 = 0
    for i in range(total0):
        if knn.predict(X0[i])[0] == y0[i]:
            correctlyclassified0 += 1
    acc0.append(correctlyclassified0/float(total0))
    total1 = len(X1)
    correctlyclassified1 = 0
    for i in range(total1):
        if knn.predict(X1[i])[0] == y1[i]:
            correctlyclassified1 += 1
    acc1.append(correctlyclassified1/float(total1))

#Saving the two plots
print 'Saving plots testr30.png and testr31.png...'
p1 = plt.figure()
plt.title('K vs. Accuracy for testr30.csv')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.plot(ks,acc0,'r-')
plt.grid(True)
plt.savefig('testr30.png')

p2 = plt.figure()
plt.title('K vs. Accuracy for testr31.csv')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.plot(ks,acc1,'b-')
plt.grid(True)
plt.savefig('testr31.png')

print 'Done.'

