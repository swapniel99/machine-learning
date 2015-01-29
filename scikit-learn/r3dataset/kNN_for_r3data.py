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

with open('trainr3.csv','rb') as f:
    read = csv.reader(f, delimiter = ',', quoting = csv.QUOTE_NONNUMERIC)
    for row in read:
        if row==[]:
            continue
        X.append(row[:-1])
        y.append(int(row[-1]))

#Load testing data from testr30.csv
X0=[]
y0=[]
with open('testr30.csv','rb') as f0:
    read = csv.reader(f0, delimiter = ',', quoting = csv.QUOTE_NONNUMERIC)
    for row in read:
        if row==[]:
            continue
        X0.append(row[:-1])
        y0.append(int(row[-1]))

#Load testing data from testr31.csv
X1=[]
y1=[]
with open('testr31.csv','rb') as f1:
    read = csv.reader(f1, delimiter = ',', quoting = csv.QUOTE_NONNUMERIC)
    for row in read:
        if row==[]:
            continue
        X1.append(row[:-1])
        y1.append(int(row[-1]))

#Generate first plot: K vs. Accuracy for first test file.
ks=[]
acc0=[]
for k in range(1,21):
    ks.append(k)
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    total=len(X0)
    correctlyclassified=0
    for i in range(total):
        if knn.predict(X0[i])[0]==y0[i]:
            correctlyclassified+=1
    acc0.append(correctlyclassified/float(total))

#Generate second plot: K vs. Accuracy for second test file.
acc1=[]
for k in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    total=len(X1)
    correctlyclassified=0
    for i in range(total):
        if knn.predict(X1[i])[0]==y1[i]:
            correctlyclassified+=1
    acc1.append(correctlyclassified/float(total))

#Save the two plots
p1=plt.figure()
plt.title('K vs. Accuracy for testr30.csv')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.plot(ks,acc0,'r-')
plt.grid(True)
plt.savefig('testr30.png')

p2=plt.figure()
plt.title('K vs. Accuracy for testr31.csv')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.plot(ks,acc1,'b-')
plt.grid(True)
plt.savefig('testr31.png')
