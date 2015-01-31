'''
Created on 30-Jan-2015

@author: swapniel
'''

import csv
import numpy as np
import sklearn.cross_validation as cv
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from random import shuffle

# Load training data from trainr3.csv
X=[]
y=[]

print 'Fetching data...'
with open('image_recognition.csv','rb') as f:
    read = csv.reader(f, delimiter = ',', quoting = csv.QUOTE_NONE)
    for row in read:
        if row==[]:
            continue
        X.append(row[1:])
        y.append(ord(row[0])-ord('A'))

m=len(y)
X = np.array(X, dtype=float)
y = np.array(y)
ks=range(1,11)

#Random sampling from data for leave one out, since the size of data set is huge. Leave-one-out takes a lot of time.
print 'Sampling data for Leave-one-out cross-validation...'
ms=2000
ids=range(m)
shuffle(ids)
ids=ids[:ms]

#Sampled data from original dataset
Xs=X[ids]
ys=y[ids]

#Leave-one-out cross-validation
print 'Generating Leave-one-out cross-validation plot for k = 1 to 10...'
accsloo=[]
loo = cv.LeaveOneOut(n=ms)
 
for k in ks:
    print 'for k =',k
    accsloo.append(cv.cross_val_score(KNeighborsClassifier(n_neighbors=k), Xs, ys, cv=loo, n_jobs=2).mean())

#5fold cross-validation
print 'Generating 5-fold cross-validation plot for k = 1 to 10...'
accskf=[]
kfold = cv.KFold(n=m, n_folds=5)
 
for k in ks:
    print 'for k =',k
    accskf.append(cv.cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=kfold, n_jobs=2).mean())

#Saving the plots
print 'Saving plots loo.png and kfold.png...'
p1=plt.figure()
plt.title('K vs. Accuracy for Leave-One-Out')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.plot(ks,accsloo,'r-')
plt.grid(True)
plt.savefig('loo.png')

p2=plt.figure()
plt.title('K vs. Accuracy for 5-Fold')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.plot(ks,accskf,'b-')
plt.grid(True)
plt.savefig('kfold.png')

print 'Done.'