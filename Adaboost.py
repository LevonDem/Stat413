import numpy as np
import csv
import random
import math
import os

def prepare_data(valid_digits=np.array((6,5))):
    if len(valid_digits)!=2:
        raise Exception("Error: you must specify exactly 2 digits for classification!")

    #num_train = 360
    csvfile=open('digits.csv','r')
    reader=csv.reader(csvfile)
    data=[]
    for line in reader:
        data.append(line)

    csvfile.close()
    digits=np.asarray(data,dtype='float')

    X = digits[(digits[:, 64] == valid_digits[0]) | (digits[:, 64] == valid_digits[1]), 0:64]
    Y = digits[(digits[:, 64] == valid_digits[0]) | (digits[:, 64] == valid_digits[1]), 64:65]
    
    X = np.asarray(map(lambda k: X[k,:]/X[k,:].max(), range(0,len(X))))

    Y[Y==valid_digits[0]]=0
    Y[Y==valid_digits[1]]=1

    training_set=random.sample(range(360),270)
    testing_set=list(set(range(360)).difference(set(training_set)))

    X_train=X[training_set,]
    Y_train=Y[training_set,]

    X_test=X[testing_set,:]
    Y_test=Y[testing_set,]

    return X_train,Y_train,X_test,Y_test

def accuracy(p,y):

    acc=np.mean((p>0.5)==(y==1))
    return acc

# Adaboost
# Use Adaboost to classify the digits data

def my_Adaboost(X_train, Y_train, X_test, Y_test,num_iterations = 200):
    n=X_train.shape[0]
    p=X_train.shape[1]
    threshold=0.8


    X_train1=2*(X_train>threshold)-1
    Y_train=2*Y_train-1

    X_test1=2*(X_test>threshold)-1
    Y_test=2*Y_test-1

    beta=np.repeat(0.,p).reshape((p,1))
    w=np.repeat(1./n,n).reshape((n,1))

    weak_results=np.multiply(Y_train,X_train1)>0

    acc_train=np.repeat(0.,num_iterations,axis=0)
    acc_test=np.repeat(0.,num_iterations,axis=0)

    for it in range(num_iterations):

        w=w/np.sum(w)
        weighted_weak_results=np.multiply(w,weak_results)
        weighted_accuracy=np.sum(weighted_weak_results,axis=0)
        e=1-weighted_accuracy
        j=np.argmin(e)
        dbeta=0.5*math.log((1-e[j])/e[j])
        beta[j]=beta[j]+dbeta
        w=np.multiply(w,np.exp(-np.multiply(np.multiply(Y_train,X_train1[:,j:(j+1)]),dbeta)))


        acc_train[it]=np.mean(np.multiply(np.sign(np.dot(X_train1,beta)),Y_train)>0)
        #print acc_train[it]
        #acc_test[it]=np.mean(np.sign(np.dot(X_test1,beta))==Y_test)
        acc_test[it]=np.mean(np.multiply(np.sign(np.dot(X_test1,beta)),Y_test)>0)
    return beta,acc_train,acc_test


X_train,Y_train,X_test,Y_test=prepare_data()
beta,acc_train,acc_test=my_Adaboost(X_train, Y_train, X_test, Y_test,num_iterations = 500)

