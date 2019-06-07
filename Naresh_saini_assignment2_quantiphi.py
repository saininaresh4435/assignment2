
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import random
import sklearn.utils
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
def data_load(fn1,fn2):
	wine_white=pd.read_csv(fn1,sep=';')
	wine_red=pd.read_csv(fn2,sep=";")
	df=pd.concat([wine_white,wine_red ],axis=0)
	df = sklearn.utils.shuffle(df)
	#df=(df- df.mean())/df.std()
	
	return df
def datapreprocessing():
	df=data_load("winequality-white.csv","winequality-red.csv")
	y=df['quality']
	df=df.drop('quality',axis=1)
	x_train=df.iloc[:5197,:] #20% data in test set
	y_train=y.iloc[:5197]
	x_test=df.iloc[5197:,:]
	y_test=y.iloc[5197:]
	y_train=y_train.reshape(x_train.shape[0],1)
	theta = np.zeros(x_train.shape[1]+1)
	theta = theta.reshape(1,x_train.shape[1]+1)
	new_col = np.ones((x_train.shape[0],1))
	x_train= np.concatenate((new_col, x_train), axis = 1)
	return x_train,y_train,x_test,y_test,theta

x_train,y_train,x_test,y_test,theta=datapreprocessing()# call datapreprocessing function


# In[34]:


iteration=3000
lr=0.0001
def pred_Cost(X,y,theta):
    rss=(np.dot(X,theta.T)-y)**2
    cost_fun=np.sum(rss)/(2 * len(X))
    return cost_fun
def gradientDescent(X,y,theta,iters,lr):
    cost = np.zeros(iters)
    for i in range(iters):
        
        theta = theta - (lr/len(X)) * np.sum(X * (np.dot(X,theta.T )- y), axis=0)
        cost[i] = pred_Cost(X, y, theta)
    
    return theta,cost

#running the gd and cost function
param,cost = gradientDescent(x_train,y_train,theta,iteration,lr)


# In[35]:
def plotdata():
	cost1 = list(cost)
	n_iterations = [x for x in range(1,3001)]
	plt.plot(n_iterations, cost1)
	plt.xlabel('No. of iterations')
	plt.ylabel('Cost')
	plt.show()
plotdata()


def model_predict(x_test,y_test,theta):
    theta0=theta[0][0]
    theta1=theta[0][1:]
    theta1=np.reshape(theta1,(x_test.shape[1],1))
    y_pred=np.ones(x_test.shape[0])
    y_pred=np.reshape(y_pred,(x_test.shape[0],1))
    for i in range(0,len(x_test)):
        x_test2=x_test.iloc[i,:].reshape(1,x_test.shape[1])
        y_pred[i]=theta0+np.dot(x_test2,theta1)
    #print(y_pred.shape)
    y_test=np.reshape(y_test,(x_test.shape[0],1))
    #print(y_test.shape)
    r2_value=r2_score(y_test, y_pred) 
    #mse=(1/3000)*np.sum((y_test-y_pred)**2)
    mse=mean_squared_error(y_test, y_pred)
    smse=np.sqrt(metrics.mean_squared_error(y_test,y_pred))
    return r2_value,mse,smse

r2_value,mse,smse=model_predict(x_test,y_test,param)
print(r2_value,mse,smse)

