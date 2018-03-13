
############################################################# 
## Stat 413 - Homework 1
## Author: 
## Date : 
## Description: This script implements linear and logistic 
## regression using the sweep operator
#############################################################

#############################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names, 
## function inputs or outputs. 
#############################################################

import numpy as np
from sklearn import datasets, linear_model

################################
## Function 1: Sweep operator ##
################################

def mySweep(A, m):
    
    """
    Perform a SWEEP operation on A with the pivot element A[m,m].
    
    :param A: a square matrix (np.array).
    :param m: the pivot element is A[m, m].
    :returns a swept matrix (np.array). Original matrix is unchanged.
    """
    
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## Fill in the missing code below:
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    B = np.copy(A)   
    n = B.shape[0]
    for k in range(m):
        for i in range(n):
            for j in range(n):
                if i!=k and j!=k:
                    B[i,j] = B[i,j] - B[i,k]*B[k,j] / B[k,k]
        for i in range(n):
            if i!=k:
                 B[i,k] = B[i,k] / B[k,k]
        for j in range(n):
            if j!=k:
                B[k,j] = B[k,j] / B[k,k]
        B[k,k] = -1/B[k,k]
    
    ## The function outputs the matrix (np.array) B
    return(B)


############################################################
## Function 2: Linear regression using the sweep operator ##
############################################################

def myLinearRegression(X, Y):
  
  """
  Find the regression coefficient estimates beta_hat
  corresponding to the model Y = X * beta + epsilon
  Your code must use the sweep operator you coded above.
  Note: we do not know what beta is. We are only 
  given a matrix X and a vector Y and we must come 
  up with an estimate beta_hat. The vector beta_hat 
  should include an intercept term: that is, if X is n 
  by p, then beta_hat should be p+1 dimensional, where 
  the first element corresponds to the intercept term.
  
  X: an 'n row' by 'p column' matrix (np.array) of input variables.
  Y: an n-dimensional vector (np.array) of responses
  
  Hint: To account for the intercept term, you may need to
  add a column of 1's to one of the matrices in your solution.

  FILL IN THE BODY OF THIS FUNCTION BELOW
  """
  
  ## Let me start things off for you...
  n = X.shape[0]
  p = X.shape[1]
  
  Z = np.hstack([np.ones([n, 1]), X, Y])
  A = np.dot(np.transpose(Z),  Z)
  S = mySweep(A, p+1)
  
  ## Function returns the (p+1)-dimensional vector (np.array) 
  ## beta_hat of regression coefficient estimates
  beta_hat = S[:(p+1), p+1]
  return beta_hat

################################
## Linear regression examples ##
################################

## 1) Simulated data
def testing_Linear_Regression():
  
  ## Test your linear regression function using 
  ## simulated data.

  ## Define parameters
  n = 100
  p = 3
  
  ## Simulate data from our assumed model.
  ## We can assume that the true intercept is 0
  X = np.random.normal(0, 1, (n, p))
  beta = np.arange(p) + 1
  Y = np.dot(X, beta) + np.random.normal(0, 1, n)
  Y = Y.reshape(n, 1)

  ## Save Python's linear regression coefficients 
    
  # Create a linear regression object
  regr = linear_model.LinearRegression()

  # Fit the linear regression to our data
  regr.fit(X, Y)

  # Print model coefficients and intercept
  Python_coef = np.concatenate((regr.intercept_.reshape(1), regr.coef_[0]), axis = 0)
      
  ## Save our linear regression coefficients
  my_coef = myLinearRegression(X, Y)
  
  ## Are these two vectors different?
  sum_square_diff = sum((Python_coef - my_coef)**2)
  if sum_square_diff <= 10**(-10):
      return 'Both results are identical'
  else:
      return 'There seems to be a problem...'
  
## Test the regression function
testing_Linear_Regression()


## 2) Real data. 
## Test your regression function using the 'iris' dataset in Python.
## This part is open-ended, you can make tables, compare your 
## estimates with those of Python's built in regression, make plots, etc.
iris = datasets.load_iris()


######################################
## Function 3: Logistic regression  ##
######################################

from scipy import linalg

def mylogistic(_x, _y):
    
    # Find the logistic regression coefficient estimates beta_hat
    # corresponding to the model _y = _x * beta + epsilon
    # Your code must use the sweep operator you wrote above.
    # Note: we do not know what beta is. We are only 
    # given a matrix _x and a vector _y and we must come 
    # up with an estimate beta_hat. The vector beta_hat 
    # does not include an intercept term: that is, if X is n 
    # by p, then beta_hat should be p dimensional.
    # 
    # _x: an 'n row' by 'p column' matrix (np.array) of input variables.
    # _y: an n-dimensional vector (np.array) of categorical responses (0's and 1's).

    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## Fill in the missing code below:
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    x = _x.copy()
    y = _y.copy()
    r, c = x.shape

    beta_hat = np.zeros((c, 1))
    epsilon = 1e-6

    while True:
        eta = np.dot(x, beta_hat)
        pr = sigmoid(eta)
        w = pr * (1 - pr)
        z = eta + (y - pr) / w
        sw = np.sqrt(w)
        mw = np.repeat(sw, c, axis=1)

        x_work = mw * x
        y_work = sw * z

        beta_new, _, _, _ = np.linalg.lstsq(x_work, y_work)
        err = np.sum(np.abs(beta_new - beta_hat))
        beta_hat = beta_new
        if err < epsilon:
            break

        ## Function returns the p-dimensional vector (np.array)
        ## beta_hat of regression coefficient estimates
        return beta_hat
    
def sigmoid(_x):
    x = _x.copy()
    y = 1 / (1 + np.exp(-x))
    return y


##################################
## Logistic regression examples ##
##################################

## 1) Simulated data
def testing_Logistic_Regression():
  
  ## This function is not graded; you can use it to 
  ## test out the 'mylogistic' function 
  
  ## Define parameters
  n = 100
  p = 2
  
  ## Simulate data from our assumed model.
  ## We can assume that the true intercept is 0
  X = np.random.normal(0, 1, (n, p))
  beta = np.ones((p, 1))
  Y = np.random.uniform(0, 1, (n, 1)) < sigmoid(np.dot(X, beta)).reshape((n, 1))
  Y = Y.reshape(n, 1)
    
  ## Save Python's logistic regression coefficients 
    
  # Create a logistic regression object
  regr = linear_model.LogisticRegression(fit_intercept = False)

  # Fit the logistic regression to our data
  regr.fit(X, Y)

  # Print model coefficients and intercept
  #Python_coef = np.concatenate((regr.intercept_.reshape(1), regr.coef_[0]), axis = 0)
  Python_coef = regr.coef_[0]
    
  ## Save our logistic regression coefficients
  my_coef = mylogistic(X, Y)

  print(Python_coef)
  print(my_coef.reshape(1, p)[0])
    
  ## Are these two vectors different?
  sum_square_diff = sum((Python_coef - my_coef.reshape(1, p)[0])**2)
  if sum_square_diff <= 10**(-10):
      return 'Both results are identical'
  else:
      return 'There seems to be a problem...'
  
testing_Logistic_Regression()    



## 2) Real data. 
## Test your regression function using the 'iris' dataset in Python.
## This part is open-ended, you can make tables, compare your 
## estimates with those of Python, make plots, etc. Note that since 
## you are performing logistic regression, the response variable
## Y must be categorical (i.e. Y is the Species variable in 'iris')
iris = datasets.load_iris()



