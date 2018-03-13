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
##
## Very important: Do not use the function "setwd" anywhere
## in your code. If you do, I will be unable to grade your 
## work since R will attempt to change my working directory
## to one that does not exist.
##
## Do not use the following functions for this assignment,
## except when debugging or in the optional examples section:
## 1) lm()
## 2) solve()
#############################################################


################################
## Function 1: Sweep operator ##
################################

mySweep <- function(A, m){
  
  # Perform a SWEEP operation on A with the pivot element A[m,m].
  # 
  # A: a square matrix.
  # m: the pivot element is A[m, m].
  # Returns a swept matrix B (which is m by m).
  
  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ## Fill in the missing code below:
  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  B <- A
  n <- nrow(B)
  
  for(k in 1:m){ 
    for(i in 1:n)     
      for(j in 1:n)   
        if(i != k  & j != k)     
          B[i,j] <- B[i,j] - B[i,k]*B[k,j]/B[k,k]    
        
        for(i in 1:n) 
          if(i != k) 
            B[i,k] <- B[i,k]/B[k,k]  
          
          for(j in 1:n) 
            if(j != k) 
              B[k,j] <- B[k,j]/B[k,k]
            
            B[k,k] <- - 1/B[k,k]
  }
  
  ## The function outputs the matrix B
  return(B)
  
}


############################################################
## Function 2: Linear regression using the sweep operator ##
############################################################

myLinearRegression <- function(X, Y){
  
  # Find the regression coefficient estimates beta_hat
  # corresponding to the model Y = X * beta + epsilon
  # Your code must use the sweep operator you wrote above.
  # Note: we do not know what beta is. We are only 
  # given a matrix X and a vector Y and we must come 
  # up with an estimate beta_hat. The vector beta_hat 
  # should include an intercept term: that is, if X is n 
  # by p, then beta_hat should be p+1 dimensional, where 
  # the first element corresponds to the intercept term. 
  # 
  # X: an 'n row' by 'p column' matrix of input variables.
  # Y: an n-dimensional vector of responses
  #
  # Hint: Compare your solution to R's built in regression
  # solver using lm(Y ~ X). Both beta hat's should be the 
  # same. To account for the intercept term, you may need to
  # add a column of 1's to one of the matrices in your solution.
  
  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ## Fill in the missing code below:
  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  n <- nrow(X)
  p <- ncol(X)
  Z <- cbind(rep(1, n), X, Y)
  A <- t(Z) %*% Z
  S <- mySweep(A, p+1)

  ## Function returns the (p+1)-dimensional vector 
  ## beta_hat of regression coefficient estimates
  beta_hat <- S[1:(p+1), p+2]
  return(beta_hat)
  
  
}

################################
## Linear regression examples ##
################################

## 1) Simulated data
testing_Linear_Regression <- function(){
  
  ## Test your linear regression function using 
  ## simulated data.

  ## Define parameters
  n    <- 100
  p    <- 3
  
  ## Simulate data from our assumed model.
  ## We can assume that the true intercept is 0
  X    <- matrix(rnorm(n * p), nrow = n)
  beta <- matrix(1:p, nrow = p)
  Y    <- X %*% beta + rnorm(n)
  
  ## Save R's linear regression coefficients
  R_coef  <- coef(lm(Y ~ X))
  
  ## Save our linear regression coefficients
  my_coef <- myLinearRegression(X, Y)
  
  ## Are these two vectors different?
  sum_square_diff <- sum((R_coef - my_coef)^2)
  if(sum_square_diff <= 10^(-10)){
    return('Both results are identical')
  }else{
    return('There seems to be a problem...')
  }
  
}

## Test the regression function
testing_Linear_Regression()

## 2) Real data. 
## Test your regression function using the 'iris' dataset in R.
## This part is open-ended, you can make tables, compare your 
## estimates with those of 'lm', make plots, etc.
data(iris)



######################################
## Function 3: Logistic regression  ##
######################################

## Expit/sigmoid function
sigmoid <- function(x){
  1 / (1 + exp(-x))
}

myLogistic <- function(X, Y){
  
  # Find the logistic regression coefficient estimates beta_hat
  # corresponding to the model Y = X * beta + epsilon
  # Your code must use the sweep operator you wrote above.
  # Note: we do not know what beta is. We are only 
  # given a matrix X and a vector Y and we must come 
  # up with an estimate beta_hat. The vector beta_hat 
  # does not include an intercept term: that is, if X is n 
  # by p, then beta_hat should be p dimensional.
  # 
  # X: an 'n row' by 'p column' matrix of input variables.
  # Y: an n-dimensional vector of categorical responses (0's and 1's).
  
  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ## Fill in the missing code below:
  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  r <- nrow(X)
  p <- ncol(X)
  
  beta    <- rep(0, p)
  epsilon <- 1e-6
  
  numIter <- 1
  while(TRUE && numIter < 100){
    eta <- X %*% beta
    pr  <- sigmoid(eta)
    w   <- pr * (1 - pr)
    z   <- eta + (Y - pr) / w
    sw  <- sqrt(w)
    mw  <- matrix(rep(sw, times = p), ncol = p)
    
    x_work <- mw * X
    y_work <- sw * z

    Z <- cbind(x_work, y_work)
    S <- mySweep(t(Z) %*% Z, ncol(x_work))
    beta_hat <- S[1:ncol(x_work), ncol(x_work)+1]
    
    beta_new <- beta_hat
    #beta_new <- as.vector(coef(lm(y_work ~ x_work + 0)))[1:p]
    err      <- sum(abs(beta_new - beta))
    beta     <- beta_new
    if(err < epsilon)
      break
    numIter <- numIter + 1
  }
  
  beta  
  
}

##################################
## Logistic regression examples ##
##################################

## 1) Simulated data
testing_Logistic_Regression <- function(){
  
  ## This function is not graded; you can use it to 
  ## test out the 'myLinearRegression' function 
  
  ## Define parameters
  n    <- 5
  p    <- 2
  
  ## Simulate data from our assumed model.
  ## We can assume that the true intercept is 0
  X    <- matrix(rnorm(n * p), nrow = n)
  beta <- matrix(1:p, nrow = p)
  Y    <- 1* (runif(n) < sigmoid(X %*% beta))
  
  ## Save R's logistic regression coefficients
  R_coef  <- coef(glm(Y ~ X + 0, family = 'binomial'))
  
  ## Save our logistic regression coefficients
  my_coef <- myLogistic(X, Y)
  
  ## Are these two vectors different?
  sum_square_diff <- sum((R_coef - my_coef)^2)
  if(sum_square_diff <= 10^(-10)){
    return('Both results are identical')
  }else{
    return('There seems to be a problem...')
  }
  
}

## Test the regression function
testing_Logistic_Regression()

## 2) Real data. 
## Test your regression function using the 'iris' dataset in R.
## This part is open-ended, you can make tables, compare your 
## estimates with those of 'glm', make plots, etc. Note that since 
## you are performing logistic regression, the response variable
## Y must be categorical (i.e. Y <- iris$Species)
data(iris)
Y <- as.numeric(iris$Species)[1:100] - 1
X <- as.matrix(iris[1:100,2:3])
myLogistic(X, Y)
coef(glm(Y ~ X + 0, family = 'binomial'))
