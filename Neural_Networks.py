import numpy as np
import pandas as pd

## Read in regression_auto dataset
Location = 'regression_auto.txt'
auto_data = pd.read_table(Location, index_col = False, names=['MPG','Weight', 'Price'], sep='\t', header = 0)

## Look at top of dataset
print auto_data.head()

## Save variables
MPG    = np.array(auto_data['MPG'], dtype=float)
Weight = np.array(auto_data['Weight'], dtype=float)
Price  = np.array(auto_data['Price'], dtype=float)

## Reshape arrays so we can use them in our neural networks
n = len(MPG) 
MPG = np.reshape(MPG, (n, 1))
Weight = np.reshape(Weight, (n, 1))
Price = np.reshape(Price, (n, 1))

# X = (MPG, Price), y = Weight
X = np.hstack([MPG, Price])
y = Weight

# scale units
X = X/np.amax(X, axis=0) 
y = y/np.amax(y, axis=0)


## Define neural network class
class Neural_Network(object):
  def __init__(self):
    
    #parameters
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 10
    
    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # (3x2) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer
    
  def forward(self, X):
    
    #forward propagation through our network
    self.z = np.dot(X, self.W1)        # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z)     # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3)          # final activation function
    return o 

  def sigmoid(self, s):
        
    # activation function 
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
        
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
        
    # backward propgate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error * self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    epsilon = 0.1 ## learning rate
    self.W1 += epsilon * X.T.dot(self.z2_delta)      # adjusting first set (input --> hidden) weights
    self.W2 += epsilon * self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights
    
  def train (self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)  

NN = Neural_Network()
for i in xrange(5000): # trains the NN 5,000 times
  NN.train(X, y)

## Print final results
print "Actual Output: \n" + str(y) 
print "Predicted Output: \n" + str(NN.forward(X)) 
print "Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))) # mean sum squared loss
print "\n"

