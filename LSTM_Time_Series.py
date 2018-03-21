
###########################################
## Author: Levon Demirdjian
## Last Updated: 02/12/2018
## Description: This script implements 
## an LSTM to forecast illness severity
## using time series of influenza 
## prevalence in a number of regions 
###########################################

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
get_ipython().magic(u'matplotlib inline')

## Create tensorflow graph
tf.reset_default_graph()    ## This would reset the graphs if there were already some running

## Read data
data = pd.read_csv('region1_data.txt', sep=" ")
ts   = data[['illness_signal']]

# Convert data into array that can be broken up into training batches
TS = np.array(ts)
n_steps   = 20
f_horizon = 20

x_data = TS[:(len(TS) - n_steps) - (len(TS) % n_steps)] ## So we can divide evenly into minibatches
y_data = TS[f_horizon:(len(TS) - n_steps) - (len(TS) % n_steps) + f_horizon]

x_batches = x_data.reshape(-1, n_steps, 1)
y_batches = y_data.reshape(-1, n_steps, 1)

## Testing data
def test_data(series, forecast, n_steps):
    test_x_setup = TS[-(n_steps + forecast):]
    testX = test_x_setup[:n_steps].reshape(-1, n_steps, 1)
    testY = TS[-(n_steps):].reshape(-1, n_steps, 1)
    return testX, testY

X_test, Y_test = test_data(TS, f_horizon, n_steps)

## Define graph
n_inputs  = 1
n_neurons = 100
n_outputs = 1

## Input and output placeholders
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
dropout = tf.placeholder(tf.float32)

#basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation = tf.nn.relu) ## Create RNN object

## Create RNN structure
#basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation = tf.nn.relu)
#outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype = tf.float32)  
#multi_layer_cell = tf.contrib.rnn.MultiRNNCell([basic_cell] * n_layers)
#outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype = tf.float32)

## Create RNN structure
#basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation = tf.nn.relu)
basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units = n_neurons, activation = tf.nn.tanh)
rnn_outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype = tf.float32)
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = fully_connected(stacked_rnn_outputs, n_outputs, activation_fn = None)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

## Define loss function and training algorithm
learning_rate = 0.01
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

## Train the network
n_iterations = 10000
batch_size = 20

with tf.Session() as sess:
   init.run()
   for iteration in range(n_iterations):
       #X_batch, y_batch = [...]
       sess.run(training_op, feed_dict = {X: x_batches, y: y_batches})
       #sess.run(training_op, feed_dict = {X: X_batch, y: y_batch})
       if iteration % 100 == 0:
           mse      = loss.eval(feed_dict = {X: x_batches, y: y_batches})
           mse_test = loss.eval(feed_dict = {X: X_test, y: Y_test})
           print(iteration, "\tMSE (train):", mse)
           print(iteration, "\tMSE (test):", mse_test)
       
       y_pred = sess.run(outputs, feed_dict = {X: X_test})


## Plot results
y_pred_series = pd.Series(np.ravel(y_pred))
print(y_pred_series)

f = plt.figure()
plt.title("Predicted illness signal in region 2", fontsize=14)
plt.plot(pd.Series(np.ravel(Y_test)), label = "Actual")
plt.plot(y_pred_series, label = "Forecast")
plt.xlabel("Days into the future")
plt.ylabel("Illness signal")
plt.xticks(np.arange(0, 20, 1.0), np.arange(1, 21, 1))
legend = plt.legend(loc='upper left', shadow=True)
plt.show()


