
## Part 1: TensorFlow basics
import tensorflow as tf
node1 = tf.constant(3.0, dtype = tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly

## Seems like the following will print [3, 4]
print(node1, node2)

## To actually evaluate the nodes, we must run the computational graph within a session
sess = tf.Session()
print(sess.run([node1, node2]))

#from __future__ import print_function
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3)", sess.run(node3))

# A graph can be parameterized to accept external inputs, known as placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

## Start a session then evaluate the adder node with specific placeholder values
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# Variables allow us to add trainable parameters to a graph
W = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows
# Until we call sess.run, the variables are uninitialized.
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# Evaluate the model on training data
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# We can manually change parameter values
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

## Define the rules for training
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

## Run 1000 rounds of gradient descent
sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))

