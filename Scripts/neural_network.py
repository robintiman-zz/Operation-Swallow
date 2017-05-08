import tensorflow as tf
import numpy as np

#Load the vectorized question pairs
train = np.load("train_vec.npy")

#Input tensors of all features x and labels y
inputX = train[:, 4:]
inputY = train[:, 0]

#Hyperparameters
learning_rate = 0.000001
training_epochs = 2000
display_step = 50
n_samples = len(inputX)
n_features = len(inputX[0])
n_labels = 1

#Feed in the feature inputs
x = tf.placeholder(tf.float32, [None, n_features])
#Create weights
W = tf.Variable(tf.zeros([n_features, n_labels]))
#Create biases
b = tf.Variable(tf.zeros([n_labels]))
#Multiply input by weights and add biases
y_values = tf.add(tf.matmul(x, W), b)
#Apply softmax to the values we just created. Softmax is the activation function
y = tf.nn.softmax(y_values)
#Feed in the label inputs
y_ = tf.placeholder(tf.float32, [None])

#Training
#Create our cost function, mean squared error
cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*n_samples)
#Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#Initialize variables and tensorflow session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#Training loop
for i in range(training_epochs):
    sess.run(optimizer, feed_dict={x: inputX, y_: inputY})

    #Write out logs of training
    if (i) % 1 == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_: inputY})
        print('Training step:', '%04d' % (i), "cost=", "{:.9f}".format(cc))

print("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b))

#Results
results = sess.run(y, feed_dict={x: inputX})
print(results)