import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# https://www.tensorflow.org/get_started/mnist/beginners

# This represents a placeholder that will store our vertices of a single image that we input
# Our image is stored as a 28px x 28px image, flattened into vertices 28 x 28 = 784
# 'None' - here this means that dimension can be of any length (in the tutorial it is 55,000)
x = tf.placeholder(tf.float32, [None, 784])

# We also need to include 'weights' and 'biases' for our model
# Variables are modifiable and can be modified by our computation
# For Machine Learning, one generally has model parameters as Variables

# We are going to create these Variables full of zeros initially. Since we are trying to 
# learn W and b, it doesnt matter what they are initially set at
# Our 'W' has a shape of [784, 10] so we can multiply the 784 image vectors to produce a 10 dimensional vetcors
# of evidence for our difference classes
W = tf.Variable(tf.zeros([784, 10]))

# Our 'b' has a shape of [10] so we can add it to the output
b = tf.Variable(tf.zeros([10]))

# Define and implement our model
# 1. Multiply x * W with the expression `tf.matmul(x, W)`.
# 2. Add the result of our matmul expression with b => tf.matmul(x, W) + b
# 3. Apply our softmax expression
y = tf.nn.softmax(tf.matmul(x, W) + b)

# To implement cross-entroy, the measure of how inefficient our predictions are, we need a placeholder
y_ = tf.placeholder(tf.float32, [None, 10])

# 1. `tf.log` computes logarithm of each element of y. 
# 2. We multiple each element of _y with corresponding element of `tf.log(y)`
# 3. `tf.reduce_sum` addes the elements in the second dimension of y, due to the `reduction_indices=[1]` parameter.
# 4. `tf.reduce_mean` computes the mean over all the examples in the batch
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Asked TensorFlow to minimize cross_entropy, our inefficient predictions, using gradient descent algorthim
# with a learning of 0.5.  TensorFlow will shift each variable a little bit in the direction that will
# reduce the cost (loss).
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Launch an InteractiveSession
sess = tf.InteractiveSession()

# Inialize our Variables
tf.global_variables_initializer().run()

# Lets train our model 1000 times
# Using small batches of random data is called 'stochastic' training
# Ideally we would want to use ALL of our data but that can be expensive,
# so by using randomly getting different subsets it much cheaper and manageable
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# How well did our model do?
# `tf.argmax` is extremely useful at providing the index of the highest tensor along some axis
# `tf.argmax(y, 1)` is the label our model thinks is most likely for each input
# While `tf.argmax(y_, 1)` is the correct one
# `tf.equal` checks to see if the prediction matches the truth
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))

# To determine what fraction are correct we are going to cast floating point #'s and take the mean
# For example [True, False, True, True] becomes [1, 0, 1, 1] which would result in 0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# This should result in roughly 92% accuracy
# Is this good? Well, not really, in fact its pretty bad. With some small changes we can get this to
# be roughly 97%...while some of the best models can get to over 99.7%

