import tensorflow as tf

W = tf.Variable( [.3], tf.float32)
b = tf.Variable( [-.3], tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b
sess = tf.Session()

# It is important to realize `init` is a handle to TensorFlow subgraph and variables remain
# unitialized until we execute `sess.run`
init = tf.global_variables_initializer()
sess.run(init)

# Used to provide values to a `loss` function
y = tf.placeholder(tf.float32)

# We will Standard Loss model for linear regression, which sums the squares of the deltas between current model and provided data
# linear_model - y => Creates a vector where each element is the corresponding examples error delta.
# tf.square => This is called to square the error
squared_deltas = tf.square(linear_model -y)

# A 'loss' function measures how far apart the current model is from the provided data
# We sum all the squared errors to create a single scalar values that abstracts the error of all examples
loss = tf.reduce_sum(squared_deltas)

# An optimizer that slowly changes each variable according to the magnitude of the derivative (below: 0.01) with respect to that variable
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(1000):
    sess.run(train, { x: [1,2,3,4], y: [0, -1, -2, -3] })

print(sess.run([W, b]))