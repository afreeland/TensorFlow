import tensorflow as tf

# Fourth Step

W = tf.Variable( [.3], tf.float32)
b = tf.Variable( [-.3], tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b
sess = tf.Session()

# It is important to realize `init` is a handle to TensorFlow subgraph and variables remain
# unitialized until we execute `sess.run`
init = tf.global_variables_initializer()
sess.run(init)


print(sess.run(linear_model, { x: [1,2,3,4] }))

# Used to provide values to a `loss` function
y = tf.placeholder(tf.float32)


# We will Standard Loss model for linear regression, which sums the squares of the deltas between current model and provided data
# linear_model - y => Creates a vector where each element is the corresponding examples error delta.
# tf.square => This is called to square the error
squared_deltas = tf.square(linear_model -y)

# A 'loss' function measures how far apart the current model is from the provided data
# We sum all the squared errors to create a single scalar values that abstracts the error of all examples
loss = tf.reduce_sum(squared_deltas)

print(sess.run(squared_deltas, { x: [1,2,3,4], y: [0, -1, -2, -3] }))
print(sess.run(loss, { x: [1,2,3,4], y: [0, -1, -2, -3]}))

# Manually reassigning to values to perfect values of -1 and 1. 
# A variable initialized can be reassigned using `tf.assign`
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])

# We guessed the 'perfect' values of W and b (our y array now has perfectly guessed the result)
#  W * x + b
# -1 * 1 + 1 => 0
# -1 * 2 + 1 => -1
# -1 * 3 + 1 => -2
# -1 * 4 + 1 => -3
print(sess.run(loss, { x: [1,2,3,4], y: [0, -1, -2, -3] }))

