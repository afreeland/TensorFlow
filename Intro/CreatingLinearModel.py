import tensorflow as tf

# Third Step

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