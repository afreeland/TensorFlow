import tensorflow as tf

# First Step

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # Also a tf.float32 implicitliy
node3 = tf.add(node1, node2)
print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))
print (sess.run(node3))