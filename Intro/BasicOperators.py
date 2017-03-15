import tensorflow as tf

# Second Step

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b # + provides a shortcut for tf.add(a, b)

add_and_triple = adder_node * 3

sess = tf.Session()

# Adds 3 + 4.5
# Result 7.5
print(sess.run(adder_node, { a: 3, b: 4.5 }))

# Adds a[0] + b[0] => 1 + 2
# Adds a[1] + b[1] => 3 + 4
# Result 3, 7
print(sess.run(adder_node, { a: [1, 3], b: [2, 4] }))

# Takes the result of adder_node and multiplies by 3 => adder_node * 3 => (a + b) * 3
# Result 24
print(sess.run(add_and_triple, { a: 3, b: 5} ))