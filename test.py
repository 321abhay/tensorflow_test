import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1,node2)
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
add_and_triple = adder_node * 3.
W = tf.Variable([.3], tf.float32)
K = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + K
init = tf.global_variables_initializer()
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
fixW = tf.assign(W, [-1.])
fixb = tf.assign(K, [1.])





sess = tf.Session()
sess.run(init)
print(sess.run(node3))
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))
print(sess.run(add_and_triple, {a: 3, b:4.5}))

print(sess.run(linear_model, {x:[1,2,3,4]}))
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
