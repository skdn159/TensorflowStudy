import tensorflow as tf

#hello
'''
hello = tf.constant("Hello")

sess = tf.Session()
print(sess.run(hello))

'''

# sum
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(2.0, tf.float32)
node3 = tf.add(node1, node2)

print("node1=", node1, " node2=", node2)   # node1= Tensor("Const:0", shape=(), dtype=float32)  node2= Tensor("Const_1:0", shape=(), dtype=float32)
print("node3=", node3)                     # node3= Tensor("Add:0", shape=(), dtype=float32)

sess = tf.Session()
print("sess.Run(node1,node2) = ", sess.run([node1,node2]))  # sess.Run(node1,node2) =  [3.0, 2.0]
print("sess.Run(node3) = ", sess.run(node3))                # sess.Run(node3) =  5.0


#place holder
a = tf.placeholder(tf.float32)
b= tf.placeholder(tf.float32)
add_node = a+b

print(sess.run(add_node, feed_dict={a:3.5, b:4.5}))         # 8.0
print(sess.run(add_node, feed_dict={a:[1,3], b:[-2, -3]}))  # [-1.  0.]







