import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name = "weight")
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hypho = W*x
cost = tf.reduce_sum(tf.square(hypho-y))

learning_rate = 0.1
gradient = tf.reduce_mean((W*x - y ) *W)   # gradient : 경사
descent = W-learning_rate*gradient
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(21):
    sess.run(update,feed_dict = {x:x_data, y:y_data})
    print(i, sess.run(cost, feed_dict={x: x_data, y:y_data}), sess.run(W))





