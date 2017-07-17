import tensorflow as tf
import matplotlib.pyplot as plt

x = [1,2,3]
y = [1,2,3]

W = tf.placeholder(tf.float32)
hypo = x * W

cost = tf.reduce_mean(tf.square(hypo-y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

for i in range(-30,50):
    feed_W = i*0.1
    cur_cost , cur_W = sess.run([cost,W], feed_dict = {W:feed_W})
    W_val.append(cur_W)
    cost_val.append(cur_cost)
    
plt.plot(W_val, cost_val)
plt.show()







