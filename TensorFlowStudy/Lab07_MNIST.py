# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import random
import matplotlib.pyplot as plt
tf.set_random_seed(777)  # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

# MNIST data image of shape 28 * 28 = 784
x = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
y = tf.placeholder(tf.float32, [None, nb_classes])

w = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypo = tf.nn.softmax(tf.matmul(x,w)+b)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypo),axis =1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

is_correct =tf.equal(tf.arg_max(hypo,1), tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

tranin_epoch = 15
batchSize = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(tranin_epoch):
        avg_cost = 0
        tot_batch = int(mnist.train.num_examples / batchSize)

        for i in range(tot_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batchSize)
            c, _ = sess.run([cost,optimizer], 
                            feed_dict = {x:batch_xs, y: batch_ys})
            avg_cost += c / tot_batch

        print("Epoch: ", "%04d" % (epoch+1), "cost=","{:.9f}".format(avg_cost))

        # For predict
    r =  random.randint(0, mnist.test.num_examples -1)
    print("label =", sess.run(tf.arg_max(mnist.test.labels[r:r+1],1)))
    print("Predict:", sess.run(tf.arg_max(hypo,1), feed_dict={x:mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap="Greys", interpolation= "nearest")
    plt.show()




