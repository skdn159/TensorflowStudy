import tensorflow as tf
import numpy as np

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype = np.float )
#x_data = np.array([0,0],[0,1],[1,0],[1,1], dtype = np.float)  why added outer []
y_data = np.array([[0],[1],[1],[0]], dtype = np.float)
#y_data = np.array([0],[1],[1],[0], dtype = np.float)


x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([2,2]), name = 'weight1')
b1 = tf.Variable(tf.random_normal([2]), name= 'bias1')
layer1 = tf.sigmoid(tf.matmul(x,w1)+b1)

w2 = tf.Variable(tf.random_normal([2,1]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([1]), name = 'bias2')
hypo = tf.sigmoid(tf.matmul(layer1,w2)+b2)


cost = -1*tf.reduce_mean( y * tf.log(hypo) + (1-y)* tf.log(1-hypo))
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

predict =tf.cast(hypo>0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,y), dtype = tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(10001):
        sess.run(train,feed_dict = {x:x_data, y:y_data})
        if i % 100 ==0 :
            print(i,sess.run(cost, feed_dict = {x:x_data,y:y_data}), "\n",sess.run([w1,w2]))

    h,c,a = sess.run([hypo,predict,accuracy], feed_dict={x:x_data,y:y_data})
    print("\nHypothesis : ",h, "\nCorrect : ",c, "\nAccuracy : ", a )

        





