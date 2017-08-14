import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype = np.float)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

x = tf.placeholder(tf.float32,shape=[None,8])
y = tf.placeholder(tf.float32,shape=[None,1])

w = tf.Variable(tf.random_normal([8,1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypo = tf.sigmoid(tf.matmul(x,w)+b)  # if x,w seq changed it Get ERROR!! (ex. matmul(w,x))
cost = -1* tf.reduce_mean(y *tf.log(hypo) + (1-y)*tf.log(1-hypo))
train = tf.train.GradientDescentOptimizer(learning_rate= 0.01).minimize(cost)

predict = tf.cast(hypo>0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype = tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed = {x:x_data, y:y_data}

    for i in range(10001):
        sess.run(train, feed_dict=feed)

        if i % 200 == 0:
            print(i,sess.run(cost, feed_dict = feed))

    h,c,a = sess.run([hypo,predict,accuracy], feed_dict = feed)
    print("\nhypo:\n",h,"\npredict\n",c, "\naccuracy\n",a)


