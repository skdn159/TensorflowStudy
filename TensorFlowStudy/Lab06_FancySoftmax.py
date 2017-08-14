import tensorflow as tf
import numpy as np

xy = np.loadtxt("data-04-zoo.csv", delimiter=',',dtype = np.float)
x_data  = xy[:,0:-1]
y_data  = xy[:,[-1]]

nb_classes = 7

x=tf.placeholder(tf.float32, [None,16])
y= tf.placeholder(tf.int32,[None,1])

y_one_hot = tf.one_hot(y,nb_classes)
y_one_hot = tf.reshape(y_one_hot, [-1,nb_classes])

w = tf.Variable(tf.random_normal([16,nb_classes]),name='weight')
b= tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

logit = tf.matmul(x,w)+b
hypo = tf.nn.softmax(logit)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits= logit , labels= y_one_hot)

cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

prediction = tf.arg_max(hypo,1)
correct_Prediction = tf.equal(prediction,tf.arg_max(y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_Prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(optimizer,feed_dict={x:x_data,y:y_data})

        if i %100 ==0 :
            loss, acc = sess.run([cost,accuracy], feed_dict={x:x_data,y:y_data})

            print("step: {:5} \t Loss:{:.3f} \t Acc: {:.2%}".format(i,loss,acc))

    pred = sess.run(prediction, feed_dict = {x:x_data})

    for p,y in zip(pred, y_data.flatten()):
        print ("[{}] Prediction:{} True y: {}".format(p==int(y),p,int(y)))


