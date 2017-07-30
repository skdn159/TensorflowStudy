import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy=np.loadtxt('QuizResults.csv',delimiter=',',dtype=np.float)
x_data = xy[1:,0:-1]
y_data = xy[1:,[-1]]

#check data 
print( x_data, len(x_data))
print( y_data)

x= tf.placeholder(tf.float32, shape=[None,3])
y= tf.placeholder(tf.float32, shape=[None,1])

w= tf.Variable(tf.random_normal([3,1]), name='weight')
b= tf.Variable(tf.random_normal([1]), name='bias')

hypo = tf.matmul(x,w)+b
cost = tf.reduce_mean(tf.square(hypo-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    cost_val, hypo_val, _  = sess.run([cost,hypo,train],
                                      feed_dict={x:x_data,y:y_data})

    if i % 20 ==0:
        print(i, "cost: ", cost_val, 
              "\nPredict:\n",hypo_val)

print("your Score will be ", sess.run(hypo,feed_dict={x:[[100,70,101]]}))
print("other Score will be ", sess.run(hypo,feed_dict={x:[[60,70,111] ,[90,100,80]] })) 
 
          

