import tensorflow as tf

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

x = tf.placeholder(tf.float32,shape=[None, 2])
y = tf.placeholder(tf.float32,shape=[None, 1])

w = tf.Variable(tf.random_normal([2,1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypo = tf.sigmoid(tf.matmul(x,w)+b)

cost = -1 * tf.reduce_mean(y*tf.log(hypo) + (1-y)*tf.log(1 - hypo))  # -1 must be added
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predict = tf.cast(hypo>0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,y), dtype = tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10001):
        cost_val, _ = sess.run([cost,train], feed_dict = {x:x_data, y: y_data})

        if i % 200 == 0:
            print(i , cost_val)
            
    h,c,a = sess.run([hypo,predict,accuracy], feed_dict={x:x_data, y:y_data})

    print("\nhypo: \n",h,"\npredict:\n",c, "\nAccuracy:\n",a)
     





