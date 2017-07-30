import tensorflow as tf

file_queue = tf.train.string_input_producer(
    ['QuizResults_batch.csv'],shuffle =False,name='file_queue')

reader = tf.TextLineReader()
key,value = reader.read(file_queue)

recordDefault= [[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value, record_defaults=recordDefault)

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

x= tf.placeholder(tf.float32,shape=[None,3])
y= tf.placeholder(tf.float32,shape=[None,1])

w=tf.Variable(tf.random_normal([3,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypo = tf.matmul(x,w)+b
cost = tf.reduce_mean(tf.square(hypo-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess= sess,coord= coord)

for i in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val,hypo_val,_ = sess.run([cost,hypo,train],feed_dict={x:x_batch, y:y_batch})
    if i % 40 == 0:
        print(i, " cost: ",cost_val,
              "\nPredict:\n",hypo_val)

coord.request_stop()
coord.join(threads)

                                   
        




    

        