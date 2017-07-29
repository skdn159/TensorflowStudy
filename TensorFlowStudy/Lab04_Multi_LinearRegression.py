import tensorflow as tf

'''
#non-matrix Version (not used)
x1_data = [73.,93.,89.,96.,73.]     #instances' 1st test Results
x2_data = [80.,88.,91.,98.,66.]     #instances' 2nd test Results
x3_data = [75.,93.,90.,100.,70.]    #instances' 3rd test Results
y_data = [152.,185.,180.,196.,142.] #instances' final test Results

x1= tf.placeholder(tf.float32)
x2= tf.placeholder(tf.float32)
x3= tf.placeholder(tf.float32)
y= tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name = 'weight1')
w2 = tf.Variable(tf.random_normal([1]), name = 'weight2')
w3 = tf.Variable(tf.random_normal([1]), name = 'weight3')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypho = x1*w1 + x2*w2 + x3*w3 +b

cost = tf.reduce_mean(tf.square(hypho-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer)

for i in range(2001):
    cost_val, hypho_val, _ = sess.run([cost,hypho, train],
                                      feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    if i % 20 == 0 :
        print(i, " cost: ", cost_val, "\n Predict:\n", hypho_val)

'''



# matrix Version
x_data = [[73. , 80., 75.],  # instance_1 's all test Result (except y)
          [93. , 88., 93.],  # instance_2 's all test Result (except y)
          [89. , 91., 90.],
          [96. , 98., 100.],
          [73. , 66., 70.]]        

y_data = [[152.],[185.],[180.],[196.],[142.]] # SHAPE!!

x = tf.placeholder(tf.float32,shape=[None,3])
y = tf.placeholder(tf.float32,shape=[None,1])

w= tf.Variable(tf.random_normal([3,1]), name= 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypo = tf.matmul(x,w)+b
cost = tf.reduce_mean(tf.square(hypo-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    cost_val, hypo_val, _ = sess.run([cost, hypo, train], feed_dict={x:x_data,y:y_data})
    
    if i % 10 == 0:
        print(i, " cost: ", cost_val, "\n Predict: \n",hypo_val)





