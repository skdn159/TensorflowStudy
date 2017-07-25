import tensorflow as tf

''' 
# input manually 

x_train = [1,2,3,]
y_train=[1,2,3,]

W= tf.Variable(tf.random_normal([1]), name='weight')
b= tf.Variable(tf.random_normal([1]), name='bias')

hypho = x_train * W +b

err = tf.reduce_mean(tf.square(hypho - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01)
train = optimizer.minimize(err)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    sess.run(train)
    
    if i %100 ==0:
        print(i,"err = ", sess.run(err)," W = ",sess.run(W), " b = ",sess.run(b))


'''

#input by Place holder
W= tf.Variable(tf.random_normal([1]), name= 'weight')
b= tf.Variable(tf.random_normal([1]) , name = 'bias')

x= tf.placeholder(tf.float32, shape=([None]))
y= tf.placeholder(tf.float32, shape=([None]))

hypo = x*W +b
err = tf.reduce_mean(tf.square(hypo- y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(err)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    err_val, W_val, b_val,_ = sess.run(
        [err, W, b,train],                  # train�� ���� �Ⱦ���� ����� ���̰� �ִ�. => �翬�� mainFunc�� ���ϸ� �����ǹ̰�...
        feed_dict={x:[1,2,3,4,5], y:[2.1,3.1,4.1,5.1,6.1]})

    if i % 100 == 0:
        print('i = ',i,' err_val = ', err_val, ' W_val = ',W_val, ' b_Val = ', b_val)









