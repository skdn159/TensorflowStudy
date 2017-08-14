import tensorflow as tf

x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

x= tf.placeholder(tf.float32, [None, 4])
y= tf.placeholder(tf.float32, [None, 3])

nb_Classes = 3

w = tf.Variable(tf.random_normal([4, nb_Classes], name = "weight"))
b = tf.Variable(tf.random_normal([nb_Classes], name = "bias"))

hypo = tf.nn.softmax(tf.matmul(x,w)+b)

cost = tf.reduce_mean( -1* tf.reduce_sum(y*tf.log(hypo), axis =1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(2001):
        sess.run(optimizer, feed_dict ={x:x_data, y:y_data})
        
        if i % 200 ==0 :
            print(i, sess.run(cost, feed_dict={x:x_data, y:y_data}))

    guessA = sess.run(hypo, feed_dict={x:[[1,11,7,9]]})
    print("GUESS A", guessA, sess.run(tf.arg_max(guessA,1)))

    #a = sess.run(hypo, feed_dict={x: [[1, 11, 7, 9]]})
    #print(a, sess.run(tf.arg_max(a, 1)))



    guessBunch = sess.run(hypo, feed_dict={x:[[1,11,7,9],[1,3,4,3],[1,1,0,1]]})
    print("Guess BUNCH", guessBunch, sess.run(tf.arg_max(guessBunch,1)))



                  





