import tensorflow as  tf
import  numpy as np
import  os
tf.app.flags.DEFINE_integer('training_iteration', 1000,'number of training iterations.')  #　配置参数－迭代次数
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.') # 配置参数－模型版本
tf.app.flags.DEFINE_string('work_dir', 'model/', 'Working directory.') # 配置参数-模型存储路径
FLAGS = tf.app.flags.FLAGS
# 创建模型存储路径
if(not os.path.exists(FLAGS.work_dir)):
    os.makedirs(FLAGS.work_dir)



sess = tf.InteractiveSession()
 
x = tf.placeholder('float', shape=[None, 5],name="inputs")  # 创建输入层  5维float
y_ = tf.placeholder('float', shape=[None, 1])   # 创建输出层 1维float
w = tf.get_variable('w', shape=[5, 1], initializer=tf.truncated_normal_initializer)  # 创建网络层链接  w 并初始化
b = tf.get_variable('b', shape=[1], initializer=tf.zeros_initializer)  # 创建网络层链接  b  并初始化

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

#saver.save(sess, './model/mylinermodel.ckpt',global_step=100,write_meta_graph=False)


y = tf.add(tf.matmul(x, w) , b,name="outputs")  # 前向计算
ms_loss = tf.reduce_mean((y - y_) ** 2)  # 误差平方损失值
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(ms_loss)  # 梯度下降法 更新
train_x = np.random.randn(1000, 5)
# let the model learn the equation of y = x1 * 1 + x2 * 2 + x3 * 3
train_y = np.sum(train_x * np.array([1, 2, 3,4,5]) + np.random.randn(1000, 5) / 100, axis=1).reshape(-1, 1)
for i in range(FLAGS.training_iteration):
    loss, _ = sess.run([ms_loss, train_step], feed_dict={x: train_x, y_: train_y})
    if i%100==0:
        print("loss is:",loss)
        #graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,["inputs", "outputs"])  # 存储变量为常数值
        #tf.train.write_graph(graph, ".", FLAGS.work_dir + "liner.pb",as_text=False)  # 保存模型文件.pb
        #saver.save(sess, './model/mylinermodel-'+str(i))
        saver.save(sess,'./model/linermodel.ckpt')
tf.train.write_graph(sess.graph_def,FLAGS.work_dir,"linermodel.pbtxt",as_text=True)
print('Done exporting!')
print('Done training!')
