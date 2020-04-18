import tensorflow as tf 

# # # x = tf.Variable(3, name = 'x')
# # # y = tf.Variable(4, name = 'y')
# # # f = x * x * y + y + 2

# # # # with tf.compat.v1.Session() as sess:
# # # sess = tf.Session()
# # # sess.run(x.initializer)
# # # sess.run(y.initializer)
# # # # x.initializer.run()
# # # # y.initializer.run()

# # #     # f = x * x * y + y + 2
# # #     # result = f.eval()
# # # result = sess.run(f)

# # # print(result)

# # # sess.close()

# # import tensorflow as tf
# # # tf.executing_eagerly() = false

# # # tf.compat.v1.disable_eager_execution()
# # # with tf.compat.v1.Session() as sess:
# # #     a = tf.constant(3)
# # #     b = tf.constant(3)
# # #     c = a+b
# # #     print(sess.run(c))
# # #     print(c.eval())

# # x = tf.Variable(3)

# # print(x.graph is tf.get_default_graph())


# import numpy as np
# from sklearn.datasets import fetch_california_housing


# housing = fetch_california_housing()

# m, n = housing.data.shape

# housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# print(housing.data)

# print(housing_data_plus_bias)

# X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

# XT = tf.transpose(X)

# theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

# with tf.Session() as ses:
#     print(XT.eval())
#     theta_value = theta.eval()

#     print("=========")
#     print(theta_value)


import numpy as np
from sklearn.datasets import fetch_california_housing

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

root_logdir = "tf_logs"

logdir = "{}/run-{}/".format(root_logdir, now)

# Download the data. California housing is a standard sklearn dataset, so we'll just use it from there.
housing = fetch_california_housing()
m, n = housing.data.shape

# Add a bias column (with all ones)
housing_data_with_bias = np.c_[np.ones((m, 1)), housing.data]

# Normalize input features
from sklearn.preprocessing import StandardScaler
housing_data_with_bias_scaled = StandardScaler().fit_transform(housing_data_with_bias)

n_epochs = 1000
learning_rate = 0.01

# Define X, y, theta
X = tf.constant(housing_data_with_bias_scaled, dtype = tf.float32, name = 'X')
y = tf.constant(housing.target.reshape(-1, 1), dtype = tf.float32, name = 'y')
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name = 'theta')
y_prediction = tf.matmul(X, theta, name = 'y_prediction')

# Compute mean squared error
error = y_prediction - y
mse = tf.reduce_mean(tf.square(error), name = 'mse')

# Compute gradients
gradients = 2/m * tf.matmul(tf.transpose(X), error)

# Update theta
theta_new = theta - learning_rate * gradients
theta_update_op = tf.assign(theta, theta_new)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Run
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print('Epoch', epoch, 'MSE =', mse.eval())
        summary_str = mse_summary.eval(feed_dict={X: housing_data_with_bias_scaled, y: housing.target.reshape(-1, 1)})
        file_writer.add_summary(summary_str, epoch)
        sess.run(theta_update_op)

    best_theta = theta.eval()

print(best_theta)
