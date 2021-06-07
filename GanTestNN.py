import numpy as np
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import generate_data as gd

N_train = 200
N_test = 100
np.random.seed(10)
data_train, data_test = gd.eg2(0.9, 0.1, N_train=N_train, N_test=N_test)

s_train = deepcopy(data_train)  # shuffled data

np.random.shuffle(s_train['X1'])  # shuffle do not return anything, the original array has been changed
np.random.shuffle(s_train['X2'])
np.random.shuffle(s_train['X3'])
np.random.shuffle(s_train['X4'])

# combine data
X_d_train = np.zeros([len(data_train['Y']), 4])
for i in range(4):
    X_d_train[:, i] = data_train['X' + str(i + 1)]

X_s_train = np.zeros([len(data_train['Y']), 4])
for i in range(4):
    X_s_train[:, i] = s_train['X' + str(i + 1)]

X_sd_train = np.concatenate([X_s_train, X_d_train], axis=0)

# class label for origin data and shuffled data
C_s_train = np.ones_like(s_train['Y'])  # shuffled data are labeled 1
C_d_train = np.zeros_like(s_train['Y'])  # origin data are labeled 0

C_sd_train = np.concatenate([C_s_train, C_d_train])
C_sd_train = C_sd_train.reshape(-1, 1)

w_sd_train = np.array([1.0] * 2 * N_train)

alpha = 0.5
n_iter = 5
w_list = []
loss_lists = []
acc_list = []
for i_iter in range(n_iter):  # run n_iter times to update weight w
    curr_w = deepcopy(w_sd_train)
    w_list.append(curr_w)
    # the model
    n_x = 4  # number of factor X1 to X4
    n_y = 1
    tf.reset_default_graph()
    X = tf.placeholder(tf.float64, [None, n_x])
    Y = tf.placeholder(tf.float64, [None, n_y])
    W1 = tf.Variable(tf.random_normal([n_x, 2 * n_x], stddev=0.001, seed=1))
    b1 = tf.get_variable("b1", [1, 2 * n_x], initializer=tf.zeros_initializer())
    W2 = tf.Variable(tf.random_normal([2 * n_x, 1], stddev=0.001, seed=1))
    b2 = tf.get_variable("b2", [1, 1], initializer=tf.zeros_initializer())
    sample_weight = tf.Variable(tf.constant(curr_w, name='sample_weight'), name='sample_weight', trainable=False)
    sample_weight = tf.reshape(sample_weight, [1, len(curr_w)])
    Z1 = tf.add(tf.cast(tf.matmul(X, tf.cast(W1, tf.float64)), tf.float64), tf.cast(b1, tf.float64))
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(A1, tf.cast(W2, tf.float64)), tf.cast(b2, tf.float64))
    A2 = tf.sigmoid(Z2)

    element_loss = -(
                Y * tf.log(tf.clip_by_value(A2, 1e-10, 1.0)) + (1 - Y) * tf.log(1 - tf.clip_by_value(A2, 1e-10, 1.0)))
    entropy = -tf.reduce_mean(tf.matmul(sample_weight,
                                        Y * tf.log(tf.clip_by_value(A2, 1e-10, 1.0)) + (1 - Y) * tf.log(
                                            1 - tf.clip_by_value(A2, 1e-10, 1.0))))  # with reweight
    loss = entropy

    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    # train this model
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        steps = 30000
        loss_list = []  # record if the algo converge
        for i in range(steps):
            sess.run(train_step, feed_dict={X: X_sd_train, Y: C_sd_train})

            if i % 100 == 0:
                print("current step:{}".format(i))

                curr_loss_train = sess.run(loss, feed_dict={X: X_sd_train, Y: C_sd_train})
                loss_list.append(curr_loss_train)

        C_sd_hat = sess.run(A2, feed_dict={X: X_sd_train, Y: C_sd_train})
        loss_perelement = sess.run(element_loss, feed_dict={X: X_sd_train, Y: C_sd_train})
    loss_lists.append(loss_list)
    curr_alpha = alpha
    w_d_temp = curr_w[-N_train:] * np.exp(curr_alpha * loss_perelement[-N_train:].reshape(-1))
    norm_factor = np.sum(w_d_temp)
    w_d_temp = N_train * w_d_temp / norm_factor
    curr_w[-N_train:] = w_d_temp
    w_sd_train = deepcopy(curr_w)

    C_sd_prd = np.where(C_sd_hat > 0.5, 1.0, 0.0)
    delta = C_sd_train - C_sd_prd
    acc = len(delta[delta == 0])
    acc1 = acc / len(delta)
    acc_list.append(acc1)

plt.figure()
# plt.plot(loss_list)
plt.plot(C_sd_hat, label='hat')
plt.plot(C_sd_train, label='origin')
plt.plot(C_sd_prd, label='prd')
plt.legend()
plt.show()

plt.figure()
for i in range(len(loss_lists)):
    curr_loss = loss_lists[i]
    plt.plot(curr_loss, label="{}".format(i))
plt.legend()
plt.show()

prd_method = 'LR'  # LR=linear regression
if prd_method == 'LR':
    from sklearn.linear_model import LinearRegression

    reg1 = LinearRegression()
    reg2 = LinearRegression()
# prepare data
X_train = X_d_train
y_train = data_train['Y'].reshape(-1, 1)

X_test = np.zeros([len(data_test['Y']), 4])
for i in range(4):
    X_test[:, i] = data_test['X' + str(i + 1)]
y_test = data_test['Y'].reshape(-1, 1)

# model with weight
w_prdmodel = w_sd_train[-N_train:]

reg1.fit(X_train, y_train, w_prdmodel)
y_test_hat_1 = reg1.predict(X_test)

mse_1 = np.sum(np.square(y_test - y_test_hat_1)) / y_test.shape[-1]
R2_1 = reg1.score(X_test, y_test)

# model without weight
reg2.fit(X_train, y_train)
y_test_hat_2 = reg2.predict(X_test)

mse_2 = np.sum(np.square(y_test - y_test_hat_2)) / y_test.shape[-1]
R2_2 = reg2.score(X_test, y_test)
print("reweighted performance:\nMSE:{:.4f}\nR2:{:.4f}".format(mse_1, R2_1))
print("traditional performance:\nMSE:{:.4f}\nR2:{:.4f}".format(mse_2, R2_2))

plt.figure()
for ws in w_list:
    plt.plot(ws, alpha=0.5)
plt.show()

for i in range(len(w_list) - 1):
    plt.plot(w_list[i + 1] - w_list[i])
    print(i)
    print(np.min(w_list[i + 1] - w_list[i]), np.max(w_list[i + 1] - w_list[i]))
plt.show()
