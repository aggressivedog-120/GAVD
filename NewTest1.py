from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import generate_data as gd


# 需要检验w_moment的参数敏感性吗？

def m(x, w):
    x = x.reshape(-1)
    w = w.reshape(-1)
    temp = np.sum(x * w) / np.sum(w)
    return temp


def cov(x, y, w):
    x = x.reshape(-1)
    y = y.reshape(-1)
    w = w.reshape(-1)
    temp = np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)
    return temp


def corr(x, y, w):
    """
    x, y是n维向量，我们会计算x,y的加权相关系数，
    w是n维向量，表示每个样本的权重
    :param x, y:
    :param w:
    :return:
    """
    x = x.reshape(-1)
    y = y.reshape(-1)
    w = w.reshape(-1)
    temp = cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))
    return temp


def corrs(X, w):
    """
    计算X中p个变量两两之间的加权相关性
    :param X:
    :param w:
    :return:
    """
    p = X.shape[1]
    w = w.reshape(-1)
    temp = np.eye(p)
    for i in range(1, p):
        for j in range(i):
            temp[i, j] = corr(X[:, i], X[:, j], w)
            temp[j, i] = temp[i, j]
    return temp


"""
先用第一个方法试一试效果看
"""

N = 2000  # 生成数据的数量
X_train, y_train, beta_train = gd.eg8(n=N, p=10, r=1.7)
N_train = X_train.shape[0]

# ========阶段1，利用GAN产生权重矩阵========
# shuffle data
# train set
td = pd.DataFrame(X_train).corr()
X_train_shuffled = deepcopy(X_train)
for i in range(X_train_shuffled.shape[1]):
    np.random.shuffle(X_train_shuffled[:, i])
td_s = pd.DataFrame(X_train_shuffled).corr()

# 利用train set和shuffled train set获取weight
c_d = np.zeros_like(y_train)  # 真正的数据
c_s = np.ones_like(y_train)  # shuffled伪样本

c = np.concatenate((c_d, c_s), axis=0)  # 分类label，真正data标为0，shuffled data标为1

w = np.ones([2 * y_train.shape[0]], dtype=np.float32)
# w是权重，初始化都为1，前n个为真正data的权重，在训练中不断变化
# 后n个为shuffled data的权重，保持1不变

# ====== 迭代优化discriminator和weight ======
alpha = 0.5
n_iter = 5
w_moment = 0.2  # 新的w和之前的w进行加权产生下一届的w
w_list = []
loss_lists = []
acc_list = []
X_ds = np.concatenate((X_train, X_train_shuffled), axis=0)
for i_iter in range(n_iter):  # run n_iter times to update weight w
    curr_w = deepcopy(w)
    w_list.append(curr_w)
    # the model
    n_x = X_train.shape[1]  # number of factor
    n_y = 1
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, n_x])
    Y = tf.placeholder(tf.float32, [None, n_y])
    W1 = tf.Variable(tf.random_normal([n_x, 2 * n_x], stddev=0.001, seed=1))
    b1 = tf.get_variable("b1", [1, 2 * n_x], initializer=tf.zeros_initializer())
    W2 = tf.Variable(tf.random_normal([2 * n_x, 1], stddev=0.001, seed=1))
    b2 = tf.get_variable("b2", [1, 1], initializer=tf.zeros_initializer())
    sample_weight = tf.Variable(tf.constant(curr_w, name='sample_weight'), name='sample_weight',
                                trainable=False, dtype=tf.float32)
    sample_weight = tf.reshape(sample_weight, [1, len(curr_w)])
    Z1 = tf.add(tf.matmul(X, W1), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(A1, W2), b2)
    A2 = tf.sigmoid(Z2)

    element_loss = -(
                Y * tf.log(tf.clip_by_value(A2, 1e-10, 1.0)) + (1 - Y) * tf.log(1 - tf.clip_by_value(A2, 1e-10, 1.0)))
    entropy = -tf.reduce_mean(tf.matmul(sample_weight,
                                        Y * tf.log(tf.clip_by_value(A2, 1e-10, 1.0)) + (1 - Y) * tf.log(
                                            1 - tf.clip_by_value(A2, 1e-10, 1.0))))  # with reweight
    loss = entropy

    train_step = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(loss)
    # train this model
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        steps = 3000
        loss_list = []  # record if the algo converge
        for i in range(steps):
            sess.run(train_step, feed_dict={X: X_ds, Y: c})

            if i % 100 == 0:
                print("current step:{}".format(i))

                curr_loss_train = sess.run(loss, feed_dict={X: X_ds, Y: c})
                loss_list.append(curr_loss_train)

        C_sd_hat = sess.run(A2, feed_dict={X: X_ds, Y: c})
        loss_perelement = sess.run(element_loss, feed_dict={X: X_ds, Y: c})
    loss_lists.append(loss_list)
    curr_alpha = alpha
    w_d_temp = curr_w[:N_train] * np.exp(curr_alpha * loss_perelement[:N_train].reshape(-1))
    norm_factor = np.sum(w_d_temp)
    w_d_temp = N_train * w_d_temp / norm_factor

    temp_w = deepcopy(curr_w)
    temp_w[:N_train] = w_d_temp

    curr_w = (1 - w_moment) * curr_w + w_moment * temp_w

    w = deepcopy(curr_w)

    # C_sd_prd = np.where(C_sd_hat>0.5, 1.0, 0.0)

# 画loss曲线，观察每次迭代中，神经网络是否收敛
plt.figure()
for i in range(len(loss_lists)):
    curr_loss = loss_lists[i]
    plt.plot(curr_loss, label="{}".format(i))
plt.title("The convergency state during iterations")
plt.legend()
plt.show()

# 画w曲线
plt.figure()
for i in range(len(loss_lists)):
    curr_w = w_list[i]
    plt.plot(curr_w, label="{}".format(i))
plt.title("The w during iterations")
plt.legend()
plt.show()

# 画未经过weighted的数据的corr，以及经过weighted的数据的corr
w1 = w[:N_train]
Xy_train = np.concatenate((X_train, y_train), axis=1)
corr_origin = corrs(Xy_train, np.ones_like(y_train))
corr_weighted = corrs(Xy_train, w1)

plt.figure()
plt.imshow(corr_origin, cmap=plt.cm.Reds, origin='lower')
plt.colorbar()
plt.title('corr without weight')
plt.show()

plt.figure()
plt.imshow(corr_weighted, cmap=plt.cm.Reds, origin='lower')
plt.colorbar()
plt.title('corr with weight')
plt.show()

# ======== 2.将得到的weight放到线性回归模型中，观察对比和不用weight的效果。
# 训练带权重和不带权重的模型
prd_method = 'LR'  # LR=linear regression
if prd_method == 'LR':
    from sklearn.linear_model import LinearRegression, Ridge, Lasso

    reg_origin = LinearRegression()
    reg_weighted = LinearRegression()
    reg_ridge = Ridge(alpha=.5)
    reg_lasso = Lasso(alpha=.001)
# prepare data
X_train = X_train
y_train = y_train.reshape(-1, 1)

reg_origin.fit(X_train, y_train)
reg_weighted.fit(X_train, y_train, w1)
reg_ridge.fit(X_train, y_train)
reg_lasso.fit(X_train, y_train)

# 生成测试数据集
test_data_dict = {}
r_test = [-3, -2, -1.7, -1.5, -1.3, 1.3, 1.5, 1.7, 2, 3]
# r_test = [2]
for curr_r in r_test:
    test_data_dict[curr_r] = [gd.eg8(n=2000, p=10, r=curr_r) for i in range(10)]

# 用不带权重的测一下结果
rmse_mean_list_origin = []
rmse_std_list_origin = []
print("test without weight")
for curr_r in r_test:
    curr_data_list = test_data_dict[curr_r]
    curr_rmse_list = []
    for curr_data, curr_y, beta_s in curr_data_list:
        curr_y_hat = reg_origin.predict(curr_data)
        curr_rmse = np.sqrt(np.sum(np.square(curr_y - curr_y_hat)) / curr_y.shape[0])
        curr_rmse_list.append(curr_rmse)
    temp_rmse_mean = np.mean(np.array(curr_rmse_list))
    temp_rmse_std = np.std(np.array(curr_rmse_list))
    rmse_mean_list_origin.append(temp_rmse_mean)
    rmse_std_list_origin.append(temp_rmse_std)

# 用带权重的测一下结果
rmse_mean_list_weighted = []
rmse_std_list_weighted = []
print("test with weight")
for curr_r in r_test:
    curr_data_list = test_data_dict[curr_r]
    curr_rmse_list = []
    for curr_data, curr_y, beta_s in curr_data_list:
        curr_y_hat = reg_weighted.predict(curr_data)
        curr_rmse = np.sqrt(np.sum(np.square(curr_y - curr_y_hat)) / curr_y.shape[0])
        curr_rmse_list.append(curr_rmse)
    temp_rmse_mean = np.mean(np.array(curr_rmse_list))
    temp_rmse_std = np.std(np.array(curr_rmse_list))
    rmse_mean_list_weighted.append(temp_rmse_mean)
    rmse_std_list_weighted.append(temp_rmse_std)

# 用Lasso测试一下
rmse_mean_list_lasso = []
rmse_std_list_lasso = []
print("test with lasso")
for curr_r in r_test:
    curr_data_list = test_data_dict[curr_r]
    curr_rmse_list = []
    for curr_data, curr_y, beta_s in curr_data_list:
        curr_y_hat = reg_lasso.predict(curr_data)
        curr_y_hat = curr_y_hat.reshape(-1, 1)
        curr_rmse = np.sqrt(np.sum(np.square(curr_y - curr_y_hat)) / curr_y.shape[0])
        curr_rmse_list.append(curr_rmse)
    temp_rmse_mean = np.mean(np.array(curr_rmse_list))
    temp_rmse_std = np.std(np.array(curr_rmse_list))
    rmse_mean_list_lasso.append(temp_rmse_mean)
    rmse_std_list_lasso.append(temp_rmse_mean)

# 用Ridge测试一下
rmse_mean_list_ridge = []
rmse_std_list_ridge = []
print("test with ridge")
for curr_r in r_test:
    curr_data_list = test_data_dict[curr_r]
    curr_rmse_list = []
    for curr_data, curr_y, beta_s in curr_data_list:
        curr_y_hat = reg_ridge.predict(curr_data)
        curr_rmse = np.sqrt(np.sum(np.square(curr_y - curr_y_hat)) / curr_y.shape[0])
        curr_rmse_list.append(curr_rmse)
    temp_rmse_mean = np.mean(np.array(curr_rmse_list))
    temp_rmse_std = np.std(np.array(curr_rmse_list))
    rmse_mean_list_ridge.append(temp_rmse_mean)
    rmse_std_list_ridge.append(temp_rmse_mean)

print("origin RMSE:{}".format(rmse_mean_list_origin))
print("lasso RMSE:{}".format(rmse_mean_list_lasso))
print("ridge RMSE:{}".format(rmse_mean_list_ridge))
print("weighted RMSE:{}".format(rmse_mean_list_weighted))

plt.figure()
plt.plot(rmse_mean_list_origin, label='OLS')
plt.plot(rmse_mean_list_lasso, label='Lasso')
plt.plot(rmse_mean_list_ridge, label='Ridge')
plt.plot(rmse_mean_list_weighted, label='our method')
plt.legend()
plt.xticks(list(range(len(rmse_mean_list_weighted))), r_test)
plt.ylabel('RMSE')
plt.title('r on test data')
plt.savefig('result/rmse1.png')
plt.show()
