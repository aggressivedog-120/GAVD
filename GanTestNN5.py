import os
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import generate_data as gd
from copy import deepcopy
"""
use eg6
add decay learning_rate
@20190918 the learning rate of adjusting w may matter

# generating train_data(r range in [0.3, 0.5, 0.7]), test_data(in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# change the size of train data, keep the size of test data the same
# getting weights with train_data
# using weights to train linear regression  
"""

#======================================================
data_seed = np.random.randint(1,10000)
np.random.seed(data_seed)

useless, all_data = gd.eg6(500000, N_test=10, feature_num=10,
                           stable_ratio=0.4, rho=0.7, quantile=0.6, r=0.99)

data_seed = np.random.randint(1,10000)
np.random.seed(data_seed)
useless, all_test_data = gd.eg6(500000, N_test=10, feature_num=10,
                           stable_ratio=0.4, rho=0.7, quantile=0.6, r=0.99)
test_sample_size = 1000

"""
there are several ways to update weights w, the first one is one-step exponential, the disadvantage 
is the range is hard to control, some samples' weight will become very large
so I will try a neural network in the next step
"""
weight_update_method = 'NN'  # 'one_step_exp'
for bias_rate in [0.3]:
    for sample_size in [1000]:
        data_train = gd.concate_biased_unbiased(all_data, size=sample_size, r_bias=bias_rate)

        s_train = deepcopy(data_train)  # shuffled data

        # shuffle data in s_train
        for i in range(s_train['stable'].shape[1]):
            np.random.shuffle(s_train['stable'][:,i])
        for i in range(s_train['noise'].shape[1]):
            np.random.shuffle(s_train['noise'][:,i])

        # combine data
        X_d = np.concatenate((data_train['stable'],data_train['noise']),axis=1)
        X_s = np.concatenate((s_train['stable'], s_train['noise']),axis=1)

        X_sd = np.concatenate([X_s, X_d], axis=0)

        # class label for origin data and shuffled data
        C_s = np.ones_like(s_train['Y'])  # shuffled data are labeled 1
        C_d = np.zeros_like(data_train['Y'])  # origin data are labeled 0

        C_sd = np.concatenate([C_s, C_d], axis=0)
        C_sd = C_sd.reshape(-1,1)

        w_sd = np.array([1.0]*2*sample_size)
        # weight for X_sd, but weights of X_s don't change, only weights of X_d change


        alpha=0.5
        n_iter = 3
        decay_adjust_weight = False
        decay_lr = False
        w_list = []
        loss_lists = []
        loss_lists_w = []
        acc_list = []
        network_seed = np.random.randint(1, 10000)
        for i_iter in range(n_iter):  # run n_iter times to update weight w
            curr_w = deepcopy(w_sd)
            w_list.append(curr_w)
            # the model
            n_x = X_sd.shape[1]  # number of factor X1 to X4
            n_y = 1
            tf.reset_default_graph()
            X = tf.placeholder(tf.float64, [None, n_x])
            Y = tf.placeholder(tf.float64, [None, n_y])
            W1 = tf.Variable(tf.random_normal([n_x, 2 * n_x], stddev=0.001, seed=network_seed))
            b1 = tf.get_variable("b1", [1, 2 * n_x], initializer=tf.zeros_initializer())
            W2 = tf.Variable(tf.random_normal([2 * n_x, 1], stddev=0.001, seed=network_seed))
            b2 = tf.get_variable("b2", [1, 1], initializer=tf.zeros_initializer())
            sample_weight = tf.Variable(tf.constant(curr_w, name='sample_weight'), name='sample_weight', trainable=False)
            sample_weight = tf.reshape(sample_weight, [1,len(curr_w)])
            Z1 = tf.add(tf.cast(tf.matmul(X, tf.cast(W1, tf.float64)), tf.float64), tf.cast(b1, tf.float64))
            A1 = tf.nn.relu(Z1)
            Z2 = tf.add(tf.matmul(A1, tf.cast(W2, tf.float64)), tf.cast(b2, tf.float64))
            A2 = tf.sigmoid(Z2)

            element_loss = -(Y * tf.log(tf.clip_by_value(A2, 1e-10, 1.0)) + (1 - Y) * tf.log(1 - tf.clip_by_value(A2, 1e-10, 1.0)))
            entropy = -tf.reduce_mean(tf.matmul(sample_weight,
                        Y * tf.log(tf.clip_by_value(A2, 1e-10, 1.0)) + (1 - Y) * tf.log(1 - tf.clip_by_value(A2, 1e-10, 1.0))))/tf.reduce_sum(sample_weight)# with reweight
            loss = entropy

            #@20190916 added decay learning rate
            if decay_lr:
                LR_BASE = 1e-3
                LR_DECAY = 0.1
                LR_STEP = 5000
                my_global_step = tf.Variable(0, trainable=False)
                decay_lr = tf.train.exponential_decay(LR_BASE, my_global_step, LR_STEP, LR_DECAY, staircase=True)
                train_step = tf.train.AdamOptimizer(learning_rate=decay_lr).minimize(loss, global_step=my_global_step)
            else:
                train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

            # train this model
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                steps = 30000
                loss_list = []  # record if the algo converge
                for i in range(steps):
                    sess.run(train_step, feed_dict={X: X_sd, Y: C_sd})

                    if i % 200 == 0:
                        if i%1000==0:
                            print("Domain classification NN current step:{}".format(i))

                        curr_loss_train = sess.run(loss, feed_dict={X: X_sd, Y: C_sd})
                        loss_list.append(curr_loss_train)

                C_sd_hat = sess.run(A2, feed_dict={X: X_sd, Y: C_sd})
                loss_perelement = sess.run(element_loss, feed_dict={X: X_sd, Y: C_sd})
            loss_lists.append(loss_list)


            if weight_update_method == 'one_step_exp':
                if decay_adjust_weight:
                    curr_alpha = alpha*np.power(0.1,i_iter)  # @20190918 set the learning rate as a decay one
                else:
                    curr_alpha = alpha
                w_d_temp = curr_w[-sample_size:] * np.exp(curr_alpha*loss_perelement[-sample_size:].reshape(-1))
                norm_factor = np.sum(w_d_temp)
                w_d_temp = sample_size * w_d_temp/norm_factor
                curr_w[-sample_size:] = w_d_temp
                w_sd = deepcopy(curr_w)

                C_sd_prd = np.where(C_sd_hat>0.5, 1.0, 0.0)
                delta = C_sd - C_sd_prd
                acc = len(delta[delta==0])
                acc1 = acc/len(delta)
                acc_list.append(acc1)
            elif weight_update_method == 'NN':
                lmbda = 20
                network_seed = np.random.randint(1, 10000)
                data_loss = loss_perelement[-sample_size:].reshape(1,-1)
                w_d_temp = curr_w[-sample_size:]
                tf.reset_default_graph()
                X_w = tf.placeholder(tf.float32, [None, n_x])
                loss_fixed = - tf.placeholder(tf.float32, [1, None]) # 加上一个负号，不加的话需要用maxmize，加的话用minimize
                W1_w = tf.Variable(tf.random_normal([n_x, 2 * n_x], stddev=0.001, seed=network_seed))
                b1_w = tf.get_variable("b1", [1, 2*n_x], initializer=tf.zeros_initializer())
                W2_w = tf.Variable(tf.random_normal([2 * n_x, 1], stddev=0.001, seed=network_seed))
                b2_w = tf.get_variable("b2", [1, 1], initializer=tf.zeros_initializer())

                Z1_w = tf.add(tf.matmul(X_w, W1_w), b1_w)
                A1_w = tf.nn.relu(Z1_w)
                Z2_w = tf.add(tf.matmul(A1_w, W2_w), b2_w)
                A2_w = tf.sigmoid(Z2_w)
                A2_w = A2_w / tf.reduce_sum(A2_w,axis=0)
                if lmbda >0:
                    loss_w = tf.reduce_mean(tf.matmul(loss_fixed, A2_w)) + lmbda * tf.reduce_mean(tf.square(A2_w)) # added l2 loss
                else:
                    loss_w = tf.reduce_mean(tf.matmul(loss_fixed, A2_w))
                train_step_w = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(loss_w)
                with tf.Session() as sess:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    steps_w = 20000
                    loss_list_w = [] # record if the algo converges
                    for i_w in range(steps_w):
                        sess.run(train_step_w, feed_dict={X_w: X_d, loss_fixed: data_loss})

                        if i_w % 200 == 0:
                            if i_w % 1000 == 0:
                                print("weight NN current step:{}".format(i_w))

                            curr_loss_train_w = sess.run(loss_w, feed_dict={X_w: X_d, loss_fixed: data_loss})
                            loss_list_w.append(curr_loss_train_w)
                    w_d_temp = sess.run(A2_w, feed_dict={X_w: X_d})
                    w_d_temp = w_d_temp.reshape(-1)
                    w_d_temp = sample_size * w_d_temp / np.sum(w_d_temp)
                curr_w[-sample_size:] = w_d_temp
                w_sd = deepcopy(curr_w)
                loss_lists_w.append(loss_list_w)






        # #plot prediction
        # plt.figure()
        # #plt.plot(loss_list)
        # plt.plot(C_sd_hat,label='hat')
        # plt.plot(C_sd,label='origin')
        # plt.plot(C_sd_prd, label='prd')
        # plt.title("prediction and real label of real data and shuffled data")
        # plt.legend()
        # plt.show()

        # plot loss curve
        plt.figure(figsize=(7,15))
        plt.subplot(311)
        for i in range(len(loss_lists)):
            curr_loss = loss_lists[i]
            plt.plot(curr_loss, label="{}th iteration".format(i))
        plt.title('Loss curve for different iteration alpha={}'.format(alpha))
        plt.legend()

        # plot weights
        plt.subplot(312)
        for i in range(len(w_list)):
            ws = w_list[i]
            plt.plot(ws[-sample_size:],alpha=0.5,label='{}th iteration'.format(i))
        plt.title("Weights during different iterations")
        plt.legend()

        plt.subplot(313)
        for i in range(len(loss_lists_w)):
            curr_loss_w = loss_lists_w[i]
            plt.plot(curr_loss_w, label="{}th iteration".format(i))
        plt.title('Loss curve for weights NN  in different iteration alpha={}'.format(alpha))
        plt.legend()
        plt.show()

        #每次iteration分别画张图
        for i in range(len(loss_lists)):
            plt.figure(figsize=(7,15))
            plt.subplot(311)

            curr_loss = loss_lists[i]
            plt.plot(curr_loss, label="{}th iteration".format(i))
            plt.title('Loss curve for different iteration alpha={}'.format(alpha))
            plt.legend()

            # plot weights
            plt.subplot(312)
            ws = w_list[i]
            plt.plot(ws[-sample_size:],alpha=0.5,label='{}th iteration'.format(i))
            plt.title("Weights during different iterations")
            plt.legend()

            plt.subplot(313)
            curr_loss_w = loss_lists_w[i]
            plt.plot(curr_loss_w, label="{}th iteration".format(i))
            plt.title('Loss curve for weights NN  in different iteration alpha={}'.format(alpha))
            plt.legend()
            plt.show()

        #prepare data
        X_train = X_d
        y_train = data_train['Y'].reshape(-1,1)
# X_test = np.concatenate((data_test['stable'],data_test['noise']), axis=1)
# y_test = data_test['Y'].reshape(-1,1)
'''
prd_method = 'LR'  # LR=linear regression
if prd_method == 'LR':
    from sklearn.linear_model import LinearRegression
    reg_common = LinearRegression()

    reg_weighted = []
    for reg_iter in range(len(w_list)):
        temp = LinearRegression()
        reg_weighted.append(temp)

    # ====== fitting ======
    # without weight
    reg_common.fit(X_train, y_train)
    # with weight
    for i in range(len(w_list)):
        w = w_list[i]
        w_prd = w[-sample_size:]
        reg_weighted[i].fit(X_train, y_train, w_prd)

    # test set performance
    mse_records = []
    x_range=[]
    y_range=[]
    test_bias_rate_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for test_br in test_bias_rate_range:
        test_data = gd.concate_biased_unbiased(all_test_data, size=test_sample_size, r_bias=test_br)
        X_test = np.concatenate((test_data['stable'], test_data['noise']), axis=1)
        y_test = test_data['Y'].reshape(-1, 1)
        x_range.append(np.std(X_test,axis=0))
        y_range.append(np.std(y_test,axis=0))
        mse_dict = {}
        # common, without weight
        y_common_prd = reg_common.predict(X_test)
        mse_common = np.sum(np.square(y_test - y_common_prd)) / y_test.shape[0]
        print("normal performance:\nMSE:{:.4f}".format(mse_common))
        mse_dict['common'] = mse_common
        # with weight
        for i in range(len(reg_weighted)):
            w = w_list[i]
            w_prd = w[-sample_size:]
            reg_weighted[i].fit(X_train, y_train, w_prd)
            y_weight_prd = reg_weighted[i].predict(X_test)
            mse_weighted = np.sum(np.square(y_test - y_weight_prd)) / y_test.shape[0]
            print("reweighted performance of {}:\nMSE:{:.4f}".format(i, mse_weighted))
            mse_dict['{}iter_mse'.format(i)] = mse_weighted

        mse_records.append(mse_dict)
    mse_records_df = pd.DataFrame(mse_records)
    mse_records_df.index = test_bias_rate_range
'''

# if prd_method == 'NN':
#     print("we use neural network now")
# loss_lists_regmodel=[]
# network_seed2 = np.random.randint(1,10000)
#
# mse = {}
# mse['train'] = []
# mse['test'] = []
# for i_w in range(len(w_list)+1):
#     if i_w == len(w_list):
#         w = np.array([1.0]*N_train)  # add the unweighted version
#     else:
#         w = w_list[i_w][-N_train:]
#     # build the network, 2 layers
#     tf.reset_default_graph()
#     X = tf.placeholder(tf.float64, [None, f_n])
#     Y = tf.placeholder(tf.float64, [None, 1])
#     W1 = tf.Variable(tf.random_normal([f_n, 2 * f_n], stddev=0.001, seed=network_seed2))
#     b1 = tf.get_variable("b1", [1, 2 * f_n], initializer=tf.zeros_initializer())
#     W2 = tf.Variable(tf.random_normal([2 * f_n, 1], stddev=0.001, seed=network_seed2))
#     b2 = tf.get_variable("b2", [1, 1], initializer=tf.zeros_initializer())
#     Z1 = tf.add(tf.cast(tf.matmul(X, tf.cast(W1, tf.float64)), tf.float64), tf.cast(b1, tf.float64))
#     A1 = tf.nn.relu(Z1)
#     Z2 = tf.add(tf.matmul(A1, tf.cast(W2, tf.float64)), tf.cast(b2, tf.float64))
#
#     sample_weight = tf.Variable(tf.constant(w, name='sample_weight'), name='sample_weight', trainable=False)
#     sample_weight = tf.reshape(sample_weight, [1,len(w)])
#
#     loss = tf.reduce_mean(tf.matmul(sample_weight, tf.square(Y-Z2)))
#
#     LR_BASE = 1e-3
#     LR_DECAY = 0.1
#     LR_STEP = 5000
#     my_global_step = tf.Variable(0, trainable=False)
#     decay_lr = tf.train.exponential_decay(LR_BASE, my_global_step, LR_STEP, LR_DECAY, staircase=True)
#     train_step = tf.train.AdamOptimizer(learning_rate=decay_lr).minimize(loss, global_step=my_global_step)
#
#     with tf.Session() as sess:
#         init = tf.global_variables_initializer()
#         sess.run(init)
#         steps = 10000
#         loss_list = []  # record if the algo converge
#         for i in range(steps):
#             sess.run(train_step, feed_dict={X: X_train, Y: y_train})
#             if i % 100 == 0:
#                 # if i % 1000 == 0:
#                 #     print("current step:{}".format(i))
#
#                 curr_loss_train = sess.run(loss, feed_dict={X: X_train, Y: y_train})
#                 loss_list.append(curr_loss_train)
#
#         y_train_hat = sess.run(Z2, feed_dict={X: X_train, Y: y_train})
#         y_test_hat = sess.run(Z2, feed_dict={X: X_test, Y: y_test})
#     mse_train = np.sum(np.square(y_train - y_train_hat)) / y_train.shape[0]
#     mse_test = np.sum(np.square(y_test - y_test_hat)) / y_test.shape[0]
#     if i_w == len(w_list):
#         print("origin performance without weight:\nMSE train:{:.4f}\nMSE test:{:.4f}".format(mse_train, mse_test))
#     else:
#         print("reweighted performance of {}:\nMSE train:{:.4f}\nMSE test:{:.4f}".format(i_w, mse_train, mse_test))
#     mse['train'].append(mse_train)
#     mse['test'].append(mse_test)
#     loss_lists_regmodel.append(loss_list)
#
# # plot loss curve
# plt.figure()
# for i in range(len(loss_lists_regmodel)):
#     curr_loss = loss_lists_regmodel[i]
#     if i == len(loss_lists_regmodel)-1:
#         plt.plot(curr_loss, label="without weight")
#     else:
#         plt.plot(curr_loss, label="weight during {}th iter".format(i))
# plt.title('Loss curve for different run times for XY prediction model')
# plt.legend()
# plt.show()
#
# mse_list.append(mse)
#
#     # for i in range(len(w_list)-1):
#     #     plt.plot(w_list[i+1]-w_list[i],label=i)
#     #     print(i)
#     #     print(np.min(w_list[i+1]-w_list[i]),np.max(w_list[i+1]-w_list[i]))
#     # plt.title("difference of w in w list  ")
#     # plt.legend()
#     # plt.show()
#     # aaa = pd.DataFrame(X_train).corr()
#     # bbb = pd.DataFrame(X_test).corr()
#
# for i in range(6):
#     error_train = []
#     error_test = []
#     for j in range(len(mse_list)):
#         error_train.append(mse_list[j]['train'][i])
#         error_test.append(mse_list[j]['test'][i])
#
#     error_train = np.array(error_train)
#     error_test = np.array(error_test)
#     print("&{}th iteration&${:.4f} \\pm {:.4f}$ & ${:.4f} \\pm {:.4f}$ \\ \\".format(i+1,np.mean(error_train),np.std(error_train),np.mean(error_test),np.std(error_test)))

for i in range(len(w_list)):
    w = w_list[i]
    w_prd = w[-sample_size:]
    bias_sumw = np.sum(w_prd[:300])/1000
    common_sumw = np.sum(w_prd[300:])/1000
    print('{}:{}'.format(bias_sumw,common_sumw))

