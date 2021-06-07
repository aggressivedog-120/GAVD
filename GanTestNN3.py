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
use eg3, eg4
add decay learning_rate
@20190918 the learning rate of adjusting w may matter
"""

#======================================================
mse_list = []

for K in range(1):
    print("This is {} run time".format(K))
    figpath = 'figure_path'
    savemyfig = False
    N_train = 2000
    N_test = 1000
    depend_r_train = 0.9
    depend_r_test = 0.1
    f_n = 10 # feature number
    stable_ratio=0.4  # the percent of stable features in all features
    data_seed = np.random.randint(1,10000)
    #data_seed = 233
    np.random.seed(data_seed)
    data_train, data_test = gd.eg4(N_train=N_train, N_test=N_test, depend_ratio_train=depend_r_train, depend_ratio_test=depend_r_test, feature_num=f_n, stable_ratio=0.4)

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
    C_d = np.zeros_like(s_train['Y'])  # origin data are labeled 0

    C_sd = np.concatenate([C_s, C_d])
    C_sd = C_sd.reshape(-1,1)

    w_sd = np.array([1.0]*2*N_train)  # weight for X_sd, but weights of X_s don't change, only weights of X_d change


    alpha=0.5
    n_iter = 5
    decay_adjust_weight = False
    w_list = []
    loss_lists = []
    acc_list = []
    network_seed = np.random.randint(1,10000)
    for i_iter in range(n_iter):  # run n_iter times to update weight w
        curr_w = deepcopy(w_sd)
        w_list.append(curr_w)
        # the model
        n_x = f_n  # number of factor X1 to X4
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
                    Y * tf.log(tf.clip_by_value(A2, 1e-10, 1.0)) + (1 - Y) * tf.log(1 - tf.clip_by_value(A2, 1e-10, 1.0))))  # with reweight
        loss = entropy

        #@20190916 added decay learning rate
        LR_BASE = 1e-3
        LR_DECAY = 0.1
        LR_STEP = 5000
        my_global_step = tf.Variable(0, trainable=False)
        decay_lr = tf.train.exponential_decay(LR_BASE, my_global_step, LR_STEP, LR_DECAY, staircase=True)

        #train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
        train_step = tf.train.AdamOptimizer(learning_rate=decay_lr).minimize(loss, global_step=my_global_step)
        # train this model
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            steps = 6000
            loss_list = []  # record if the algo converge
            for i in range(steps):
                sess.run(train_step, feed_dict={X: X_sd, Y: C_sd})

                if i % 100 == 0:
                    if i%1000==0:
                        print("current step:{}".format(i))

                    curr_loss_train = sess.run(loss, feed_dict={X: X_sd, Y: C_sd})
                    loss_list.append(curr_loss_train)

            C_sd_hat = sess.run(A2, feed_dict={X: X_sd, Y: C_sd})
            loss_perelement = sess.run(element_loss, feed_dict={X: X_sd, Y: C_sd})
        loss_lists.append(loss_list)
        if decay_adjust_weight:
            curr_alpha = alpha*np.power(0.1,i_iter)  # @20190918 set the learning rate as a decay one
        else:
            curr_alpha = alpha
        w_d_temp = curr_w[-N_train:] * np.exp(curr_alpha*loss_perelement[-N_train:].reshape(-1))
        norm_factor = np.sum(w_d_temp)
        w_d_temp = N_train * w_d_temp/norm_factor
        curr_w[-N_train:] = w_d_temp
        w_sd = deepcopy(curr_w)

        C_sd_prd = np.where(C_sd_hat>0.5, 1.0, 0.0)
        delta = C_sd - C_sd_prd
        acc = len(delta[delta==0])
        acc1 = acc/len(delta)
        acc_list.append(acc1)

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
    plt.figure()
    for i in range(len(loss_lists)):
        curr_loss = loss_lists[i]
        plt.plot(curr_loss, label="{}th iteration".format(i))
    plt.title('Loss curve for different iteration')
    plt.legend()
    if savemyfig:
        plt.savefig(os.path.join(figpath, 'Loss_for_domain_classification_decaygamma.png'))
    plt.show()

    # plot weights
    plt.figure()
    for i in range(len(w_list)):
        ws = w_list[i]
        plt.plot(ws,alpha=0.5,label='{}th iteration'.format(i))
    plt.title("Weights during different iterations")
    plt.legend()
    if savemyfig:
        plt.savefig(os.path.join(figpath, 'weights_decaygamma.png'))
    plt.show()

    #prepare data
    X_train = X_d
    y_train = data_train['Y'].reshape(-1,1)
    X_test = np.concatenate((data_test['stable'],data_test['noise']), axis=1)
    y_test = data_test['Y'].reshape(-1,1)

    prd_method = 'NN'  # LR=linear regression
    if prd_method == 'LR':
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()

        print_params = False  # print the parameter or not
        # without weight
        reg.fit(X_train, y_train)
        y_test_hat = reg.predict(X_test)
        mse = np.sum(np.square(y_test - y_test_hat)) / y_test.shape[0]
        R2 = reg.score(X_test, y_test)
        print("normal performance:\nMSE:{:.4f}\nR2:{:.4f}".format(mse, R2))
        if print_params:
            print("parameters:{}".format(reg.coef_))
        # with weight
        for i in range(len(w_list)):
            w = w_list[i]
            w_prd = w[-N_train:]
            reg.fit(X_train, y_train, w_prd)
            y_test_hat = reg.predict(X_test)
            mse = np.sum(np.square(y_test - y_test_hat)) / y_test.shape[0]
            R2 = reg.score(X_test, y_test)
            print("reweighted performance of {}:\nMSE:{:.4f}\nR2:{:.4f}".format(i, mse, R2))
            if print_params:
                print("parameters:{}".format(reg.coef_))
        aaa = pd.DataFrame(X_train).corr()
        bbb = pd.DataFrame(X_test).corr()
        print("real parameter:{}".format(data_train['params']))
        sss = pd.DataFrame(X_s).corr()

    if prd_method == 'NN':
        print("we use neural network now")
    loss_lists_regmodel=[]
    network_seed2 = np.random.randint(1,10000)

    mse = {}
    mse['train'] = []
    mse['test'] = []
    for i_w in range(len(w_list)+1):
        if i_w == len(w_list):
            w = np.array([1.0]*N_train)  # add the unweighted version
        else:
            w = w_list[i_w][-N_train:]
        # build the network, 2 layers
        tf.reset_default_graph()
        X = tf.placeholder(tf.float64, [None, f_n])
        Y = tf.placeholder(tf.float64, [None, 1])
        W1 = tf.Variable(tf.random_normal([f_n, 2 * f_n], stddev=0.001, seed=network_seed2))
        b1 = tf.get_variable("b1", [1, 2 * f_n], initializer=tf.zeros_initializer())
        W2 = tf.Variable(tf.random_normal([2 * f_n, 1], stddev=0.001, seed=network_seed2))
        b2 = tf.get_variable("b2", [1, 1], initializer=tf.zeros_initializer())
        Z1 = tf.add(tf.cast(tf.matmul(X, tf.cast(W1, tf.float64)), tf.float64), tf.cast(b1, tf.float64))
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(A1, tf.cast(W2, tf.float64)), tf.cast(b2, tf.float64))

        sample_weight = tf.Variable(tf.constant(w, name='sample_weight'), name='sample_weight', trainable=False)
        sample_weight = tf.reshape(sample_weight, [1,len(w)])

        loss = tf.reduce_mean(tf.matmul(sample_weight, tf.square(Y-Z2)))

        LR_BASE = 1e-3
        LR_DECAY = 0.1
        LR_STEP = 5000
        my_global_step = tf.Variable(0, trainable=False)
        decay_lr = tf.train.exponential_decay(LR_BASE, my_global_step, LR_STEP, LR_DECAY, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate=decay_lr).minimize(loss, global_step=my_global_step)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            steps = 10000
            loss_list = []  # record if the algo converge
            for i in range(steps):
                sess.run(train_step, feed_dict={X: X_train, Y: y_train})
                if i % 100 == 0:
                    # if i % 1000 == 0:
                    #     print("current step:{}".format(i))

                    curr_loss_train = sess.run(loss, feed_dict={X: X_train, Y: y_train})
                    loss_list.append(curr_loss_train)

            y_train_hat = sess.run(Z2, feed_dict={X: X_train, Y: y_train})
            y_test_hat = sess.run(Z2, feed_dict={X: X_test, Y: y_test})
        mse_train = np.sum(np.square(y_train - y_train_hat)) / y_train.shape[0]
        mse_test = np.sum(np.square(y_test - y_test_hat)) / y_test.shape[0]
        if i_w == len(w_list):
            print("origin performance without weight:\nMSE train:{:.4f}\nMSE test:{:.4f}".format(mse_train, mse_test))
        else:
            print("reweighted performance of {}:\nMSE train:{:.4f}\nMSE test:{:.4f}".format(i_w, mse_train, mse_test))
        mse['train'].append(mse_train)
        mse['test'].append(mse_test)
        loss_lists_regmodel.append(loss_list)

    # plot loss curve
    plt.figure()
    for i in range(len(loss_lists_regmodel)):
        curr_loss = loss_lists_regmodel[i]
        if i == len(loss_lists_regmodel)-1:
            plt.plot(curr_loss, label="without weight")
        else:
            plt.plot(curr_loss, label="weight during {}th iter".format(i))
    plt.title('Loss curve for different run times for XY prediction model')
    plt.legend()
    plt.show()

    mse_list.append(mse)

    # for i in range(len(w_list)-1):
    #     plt.plot(w_list[i+1]-w_list[i],label=i)
    #     print(i)
    #     print(np.min(w_list[i+1]-w_list[i]),np.max(w_list[i+1]-w_list[i]))
    # plt.title("difference of w in w list  ")
    # plt.legend()
    # plt.show()
    # aaa = pd.DataFrame(X_train).corr()
    # bbb = pd.DataFrame(X_test).corr()

for i in range(6):
    error_train = []
    error_test = []
    for j in range(len(mse_list)):
        error_train.append(mse_list[j]['train'][i])
        error_test.append(mse_list[j]['test'][i])

    error_train = np.array(error_train)
    error_test = np.array(error_test)
    print("&{}th iteration&${:.4f} \\pm {:.4f}$ & ${:.4f} \\pm {:.4f}$ \\ \\".format(i+1,np.mean(error_train),np.std(error_train),np.mean(error_test),np.std(error_test)))



