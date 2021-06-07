"""
与NewTest5的区别在于
需要的变量
paras
n 产生的样本数
p 特征个数
r bias rate
real_n 选择出来的样本数
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import generate_data as gd
import pandas as pd
import tensorflow as tf
from misc import corrs
from tensorflow.python.framework import ops
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf_config = tf.ConfigProto(inter_op_parallelism_threads = 1,
                        intra_op_parallelism_threads = 1,
                        log_device_placement=True)
tf_config.gpu_options.allow_growth = True

import multiprocessing as mp
import argparse
# 与NewTest3的区别在于使用了tanh，而不是sigmoid
# 与NewTest4的区别在于使用了并行计算

parser = argparse.ArgumentParser()
parser.description = "arguments for stable learning"
parser.add_argument("--n", help="number of all generated samples", type=int)
parser.add_argument("--p", help="number of features", type=int)
parser.add_argument("--r", help="bias rate of bias selection", type=float)
parser.add_argument("--real_n", help="number of selected data samples", type=int)
parser.add_argument("--NNepoch", help="number of epoch in Neural Network", type=int)
parser.add_argument("--NNiter", help="number of iter in Step 1", type=int)
args = parser.parse_args()

# paras = {'n': 150000,
#          'p': 10,
#          'r': 1.7,
#          'real_n': 20000}

path_fig_server = 'figs_new'
path_data_server = 'result_new'
'''
def run_stable_learning_once(paras):
'''
# 先写测试，之后定义成函数，可以并行跑起来
n = args.n
p = args.p
r = args.r
real_n = args.real_n

# seeds = [22, 919, 815, 218, 369, 735, 248, 264, 350, 34, 3567]
seeds = list(np.random.randint(0,10000,size=11))
# 1. 生成训练数据集
# 记录随机数， 同一个环境下，共10个测试数据集
data_train_list = []
for s in seeds:
    temp = gd.eg8(n, p, r, random_state=s)
    select_label = np.random.choice(temp[0].shape[0], real_n, replace=False)
    temp_X = temp[0][select_label]
    temp_Y = temp[1][select_label]
    data_train_list.append((temp_X, temp_Y, temp[2]))

# 2. 生成测试数据集
# 记录随机数， 每个环境下，有一个
# 除了r不同，n，p与训练集都相同
#seeds2 = [810, 689, 839, 925, 480, 947, 743, 245, 294, 973]
seeds2 = list(np.random.randint(0,10000,size=12))
# seeds2 = list(np.random.randint(10000,size=10))
rs = [-3, -2, -1.7, -1.5, -1.3, 1.3, 1.5, 1.7, 2, 3]
data_test_dict = {}
for i in range(len(rs)):
    test_r = rs[i]
    temp = gd.eg8(n, p, test_r, random_state=seeds2[i])
    select_label = np.random.choice(temp[0].shape[0], real_n, replace=False)
    temp_X = temp[0][select_label]
    temp_Y = temp[1][select_label]
    data_test_dict[test_r] = (temp_X, temp_Y, temp[2])

paral_para = {
    'n': n,
    'p': p,
    'r': r,
    'rs': rs,
    'real_n': real_n,
    'data_test_dict': data_test_dict
}

paral_para_list = []
for i in range(len(data_train_list)):
    temp = deepcopy(paral_para)
    temp['data_train'] = data_train_list[i]
    temp['num'] = i
    paral_para_list.append(temp)


def single_train_data_result(para):
    num = para['num']
    real_n = para['real_n']
    n = para['n']
    p = para['p']
    r = para['r']
    rs = para['rs']
    data_test_dict = para['data_test_dict']
    data_train = para['data_train']
    X_train, y_train, beta_train = data_train
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
    c_d = np.ones_like(y_train)  # 真正的数据
    c_s = -np.ones_like(y_train)  # shuffled伪样本

    c = np.concatenate((c_d, c_s), axis=0)  # 分类label，真正data标为1，shuffled data标为-1

    w = np.ones([2 * y_train.shape[0]], dtype=np.float32)
    # w是权重，初始化都为1，前n个为真正data的权重，在训练中不断变化
    # 后n个为shuffled data的权重，保持1不变

    # ====== 迭代优化discriminator和weight ======

    alpha = 0.5
    n_iter = args.NNiter
    w_moment = 0.5  # 新的w和之前的w进行加权产生下一届的w
    w_list = []
    loss_lists = []
    acc_list = []
    X_ds = np.concatenate((X_train, X_train_shuffled), axis=0)
    lr = 1e-4
    for i_iter in range(n_iter):  # run n_iter times to update weight w
        curr_w = deepcopy(w)
        w_list.append(curr_w)
        # the model
        n_x = X_train.shape[1]  # number of factor
        n_y = 1
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, [None, n_x])
        Y = tf.placeholder(tf.float32, [None, n_y])
        W1 = tf.Variable(tf.random_normal([n_x, n_x], stddev=0.001, seed=1))
        b1 = tf.get_variable("b1", [1, n_x], initializer=tf.zeros_initializer())
        W2 = tf.Variable(tf.random_normal([n_x, 1], stddev=0.001, seed=1))
        b2 = tf.get_variable("b2", [1, 1], initializer=tf.zeros_initializer())
        sample_weight = tf.Variable(tf.constant(curr_w, name='sample_weight'), name='sample_weight',
                                    trainable=False, dtype=tf.float32)
        sample_weight = tf.reshape(sample_weight, [1, len(curr_w)])
        Z1 = tf.add(tf.matmul(X, W1), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(A1, W2), b2)
        A2 = tf.tanh(Z2)
        A3 = 0.5 + 0.5 * A2

        element_loss = -((0.5 + 0.5 * Y) * tf.log(tf.clip_by_value(A3, 1e-10, 1.0)) + (0.5 - 0.5 * Y) * tf.log(
            1 - tf.clip_by_value(A3, 1e-10, 1.0)))
        entropy = -tf.reduce_mean(tf.matmul(sample_weight,
                                            (0.5 + 0.5 * Y) * tf.log(tf.clip_by_value(A3, 1e-10, 1.0)) + (
                                                    0.5 - 0.5 * Y) * tf.log(
                                                1 - tf.clip_by_value(A3, 1e-10, 1.0))))  # with reweight
        loss = entropy

        train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        # train this model
        with tf.Session(config=deepcopy(tf_config)) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            steps = 20000
            steps = args.NNepoch
            loss_list = []  # record if the algo converge
            for i in range(steps):
                sess.run(train_step, feed_dict={X: X_ds, Y: c})

                if i % 1000 == 0:
                    print("{} iter,current step:{}".format(i_iter, i))

                    curr_loss_train = sess.run(loss, feed_dict={X: X_ds, Y: c})
                    loss_list.append(curr_loss_train)

            C_sd_hat = sess.run(A2, feed_dict={X: X_ds, Y: c})
            loss_perelement = sess.run(element_loss, feed_dict={X: X_ds, Y: c})
        loss_lists.append(loss_list)
        curr_alpha = alpha
        w_d_temp = curr_w[:N_train] * np.exp(curr_alpha * loss_perelement[:N_train].reshape(-1))
        print("sample of w:{}".format(w_d_temp[:5]))
        if np.isnan(w_d_temp[0]):
            print("the model is lost, rerun this epoch ")
            lr = lr * 0.8
            continue
        norm_factor = np.sum(w_d_temp)
        w_d_temp = N_train * w_d_temp / norm_factor

        temp_w = deepcopy(curr_w)
        temp_w[:N_train] = w_d_temp

        curr_w = (1 - w_moment) * curr_w + w_moment * temp_w

        w = deepcopy(curr_w)

        # C_sd_prd = np.where(C_sd_hat>0.5, 1.0, 0.0)

    # 画loss曲线，观察每次迭代中，神经网络是否收敛
    if not os.path.exists(os.path.join(path_fig_server, 'n={}_p={}_r={}'.format(real_n, p, r))):
        os.makedirs(os.path.join(path_fig_server, 'n={}_p={}_r={}'.format(real_n, p, r)))
    current_fig_path = os.path.join(path_fig_server, 'n={}_p={}_r={}'.format(real_n, p, r))
    plt.figure()
    for i in range(len(loss_lists)):
        curr_loss = loss_lists[i]
        plt.plot(curr_loss, label="{}".format(i))
    plt.title("The convergency state during iterations n={}_p={}_r={}".format(real_n, p, r))
    plt.legend()
    plt.savefig(os.path.join(current_fig_path, "No.{} n={}_p={}_r={}.png".format(num, real_n, p, r)))
    plt.close()
    #plt.show()

    # 画w曲线
    # plt.figure()
    # for i in range(len(loss_lists)):
    #     curr_w = w_list[i]
    #     plt.plot(curr_w, label="{}".format(i))
    # plt.title("The w during iterations n={}_p={}_r={}".format(real_n, p, r))
    # plt.legend()
    # plt.show()

    # 画未经过weighted的数据的corr，以及经过weighted的数据的corr
    w1 = w[:N_train]
    Xy_train = np.concatenate((X_train, y_train), axis=1)
    corr_origin = corrs(Xy_train, np.ones_like(y_train))
    corr_weighted = corrs(Xy_train, w1)

    # plt.figure()
    # plt.imshow(corr_origin, cmap=plt.cm.Reds, origin='lower')
    # plt.colorbar()
    # plt.title('No.{} corr without weight n={}_p={}_r={}'.format(num,n, p, r))
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(corr_weighted, cmap=plt.cm.Reds, origin='lower')
    # plt.colorbar()
    # plt.title('No.{} corr with weight n={}_p={}_r={}'.format(num,n, p, r))
    # plt.show()

    # apply 到各种方法上
    # ======== 2.将得到的weight放到线性回归模型中，观察对比和不用weight的效果。======
    prd_method = 'LR'  # LR=linear regression
    if prd_method == 'LR':
        from sklearn.linear_model import LinearRegression, Ridge, Lasso

        reg_origin = LinearRegression()
        reg_weighted = LinearRegression()
        reg_ridge = Ridge(alpha=.5)
        reg_lasso = Lasso(alpha=.001)

    X_train = X_train
    y_train = y_train.reshape(-1, 1)

    reg_origin.fit(X_train, y_train)
    reg_weighted.fit(X_train, y_train, w1)
    reg_ridge.fit(X_train, y_train)
    reg_lasso.fit(X_train, y_train)

    # 在不同的r上测试效果
    # OLS
    rmse_list_origin = []
    print("test without weight")
    for curr_r in rs:
        curr_data, curr_y, beta_s = data_test_dict[curr_r]
        curr_y_hat = reg_origin.predict(curr_data)
        curr_rmse = np.sqrt(np.sum(np.square(curr_y - curr_y_hat)) / curr_y.shape[0])
        rmse_list_origin.append(curr_rmse)

    # 在不同的r上测试效果
    # our method
    rmse_list_weighted = []
    print("test with GAN generated weights")
    for curr_r in rs:
        curr_data, curr_y, beta_s = data_test_dict[curr_r]
        curr_y_hat = reg_weighted.predict(curr_data)
        curr_rmse = np.sqrt(np.sum(np.square(curr_y - curr_y_hat)) / curr_y.shape[0])
        rmse_list_weighted.append(curr_rmse)

    # 在不同的r上测试效果
    # Lasso method
    rmse_list_lasso = []
    print("test with lasso")
    for curr_r in rs:
        curr_data, curr_y, beta_s = data_test_dict[curr_r]
        curr_y_hat = reg_lasso.predict(curr_data)
        curr_y_hat = curr_y_hat.reshape(-1, 1)
        curr_rmse = np.sqrt(np.sum(np.square(curr_y - curr_y_hat)) / curr_y.shape[0])
        rmse_list_lasso.append(curr_rmse)

    # 在不同的r上测试效果
    # Ridge
    rmse_list_ridge = []
    print("test without weight")
    for curr_r in rs:
        curr_data, curr_y, beta_s = data_test_dict[curr_r]
        curr_y_hat = reg_ridge.predict(curr_data)
        curr_rmse = np.sqrt(np.sum(np.square(curr_y - curr_y_hat)) / curr_y.shape[0])
        rmse_list_ridge.append(curr_rmse)

    print("origin RMSE:{}".format(rmse_list_origin))
    print("lasso RMSE:{}".format(rmse_list_lasso))
    print("ridge RMSE:{}".format(rmse_list_ridge))
    print("weighted RMSE:{}".format(rmse_list_weighted))

    def get_stable_error(rmse):
        avg_error = np.mean(rmse)
        stability_error = np.std(rmse)
        return avg_error, stability_error

    avg_error_origin, stable_error_origin = get_stable_error(rmse_list_origin)
    avg_error_lasso, stable_error_lasso = get_stable_error(rmse_list_lasso)
    avg_error_ridge, stable_error_ridge = get_stable_error(rmse_list_ridge)
    avg_error_weighted, stable_error_weighted = get_stable_error(rmse_list_weighted)

    def get_beta_error(model, real_beta):
        # 得到beta_s_error, beta_v_error
        real_beta = real_beta.reshape(-1)
        v_num = len(real_beta)
        beta_hat = model.coef_.reshape(-1)
        beta_s_hat = beta_hat[:v_num]
        beta_v_hat = beta_hat[v_num:]
        beta_s_error = np.sum(np.abs(real_beta - beta_s_hat))
        beta_v_error = np.sum(np.abs(beta_v_hat))
        return beta_s_error, beta_v_error

    beta_s_error_origin, beta_v_error_origin = get_beta_error(reg_origin, beta_train)
    beta_s_error_lasso, beta_v_error_lasso = get_beta_error(reg_lasso, beta_train)
    beta_s_error_ridge, beta_v_error_ridge = get_beta_error(reg_ridge, beta_train)
    beta_s_error_weighted, beta_v_error_weighted = get_beta_error(reg_weighted, beta_train)

    # 记录各个指标
    OLS_rcd = {'method': 'OLS',
               'n': n, 'p': p, 'r': r,
               'rmses': rmse_list_origin,
               'avg_error': avg_error_origin,
               'stable_error': stable_error_origin,
               'beta_s_error': beta_s_error_origin,
               'beta_v_error': beta_v_error_origin
               }

    Lasso_rcd = {'method': 'Lasso',
                 'n': n, 'p': p, 'r': r,
                 'rmses': rmse_list_lasso,
                 'avg_error': avg_error_lasso,
                 'stable_error': stable_error_lasso,
                 'beta_s_error': beta_s_error_lasso,
                 'beta_v_error': beta_v_error_lasso
                 }

    Ridge_rcd = {'method': 'ridge',
                 'n': n, 'p': p, 'r': r,
                 'rmses': rmse_list_ridge,
                 'avg_error': avg_error_ridge,
                 'stable_error': stable_error_ridge,
                 'beta_s_error': beta_s_error_ridge,
                 'beta_v_error': beta_v_error_ridge
                 }

    Gan_rcd = {'method': 'GAN',
               'n': n, 'p': p, 'r': r,
               'rmses': rmse_list_weighted,
               'avg_error': avg_error_weighted,
               'stable_error': stable_error_weighted,
               'beta_s_error': beta_s_error_weighted,
               'beta_v_error': beta_v_error_weighted
               }
    # 用 OLS，Lasso，Ridge，Our method分别得到的结果,每一个方法的结果用字典记录
    # 记录这个senerio的参数，n,p,r,数据generate的随机种子,先在程序里体现吧
    # 记录各个方法的beta参数，beta_s,beta_v分开，
    # 用估计的beta_s, beta_v和真实的beta_s, beta_v计算beta_s_error, beta_v_error
    # 每个测试集合下的rmse的，记录成一个表
    # 用这个rmse表求得average_error, stability_error
    plt.figure()
    plt.plot(rmse_list_origin, label='OLS')
    plt.plot(rmse_list_lasso, label='Lasso')
    plt.plot(rmse_list_ridge, label='Ridge')
    plt.plot(rmse_list_weighted, label='our method')
    plt.legend()
    plt.xticks(list(range(len(rmse_list_weighted))), rs)
    plt.ylabel('RMSE')
    plt.title('No.{} r on test data n={}_p={}_r={}'.format(num, real_n, p, r))
    # plt.savefig('result/rmse1.png')
    plt.savefig(os.path.join(current_fig_path, 'No.{} r on test data n={}_p={}_r={}.png'.format(num, real_n, p, r)))
    plt.close()
    #plt.show()

    temp_result = (OLS_rcd, Lasso_rcd, Ridge_rcd, Gan_rcd)
    return temp_result


def multiprocess_run(para_list, processes=10):
    pool = mp.Pool(processes=processes)
    result = pool.map(single_train_data_result, para_list)
    pool.close()
    pool.join()
    return result


if __name__ == '__main__':
    results_in_diff_data = multiprocess_run(paral_para_list, processes=12)

    #store_path = 'E:/Research Project/stable_learning/result2'
    store_path = path_data_server
    with open(os.path.join(store_path, 'result_n={}_p={}_r={}'.format(real_n, p, r)), 'wb') as f:
        pickle.dump(results_in_diff_data, f)
    '''
        return results_in_diff_data
    '''
    # def multiprocess_run(para_list, processes=10):
    #     pool = mp.Pool(processes=processes)
    #     result = pool.map(run_stable_learning_once, para_list)
    #     pool.close()
    #     pool.join()
    #     return result
    '''
    if __name__ == '__main__':
        para_list = [
            {"n":1000, "p":10, "r":1.7},
            {"n": 2000, "p": 10, "r": 1.7},
            {"n": 4000, "p": 10, "r": 1.7},

            {"n": 2000, "p": 10, "r": 1.5},
            {"n": 2000, "p": 20, "r": 1.5},
            {"n": 2000, "p": 40, "r": 1.5},

            {"n": 2000, "p": 10, "r": 1.5},
            {"n": 2000, "p": 10, "r": 1.7},
            {"n": 2000, "p": 10, "r": 2.0},

        ]

        #my_result = multiprocess_run(para_list[:3], processes=3)
        my_result = run_stable_learning_once(para_list[0])
    '''

    # def get_coef(model):
    #     print(model.coef_)
    #     print('model shape:{}'.format(model.coef_.shape))
    #     print('type:{}'.format(type(model.coef_)))

    # get_coef(reg_origin)

    # for rcd in temp_result:
    #     print("{} beta_s_error:{}".format(rcd['method'], rcd['beta_s_error']))
    # print("")
    # for rcd in temp_result:
    #     print("{} beta_v_error:{}".format(rcd['method'], rcd['beta_v_error']))
    # print("")
    # for rcd in temp_result:
    #     print("{} average error:{}".format(rcd['method'], rcd['avg_error']))
    # print("")
    # for rcd in temp_result:
    #     print("{} stable error:{}".format(rcd['method'], rcd['stable_error']))

    # with open(os.path.join(store_path, 'result_n={}_p={}_r={}'.format(n,p,r)), 'rb')as f:
    #     a = pickle.load(f)
    # def get_average_performance(my_result, n, p, r):
    # 得到平均的rmses，stable error，avg error，beta_s_error, beta_v_error

    # 分析的时候运行的
    """
    my_result = results_in_diff_data
    rmse_all = {'OLS': [],
                'Lasso': [],
                'Ridge': [],
                'GAN': []}
    for rslt in my_result:
        rmse_all['OLS'].append(np.array(rslt[0]['rmses']).reshape(1, -1))
        rmse_all['Lasso'].append(np.array(rslt[1]['rmses']).reshape(1, -1))
        rmse_all['Ridge'].append(np.array(rslt[2]['rmses']).reshape(1, -1))
        rmse_all['GAN'].append(np.array(rslt[3]['rmses']).reshape(1, -1))

    rmse_mean = {'OLS': None,
                 'Lasso': None,
                 'Ridge': None,
                 'GAN': None}
    for method in ['OLS', 'Lasso', 'Ridge', 'GAN']:
        rmse_mean[method] = np.mean(np.concatenate(rmse_all[method], axis=0), axis=0)

    rs = [-3, -2, -1.7, -1.5, -1.3, 1.3, 1.5, 1.7, 2, 3]
    plt.figure()
    for method in ['OLS', 'Lasso', 'Ridge', 'GAN']:
        plt.plot(rmse_mean[method], label=method)
    plt.legend()
    plt.xticks(list(range(len(rmse_mean[method]))), rs)
    plt.ylabel('RMSE')
    plt.title('r on test data n={}_p={}_r={}'.format(n, p, r))
    # plt.savefig('result/rmse1.png')
    plt.show()
    # return rmse_mean

    method_map = {'OLS': 0, 'Lasso': 1, 'ridge': 2, 'GAN': 3}

    info = {}
    info_mean = {}
    info_std = {}

    for attr in ['avg_error', 'stable_error', 'beta_s_error', 'beta_v_error']:
        info[attr] = {}
        info_mean[attr] = {}
        info_std[attr] = {}

    for attr in ['avg_error', 'stable_error', 'beta_s_error', 'beta_v_error']:
        for m in ['OLS', 'Lasso', 'ridge', 'GAN']:
            info[attr][m] = []
            info_mean[attr][m] = 0
            info_std[attr][m] = 0
    for rslt in my_result:
        for attr in ['avg_error', 'stable_error', 'beta_s_error', 'beta_v_error']:
            for m in ['OLS', 'Lasso', 'ridge', 'GAN']:
                info[attr][m].append(rslt[method_map[m]][attr])

    for attr in ['avg_error', 'stable_error', 'beta_s_error', 'beta_v_error']:
        for m in ['OLS', 'Lasso', 'ridge', 'GAN']:
            info_mean[attr][m] = np.mean(info[attr][m])
            info_std[attr][m] = np.std(info[attr][m])

    print('info_mean:')
    for attr in ['avg_error', 'stable_error', 'beta_s_error', 'beta_v_error']:
        print(attr)
        print(info_mean[attr])

    print('info_std:')
    for attr in ['avg_error', 'stable_error', 'beta_s_error', 'beta_v_error']:
        print(attr)
        print(info_std[attr])
    """




