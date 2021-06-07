"""
Functions for generating synthetic data.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
@20190923 added new function to store the expression
"""


def eg1(r_train, r_test, N_train=1000, N_test=500):
    """
    This is a wrong example
    variables
    x1 ~ N(0,1),
    x2 = exp(x1)+N(0,1) for probability r, N(0,1) for probability 1-r
    x3 ~ N(0,1)
    x4 ~ N(0,1)
    Y = 210 + 27.4X1 + 13.7X2 + 13.7X3 + 13.7X4 + epsilon, epsilon~N(0,1)
    :param r_train:
    :param r_test
    :param Ntrain:
    :param Ntest:
    :return:
    """

    def eg1_kernel(r, N):
        X1 = np.random.randn(N)
        X2_1 = np.exp(X1)
        X2_2 = np.random.randn(N)
        X2_prob = np.random.uniform(0, 1, N)
        X2 = np.where(X2_prob < r, X2_1, X2_2)
        X3 = np.random.randn(N)
        X4 = np.random.randn(N)
        Y = 210 + 27.4 * X1 + 13.7 * X2 + 13.7 * X3 + 13.7 * X4 + np.random.randn(N)

        data = {}
        data['X1'] = X1
        data['X2'] = X2
        data['X3'] = X3
        data['X4'] = X4
        data['Y'] = Y
        return data

    data_train = eg1_kernel(r_train, N_train)
    data_test = eg1_kernel(r_test, N_test)

    return data_train, data_test


def eg2(r_train, r_test, N_train=1000, N_test=500):
    """
    This is a wrong example
    variables
    x1 ~ N(0,1),
    x2 = exp(x1)+N(0,1) for probability r, N(0,1) for probability 1-r
    x3 ~ N(0,1)
    x4 ~ N(0,1)
    Y = 210 + 27.4X1 + 13.7X2 + 13.7X3 + 13.7X4 + epsilon, epsilon~N(0,1)
    :param r_train:
    :param r_test
    :param Ntrain:
    :param Ntest:
    :return:
    """

    def eg2_kernel(r, N):
        X1 = np.random.randn(N)
        X2_1 = np.exp(X1) + 0.1 * np.random.randn(N)  # add noise or not?
        X2_2 = np.random.randn(N)
        X2_prob = np.random.uniform(0, 1, N)
        X2 = np.where(X2_prob < r, X2_1, X2_2)
        X3 = np.random.randn(N)
        X4 = np.random.randn(N)
        Y = 210 + 27.4 * X1 + 13.7 * X3 + 13.7 * X4 + np.random.randn(N)

        data = {}
        data['X1'] = X1
        data['X2'] = X2
        data['X3'] = X3
        data['X4'] = X4
        data['Y'] = Y
        return data

    data_train = eg2_kernel(r_train, N_train)
    data_test = eg2_kernel(r_test, N_test)

    return data_train, data_test


def eg3(N_train=1000, N_test=500, depend_ratio_train=0.8, depend_ratio_test=0.2, feature_num=10, stable_ratio=0.4):
    """
    noise variables are dependent on stable variables
    :param N_train:
    :param N_test:
    :param depend_ratio_train:
    :param depend_ratio_test:
    :param feature_num:
    :param stable_ratio:
    :return:
    """

    def eg3_kernel(n, p, stable_ratio=0.4, depend_ratio=0.8):
        p_stable = int(p * stable_ratio)
        p_noise = p - p_stable
        stable_feature = np.random.randn(n, p_stable)
        noise_feature_dependent = np.zeros([n, p_noise])
        noise_feature_independent = np.random.randn(n, p_noise)
        for i in range(p_noise):
            noise_feature_dependent[:, i] = stable_feature[:, i % p_stable] + stable_feature[:,
                                                                              (i + 1) % p_stable] + 2 * np.random.randn(
                n)  # still need noise
        noise_depend_label = np.random.uniform(0, 1, n).reshape(-1, 1)
        noise_depend_label = np.concatenate([noise_depend_label] * p_noise, axis=1)
        noise_feature = np.where(noise_depend_label < depend_ratio, noise_feature_dependent, noise_feature_independent)

        b = np.zeros([p_stable, 1])
        linear_len = int(p_stable / 2)

        for i in range(linear_len):  # linear part
            b[i, 0] = (-1) ** i * (i % 3 + 1) * p / 3
        for i in range(linear_len, b.shape[0]):  # nonlinear part
            b[i, 0] = p / 2

        linear_part = np.matmul(stable_feature[:, :linear_len], b[:linear_len, 0])
        nolinear_part = np.zeros([n, 1])
        for i in range(linear_len, b.shape[0]):
            temp = stable_feature[:, i % p_stable] * stable_feature[:, (i + 1) % p_stable] * b[i, 0]
            temp = temp.reshape(-1, 1)
            nolinear_part += temp

        Y = linear_part.reshape(-1, 1) + nolinear_part + np.random.randn(n, 1)

        data = {}
        data['stable'] = stable_feature
        data['noise'] = noise_feature
        data['Y'] = Y
        data['params'] = b
        data['kernel'] = 'eg3'
        return data

    data_train = eg3_kernel(n=N_train, p=feature_num, stable_ratio=stable_ratio, depend_ratio=depend_ratio_train)
    data_test = eg3_kernel(n=N_test, p=feature_num, stable_ratio=stable_ratio, depend_ratio=depend_ratio_test)
    return data_train, data_test


def eg4(N_train=1000, N_test=500, depend_ratio_train=0.8, depend_ratio_test=0.2, feature_num=10, stable_ratio=0.4):
    """
    stable variables are dependent on noise variables
    :param N_train:
    :param N_test:
    :param depend_ratio_train:
    :param depend_ratio_test:
    :param feature_num:
    :param stable_ratio:
    :return:
    """

    def eg4_kernel(n, p, stable_ratio=0.4, depend_ratio=0.8):
        p_stable = int(p * stable_ratio)
        p_noise = p - p_stable
        noise_feature = np.random.randn(n, p_noise)
        stable_feature_dependent = np.zeros([n, p_stable])
        stable_feature_independent = np.random.randn(n, p_stable)
        for i in range(p_stable):
            stable_feature_dependent[:, i] = noise_feature[:, i % p_noise] + noise_feature[:,
                                                                             (i + 1) % p_noise] + 2 * np.random.randn(
                n)  # still need noise
        stable_depend_label = np.random.uniform(0, 1, n).reshape(-1, 1)
        stable_depend_label = np.concatenate([stable_depend_label] * p_stable, axis=1)
        stable_feature = np.where(stable_depend_label < depend_ratio, stable_feature_dependent,
                                  stable_feature_independent)

        b = np.zeros([p_stable, 1])
        linear_len = int(p_stable / 2)

        for i in range(linear_len):  # linear part
            b[i, 0] = (-1) ** i * (i % 3 + 1) * p / 3
        for i in range(linear_len, b.shape[0]):  # nonlinear part
            b[i, 0] = p / 2

        Y = np.matmul(stable_feature, b) + np.random.randn(n, 1)

        data = {}
        data['stable'] = stable_feature
        data['noise'] = noise_feature
        data['Y'] = Y
        data['params'] = b
        data['kernel'] = 'eg4'
        return data

    data_train = eg4_kernel(n=N_train, p=feature_num, stable_ratio=stable_ratio, depend_ratio=depend_ratio_train)
    data_test = eg4_kernel(n=N_test, p=feature_num, stable_ratio=stable_ratio, depend_ratio=depend_ratio_test)
    return data_train, data_test


def eg5(N_train=1000, N_test=500, feature_num=20, stable_ratio=0.4):
    """"""

    def eg5_kernel(n, p, stable_ratio):
        p_stable = int(p * stable_ratio)
        p_noise = p - p_stable
        stable_feature = np.random.randn(n, p_stable)
        noise_feature = np.random.randn(n, p_noise)

        b = np.zeros([p_stable, 1])
        linear_len = int(3 * p_stable / 4)

        for i in range(linear_len):  # linear part
            b[i, 0] = (-1) ** i * (i % 3 + 1) * p / 3
        for i in range(linear_len, b.shape[0]):  # nonlinear part
            b[i, 0] = p / 2

        linear_part = np.matmul(stable_feature[:, :linear_len], b[:linear_len, 0])
        nolinear_part = np.zeros([n, 1])
        for i in range(linear_len, b.shape[0]):
            temp = stable_feature[:, i % p_stable] * stable_feature[:, (i + 1) % p_stable] * b[i, 0]
            temp = temp.reshape(-1, 1)
            nolinear_part += temp

        Y_dn = linear_part.reshape(-1, 1) + nolinear_part
        Y = Y_dn + np.random.randn(n, 1)

        data = {}
        data['stable'] = stable_feature
        data['noise'] = noise_feature
        data['Y_dn'] = Y_dn  # Y without noise
        data['Y'] = Y
        data['params'] = b
        data['kernel'] = 'eg4'
        return data

    data_train = eg5_kernel(n=N_train, p=feature_num, stable_ratio=stable_ratio)
    data_test = eg5_kernel(n=N_test, p=feature_num, stable_ratio=stable_ratio)
    return data_train, data_test


# all of the data generation methods are wrong. Now a proper one eg6 is employed
# one thing we are not sure is that how many samples we will get with selection bias, need test
def eg6(N_train, N_test, feature_num=10, stable_ratio=0.4, rho=0.7, quantile=0.6, r=0.99):
    """
    stable features and noise features are independent with each other
    @20190924 同时返回哪些是bias样本，哪些是一般样本，分别占多少个，标签
    :param N_train: train set size
    :param N_test: test set size
    :param feature_num:
    :param stable_ratio:
    :param rho:
    :param quantile:
    :param r:
    :return:
    """
    from scipy.stats import multivariate_normal
    def eg6_kernel(n, p=feature_num, stable_ratio=stable_ratio, rho=rho, quantile=quantile, r=r):
        p_stable = int(p * stable_ratio)
        p_noise = p - p_stable
        stable_feature = np.random.randn(n, p_stable)  # iid N(0,1)
        noise_feature = np.random.randn(n, p_noise)  # iid N(0,1)

        b = np.zeros([p_stable, 1])
        linear_len = int(3 * p_stable / 4)

        for i in range(linear_len):  # linear part
            b[i, 0] = (-1) ** i * (i % 3 + 1) * p / 3
        for i in range(linear_len, b.shape[0]):  # nonlinear part
            b[i, 0] = p / 2

        linear_part = np.matmul(stable_feature[:, :linear_len], b[:linear_len, 0])
        nolinear_part = np.zeros([n, 1])
        for i in range(linear_len, b.shape[0]):
            temp = stable_feature[:, i % p_stable] * stable_feature[:, (i + 1) % p_stable] * b[i, 0]
            temp = temp.reshape(-1, 1)
            nolinear_part += temp

        Y_dn = linear_part.reshape(-1, 1) + nolinear_part  # without noise
        Y = Y_dn + np.random.randn(n, 1)

        # conducting selection bias
        rv = multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]])  # distribution
        # rv2 = multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])  # distribution
        threshold = quantile * rv.pdf([0.0, 0.0])  # this is the point with the highest pdf value
        # threshold = quantile * (rv.pdf([0.0, 0.0]) - rv2.pdf([0.0, 0.0]))  # this is the point with the highest pdf value

        Y_normed = (Y_dn - np.mean(Y_dn)) / np.std(Y_dn)

        YV_pdf = np.zeros_like(noise_feature)

        for i in range(noise_feature.shape[1]):
            curr_noise = noise_feature[:, i:i + 1]
            combined = np.concatenate([Y_normed, curr_noise], axis=1)
            curr_pdf = rv.pdf(combined)
            # curr_pdf = rv.pdf(combined) - rv2.pdf(combined)
            YV_pdf[:, i] = curr_pdf

        YV_pdf_bool = YV_pdf > threshold

        YV_pdf_ANDlabel = np.array([True] * YV_pdf_bool.shape[0]).reshape(-1, 1)
        for i in range(YV_pdf_bool.shape[1]):
            YV_pdf_ANDlabel = YV_pdf_ANDlabel & YV_pdf_bool[:, i:i + 1]

        YV_pdf_ANDlabel = YV_pdf_ANDlabel.reshape(-1)

        # select according to probability r
        r_select = np.random.uniform(0, 1, YV_pdf_ANDlabel.shape)
        r_select_bool = np.where(r_select < r, True, False)
        YV_pdf_ANDlabel2 = [None] * len(YV_pdf_ANDlabel)
        for i in range(len(YV_pdf_ANDlabel)):
            if YV_pdf_ANDlabel[i]:
                if r_select_bool[i]:
                    YV_pdf_ANDlabel2[i] = True
                else:
                    YV_pdf_ANDlabel2[i] = False
            elif not YV_pdf_ANDlabel[i]:
                if not r_select_bool[i]:
                    YV_pdf_ANDlabel2[i] = True
                else:
                    YV_pdf_ANDlabel2[i] = False

        YV_pdf_ANDlabel2 = np.array(YV_pdf_ANDlabel2)

        data = {}
        data['stable'] = stable_feature
        data['noise'] = noise_feature
        data['Y_dn'] = Y_dn  # Y without noise
        data['Y'] = Y
        data['Y_normed'] = Y_normed
        data['params'] = b
        data['kernel'] = 'eg4'
        data['biased_label'] = YV_pdf_ANDlabel

        bias_data = {}
        bias_data['Y'] = data['Y'][YV_pdf_ANDlabel2 == True].reshape(-1, 1)
        bias_data['Y_normed'] = data['Y_normed'][YV_pdf_ANDlabel2 == True]
        bias_data['noise'] = data['noise'][YV_pdf_ANDlabel2 == True]
        bias_data['stable'] = data['stable'][YV_pdf_ANDlabel2 == True]

        return bias_data, data

    bias, origin = eg6_kernel(n=N_train, p=feature_num, stable_ratio=stable_ratio,
                              rho=rho, quantile=quantile, r=r)
    return bias, origin


def eg7(N_train, N_test, feature_num=10, stable_ratio=0.4, rho=0.7, alpha=5):
    """
    stable features and noise features are independent with each other
    the above function eg6 is too complex when selecting biased data
    这里用alpha这个值来控制被选中的概率，p_selected = exp(alpha * pdf value/pdf max value)
    :param N_train: train set size
    :param N_test: test set size
    :param feature_num:
    :param stable_ratio:
    :param rho:
    :param alpha range from -inf to inf
    :return:
    """
    from scipy.stats import multivariate_normal
    def eg7_kernel(n, p=feature_num, stable_ratio=stable_ratio, rho=rho, alpha=alpha):
        p_stable = int(p * stable_ratio)
        p_noise = p - p_stable
        stable_feature = np.random.randn(n, p_stable)  # iid N(0,1)
        noise_feature = np.random.randn(n, p_noise)  # iid N(0,1)

        b = np.zeros([p_stable, 1])
        linear_len = int(3 * p_stable / 4)

        for i in range(linear_len):  # linear part
            b[i, 0] = (-1) ** i * (i % 3 + 1) * p / 3
        for i in range(linear_len, b.shape[0]):  # nonlinear part
            b[i, 0] = p / 2

        linear_part = np.matmul(stable_feature[:, :linear_len], b[:linear_len, 0])
        nolinear_part = np.zeros([n, 1])
        for i in range(linear_len, b.shape[0]):
            temp = stable_feature[:, i % p_stable] * stable_feature[:, (i + 1) % p_stable] * b[i, 0]
            temp = temp.reshape(-1, 1)
            nolinear_part += temp

        Y_dn = linear_part.reshape(-1, 1) + nolinear_part  # without noise
        Y = Y_dn + np.random.randn(n, 1)

        # conducting selection bias
        rv = multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]])  # distribution
        maxpdf = rv.pdf([0.0, 0.0])  # this is the point with the highest pdf value
        # threshold = quantile * (rv.pdf([0.0, 0.0]) - rv2.pdf([0.0, 0.0]))  # this is the point with the highest pdf value

        Y_normed = (Y_dn - np.mean(Y_dn)) / np.std(Y_dn)

        YV_pdf = np.zeros_like(noise_feature)

        for i in range(noise_feature.shape[1]):
            curr_noise = noise_feature[:, i:i + 1]
            combined = np.concatenate([Y_normed, curr_noise], axis=1)
            curr_pdf = rv.pdf(combined)
            # curr_pdf = rv.pdf(combined) - rv2.pdf(combined)
            YV_pdf[:, i] = curr_pdf

        YV_pdf_scaled = np.exp(alpha * (YV_pdf / maxpdf - 1))

        select_mat = np.random.uniform(0, 1, YV_pdf.shape)

        YV_selected_mat = YV_pdf_scaled > select_mat

        YV_pdf_ANDlabel = (np.sum(YV_selected_mat, axis=1) == YV_selected_mat.shape[1])

        data = {}
        data['stable'] = stable_feature
        data['noise'] = noise_feature
        data['Y_dn'] = Y_dn  # Y without noise
        data['Y'] = Y
        data['Y_normed'] = Y_normed
        data['params'] = b
        data['kernel'] = 'eg4'

        bias_data = {}
        bias_data['Y'] = data['Y'][YV_pdf_ANDlabel == True].reshape(-1, 1)
        bias_data['Y_normed'] = data['Y_normed'][YV_pdf_ANDlabel == True]
        bias_data['noise'] = data['noise'][YV_pdf_ANDlabel == True]
        bias_data['stable'] = data['stable'][YV_pdf_ANDlabel == True]

        return bias_data, data

    bias, origin = eg7_kernel(n=N_train, p=feature_num, stable_ratio=stable_ratio,
                              rho=rho, alpha=alpha)
    return bias, origin


def eg8(n=2000, p=10, r=1.7, random_state=10):
    """
    S与V不相关
    p_s = 0.5*p
    p_v = 0.5*p
    z_1,...,z_p ~ N(0,1),独立分布，WHY？
    v_1,...,v_pv ~ N(0,1)
    s_i = 0.8*z_i + 0.2*z_{i+1} i=1,2,3...p_s

    y_poly = [S,V] [beta_s, beta_v] + S_1* S_2* S_3 + epsilon
    y_exp = [S,V] [beta_s, beta_v] + exp(S_1* S_2* S_3) + epsilon

    beta_s = mod(i+1,s)/3*(-1)^i
    beta_v = 0
    epsilon ~ N(0, 0.3)

    GENERATE ENV
    varying P(V|S), only change P(V_b|S) on a subset of unstable features V_b belongs to V,
    dimension of V_b is 0.1*p

    r in [-3,-1) and (1,3]
    Pr = prod(abs(r)^(-5*D_i))
    D_i = |f(S)-sign(r)*V_i|
    :param n:
    :param p:
    :param r:
    :return:
    """
    # for S与V不相关
    # p=10
    # n=20000
    # r = 1.7
    np.random.seed(random_state)
    p_s = int(p * 0.5)
    p_v = p - p_s
    Z = np.random.randn(n, p_s + 1)
    S = np.zeros([n, p_s])
    V = np.random.randn(n, p_v)
    for i in range(p_s):
        S[:, i] = 0.8 * Z[:, i] + 0.2 * Z[:, i + 1]
    beta_s = [(i % 3 + 1) / 3 * (-1) ** i for i in range(p_s)]
    beta_s = np.array(beta_s).reshape(-1, 1)

    fS_linear = np.matmul(S, beta_s)
    fS_nonlinear = (S[:, 0] * S[:, 1] * S[:, 2]).reshape(-1, 1)
    noise = np.sqrt(0.3) * np.random.randn(n, 1)
    y = fS_linear + fS_nonlinear + noise
    # print("Z:{}\n".format(Z.shape),
    #       "S:{}\n".format(S.shape),
    #       "V:{}\n".format(V.shape),
    #       "beta_s:{}\n".format(beta_s.shape),
    #       "fS_linear:{}\n".format(fS_linear.shape),
    #       "fS_nonlinear:{}\n".format(fS_nonlinear.shape),
    #       "noise:{}\n".format(noise.shape),
    #       "y:{}\n".format(y.shape))

    p_b = int(0.1 * p)

    D = np.zeros([n, p_b])
    for i in range(p_b):
        D[:, i] = np.abs(fS_linear + fS_nonlinear - np.sign(r) * V[:, -(i + 1)].reshape(-1, 1)).reshape(-1)
    print("D shape:{}".format(D.shape))

    r_D = np.abs(r) ** (-5 * D)

    Pr = np.prod(r_D, axis=1).reshape(-1, 1)

    uniform_label = np.random.uniform(0, 1, [n, 1])

    label = uniform_label < Pr
    label = label.reshape(-1)
    S_selected = S[label]
    V_selected = V[label]
    y_selected = y[label]

    all_data = np.concatenate((S_selected, V_selected, y_selected), axis=1)
    all_data_df = pd.DataFrame(all_data)
    all_data_df.columns = ['S_{}'.format(i) for i in range(1, p_s + 1)] + ['V_{}'.format(i) for i in
                                                                           range(1, p_v + 1)] + ['y']
    # plt.imshow(all_data_df.corr(), cmap=plt.cm.Reds, origin='lower')
    # plt.colorbar()
    # plt.show()
    return np.concatenate((S_selected, V_selected), axis=1), y_selected, beta_s
    # 加入了参数beta_s作为返回值


def get_expression(data):
    """
    输出对应的英文表达式，就是3/4是线性的，1/4是非线性的
    :param data: eg6的输出
    :return:
    """
    expression = 'Y = '
    params = data['params']
    p_stable = data['stable'].shape[1]
    p_linear = int(p_stable * 3 / 4)
    # linear part
    for i in range(p_linear):
        if i != 0:
            expression += '+'
        expression += '({:.2f} X{})'.format(params[i, 0], i)
    # nonlinear part
    for i in range(p_linear, p_stable):
        expression += '+'
        expression += '({:.2f} X{}*X{})'.format(params[i, 0], i % p_stable, (i + 1) % p_stable)
    return expression


def plot_data(bias_data, raw_data, rho):
    from scipy.stats import multivariate_normal
    rv = multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]])  # distribution
    rv2 = multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])  # distribution
    for i in range(bias_data['noise'].shape[1]):
        plt.figure()
        plt.scatter(bias_data['noise'][:, i], bias_data['Y'], c='red', alpha=0.1, label='biased data')
        # plt.xlim(-4,4)
        # plt.ylim(-4,4)
        # plt.scatter(raw_data['noise'][::100,i], raw_data['Y_normed'][::100], c='blue',alpha=0.05,label='original data')
        # x, y = np.mgrid[-4:4:.01, -4:4:.01]
        # pos = np.empty(x.shape + (2,))
        # pos[:, :, 0] = x
        # pos[:, :, 1] = y
        # C = plt.contour(x, y, rv.pdf(pos))
        # # C = plt.contour(x, y, rv.pdf(pos)-rv2.pdf(pos))
        # plt.clabel(C, inline=True, fontsize=10)
        plt.xlabel('{}th noise variable'.format(i))
        plt.ylabel('Y')
        plt.title('{}th noise variable'.format(i))
        plt.show()
    for i in range(bias_data['stable'].shape[1]):
        plt.figure()
        plt.scatter(bias_data['stable'][:, i], bias_data['Y'], c='blue', alpha=0.1, label='biased data')
        plt.xlabel('{}th stable variable'.format(i))
        plt.ylabel('Y')
        plt.title('{}th stable variable'.format(i))
        plt.show()


def plot_data_origin(raw_data):
    for i in range(raw_data['stable'].shape[1]):
        plt.figure()
        plt.scatter(raw_data['stable'][:, i], raw_data['Y_normed'], c='red', alpha=0.05, label='raw_data')
        plt.xlabel('{}th stable variable'.format(i))
        plt.ylabel('Y')
        plt.title('{}th variable'.format(i))
        plt.show()


def concate_biased_unbiased(data, size=1000, r_bias=0.4):
    """
    input real_data from eg6
    :param data:
    :return:
    """
    size_bias = int(size * r_bias)
    size_common = size - size_bias
    label = np.array(list(range(len(data['Y']))))
    data_bias_label = label[data['biased_label']]
    data_common_label = label[~data['biased_label']]
    bias_select_label = random.sample(list(data_bias_label), size_bias)
    unbias_select_label = random.sample(list(data_common_label), size_common)
    elements = ['stable', 'noise', 'Y', 'Y_dn', 'Y_normed']

    selected_data = {}
    for e in elements:
        biased = data[e][bias_select_label]
        common = data[e][unbias_select_label]
        selected_data[e] = np.concatenate((biased, common), axis=0)
        print("{} has length:{}".format(e, selected_data[e].shape))

    selected_data['params'] = data['params']
    selected_data['kernel'] = data['kernel']

    return selected_data


if __name__ == '__main__':
    # mydata = eg1(r_train=0.8, r_test=0.2, N_train=1000, N_test=500)
    # mydata = eg4(1000,500,0.8,0.2,10,0.6)
    #
    # train = mydata[0]
    # test = mydata[1]
    # aaa = pd.DataFrame(np.concatenate((train['stable'], train['noise']), axis=1)).corr()
    # bbb = pd.DataFrame(np.concatenate((test['stable'], test['noise']), axis=1)).corr()
    bias_data, real_data = eg6(500000, 10, 10, 0.4, 0.7, 0.6, 0.99)
    # bias_data, real_data = eg7(100000, 10, 10, 0.4, 0.7, 5) # 这方法还没有原先eg6好用
    # plot_data(bias_data, real_data, rho=0.7)
    # #plot_data_origin(bias_data)
    # print(get_expression(real_data))
    # print("Bias Selected Data:{}".format(bias_data['noise'].shape[0]))
    # print("Bias Selected Data Ratio:{}".format(bias_data['noise'].shape[0]/real_data['noise'].shape[0]))
    selected_data = concate_biased_unbiased(real_data, 2000, 0.8)
    plot_data(selected_data, real_data, 0.7)

    alls = np.concatenate([selected_data['stable'], selected_data['noise']], axis=1)
    alls = pd.DataFrame(alls)
    corralls = alls.corr()
    plt.figure()
    x = alls.iloc[:, 1]
    y = alls.iloc[:, 2]
    plt.scatter(x, y, alpha=0.1)
    plt.show()

    stable = pd.DataFrame(selected_data['stable'])
    noise = pd.DataFrame(selected_data['noise'])

    corr_stable = stable.corr()
    corr_noise = noise.corr()

    corr_real_stable = pd.DataFrame(real_data['stable']).corr()
