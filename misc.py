import numpy as np

def m(x, w):
    x = x.reshape(-1)
    w = w.reshape(-1)
    temp = np.sum(x *w) / np.sum(w)
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
    temp = cov(x, y, w) / np.sqrt(cov(x, x, w)*cov(y, y, w))
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
    for i in range(1,p):
        for j in range(i):
            temp[i,j] = corr(X[:,i], X[:,j], w)
            temp[j,i] = temp[i,j]
    return temp