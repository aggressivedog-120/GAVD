"""
test for GAN style reweight method
"""
from copy import deepcopy

import numpy as np

import generate_data as gd

# ********1. shuffle data********
N_train = 100
N_test = 5
np.random.seed(10)
data_train, data_test = gd.eg1(0.8, 0.2, N_train=N_train, N_test=5)

s_train = deepcopy(data_train)  # shuffled data

np.random.shuffle(s_train['X1'])  # shuffle do not return anything, the original array has been changed
np.random.shuffle(s_train['X2'])
np.random.shuffle(s_train['X3'])
np.random.shuffle(s_train['X4'])

# combine data
d_X = np.zeros([len(data_train['Y']), 4])
for i in range(4):
    d_X[:, i] = data_train['X' + str(i + 1)]

s_X = np.zeros([len(data_train['Y']), 4])
for i in range(4):
    s_X[:, i] = s_train['X' + str(i + 1)]
s_X = s_X

X_sd = np.concatenate([s_X, d_X], axis=0)

# ********2. initial data ********
c_d = np.zeros_like(s_train['Y'])  # origin data are labeled 0
c_s = np.ones_like(s_train['Y'])  # shuffled data are labeled 1

c_sd = np.concatenate([c_s, c_d])

w = np.ones_like(s_train['Y']) / len(s_train['Y'])  # w is initialized as (1/n,1/n,...,1/n)
w_s = np.ones_like(s_train['Y']) / len(s_train['Y'])  # keep unchange during the optimization

w_sd = np.concatenate([w_s, w])

# ********3. update w********
# for n loops
# predict c_d and c_s
# update w using exponential gradient
# since this is a classification problem, use log loss

# logistic regression(LR), SVM are easy to apply, good implement in sklearn
cls_method = 'SVM'  # 'LR' or 'SVM' or 'MLP' for multilayer perceptron
prob = True
if cls_method == 'LR':
    from sklearn.linear_model import LogisticRegression

    cls = LogisticRegression(solver='lbfgs')
elif cls_method == 'SVM':
    from sklearn.svm import SVC

    cls = SVC(probability=True, gamma='scale')

else:
    print("WRONG TYPE OF CLS METHOD")

n_iter = 1
alpha = 0.5
scores = []
for i in range(n_iter):
    print("======w======")
    print(w_sd)
    curr_alpha = alpha
    # cls.fit(X_sd, c_sd, w_sd)
    cls.fit(X_sd, c_sd, sample_weight=w_sd)
    # c_d_hat = cls.predict(X_sd[-N_train:])
    if prob == True:
        c_d_hat = cls.predict_proba(X_sd[-N_train:])
        c_d_hat = c_d_hat[:, 0]
    else:
        c_d_hat = cls.predict(X_sd[-N_train:])

    # w_d = w_sd[-N_train:] * np.exp(curr_alpha*(-np.log(1-c_d_hat)))
    w_d = w_sd[-N_train:] * np.exp(curr_alpha * (-1 + c_d_hat))
    norm_factor = np.sum(w_d)
    w_d = w_d / norm_factor
    w_sd[-N_train:] = w_d
    score = cls.score(X_sd, c_sd)
    scores.append(score)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(w_sd)
plt.show()
