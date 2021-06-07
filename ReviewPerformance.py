import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

n = 10000
p = 10
r = 2.0
# store_path = 'E:/Research Project/stable_learning/result'
store_path = 'result'
with open(os.path.join(store_path, 'result_n={}_p={}_r={}'.format(n, p, r)), 'rb') as f:
    my_result = pickle.load(f)

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
for attr in ['beta_s_error', 'beta_v_error', 'avg_error', 'stable_error']:
    print(attr)
    print(info_mean[attr])

print('info_std:')
for attr in ['beta_s_error', 'beta_v_error', 'avg_error', 'stable_error']:
    print(attr)
    print(info_std[attr])
