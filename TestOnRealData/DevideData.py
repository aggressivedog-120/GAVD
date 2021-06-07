"""
划分数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import pickle as pk

from TestOnRealData.futures_info import ex_dl,ex_zz,ex_cf,ex_sh

def sum_pos(s: pd.Series):
    return np.sum(s>0) / len(s)

def sum_neg(s: pd.Series):
    return np.sum(s<0) / len(s)


store_path = 'E:\\Research Project\\stable_learning\\RealData'

# for this_future in list(ex_sh.keys()):
for this_future in ['CU']:
    future = this_future.upper()
    print('current future: {}'.format(future))

    data = pd.read_csv(os.path.join(store_path, future+'.csv'))

    # 做判断，数据是否满足总长度要求
    if data.shape[0]< 2000:
        print('    not suitable enough')
        continue

    window_len = 500  # 一段连续风格的市场至少要有500个数据点

    data.loc[:,'return_mean_{}'.format(window_len)] = data.loc[:,'chgPct'].rolling(window=window_len).mean()

    data.loc[:,'return_std_{}'.format(window_len)] = data.loc[:,'chgPct'].rolling(window=window_len).std()

    data.loc[:,'Sharpe_{}'.format(window_len)] = data.loc[:,'return_mean_{}'.format(window_len)]*np.sqrt(245)/data.loc[:,'return_std_{}'.format(window_len)]

    data.loc[:, 'chg_sign'] = np.sign(data.loc[:, 'chgPct'])

    data.loc[:, 'up_ratio_{}'.format(window_len)] = data.loc[:,'chg_sign'].rolling(window=window_len).apply(sum_pos)

    data.loc[:, 'down_ratio_{}'.format(window_len)] = data.loc[:,'chg_sign'].rolling(window=window_len).apply(sum_neg)


    # plt.figure()
    # plt.plot(data.loc[:,'sharpe_{}'.format(window_len)])
    # plt.title('{} sharpe ratio'.format(future.upper()))
    # plt.show()
    #
    # plt.figure()
    # plt.plot(data.loc[:,'closePrice'])
    # plt.title('{} daily close px'.format(future.upper()))
    # plt.show()
    # 画夏普率图
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    l1, = ax1.plot(data['Sharpe_{}'.format(window_len)], 'b')
    plt.hlines(0,xmin=data.index[0], xmax=data.index[-1])
    plt.hlines(1,xmin=data.index[0], xmax=data.index[-1])
    plt.hlines(-1,xmin=data.index[0], xmax=data.index[-1])

    ax1.set_ylabel('{} Sharpe ratio in {} days'.format(future, window_len))

    ax2 = ax1.twinx()
    l2, = ax2.plot(data['closePrice'], 'r')
    ax2.set_ylabel('{} close px'.format(future))
    plt.legend([l1,l2],['Sharpe_{}'.format(window_len), 'Close Px'])
    plt.title('sharpe and close px')

    plt.show()


    # 画胜率图
    # fig = plt.figure(figsize=(10,5))
    # ax1 = fig.add_subplot(111)
    # l1, = ax1.plot(data['up_ratio_{}'.format(window_len)], 'b')
    # plt.hlines(0.55,xmin=data.index[0], xmax=data.index[-1])
    # plt.hlines(0.45,xmin=data.index[0], xmax=data.index[-1])
    # ax1.set_ylabel('{} up ratio in {} days'.format(future, window_len))
    #
    # ax2 = ax1.twinx()
    # l2, = ax2.plot(data['closePrice'], 'r')
    # ax2.set_ylabel('{} close px'.format(future))
    # plt.legend([l1,l2],['up_ratio_{}'.format(window_len), 'Close Px'])
    # plt.title('up_ratio and close px')
    #
    # plt.show()


    # 选取数据并保存
    classified_store_path = 'E:\\Research Project\\stable_learning\\RealDataClassified'
    select_mode = 'auto'

    # 牛市，熊市，震荡市
    # 除了保存数据外，还要记录牛市，熊市，震荡市的时间起点和终点，序号以及日期
    # 手动选择
    # 牛市
    if select_mode == 'manual':
        bull_time = {'type': 'bull',
                     'begin': 1700,
                     'end': 2300}

        # 熊市
        bear_time = {'type': 'bear',
                     'begin': 1000,
                     'end': 1700}

        # 震荡状态
        nodirect_time = {'type': 'nodirect',
                         'begin': 2500,
                         'end': 3200}

    # 自动选择
    elif select_mode == 'auto':
        bull_end = data['Sharpe_{}'.format(window_len)].idxmax()
        bear_end = data['Sharpe_{}'.format(window_len)].idxmin()
        normal_end = data['Sharpe_{}'.format(window_len)].abs().idxmin()
        # 牛市
        bull_time = {'type': 'bull',
                     'begin': bull_end-window_len,
                     'end': bull_end}

        # 熊市
        bear_time = {'type': 'bear',
                     'begin': bear_end-window_len,
                     'end': bear_end}

        # 震荡状态
        nodirect_time = {'type': 'normal',
                         'begin': normal_end-window_len,
                         'end': normal_end}

    saved_data = {'bull':{},
                  'bear':{},
                  'normal':{}}

    for period in [bull_time, bear_time, nodirect_time]:
        temp_data = data.iloc[period['begin']:period['end']]

        fig = plt.figure(figsize=(10, 5))

        ax1 = fig.add_subplot(111)
        l1, = ax1.plot(temp_data['Sharpe_{}'.format(window_len)], 'b')
        ax1.set_ylabel('Sharpe_{}'.format(window_len))

        ax2 = ax1.twinx()
        l2, = ax2.plot(temp_data['closePrice'], 'r')
        ax2.set_ylabel('Close Px')
        plt.legend([l1, l2], ['Sharpe_{}'.format(window_len), 'Close Px'])
        plt.title('{} {} sharpe and close px'.format(period['type'], future))

        plt.show()

        saved_data[period['type']]['begin'] = period['begin']
        saved_data[period['type']]['end'] = period['end']
        saved_data[period['type']]['data'] = temp_data

    with open(os.path.join(classified_store_path,'{}_classified_data.pkl'.format(future)), 'wb') as f:
        pk.dump(saved_data, f)


    with open(os.path.join(classified_store_path,'{}_classified_data.pkl'.format(future)), 'rb') as f:
        data_loaded = pk.load(f)















