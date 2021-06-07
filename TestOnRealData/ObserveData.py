"""
用于观察某一品种的数据，方便选择
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import pickle as pk

future = 'RB'
window_len = 500
classified_store_path = 'E:\\Research Project\\stable_learning\\RealDataClassified'

with open(os.path.join(classified_store_path, '{}_classified_data.pkl'.format(future)), 'rb') as f:
    data_loaded = pk.load(f)

for mkt_style in ['bull', 'bear', 'normal']:
    temp_data = data_loaded[mkt_style]['data']
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(111)
    l1, = ax1.plot(temp_data['Sharpe_{}'.format(window_len)], 'b')
    ax1.set_ylabel('Sharpe_{}'.format(window_len))

    ax2 = ax1.twinx()
    l2, = ax2.plot(temp_data['closePrice'], 'r')
    ax2.set_ylabel('Close Px')
    plt.legend([l1, l2], ['Sharpe_{}'.format(window_len), 'Close Px'])
    plt.title('{} {} sharpe and close px'.format(mkt_style.upper(), future))

    plt.show()





