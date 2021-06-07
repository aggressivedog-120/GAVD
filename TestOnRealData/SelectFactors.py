"""
用于预选择一些技术指标
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import pickle as pk
import talib as ta

store_path = 'E:\\Research Project\\stable_learning\\RealData'

this_future = 'CU'
future = this_future.upper()
print('current future: {}'.format(future))

data = pd.read_csv(os.path.join(store_path, future+'.csv'))

# 做判断，数据是否满足总长度要求
assert data.shape[0] > 2000

# MACD, difference between DIF, DEM
# ASI
# ATR
# BIAS
# BOLL
# CCI /100
# DMA
# KDJ
# MTM，20
# OBV
# OSC
# PSY
# RSI
# TWR
# VR
# 
#
#

# 技术指标应该都是标准化的，在0-1附近这样

# 第一种选择
# X:各种技术指标
# Y:（下一个交易日收盘价 - 本日收盘价） / 本日收盘价


# 第二种选择
# X:各种技术指标
# Y:（下一个交易日收盘价 - 下一个交易日开盘价） / 下一个交易日开盘价










