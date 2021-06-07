"""
获取各种期货数据
"""
import os

import uqer

import TestOnRealData.futures_info as future_info

client = uqer.Client(token='3463931d00f1a37acb1028ea5b8a7a5c4fb7f14a9819177c71ef3c7781bd770c')

#
# b = uqer.DataAPI.MktMFutdGet(mainCon=u"1",contractMark=u"",contractObject=u"CU",
#                              tradeDate=u"",startDate=u"20000101",endDate=u"20190101",
#                              field=u"",pandas="1")

save_path = 'E:\\Research Project\\stable_learning\\RealData'
for ex in [future_info.ex_sh, future_info.ex_cf, future_info.ex_zz, future_info.ex_dl]:
    futures = list(ex.keys())
    for f in futures:
        print("current: {}".format(f.upper()))
        if f == 'cu':
            continue
        temp = uqer.DataAPI.MktMFutdGet(mainCon=u"1", contractMark=u"", contractObject=f.upper(),
                                        tradeDate=u"", startDate=u"20000101", endDate=u"20191231",
                                        field=u"", pandas="1")
        temp.to_csv(os.path.join(save_path, f.upper() + '.csv'), index=False)
