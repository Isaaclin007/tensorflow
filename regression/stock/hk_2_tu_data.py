# -*- coding:UTF-8 -*-


import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
from skimage import feature
from dask.dataframe.methods import size
from llvmlite.ir.types import LabelType

reload(sys)
sys.setdefaultencoding('utf-8')

hk_code_list_value = [\
1, \
2, \
3, \
4, \
5, \
6, \
8, \
10, \
11, \
12, \
14, \
16, \
17, \
19, \
20, \
23, \
27, \
38, \
45, \
54, \
66, \
69, \
83, \
87, \
101, \
107, \
119, \
123, \
127, \
135, \
136, \
142, \
144, \
148, \
151, \
152, \
165, \
168, \
173, \
175, \
177, \
178, \
179, \
187, \
200, \
215, \
220, \
241, \
242, \
257, \
267, \
270, \
272, \
285, \
288, \
291, \
293, \
303, \
308, \
315, \
316, \
317, \
322, \
323, \
336, \
338, \
341, \
345, \
358, \
363, \
371, \
384, \
386, \
388, \
390, \
392, \
410, \
425, \
439, \
440, \
460, \
489, \
493, \
494, \
506, \
511, \
522, \
525, \
548, \
551, \
552, \
553, \
564, \
570, \
576, \
581, \
582, \
586, \
588, \
590, \
604, \
606, \
607, \
636, \
656, \
658, \
659, \
665, \
669, \
670, \
683, \
688, \
694, \
696, \
698, \
699, \
700, \
728, \
737, \
751, \
753, \
754, \
762, \
772, \
777, \
788, \
806, \
811, \
813, \
817, \
836, \
839, \
857, \
867, \
868, \
874, \
880, \
881, \
883, \
884, \
902, \
914, \
916, \
934, \
939, \
941, \
958, \
960, \
966, \
968, \
981, \
991, \
992, \
995, \
998, \
1030, \
1038, \
1044, \
1053, \
1055, \
1060, \
1065, \
1066, \
1071, \
1072, \
1083, \
1088, \
1093, \
1099, \
1108, \
1109, \
1112, \
1113, \
1114, \
1128, \
1138, \
1141, \
1169, \
1171, \
1177, \
1186, \
1193, \
1199, \
1208, \
1212, \
1233, \
1269, \
1282, \
1288, \
1293, \
1299, \
1308, \
1313, \
1316, \
1330, \
1333, \
1336, \
1339, \
1347, \
1357, \
1359, \
1375, \
1378, \
1382, \
1398, \
1458, \
1508, \
1515, \
1528, \
1530, \
1548, \
1618, \
1628, \
1635, \
1638, \
1658, \
1668, \
1728, \
1766, \
1787, \
1788, \
1800, \
1813, \
1816, \
1833, \
1882, \
1888, \
1898, \
1918, \
1919, \
1928, \
1929, \
1958, \
1972, \
1988, \
1997, \
1999, \
2007, \
2009, \
2018, \
2020, \
2038, \
2048, \
2068, \
2128, \
2186, \
2196, \
2232, \
2238, \
2269, \
2282, \
2313, \
2314, \
2318, \
2319, \
2328, \
2333, \
2356, \
2357, \
2359, \
2380, \
2382, \
2386, \
2388, \
2588, \
2600, \
2601, \
2607, \
2611, \
2628, \
2688, \
2689, \
2727, \
2768, \
2777, \
2799, \
2858, \
2866, \
2880, \
2883, \
2899, \
3311, \
3320, \
3323, \
3328, \
3333, \
3360, \
3369, \
3377, \
3380, \
3383, \
3396, \
3606, \
3618, \
3669, \
3799, \
3800, \
3808, \
3888, \
3898, \
3899, \
3900, \
3908, \
3958, \
3968, \
3969, \
3988, \
3993, \
6030, \
6060, \
6066, \
6088, \
6098, \
6099, \
6116, \
6158, \
6178, \
6808, \
6818, \
6837, \
6862, \
6869, \
6881, \
6886, \
]
train_test_date = '20190215'

def HKCodeList():
    hk_code_list_str = []
    for code_value in hk_code_list_value:
        temp_str = '%06d.HK' % code_value
        hk_code_list_str.append(temp_str)
    return hk_code_list_str


if __name__ == "__main__":
    for code_value in hk_code_list_value:
        input_file_name = "hk_stock_data/%05d.csv" % code_value
        output_file_name = "download_data/%06d.HK_%s.csv" % (code_value, train_test_date)

        src_df = pd.read_csv(input_file_name, encoding = 'gbk')
        src_df.columns = ['ts_code','trade_date','open', 'high', 'low', 'close', 'pre_close', 'chg_ratio', 'vol', 'amount']
        # src_df.rename(columns={\
        #     '股票代码':'ts_code', \
        #     '交易日期':'trade_date', \
        #     '开盘价':'open', \
        #     '最高价':'high', \
        #     '最低价':'low', \
        #     '收盘价':'close', \
        #     '昨收盘':'pre_close', \
        #     '涨跌幅':'chg_ratio', \
        #     '成交量':'vol', \
        #     '成交额':'amount', \
        #     }, inplace = True)
        src_df['ts_code']='%06u.HK' % code_value
        src_df['total_share']=0.0
        src_df['float_share']=0.0
        src_df['free_share']=0.0
        src_df['total_mv']=0.0
        src_df['circ_mv']=0.0
        src_df['turnover_rate_f']=0.0
        src_df['turnover_rate_f']=src_df['amount']
        for day_loop in range(0, len(src_df)): 
            src_df.loc[day_loop,'trade_date'] = src_df.loc[day_loop,'trade_date'].replace('-', '')
        src_df['trade_date'] = src_df['trade_date'].astype(np.int64)
        src_df.to_csv(output_file_name)
        print('%s 100%%' % input_file_name)
        
