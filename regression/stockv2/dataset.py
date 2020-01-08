# -*- coding:UTF-8 -*-


import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
import math
sys.path.append("..")
from common import base_common
from common import np_common
import tushare_data
import feature

def CreateDSFa3DSplitMTFunc(param, msg):
    param.CreateDSFa3DSplitStock(msg)

class Dataset():
    def __init__(self, 
                 o_data_source, 
                 o_feature):
        self.data_source = o_data_source
        self.feature = o_feature
        self.date_list = self.data_source.date_list
        self.code_list = self.data_source.code_list
        self.date_index_map = base_common.ListToIndexMap(self.date_list, True)
        self.code_index_map = base_common.ListToIndexMap(self.code_list, False)


    def FileNameDSFa3DDataset(self):
        temp_file_name = './data/dataset/DSFa3D_%s_%s.npy' %(\
            self.data_source.setting_name, \
            self.feature.setting_name)
        return temp_file_name

    def FileNameDSFa3DSplit(self, ts_code):
        temp_file_name = './data/DSFa3D/%s_%s_%s.npy' %(\
            self.data_source.setting_name_stock, \
            self.feature.setting_name, \
            ts_code)
        return temp_file_name

    def CreateDSFa3DSplitStock(self, ts_code):
        data_unit_date_index = self.feature.index_date
        split_file_name = self.FileNameDSFa3DSplit(ts_code)
        if not os.path.exists(split_file_name):
            pp_data = self.data_source.LoadStockPPData(ts_code)
            if len(pp_data) == 0:
                return
            
            dataset = np.zeros((len(self.date_list), 1, self.feature.unit_size))
            for day_loop in range(0, len(pp_data)):
                data_unit = self.feature.GetDataUnit(pp_data, day_loop)
                if len(data_unit) == 0:
                    continue
                temp_date = int(data_unit[data_unit_date_index])
                if temp_date > self.data_source.end_date:
                    continue
                if temp_date < self.data_source.start_date:
                    break
                dateset_index1 = self.date_index_map[temp_date]
                dataset[dateset_index1][0] = data_unit

            base_common.MKFileDirs(split_file_name)
            np.save(split_file_name, dataset)
            sys.stdout.write("%-4d : %s 100%%\n" % (self.code_index_map[ts_code], ts_code))

    def CreateDSFa3DSplit(self):
        base_common.ListMultiThread(CreateDSFa3DSplitMTFunc, self, 8, self.code_list)

    def CreateDSFa3DDataset(self):
        self.CreateDSFa3DSplit()
        dataset_file_name = self.FileNameDSFa3DDataset()
        dataset = np.zeros((len(self.date_list), len(self.code_list), self.feature.unit_size))
        for code_index in range(0, len(self.code_list)):
            ts_code = self.code_list[code_index]
            dataset_split_file_name = self.FileNameDSFa3DSplit(ts_code)
            if not os.path.exists(dataset_split_file_name):
                print('CreateDSFa3DDataset.Error: %s not exist' % dataset_split_file_name)
                return
            split_data = np.load(dataset_split_file_name)
            dateset_index2 = self.code_index_map[ts_code]
            for iloop in range(len(self.date_list)):
                dataset[iloop][dateset_index2] = split_data[iloop][0]
        # print("dataset: {}".format(dataset.shape))
        # print("file_name: %s" % dataset_file_name)
        base_common.MKFileDirs(dataset_file_name)
        np.save(dataset_file_name, dataset)

    def GetDSFa3DDataset(self):
        dataset_file_name = self.FileNameDSFa3DDataset()
        if not os.path.exists(dataset_file_name):
            self.CreateDSFa3DDataset()
        return np.load(dataset_file_name)


if __name__ == "__main__":
    data_source = tushare_data.DataSource(20000101, '软件服务', '', 1, 20000101, 20200106, False, False, True)
    data_source.ShowStockCodes()

    o_feature = feature.Feature(30, feature.FUT_D5_NORM, 1, False, False)

    o_dataset = Dataset(data_source, o_feature)
    temp_dataset = o_dataset.GetDSFa3DDataset()
    print("dataset: {}".format(temp_dataset.shape))

