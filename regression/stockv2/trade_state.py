# -*- coding:UTF-8 -*-

import sys
sys.path.append("..")
from common import base_common
from common.const_def import *

reload(sys)
sys.setdefaultencoding('utf-8')

class TradePos():
    # DQN Agent
    def __init__(self, short_trade=False):
        self.Reset()
        self.short_trade = short_trade

    def Reset(self):
        self.status = TS_OFF
        self.ts_code = 0.0
        self.code_index = 0
        self.pre_on_date = INVALID_DATE
        self.on_date = INVALID_DATE
        self.pre_off_date = INVALID_DATE
        self.off_date = INVALID_DATE
        self.on_price = 0.0
        self.on_open_inc = 0.0
        self.off_price = 0.0
        self.current_price = 0.0
        self.holding_days = 0
        self.pre_on_pred = 0.0
        self.current_pred = 0.0
        self.inc = 0.0

    def UpdateInc(self):
        if self.on_price == 0 or self.off_price == 0:
            self.inc = 0.0
        else:
            if self.short_trade:
                self.inc = self.on_price / self.off_price - 1.0
            else:
                self.inc = self.off_price / self.on_price - 1.0

    def Print(self, trade_index, sum_increase, capital_ratio):
        print('%-8u%-12s%-12s%-10s%-8.2f%-8.2f%-8.2f%-10s%-10s%-10s%-6u%-8.2f%-8s' % (
                trade_index, 
                base_common.TradeDateStr(self.pre_on_date, self.on_date),
                base_common.TradeDateStr(self.pre_off_date, self.off_date),
                base_common.CodeStr(self.ts_code),
                self.pre_on_pred,
                self.current_pred,
                self.on_open_inc,
                base_common.PriceStr(self.on_price),
                base_common.PriceStr(self.off_price),
                base_common.IncreaseStr(self.on_price, self.off_price, self.short_trade),
                self.holding_days,
                sum_increase,
                base_common.FloatStr(capital_ratio)))

    def PrintCaption(self):
        print('%-8s%-12s%-12s%-10s%-8s%-8s%-8s%-10s%-10s%-10s%-6s%-8s%-8s' % (
            'index', 
            'on_date', 
            'off_date', 
            'ts_code', 
            'p_pred',
            'c_pred',
            'oo_inc',
            'on', 
            'off', 
            'inc', 
            'hold',
            'inc_sum',
            'capi'))
        print('-' * 120)


class MaxDrawdown():
    def __init__(self):
        self.ResetMaxDrawdown()

    def ResetMaxDrawdown(self):
        self.capital_ratio_max_value = 1.0
        self.capital_ratio_max_drawdown = 0.0

    def UpdateMaxDrawdown(self, capital_ratio):
        if capital_ratio > self.capital_ratio_max_value:
            self.capital_ratio_max_value = capital_ratio
        drawdown = 1.0 - (capital_ratio / self.capital_ratio_max_value)
        if drawdown > self.capital_ratio_max_drawdown:
            self.capital_ratio_max_drawdown = drawdown


class TradeState():
    def __init__(self):
        self.mdd = MaxDrawdown()
        self.pos_sum = 0.0
        self.pos_num = 0
        self.neg_sum = 0.0
        self.neg_num = 0
        self.hold_days_sum = 0

    def AppendTrade(self, trade_inc, capi_ratio, hold_days):
        self.mdd.UpdateMaxDrawdown(capi_ratio)
        if trade_inc > 0:
            self.pos_sum += trade_inc
            self.pos_num += 1
        else:
            self.neg_sum += trade_inc
            self.neg_num += 1
        self.hold_days_sum += hold_days

    def Print(self):
        print('hold_days_sum: %u' % (self.hold_days_sum))
        print('capital_ratio_max_drawdown: %.2f' % self.mdd.capital_ratio_max_drawdown)
        pos_avg = self.pos_sum / self.pos_num if self.pos_num > 0 else 0.0
        neg_avg = self.neg_sum / self.neg_num if self.neg_num > 0 else 0.0
        print('pos: %6.2f, %4u, %6.2f' % (self.pos_sum, self.pos_num, pos_avg))
        print('neg: %6.2f, %4u, %6.2f' % (self.neg_sum, self.neg_num, neg_avg))