# -*- coding:UTF-8 -*-

WS_NONE = 0
WS_UP = 1
WS_DOWN = 2
WS_STR = ['NONE', 'UP', 'DOWN']

TS_NONE = 0
TS_ON = 1
TS_OFF = 2
TS_PRE_ON = 3
TS_PRE_OFF = 4
TS_STR = ['NONE', 'ON', 'OFF', 'PRE_ON', 'PRE_OFF']

INVALID_DATE = 0.0
INVALID_INDEX = -1

PPI_ts_code = 0
PPI_trade_date = 1
PPI_open = 2
PPI_close = 3
PPI_high = 4
PPI_low = 5
PPI_vol = 6
PPI_turnover_rate_f = 7
PPI_buy_sm_vol = 8
PPI_sell_sm_vol = 9
PPI_buy_md_vol = 10
PPI_sell_md_vol = 11
PPI_buy_lg_vol = 12
PPI_sell_lg_vol = 13
PPI_buy_elg_vol = 14
PPI_sell_elg_vol = 15
PPI_net_mf_vol = 16
PPI_adj_factor = 17
PPI_open_increase = 18
PPI_close_increase = 19
PPI_high_increase = 20
PPI_low_increase = 21
PPI_open_5 = 22
PPI_close_5 = 23
PPI_high_5 = 24
PPI_low_5 = 25
PPI_turnover_rate_f_5 = 26
PPI_vol_5 = 27
PPI_close_5_avg = 28
PPI_close_10_avg = 29
PPI_close_30_avg = 30
PPI_close_100_avg = 31
PPI_vol_5_avg = 32
PPI_vol_10_avg = 33
PPI_vol_30_avg = 34
PPI_vol_100_avg = 35
PPI_pre_close = 36
PPI_suspend = 37
PPI_NUM = 38

PPI_name = {}
PPI_name[PPI_ts_code] = "ts_code"
PPI_name[PPI_trade_date] = "trade_date"
PPI_name[PPI_open] = "open"
PPI_name[PPI_close] = "close"
PPI_name[PPI_high] = "high"
PPI_name[PPI_low] = "low"
PPI_name[PPI_vol] = "vol"
PPI_name[PPI_turnover_rate_f] = "turnover_rate_f"
PPI_name[PPI_buy_sm_vol] = "buy_sm_vol"
PPI_name[PPI_sell_sm_vol] = "sell_sm_vol"
PPI_name[PPI_buy_md_vol] = "buy_md_vol"
PPI_name[PPI_sell_md_vol] = "sell_md_vol"
PPI_name[PPI_buy_lg_vol] = "buy_lg_vol"
PPI_name[PPI_sell_lg_vol] = "sell_lg_vol"
PPI_name[PPI_buy_elg_vol] = "buy_elg_vol"
PPI_name[PPI_sell_elg_vol] = "sell_elg_vol"
PPI_name[PPI_net_mf_vol] = "net_mf_vol"
PPI_name[PPI_adj_factor] = "adj_factor"
PPI_name[PPI_open_increase] = "open_increase"
PPI_name[PPI_close_increase] = "close_increase"
PPI_name[PPI_high_increase] = "high_increase"
PPI_name[PPI_low_increase] = "low_increase"
PPI_name[PPI_open_5] = "open_5"
PPI_name[PPI_close_5] = "close_5"
PPI_name[PPI_high_5] = "high_5"
PPI_name[PPI_low_5] = "low_5"
PPI_name[PPI_turnover_rate_f_5] = "turnover_rate_f_5"
PPI_name[PPI_vol_5] = "vol_5"
PPI_name[PPI_close_5_avg] = "close_5_avg"
PPI_name[PPI_close_10_avg] = "close_10_avg"
PPI_name[PPI_close_30_avg] = "close_30_avg"
PPI_name[PPI_close_100_avg] = "close_100_avg"
PPI_name[PPI_vol_5_avg] = "vol_5_avg"
PPI_name[PPI_vol_10_avg] = "vol_10_avg"
PPI_name[PPI_vol_30_avg] = "vol_30_avg"
PPI_name[PPI_vol_100_avg] = "vol_100_avg"
PPI_name[PPI_pre_close] = "pre_close"
PPI_name[PPI_suspend] = "suspend"

PPI_vol_list = [PPI_vol,
                PPI_vol_5_avg,
                PPI_vol_10_avg,
                PPI_vol_30_avg,
                PPI_vol_100_avg]