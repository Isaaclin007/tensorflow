#ifndef _SRC_PPI_H
#define _SRC_PPI_H

#define PPI_ts_code 0
#define PPI_trade_date 1
#define PPI_open 2
#define PPI_close 3
#define PPI_high 4
#define PPI_low 5
#define PPI_vol 6
#define PPI_turnover_rate_f 7
#define PPI_buy_sm_vol 8
#define PPI_sell_sm_vol 9
#define PPI_buy_md_vol 10
#define PPI_sell_md_vol 11
#define PPI_buy_lg_vol 12
#define PPI_sell_lg_vol 13
#define PPI_buy_elg_vol 14
#define PPI_sell_elg_vol 15
#define PPI_net_mf_vol 16
#define PPI_adj_factor 17
#define PPI_open_increase 18
#define PPI_close_increase 19
#define PPI_high_increase 20
#define PPI_low_increase 21
#define PPI_open_5 22
#define PPI_close_5 23
#define PPI_high_5 24
#define PPI_low_5 25
#define PPI_turnover_rate_f_5 26
#define PPI_vol_5 27
#define PPI_close_5_avg 28
#define PPI_close_10_avg 29
#define PPI_close_30_avg 30
#define PPI_close_100_avg 31
#define PPI_vol_5_avg 32
#define PPI_vol_10_avg 33
#define PPI_vol_30_avg 34
#define PPI_vol_100_avg 35
#define PPI_pre_close 36
#define PPI_suspend 37
#define PPI_NUM 38

static const char *PPI_name[PPI_NUM] = 
{
"ts_code",
"trade_date",
"open",
"close",
"high",
"low",
"vol",
"turnover_rate_f",
"buy_sm_vol",
"sell_sm_vol",
"buy_md_vol",
"sell_md_vol",
"buy_lg_vol",
"sell_lg_vol",
"buy_elg_vol",
"sell_elg_vol",
"net_mf_vol",
"adj_factor",
"open_increase",
"close_increase",
"high_increase",
"low_increase",
"open_5",
"close_5",
"high_5",
"low_5",
"turnover_rate_f_5",
"vol_5",
"close_5_avg",
"close_10_avg",
"close_30_avg",
"close_100_avg",
"vol_5_avg",
"vol_10_avg",
"vol_30_avg",
"vol_100_avg",
"pre_close",
"suspend"
};



#endif // _SRC_PPI_H


