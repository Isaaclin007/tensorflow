
# LSTM state 重构方案（keras不支持，推理过程不同的batch size结果一样）：
## 需求
- 支持 LSTM batch state 特性，LSTM 每个 batch 过程中 state 不会重置，因此每个 batch 需要改为一个 tscode 的顺序数据，训练过程中每个 tscode 的数据作为一个 batch fit 一次


# dqn_test 重构方案（已完成）：
## 需求：
- 支持daily update（已完成）
- 支持持有多个 code

## 支持daily update

### 相关信息：
- dqn_test 依赖数据是 dsfa dataset
- dsfa dataset 依赖数据是 DataSource date_list(map), code_list(map), pp_data, setting_name, start_date, end_date

### 实现方案：
- preprocess 重构，拆分成填充数据和计算数据两个部分，计算数据接口只计算衍生数据为0的部分，计算数据的接口需要传入 use_daily_basic, use_money_flow, use_adj_factor, adj_mode，填充数据时生成前三个参数，第四个参数 data_source 透传
- DataSource 添加 SetPPDataDailyUpdate 接口，下载并加载 daily 数据，设置 pp_data_daily_update=True
- DataSource.LoadStockPPData 判断 pp_data_daily_update 为 True 时调用 LoadStockPPDataDailyUpdate，根据 pp_data 文件数据和 daily 数据更新 pp_data 并返回，不保存文件。

备选方案：
- DataSource 添加 dspp_dataset，date stock pp 格式的数据集
- DataSource 构造函数添加参数 daily_date_set 开关，当此开关打开后，LoadStockPPData 将从 dspp_dataset 中读取 pp_data

### 问题记录：
- preprocess 重构计算性能测试：修改前 45 ms，修改后 45 ms


## 支持持有多个 code

### 实现方案
维护持有状态机 code_pool(numpy 数据类型)，每行为一个 code 仓位，具有以下属性：
- status: trade status，初始值 TS_OFF
- ts_code
- pre_on_index
- on_index
- pre_off_index
- off_index
- on_price
- off_price
- holding_days

反向遍历 trade_date，对每个 trade_date：
- UpdateHoldingCodeStatus，更新 code_pool 内 status 非 OFF 的数据
- 如果 code_pool 有空槽位，则 AppendNewCode，选择 code 加入 code_pool，要求新选择的 code 与 code_pool 中的不重复
- UpdateHoldingCodeStatus，过程中转换为 OFF 状态时更新全局统计信息
- 遍历完成后打印 code_pool 中非 OFF 数据