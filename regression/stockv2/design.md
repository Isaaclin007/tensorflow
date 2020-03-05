# dqn_test 重构方案：
## 需求：
- 支持daily update
- 支持持有多个 code

## 相关信息：
- dqn_test 依赖数据是 dsfa dataset
- dsfa dataset 依赖数据是 DataSource pp_data

## 实现方案：
- preprocess 重构，拆分成填充数据和计算数据两个部分，计算数据接口只计算衍生数据为0的部分，计算数据的接口需要传入 use_daily_basic, use_money_flow, use_adj_factor, adj_mode，填充数据时生成前三个参数，第四个参数 data_source 透传
- DataSource 添加 SetPPDataDailyUpdate 接口，下载并加载 daily 数据，设置 pp_data_daily_update=True
- DataSource.LoadStockPPData 判断 pp_data_daily_update 为 True 时调用 LoadStockPPDataDailyUpdate，根据 pp_data 文件数据和 daily 数据更新 pp_data 并返回，不保存文件。

备选方案：
- DataSource 添加 dspp_dataset，date stock pp 格式的数据集
- DataSource 构造函数添加参数 daily_date_set 开关，当此开关打开后，LoadStockPPData 将从 dspp_dataset 中读取 pp_data


## 问题记录：
- preprocess 重构计算性能测试：修改前 45 ms，修改后 45 ms



