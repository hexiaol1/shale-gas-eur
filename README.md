# 页岩气产能预测与 EUR 符合率评价系统

基于气井生产动态数据的流动状态智能诊断与多递减模型优选系统。

### 功能特点
- **流态诊断**：双对数（Log-Log Plot）流动突变点自适应识别与斜率判定。
- **模型优选**：支持双曲递减 (Arps)、改进延伸指数 (YM-SEPD)、改进 Duong 递减及加权组合模型。
- **符合率验收**：内置时间截断回溯盲测（Time-Split Backtesting）引擎，用于甲方“EUR 符合率 > 80%”的硬指标考核验收。
- **终采截断**：支持设定废弃日产气量 ($q_{ab}$) 与最大经济开采寿命。

### 运行方式
```bash
pip install -r requirements.txt
streamlit run app.py
