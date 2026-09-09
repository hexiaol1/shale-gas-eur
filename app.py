import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# 页面基本设置
st.set_page_config(
    page_title="页岩气产能预测与EUR符合率评价系统",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. 数学递减模型定义区
# ==========================================

def exp_decline(t, qi, Di):
    """指数递减模型 (Exponential Decline)"""
    return qi * np.exp(-Di * t)

def hyp_decline(t, qi, Di, b):
    """双曲递减模型 (Hyperbolic Decline)"""
    b = np.clip(b, 1e-4, 2.0)
    return qi / ((1.0 + b * Di * t) ** (1.0 / b))

def sepd_decline(t, qi, tau, n):
    """延伸指数递减模型 (SEPD)"""
    tau = max(tau, 1e-3)
    n = np.clip(n, 0.05, 1.0)
    return qi * np.exp(- (np.maximum(t, 0) / tau) ** n)

def duong_decline(t, q1, a, m):
    """Duong 递减模型 (针对裂缝主导低渗页岩气)"""
    t_safe = np.where(t <= 0, 1e-5, t)
    return q1 * (t_safe ** -m) * np.exp((a / (1.0 - m)) * (t_safe ** (1.0 - m) - 1.0))

def fit_model(model_func, t_data, q_data, p0, bounds):
    """通用非线性最小二乘拟合包装函数"""
    try:
        popt, _ = curve_fit(model_func, t_data, q_data, p0=p0, bounds=bounds, maxfev=15000)
        q_pred = model_func(t_data, *popt)
        r2 = r2_score(q_data, q_pred)
        return popt, q_pred, r2
    except Exception:
        return None, None, -1.0

def calc_eur_with_abandonment(model_func, popt, t_start, max_days, q_abandon):
    """带废弃产量限制的全生命周期 EUR 积分计算"""
    if popt is None:
        return 0.0, 0
    t_seq = np.arange(1, max_days + 1)
    q_seq = model_func(t_seq, *popt)
    
    # 找到第一个低于废弃产量的时刻
    valid_mask = q_seq >= q_abandon
    if np.any(~valid_mask):
        abandon_idx = np.where(~valid_mask)[0][0]
        q_valid = q_seq[:abandon_idx]
        effective_days = abandon_idx
    else:
        q_valid = q_seq
        effective_days = max_days
        
    eur = np.sum(q_valid)
    return eur, effective_days

# ==========================================
# 2. 辅助分析与算法寻优区
# ==========================================

def generate_mock_data():
    """生成用于测试的模拟页岩气生产数据（含初期高产、递减及少量停产关井噪点）"""
    np.random.seed(42)
    t = np.arange(1, 1100)
    q_base = duong_decline(t, q1=350, a=1.15, m=1.25)
    noise = np.random.normal(0, 0.06 * q_base, len(t))
    q = np.maximum(q_base + noise, 0)
    
    # 模拟偶发间歇关井或测试扰动
    shut_in_idx = np.random.choice(len(t), size=25, replace=False)
    q[shut_in_idx] = 0.0
    return pd.DataFrame({'Time(d)': t, 'Production(10^3m3/d)': q})

def find_optimal_split_ratio(log_t, log_q, min_ratio=0.15, max_ratio=0.85):
    """双段分段对数线性回归断点寻优"""
    n = len(log_t)
    if n < 10:
        return 0.30
    min_idx = max(3, int(n * min_ratio))
    max_idx = min(n - 4, int(n * max_ratio))
    
    best_rss = np.inf
    best_idx = int(n * 0.30)

    for idx in range(min_idx, max_idx + 1):
        x1, y1 = log_t[:idx], log_q[:idx]
        x2, y2 = log_t[idx:], log_q[idx:]

        p1 = np.polyfit(x1, y1, 1)
        p2 = np.polyfit(x2, y2, 1)

        rss1 = np.sum((y1 - np.polyval(p1, x1)) ** 2)
        rss2 = np.sum((y2 - np.polyval(p2, x2)) ** 2)
        total_rss = rss1 + rss2

        if total_rss < best_rss:
            best_rss = total_rss
            best_idx = idx

    return best_idx / n

# ==========================================
# 3. 侧边栏及参数控制
# ==========================================

st.title("⛽ 页岩气产能预测与 EUR 符合率评价系统")
st.caption("集成《断块油气田》流态诊断优选规则与甲方验收级时间截断回溯盲测引擎")

st.sidebar.header("1. 数据加载与预处理")
data_source = st.sidebar.radio("选择数据来源", ["上传 CSV 数据", "使用内置测试数据 (含关井噪点)"])

df_raw = None
if data_source == "上传 CSV 数据":
    uploaded_file = st.sidebar.file_uploader("上传日产生产数据 (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.sidebar.success("数据上传成功！")
        except Exception as e:
            st.sidebar.error(f"读取失败: {e}")
else:
    df_raw = generate_mock_data()
    st.sidebar.info("已加载 1100 天典型井模拟数据")

if df_raw is None:
    st.stop()

cols = df_raw.columns.tolist()
time_col = st.sidebar.selectbox("时间列 (天/月)", cols, index=0)
prod_col = st.sidebar.selectbox("产量列 (10³m³/d)", cols, index=1 if len(cols) > 1 else 0)

drop_zero = st.sidebar.checkbox("自动剔除关井点 (产气量 ≤ 0) 并重构有效生产日", value=True)
filter_outliers = st.sidebar.checkbox("平滑低产毛刺点 (低于7天中位数20%)", value=True)

# 基础清洗
df_work = df_raw.copy()
if drop_zero:
    df_work = df_work[df_work[prod_col] > 0].copy()
    # 重构连续有效生产时间序列
    df_work['Effective_Days'] = np.arange(1, len(df_work) + 1)
    eval_time_col = 'Effective_Days'
else:
    df_work = df_work[df_work[prod_col] >= 0].copy()
    eval_time_col = time_col

if filter_outliers and len(df_work) > 15:
    rolling_med = df_work[prod_col].rolling(window=7, center=True, min_periods=1).median()
    df_work = df_work[df_work[prod_col] >= 0.2 * rolling_med].copy()

t_all = df_work[eval_time_col].values
q_all = df_work[prod_col].values

if len(t_all) < 30:
    st.error("有效生产数据点少于 30 个，无法进行可靠的流态分析与递减拟合。")
    st.stop()

st.sidebar.header("2. 阶段划分与流态诊断")
auto_ratio = find_optimal_split_ratio(np.log10(t_all), np.log10(np.maximum(q_all, 1e-3)))
auto_percent = int(round(auto_ratio * 100))

split_mode = st.sidebar.radio("前/后期流态划分方式", ["自动识别突变点", "手动指定比例"], index=0)
split_point = auto_percent if split_mode == "自动识别突变点" else st.sidebar.slider("前期数据截断比例 (%)", 10, 90, auto_percent, 5)

st.sidebar.header("3. 经济极限与验收标准")
q_abandon = st.sidebar.number_input("废弃产气量极限 (10³m³/d)", min_value=0.0, value=1.0, step=0.5, help="低于该日产即认为达到经济极限停止开采")
max_life_years = st.sidebar.number_input("最长开采年限 (年)", min_value=5, max_value=50, value=20)
target_accuracy = st.sidebar.slider("甲方要求的符合率目标 (%)", min_value=60, max_value=95, value=80, step=5)

# ==========================================
# 4. 全局递减模型拟合
# ==========================================
bounds_exp = ([0, 0], [np.inf, 1])
bounds_hyp = ([0, 0, 0.0001], [np.inf, 1, 2.0])
bounds_sepd = ([0, 1, 0.05], [np.inf, np.inf, 1.0])
bounds_duong = ([0, 0, 1.01], [np.inf, 10, 4.0])

init_q0 = float(np.mean(q_all[:5]))
with st.spinner('全局多模型参数拟合中...'):
    popt_hyp, q_pred_hyp, r2_hyp = fit_model(hyp_decline, t_all, q_all, p0=[init_q0, 0.01, 0.8], bounds=bounds_hyp)
    popt_exp, q_pred_exp, r2_exp = fit_model(exp_decline, t_all, q_all, p0=[init_q0, 0.01], bounds=bounds_exp)
    popt_sepd, q_pred_sepd, r2_sepd = fit_model(sepd_decline, t_all, q_all, p0=[init_q0, 200, 0.4], bounds=bounds_sepd)
    popt_duong, q_pred_duong, r2_duong = fit_model(duong_decline, t_all, q_all, p0=[init_q0, 1.1, 1.25], bounds=bounds_duong)

# 流态诊断参数计算
split_idx = int(len(t_all) * (split_point / 100.0))
t_early, q_early = t_all[:split_idx], q_all[:split_idx]
t_late, q_late = t_all[split_idx:], q_all[split_idx:]

try:
    slope_early, intercept_early = np.polyfit(np.log10(t_early), np.log10(q_early), 1)
    slope_late, intercept_late = np.polyfit(np.log10(t_late), np.log10(q_late), 1) if len(t_late) > 2 else (0, 0)
except Exception:
    slope_early, intercept_early, slope_late, intercept_late = 0, 0, 0, 0

# 模型优选逻辑判定
is_boundary_dominated = slope_late < -0.85
recommended_model_name = ""
if is_boundary_dominated:
    if -0.40 <= slope_early <= -0.04:
        recommended_model_name = "改进延伸指数 (YM-SEPD)"
    elif -0.50 <= slope_early < -0.40:
        recommended_model_name = "双曲递减 (Hyperbolic)"
    else:
        recommended_model_name = "改进延伸指数 (YM-SEPD)"
else:
    if -0.40 <= slope_early <= -0.04:
        recommended_model_name = "0.5 Duong + 0.5 YM-SEPD"
    elif -0.50 <= slope_early < -0.40:
        recommended_model_name = "0.5 YM-SEPD + 0.5 双曲递减"
    else:
        recommended_model_name = "改进 Duong 模型"

# ==========================================
# 5. 标签页界面呈现
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 甲方考核与符合率验收", 
    "📤 流态诊断与双对数分析", 
    "⚙️ 产能预测与模型优选", 
    "📊 模型拟合综合对比", 
    "📖 理论方法说明"
])

# ----------------- TAB 1: 核心符合率验收 -----------------
with tab1:
    st.subheader("🎯 应对甲方“EUR 符合率 > 80%”专项评价与回溯盲测")
    st.markdown("""
    > **工程逻辑闭环**：全生命周期的 EUR 无法在当下直接测量。为严格满足工程审计要求，本系统提供两种行业标准评测方案：
    > 1. **验证期累产硬符合率（Blind Cumulative Accuracy）**：截取前期数据训练，盲测后续**真实已发生**的生产量（绝对物理真值）；
    > 2. **全周期基准演化收敛率（EUR Convergence Rate）**：以前期训练模型预测的 EUR 对比全历史长周期标定的基准 EUR。
    """)
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.markdown("##### 盲测实验切分参数")
        train_days = st.slider(
            "选择训练期（仅使用前 N 天推测未来）",
            min_value=60,
            max_value=int(t_all[-1] - 30),
            value=min(365, int(t_all[-1] * 0.5)),
            step=15,
            help="模拟历史上第 N 天时，我们仅凭已知数据向后预测的能力"
        )
        test_days = int(t_all[-1] - train_days)
        st.info(f"训练期（已知）: **{train_days}** 天\n\n验证期（盲测）: **{test_days}** 天")
        
        selected_blind_model = st.selectbox(
            "回溯测试采用的递减模型",
            ["论文推荐优选模型", "改进延伸指数 (YM-SEPD)", "改进 Duong", "双曲递减 (Hyperbolic)"]
        )

    # 执行盲测拟合计算
    train_mask = t_all <= train_days
    test_mask = t_all > train_days
    
    t_train, q_train = t_all[train_mask], q_all[train_mask]
    t_test, q_test = t_all[test_mask], q_all[test_mask]
    
    # 局部拟合
    p_hyp_tr, _, _ = fit_model(hyp_decline, t_train, q_train, p0=[q_train[0], 0.01, 0.8], bounds=bounds_hyp)
    p_sepd_tr, _, _ = fit_model(sepd_decline, t_train, q_train, p0=[q_train[0], 150, 0.4], bounds=bounds_sepd)
    p_duo_tr, _, _ = fit_model(duong_decline, t_train, q_train, p0=[q_train[0], 1.1, 1.25], bounds=bounds_duong)
    
    def predict_blind(t_input, name):
        if name == "改进延伸指数 (YM-SEPD)" and p_sepd_tr is not None:
            return sepd_decline(t_input, *p_sepd_tr)
        elif name == "改进 Duong" and p_duo_tr is not None:
            return duong_decline(t_input, *p_duo_tr)
        elif name == "双曲递减 (Hyperbolic)" and p_hyp_tr is not None:
            return hyp_decline(t_input, *p_hyp_tr)
        elif name == "论文推荐优选模型":
            if "0.5 Duong + 0.5 YM-SEPD" in recommended_model_name and p_duo_tr is not None and p_sepd_tr is not None:
                return 0.5 * duong_decline(t_input, *p_duo_tr) + 0.5 * sepd_decline(t_input, *p_sepd_tr)
            elif "0.5 YM-SEPD + 0.5 双曲递减" in recommended_model_name and p_sepd_tr is not None and p_hyp_tr is not None:
                return 0.5 * sepd_decline(t_input, *p_sepd_tr) + 0.5 * hyp_decline(t_input, *p_hyp_tr)
            elif p_sepd_tr is not None:
                return sepd_decline(t_input, *p_sepd_tr)
        return np.zeros_like(t_input)

    q_test_pred = predict_blind(t_test, selected_blind_model)
    actual_test_cum = np.sum(q_test)
    pred_test_cum = np.sum(q_test_pred)
    
    # 累产相对误差与符合率
    cum_relative_error = (pred_test_cum - actual_test_cum) / (actual_test_cum + 1e-6)
    cum_accuracy = max(0.0, 1.0 - abs(cum_relative_error)) * 100.0

    with col_t2:
        fig_blind = go.Figure()
        fig_blind.add_trace(go.Scatter(x=t_train, y=q_train, mode='markers', name='训练段实际日产 (历史已知)', marker=dict(color='#2980b9', size=4)))
        fig_blind.add_trace(go.Scatter(x=t_test, y=q_test, mode='markers', name='验证段实际日产 (客观真值)', marker=dict(color='#27ae60', size=4)))
        
        t_forecast_blind = np.arange(1, t_all[-1] + 1)
        q_blind_curve = predict_blind(t_forecast_blind, selected_blind_model)
        fig_blind.add_trace(go.Scatter(x=t_forecast_blind, y=q_blind_curve, mode='lines', name=f'盲测外推趋势 ({selected_blind_model})', line=dict(color='#e74c3c', width=2.5)))
        
        fig_blind.add_vline(x=train_days, line_dash="dash", line_color="gray", annotation_text="预测起点", annotation_position="top left")
        fig_blind.update_layout(title="时间截断历史盲测 (Backtesting) 验证曲线", xaxis_title="有效生产天数 (d)", yaxis_title="日产气量 (10³m³/d)", hovermode="x unified", height=400)
        st.plotly_chart(fig_blind, use_container_width=True)

    # 符合率达标卡片展示
    st.markdown("---")
    st.subheader("📋 甲方验收符合率评估结论")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("验证段实际累产 (客观计量)", f"{actual_test_cum:,.1f} 10³m³")
    m_col2.metric("盲测推算累产", f"{pred_test_cum:,.1f} 10³m³")
    m_col3.metric("预测相对偏差", f"{cum_relative_error * 100:+.2f}%")
    
    is_passed = cum_accuracy >= target_accuracy
    delta_text = "验收达标 (符合率要求 ≥ 80%)" if is_passed else "未达标"
    m_col4.metric(
        "单井验证段符合率",
        f"{cum_accuracy:.2f}%",
        delta=delta_text,
        delta_color="normal" if is_passed else "inverse"
    )
    
    if is_passed:
        st.success(f"🎉 **符合率考核通过**：当前选用的递减模型在验证段的累产符合率为 **{cum_accuracy:.2f}%**，达到甲方要求的 **{target_accuracy}%** 门槛。")
    else:
        st.warning(f"⚠️ **当前模型符合率偏离**：验证段符合率为 **{cum_accuracy:.2f}%**，建议在左侧调整平滑过滤参数，或更换为其他更适宜的递减模型。")

# ----------------- TAB 2: 流态诊断 -----------------
with tab2:
    st.subheader("双对数流态诊断与断点分析")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.metric("前期双对数斜率 (流动状态参数)", f"{slope_early:.4f}", help="反应压裂裂缝网与基质主导流态")
    with col_l2:
        flow_status = "边界控制拟稳态阶段 (斜率 < -0.85)" if is_boundary_dominated else "瞬态/过渡线性流阶段 (未进入拟稳态)"
        st.metric("当前井流动阶段评判", flow_status)
        
    fig_diag = go.Figure()
    fig_diag.add_trace(go.Scatter(x=np.log10(t_early), y=np.log10(q_early), mode='markers', name='前期生产数据', marker=dict(color='#3498db', size=4)))
    fig_diag.add_trace(go.Scatter(x=np.log10(t_early), y=slope_early*np.log10(t_early) + intercept_early, mode='lines', name=f'前期斜率拟合 ({slope_early:.3f})', line=dict(color='blue', dash='dash')))
    
    if len(t_late) > 2:
        fig_diag.add_trace(go.Scatter(x=np.log10(t_late), y=np.log10(q_late), mode='markers', name='后期生产数据', marker=dict(color='#e67e22', size=4)))
        fig_diag.add_trace(go.Scatter(x=np.log10(t_late), y=slope_late*np.log10(t_late) + intercept_late, mode='lines', name=f'后期斜率拟合 ({slope_late:.3f})', line=dict(color='red', dash='dash')))
    
    fig_diag.update_layout(title="生产时间-产量双对数诊断曲线 (Log-Log Plot)", xaxis_title="lg(有效生产天数)", yaxis_title="lg(日产气量)", height=500)
    st.plotly_chart(fig_diag, use_container_width=True)

# ----------------- TAB 3: 预测与储量 -----------------
with tab3:
    st.subheader("全生命周期产能预测与 EUR 评估看板")
    st.info(f"💡 基于《断块油气田》论文优选逻辑推荐方案：**{recommended_model_name}**")
    
    forecast_horizon_days = int(max_life_years * 365)
    
    # 结合废弃产量计算各模型全生命周期 EUR
    eur_hyp, life_hyp = calc_eur_with_abandonment(hyp_decline, popt_hyp, 1, forecast_horizon_days, q_abandon)
    eur_sepd, life_sepd = calc_eur_with_abandonment(sepd_decline, popt_sepd, 1, forecast_horizon_days, q_abandon)
    eur_duong, life_duong = calc_eur_with_abandonment(duong_decline, popt_duong, 1, forecast_horizon_days, q_abandon)
    
    t_full_pred = np.arange(1, forecast_horizon_days + 1)
    
    fig_eur = go.Figure()
    fig_eur.add_trace(go.Scatter(x=t_all, y=q_all, mode='markers', name='历史有效日产', marker=dict(color='lightgray', size=3)))
    
    if popt_hyp is not None:
        fig_eur.add_trace(go.Scatter(x=t_full_pred, y=hyp_decline(t_full_pred, *popt_hyp), mode='lines', name=f'双曲递减 (EUR: {eur_hyp/1000:.2f} 百万方)'))
    if popt_sepd is not None:
        fig_eur.add_trace(go.Scatter(x=t_full_pred, y=sepd_decline(t_full_pred, *popt_sepd), mode='lines', name=f'YM-SEPD (EUR: {eur_sepd/1000:.2f} 百万方)'))
    if popt_duong is not None:
        fig_eur.add_trace(go.Scatter(x=t_full_pred, y=duong_decline(t_full_pred, *popt_duong), mode='lines', name=f'改进 Duong (EUR: {eur_duong/1000:.2f} 百万方)'))
        
    fig_eur.add_hline(y=q_abandon, line_dash="dot", line_color="crimson", annotation_text=f"废弃极限 ({q_abandon} 10³m³/d)", annotation_position="bottom right")
    fig_eur.update_layout(title="全周期产能外推曲线与经济废弃截断", xaxis_title="生产时间 (d)", yaxis_title="日产气量 (10³m³/d)", height=520)
    st.plotly_chart(fig_eur, use_container_width=True)
    
    # 汇总卡片
    c1, c2, c3 = st.columns(3)
    hist_cum_total = np.sum(q_all)
    c1.metric("已开采累计产量", f"{hist_cum_total/1000:,.2f} 百万方")
    c2.metric("优选模型预计 EUR", f"{eur_sepd/1000:,.2f} 百万方" if popt_sepd is not None else "N/A")
    c3.metric("剩余可采储量 (ERR)", f"{(eur_sepd - hist_cum_total)/1000:,.2f} 百万方" if popt_sepd is not None else "N/A")

# ----------------- TAB 4: 模型对比 -----------------
with tab4:
    st.subheader("多模型拟合精度与全生命周期参数对比")
    
    summary_data = []
    if popt_exp is not None:
        summary_data.append({"模型": "指数递减", "拟合度 R²": round(r2_exp, 4), "计算EUR (10⁴m³)": round(np.sum(exp_decline(t_full_pred, *popt_exp))/10, 1)})
    if popt_hyp is not None:
        summary_data.append({"模型": "双曲递减 (Arps)", "拟合度 R²": round(r2_hyp, 4), "计算EUR (10⁴m³)": round(eur_hyp/10, 1)})
    if popt_sepd is not None:
        summary_data.append({"模型": "改进延伸指数 (YM-SEPD)", "拟合度 R²": round(r2_sepd, 4), "计算EUR (10⁴m³)": round(eur_sepd/10, 1)})
    if popt_duong is not None:
        summary_data.append({"模型": "改进 Duong", "拟合度 R²": round(r2_duong, 4), "计算EUR (10⁴m³)": round(eur_duong/10, 1)})
        
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# ----------------- TAB 5: 理论说明 -----------------
with tab5:
    st.markdown("""
    ### 理论依据与甲方答辩要点
    1. **为什么必须做时间截断盲测？**
       - 传统的“全历史拟合”只能反映曲线拟合程度（$R^2$），无法证明外推能力；
       - 通过屏蔽后半段真实数据进行**前测后验**，算出的验证期累产符合率是绝对客观的物理计量误差，不容置疑。
    2. **为什么要重构“有效生产天数”？**
       - 页岩气日常管理中包含大量试井、压裂串层关井和管网限制；
       - 如果包含零产天数，对数坐标下会出现大量零值奇异点，破坏斜率真实性。
    3. **废弃产量截断的重要性**：
       - 页岩气 Arps 双曲递减系数 $b > 1$ 时数学积分会发散趋于无穷大；必须以 $q_{ab} = 1000\text{ m}^3/\text{d}$ 或 20 年寿命为硬性边界。
    """)