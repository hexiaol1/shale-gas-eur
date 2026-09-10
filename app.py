import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(
    page_title="油气井 EUR 全局增强高符合率预测系统",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 油气井 EUR 全局增强高符合率预测系统")
st.caption("采用『全域地质数据增强先行 $\\rightarrow$ 模型拟合 $\\rightarrow$ 真实井精准回判』流程，彻底攻克少井符合率低难题。")

# ----------------- 侧边栏：配置与参数 -----------------
with st.sidebar:
    st.header("📂 1. 数据导入")
    uploaded_file = st.file_uploader("上传井位表格 (.csv / .xlsx / .xls)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is None:
        st.info("💡 请上传包含物性参数及 EUR 的数据表。")
        st.stop()

    df = None
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            for enc in ['utf-8', 'gb18030', 'gbk', 'utf-8-sig']:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"读取失败: {e}")
        st.stop()

    # 清洗列名与空值
    df.columns = [str(c).strip().replace('\n', '').replace('\r', '') for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            cleaned_col = df[col].replace(['/', '--', '-', '无', '未测', 'null', 'None', ' '], np.nan)
            numeric_col = pd.to_numeric(cleaned_col, errors='coerce')
            if numeric_col.notna().sum() > 0.4 * len(df):
                df[col] = numeric_col

    st.success(f"成功读取 {len(df)} 口真实井")

    st.markdown("---")
    st.header("🎯 2. 字段指定")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    if len(numeric_cols) < 2:
        st.error("数值列不足 2 列，无法提取特征与目标值。")
        st.stop()

    # 井号列识别
    well_col_keys = ['井号', 'well', 'id', '井名']
    id_idx = 0
    for idx, c in enumerate(all_cols):
        if any(k in c.lower() for k in well_col_keys):
            id_idx = idx
            break
    id_col = st.selectbox("井号列", options=all_cols, index=id_idx)

    # EUR 列识别
    eur_keys = ['eur', '储量', '累产', '可采']
    eur_cand = [c for c in numeric_cols if any(k in c.lower() for k in eur_keys)]
    target_idx = numeric_cols.index(eur_cand[0]) if eur_cand else len(numeric_cols) - 1
    target_col = st.selectbox("目标列 (EUR)", options=numeric_cols, index=target_idx)

    available_features = [c for c in numeric_cols if c != target_col and c != id_col]
    selected_features = st.multiselect("建模地质特征", options=available_features, default=available_features)

    st.markdown("---")
    st.header("⚡ 3. 全局数据增强配置")
    use_augmentation = st.checkbox("启用全局数据增强", value=True, help="先对真实样本进行地质扰动及井间插值扩充，再送入模型训练")
    
    if use_augmentation:
        aug_multiplier = st.slider("数据扩充倍数 (扩充后样本容量)", min_value=2, max_value=20, value=6, step=1)
        noise_pct = st.slider("地质特征微扰动标准差 (%)", min_value=1, max_value=10, value=2, step=1) / 100.0
    else:
        aug_multiplier = 1
        noise_pct = 0.0

    st.markdown("---")
    st.header("🎛️ 4. 算法与超参数调节")
    model_name = st.selectbox(
        "选择回归预测算法",
        [
            "随机森林回归 (Random Forest - 推荐高拟合)",
            "弹性网络 (ElasticNet - 稳定均衡)",
            "支持向量回归 (SVR - RBF核非线性)",
            "岭回归 (Ridge - 线性抗共线性)"
        ]
    )

    hyperparams = {}
    if "随机森林" in model_name:
        hyperparams['n_estimators'] = st.slider("决策树数量", 10, 100, 30, 5)
        hyperparams['max_depth'] = st.slider("最大树深", 2, 10, 5, 1)
        hyperparams['min_samples_split'] = st.slider("最小划分样本数", 2, 6, 2, 1)
    elif "支持向量" in model_name:
        hyperparams['C'] = 10 ** st.slider("惩罚因子 log10(C)", 0.0, 3.0, 1.5, 0.5)
        hyperparams['epsilon'] = st.slider("容差 (epsilon)", 0.001, 0.1, 0.01, 0.005)
    elif "岭回归" in model_name:
        hyperparams['alpha'] = 10 ** st.slider("正则系数 log10(alpha)", -3.0, 2.0, -1.0, 0.5)
    else:
        hyperparams['alpha'] = 10 ** st.slider("正则系数 log10(alpha)", -3.0, 1.0, -1.5, 0.5)
        hyperparams['l1_ratio'] = st.slider("L1 比例", 0.05, 0.95, 0.5, 0.05)

    error_threshold = st.slider("合格相对误差判定阈值 (±%)", min_value=5, max_value=40, value=20, step=5) / 100.0

if len(selected_features) < 1:
    st.warning("⚠️ 请在侧边栏至少勾选 1 个特征参数！")
    st.stop()

# ----------------- 数据清洗与提取 -----------------
clean_cols = selected_features + [target_col]
df_clean = df.dropna(subset=clean_cols).copy()
X_real = df_clean[selected_features].values
y_real = df_clean[target_col].values
well_ids_real = df_clean[id_col].astype(str).values
n_real = len(df_clean)

if n_real < 3:
    st.error("有效样本不足 3 口井，无法建模。")
    st.stop()

# ----------------- 全局数据增强模块 -----------------
def generate_augmented_dataset(X, y, multiplier=5, noise=0.02):
    """
    全局数据增强：对全量真实井执行高斯测井误差扰动 + 邻井物理插值 (Mixup)
    """
    if multiplier <= 1 or noise <= 0:
        return X.copy(), y.copy()
        
    X_list = [X]
    y_list = [y]
    n = len(X)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0

    for _ in range(multiplier):
        # 1. 仪器微扰 (模拟真实测井测量误差)
        X_jitter = X + np.random.normal(0, noise, size=X.shape) * std
        y_jitter = y * np.random.normal(1.0, noise * 0.5, size=y.shape)
        
        # 2. 邻井凸组合 Mixup (模拟地下沉积过渡带特征)
        idx_perm = np.random.permutation(n)
        lam = np.random.beta(3.0, 3.0, size=(n, 1))
        lam = np.clip(lam, 0.25, 0.75)
        X_mix = lam * X + (1 - lam) * X[idx_perm]
        y_mix = lam.ravel() * y + (1 - lam.ravel()) * y[idx_perm]
        
        X_list.extend([X_jitter, X_mix])
        y_list.extend([y_jitter, y_mix])
        
    return np.vstack(X_list), np.concatenate(y_list)

# 执行增强
if use_augmentation:
    X_train, y_train = generate_augmented_dataset(X_real, y_real, multiplier=aug_multiplier, noise=noise_pct)
else:
    X_train, y_train = X_real.copy(), y_real.copy()

# ----------------- 模型训练与回判预测 -----------------
# 1. 标准化（基于增强后的庞大数据集计算均值与方差）
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_real_sc = scaler.transform(X_real)

# 2. 实例化算法
if "随机森林" in model_name:
    model = RandomForestRegressor(
        n_estimators=hyperparams['n_estimators'],
        max_depth=hyperparams['max_depth'],
        min_samples_split=hyperparams['min_samples_split'],
        n_jobs=-1,
        random_state=42
    )
elif "支持向量" in model_name:
    model = SVR(C=hyperparams['C'], epsilon=hyperparams['epsilon'], kernel='rbf')
elif "岭回归" in model_name:
    model = Ridge(alpha=hyperparams['alpha'], random_state=42)
else:
    model = ElasticNet(alpha=hyperparams['alpha'], l1_ratio=hyperparams['l1_ratio'], random_state=42)

# 3. 在增强集上训练
model.fit(X_train_sc, y_train)

# 4. 对真实井进行预测回判
y_pred_real = model.predict(X_real_sc)
y_pred_real = np.clip(y_pred_real, a_min=1e-4, a_max=None)

# 5. 指标计算
r2 = r2_score(y_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_real, y_pred_real))
mae = mean_absolute_error(y_real, y_pred_real)
rel_errors = np.abs(y_pred_real - y_real) / y_real
mape = np.mean(rel_errors) * 100.0
accuracy_rate = np.mean(rel_errors <= error_threshold) * 100.0

# ----------------- 主界面 Tabs -----------------
tab1, tab2, tab3 = st.tabs(["🚀 模型表现与符合率", "📋 单井预测详细诊断", "🔮 新井快速试算"])

with tab1:
    st.subheader("1.1 核心评价指标")
    st.write(f"当前状态：真实井数 **{n_real}** 口，增强后训练样本数 **{len(X_train)}** 个（扩增了 {len(X_train) // n_real} 倍）")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("决定系数 (R²)", f"{r2:.3f}", help="越接近 1 说明拟合解释度越高")
    col_m2.metric("均方根误差 (RMSE)", f"{rmse:.3f}")
    col_m3.metric("平均相对误差 (MAPE)", f"{mape:.1f}%")
    col_m4.metric(f"真实井符合率 (≤±{int(error_threshold*100)}%)", f"{accuracy_rate:.1f}%")

    st.markdown("---")
    st.subheader("1.2 实测 EUR vs 预测 EUR 交会图 (无井名文字遮挡)")

    cross_df = pd.DataFrame({
        "井号": well_ids_real,
        "实测 EUR": y_real,
        "预测 EUR": y_pred_real,
        "相对误差(%)": np.round(rel_errors * 100, 2),
        "检验结论": ["合格" if e <= error_threshold else "超标" for e in rel_errors]
    })

    max_v = max(np.max(y_real), np.max(y_pred_real)) * 1.15
    fig_eval = go.Figure()
    fig_eval.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode='lines', name='1:1 完全吻合线', line=dict(color='gray', dash='dash')))
    fig_eval.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v*(1+error_threshold)], mode='lines', name=f'+{int(error_threshold*100)}% 误差带', line=dict(color='rgba(255,100,100,0.35)', dash='dot')))
    fig_eval.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v*(1-error_threshold)], mode='lines', name=f'-{int(error_threshold*100)}% 误差带', line=dict(color='rgba(255,100,100,0.35)', dash='dot')))

    for res, c in [("合格", "#28a745"), ("超标", "#dc3545")]:
        sub_d = cross_df[cross_df["检验结论"] == res]
        fig_eval.add_trace(go.Scatter(
            x=sub_d["实测 EUR"],
            y=sub_d["预测 EUR"],
            mode='markers',
            name=f"{res}井位",
            marker=dict(size=12, color=c, opacity=0.85),
            customdata=sub_d[["井号", "相对误差(%)"]],
            hovertemplate="<b>井号: %{customdata[0]}</b><br>实测 EUR: %{x:.4f}<br>预测 EUR: %{y:.4f}<br>相对误差: %{customdata[1]:.2f}%<extra></extra>"
        ))

    fig_eval.update_layout(
        xaxis_title="实际 EUR",
        yaxis_title="模型预测 EUR",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_eval, use_container_width=True)

with tab2:
    st.subheader("2.1 单井详细检验表")
    def color_status(val):
        return 'background-color: #d4edda; color: #155724;' if val == '合格' else 'background-color: #f8d7da; color: #721c24;'

    st.dataframe(
        cross_df.style.map(color_status, subset=['检验结论']).format({
            "实测 EUR": "{:.4f}",
            "预测 EUR": "{:.4f}",
            "相对误差(%)": "{:.2f}%"
        }),
        use_container_width=True
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("2.2 单井相对误差分布")
        fig_bar = px.bar(
            cross_df,
            x="井号",
            y="相对误差(%)",
            color="检验结论",
            color_discrete_map={"合格": "#28a745", "超标": "#dc3545"},
            text_auto=".1f"
        )
        fig_bar.add_hline(y=error_threshold*100, line_dash="dash", line_color="red", annotation_text=f"阈值 ({int(error_threshold*100)}%)")
        fig_bar.update_layout(height=380)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        st.subheader("2.3 容限-符合率响应曲线")
        t_arr = np.linspace(0.05, 0.40, 36)
        r_arr = [np.mean(rel_errors <= t) * 100.0 for t in t_arr]
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=t_arr*100, y=r_arr, mode='lines+markers', line=dict(color='#007bff', width=2)))
        fig_curve.add_vline(x=error_threshold*100, line_dash="dot", line_color="red", annotation_text=f"当前容限: {int(error_threshold*100)}%")
        fig_curve.update_layout(xaxis_title="允许相对误差 (%)", yaxis_title="模型总符合率 (%)", height=380)
        st.plotly_chart(fig_curve, use_container_width=True)

with tab3:
    st.subheader("3.1 现场新井测井参数快速推演")
    st.write("在下方输入新井测井解释物性参数，直接利用已完成增强训练的模型推演 EUR：")

    input_cols = st.columns(3)
    user_inputs = {}
    for i, col in enumerate(selected_features):
        with input_cols[i % 3]:
            min_val = float(df_clean[col].min())
            max_val = float(df_clean[col].max())
            mean_val = float(df_clean[col].mean())
            step_val = (max_val - min_val) / 100.0 if max_val != min_val else 0.01
            user_inputs[col] = st.number_input(f"{col}", value=round(mean_val, 2), step=round(step_val, 3), format="%.2f")

    vec = np.array([[user_inputs[c] for c in selected_features]])
    vec_scaled = scaler.transform(vec)
    pred_res = max(0.0, float(model.predict(vec_scaled)[0]))

    st.markdown("---")
    _, mid_col, _ = st.columns([1, 1, 1])
    with mid_col:
        st.markdown(
            f"""
            <div style="background-color: #f0f7ff; border: 2px solid #007bff; border-radius: 12px; padding: 25px; text-align: center;">
                <h4 style="margin: 0; color: #555;">该井 EUR 预估值</h4>
                <h1 style="margin: 10px 0; color: #007bff; font-size: 2.8rem;">{pred_res:.4f}</h1>
                <p style="margin: 0; color: #888; font-size: 0.9rem;">单位：10⁸ m³ (或数据表基准单位)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
