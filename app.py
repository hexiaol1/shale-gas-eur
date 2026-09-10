import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(
    page_title="油气井 EUR 数据增强与超参数优化系统",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⛏️ 油气井 EUR 预测与符合率检验系统 (数据增强与超参调优版)")
st.caption("针对少井工况：去除 PCA，引入『地质特征数据增强』与『模型超参数动态微调』突破符合率瓶颈。")

# ----------------- 侧边栏：文件上传与参数选择 -----------------
with st.sidebar:
    st.header("📂 1. 数据上传")
    uploaded_file = st.file_uploader("上传井位数据表 (支持 .csv / .xlsx / .xls)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is None:
        st.info("💡 请先上传包含地质参数和 EUR 的数据表格。")
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
            if df is None:
                st.error("CSV 编码格式解析失败，请在 Excel 中另存为 xlsx 后上传。")
                st.stop()
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        st.stop()

    df.columns = [str(c).strip().replace('\n', '').replace('\r', '') for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            cleaned_col = df[col].replace(['/', '--', '-', '无', '未测', 'null', 'None', ' '], np.nan)
            numeric_col = pd.to_numeric(cleaned_col, errors='coerce')
            if numeric_col.notna().sum() > 0.4 * len(df):
                df[col] = numeric_col

    st.success(f"成功载入 {len(df)} 行数据")
    
    st.markdown("---")
    st.header("🎯 2. 字段映射")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    if len(numeric_cols) < 2:
        st.error("识别到的数值列不足 2 列，请检查表格是否包含纯文本或缺失过多。")
        st.stop()

    well_col_keywords = ['井号', 'well', 'id', '井名', '井号名称']
    default_id_idx = 0
    for idx, col in enumerate(all_cols):
        if any(k in col.lower() for k in well_col_keywords):
            default_id_idx = idx
            break
    id_col = st.selectbox("井号 / ID 列", options=all_cols, index=default_id_idx)
    
    eur_keywords = ['eur', '储量', '累产', '可采', '最终可采', '估算可采']
    eur_candidates = [c for c in numeric_cols if any(k in c.lower() for k in eur_keywords)]
    default_target_idx = numeric_cols.index(eur_candidates[0]) if eur_candidates else len(numeric_cols) - 1
    target_col = st.selectbox("目标变量 (EUR / 累产)", options=numeric_cols, index=default_target_idx)

    available_features = [c for c in numeric_cols if c != target_col and c != id_col]
    selected_features = st.multiselect("参与建模的地质/工程特征", options=available_features, default=available_features)

    st.markdown("---")
    st.header("🧪 3. 数据增强与物理变换")
    
    use_log_transform = st.checkbox(
        "启用目标值对数变换 (Log-Transform)", 
        value=True, 
        help="推荐开启：油气产能常呈对数正态分布，取对数后拟合可抑制大井误差绑架整体模型。"
    )
    
    use_augmentation = st.checkbox(
        "启用训练集数据增强 (Data Augmentation)", 
        value=True, 
        help="根据测井仪器 $2\%-5\%$ 测量误差生成扰动合成样本，大幅扩充小样本训练集容量。"
    )
    
    if use_augmentation:
        aug_ratio = st.slider("数据倍增倍数 (合成样本/原始样本)", min_value=1, max_value=8, value=3, step=1)
        noise_level = st.slider("特征扰动标准差比例 (%)", min_value=1, max_value=8, value=3, step=1) / 100.0
    else:
        aug_ratio = 0
        noise_level = 0.0

    st.markdown("---")
    st.header("🎛️ 4. 模型选择与超参数调节")
    
    model_name = st.selectbox(
        "选择回归预测模型",
        [
            "随机森林回归 (Random Forest)",
            "Lasso 回归 (L1 稀疏约束)",
            "岭回归 (Ridge - L2 抗共线性)",
            "支持向量回归 (SVR - 核函数非线性映射)",
            "弹性网络 (ElasticNet - L1+L2 混合)"
        ]
    )
    
    # 动态超参数面板
    hyperparams = {}
    if "随机森林" in model_name:
        st.markdown("**随机森林超参数配置**")
        hyperparams['n_estimators'] = st.slider("决策树数量 (n_estimators)", 10, 100, 30, 5)
        hyperparams['max_depth'] = st.slider("最大树深 (max_depth)", 1, 6, 3, 1, help="小样本建议 <= 3，防止强行拟合噪声")
        hyperparams['min_samples_split'] = st.slider("节点分裂最小样本数", 2, 6, 2, 1)
    elif "Lasso" in model_name:
        st.markdown("**Lasso 超参数配置**")
        alpha_exp = st.slider("正则化强度 log10(alpha)", -4.0, 1.0, -1.5, 0.25)
        hyperparams['alpha'] = 10 ** alpha_exp
        st.caption(f"当前 alpha = {hyperparams['alpha']:.5f}")
    elif "岭回归" in model_name:
        st.markdown("**岭回归超参数配置**")
        alpha_exp = st.slider("L2 正则强度 log10(alpha)", -3.0, 3.0, 0.5, 0.25)
        hyperparams['alpha'] = 10 ** alpha_exp
        st.caption(f"当前 alpha = {hyperparams['alpha']:.5f}")
    elif "支持向量回归" in model_name:
        st.markdown("**SVR 超参数配置**")
        c_exp = st.slider("惩罚因子 log10(C)", -1.0, 3.0, 1.0, 0.5)
        hyperparams['C'] = 10 ** c_exp
        hyperparams['epsilon'] = st.slider("容忍误差裕度 (epsilon)", 0.01, 0.30, 0.05, 0.01)
        hyperparams['kernel'] = st.selectbox("核函数", ["rbf", "linear", "poly"], index=0)
    else:
        st.markdown("**弹性网络超参数配置**")
        alpha_exp = st.slider("总正则强度 log10(alpha)", -3.0, 1.0, -1.5, 0.25)
        hyperparams['alpha'] = 10 ** alpha_exp
        hyperparams['l1_ratio'] = st.slider("L1 比例权重 (l1_ratio)", 0.05, 0.95, 0.5, 0.05)

    st.markdown("---")
    st.header("📊 5. 验证与符合率阈值")
    cv_method = st.radio("交叉验证方式", ["留一交叉验证 (LOOCV - 小样本推荐)", "K折交叉验证 (5-Fold)"])
    error_threshold = st.slider("合格相对误差容限 (±%)", min_value=5, max_value=50, value=25, step=5) / 100.0

if len(selected_features) < 1:
    st.warning("⚠️ 请在侧边栏至少勾选 1 个特征参数！")
    st.stop()

# ----------------- 数据清洗 -----------------
clean_cols = selected_features + [target_col]
df_clean = df.dropna(subset=clean_cols).copy()
if len(df_clean) < len(df):
    st.warning(f"注意：排除了包含空缺值的 {len(df) - len(df_clean)} 口井，当前可用建模样本数: {len(df_clean)}")

if len(df_clean) < 3:
    st.error("有效样本量不足 3 个，无法进行交叉验证。")
    st.stop()

X = df_clean[selected_features].values
y = df_clean[target_col].values
well_ids = df_clean[id_col].astype(str).values
n_samples = len(df_clean)

# ----------------- 数据增强核心函数 -----------------
def augment_training_data(X_train, y_train, multiplier=2, noise=0.03):
    """
    仅在训练集内部进行小样本高斯微扰与 Mixup 插值合成
    """
    if multiplier <= 0 or noise <= 0:
        return X_train, y_train
        
    augmented_X = [X_train]
    augmented_y = [y_train]
    
    n_train = len(X_train)
    std_features = np.std(X_train, axis=0)
    std_features[std_features == 0] = 1.0  # 防止除以零
    
    for _ in range(multiplier):
        # 1. 高斯噪声微扰增强（模拟仪器解释偏差）
        jitter = np.random.normal(0, noise, size=X_train.shape) * std_features
        X_jittered = X_train + jitter
        y_jittered = y_train * np.random.normal(1.0, noise * 0.5, size=y_train.shape)
        
        # 2. 局部 Mixup 连续插值增强（模拟邻井过渡层）
        perm = np.random.permutation(n_train)
        lam = np.random.beta(2.0, 2.0, size=(n_train, 1))
        lam = np.clip(lam, 0.35, 0.65)  # 避免偏激外推
        X_mix = lam * X_train + (1 - lam) * X_train[perm]
        y_mix = lam.ravel() * y_train + (1 - lam.ravel()) * y_train[perm]
        
        augmented_X.extend([X_jittered, X_mix])
        augmented_y.extend([y_jittered, y_mix])
        
    return np.vstack(augmented_X), np.concatenate(augmented_y)

# ----------------- 模型工厂函数 -----------------
def create_model(model_name, hyperparams):
    if "随机森林" in model_name:
        return RandomForestRegressor(
            n_estimators=hyperparams['n_estimators'],
            max_depth=hyperparams['max_depth'],
            min_samples_split=hyperparams['min_samples_split'],
            n_jobs=-1,
            random_state=42
        )
    elif "Lasso" in model_name:
        return Lasso(alpha=hyperparams['alpha'], random_state=42, max_iter=5000)
    elif "岭回归" in model_name:
        return Ridge(alpha=hyperparams['alpha'], random_state=42)
    elif "支持向量回归" in model_name:
        return SVR(
            C=hyperparams['C'],
            epsilon=hyperparams['epsilon'],
            kernel=hyperparams['kernel']
        )
    else:
        return ElasticNet(
            alpha=hyperparams['alpha'],
            l1_ratio=hyperparams['l1_ratio'],
            random_state=42,
            max_iter=5000
        )

# ----------------- 页面主 Tabs -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 数据检查与相关性", 
    "🚀 增强建模与超参调优验证", 
    "🎯 符合率与误差诊断", 
    "🔮 单井新参数估算"
])

# ----------------- TAB 1: 数据检查 -----------------
with tab1:
    st.subheader("1.1 导入数据概览")
    st.dataframe(df_clean[[id_col] + clean_cols], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1.2 数值统计指标")
        st.dataframe(df_clean[clean_cols].describe().T.style.format("{:.3f}"), use_container_width=True)
    with col2:
        st.subheader("1.3 特征相关性热力图")
        corr_matrix = df_clean[clean_cols].corr()
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            aspect="auto", 
            color_continuous_scale="RdBu_r",
            labels=dict(color="相关系数")
        )
        fig_corr.update_layout(height=420, margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

# ----------------- TAB 2: 增强建模与预测 -----------------
with tab2:
    st.subheader("2.1 交叉验证评估机制")
    st.write(f"当前方案：数据增强 = **{'已开启 (×' + str(aug_ratio*2 + 1) + '倍扩展)' if use_augmentation else '已关闭'}**，目标对数变换 = **{'已开启' if use_log_transform else '已关闭'}**")

    if cv_method.startswith("留一"):
        cv_splitter = LeaveOneOut()
    else:
        k_val = min(5, n_samples)
        cv_splitter = KFold(n_splits=k_val, shuffle=True, random_state=42)

    y_preds = np.zeros(n_samples)

    # 交叉验证循环
    for train_idx, test_idx in cv_splitter.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # 1. 仅在训练折做数据增强（严防泄露）
        if use_augmentation:
            X_tr_fit, y_tr_fit = augment_training_data(X_tr, y_tr, multiplier=aug_ratio, noise=noise_level)
        else:
            X_tr_fit, y_tr_fit = X_tr, y_tr

        # 2. 目标对数变换
        if use_log_transform:
            y_tr_fit = np.log(np.clip(y_tr_fit, a_min=1e-6, a_max=None))

        # 3. 特征标准化（在增强后的训练集上 fit）
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_fit)
        X_te_sc = scaler.transform(X_te)

        # 4. 根据当前侧边栏超参实例化并拟合
        m = create_model(model_name, hyperparams)
        m.fit(X_tr_sc, y_tr_fit)
        pred_sub = m.predict(X_te_sc)

        # 5. 反变换
        if use_log_transform:
            pred_sub = np.exp(pred_sub)

        y_preds[test_idx] = pred_sub

    # 截断非物理负数
    y_preds = np.clip(y_preds, a_min=1e-4, a_max=None)

    # 计算指标
    r2 = r2_score(y, y_preds)
    rmse = np.sqrt(mean_squared_error(y, y_preds))
    mae = mean_absolute_error(y, y_preds)
    rel_errors = np.abs(y_preds - y) / y
    mape = np.mean(rel_errors) * 100.0
    accuracy_rate = np.mean(rel_errors <= error_threshold) * 100.0

    st.markdown("---")
    st.write("##### 盲测指标表现 (随侧边栏超参调节实时变动)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("决定系数 (R²)", f"{r2:.3f}", help="越接近 1 越好")
    m2.metric("均方根误差 (RMSE)", f"{rmse:.3f}")
    m3.metric("平均相对误差 (MAPE)", f"{mape:.1f}%")
    m4.metric(f"符合率 (≤±{int(error_threshold*100)}%)", f"{accuracy_rate:.1f}%")

    cross_df = pd.DataFrame({
        "井号": well_ids,
        "实测 EUR": y,
        "预测 EUR": y_preds,
        "相对误差(%)": np.round(rel_errors * 100, 2),
        "检验结论": ["合格" if e <= error_threshold else "超标" for e in rel_errors]
    })

    # 交会图（无散点文本遮挡）
    max_v = max(np.max(y), np.max(y_preds)) * 1.15
    fig_eval = go.Figure()
    fig_eval.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode='lines', name='1:1 理想线', line=dict(color='gray', dash='dash')))
    fig_eval.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v*(1+error_threshold)], mode='lines', name=f'+{int(error_threshold*100)}% 误差带', line=dict(color='rgba(255,100,100,0.35)', dash='dot')))
    fig_eval.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v*(1-error_threshold)], mode='lines', name=f'-{int(error_threshold*100)}% 误差带', line=dict(color='rgba(255,100,100,0.35)', dash='dot')))

    for res, c in [("合格", "#28a745"), ("超标", "#dc3545")]:
        sub_d = cross_df[cross_df["检验结论"] == res]
        fig_eval.add_trace(go.Scatter(
            x=sub_d["实测 EUR"],
            y=sub_d["预测 EUR"],
            mode='markers',
            name=f"{res}井位",
            marker=dict(size=10, color=c, opacity=0.85),
            customdata=sub_d[["井号", "相对误差(%)"]],
            hovertemplate="<b>井号: %{customdata[0]}</b><br>实测: %{x:.4f}<br>预测: %{y:.4f}<br>相对误差: %{customdata[1]:.2f}%<extra></extra>"
        ))

    fig_eval.update_layout(
        xaxis_title="实际 EUR",
        yaxis_title="盲测预测 EUR",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_eval, use_container_width=True)

# ----------------- TAB 3: 符合率与误差诊断 -----------------
with tab3:
    st.subheader("3.1 单井详细检验表")
    def color_status(val):
        return 'background-color: #d4edda;' if val == '合格' else 'background-color: #f8d7da;'

    st.dataframe(
        cross_df.style.map(color_status, subset=['检验结论']).format({
            "实测 EUR": "{:.4f}",
            "预测 EUR": "{:.4f}",
            "相对误差(%)": "{:.2f}%"
        }),
        use_container_width=True
    )

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        st.subheader("3.2 单井误差分布柱状图")
        fig_bars = px.bar(
            cross_df,
            x="井号",
            y="相对误差(%)",
            color="检验结论",
            color_discrete_map={"合格": "#28a745", "超标": "#dc3545"},
            text_auto=".1f"
        )
        fig_bars.add_hline(y=error_threshold*100, line_dash="dash", line_color="red", annotation_text=f"阈值 ({int(error_threshold*100)}%)")
        fig_bars.update_layout(height=400)
        st.plotly_chart(fig_bars, use_container_width=True)

    with c_b2:
        st.subheader("3.3 容限-符合率响应曲线")
        t_arr = np.linspace(0.05, 0.50, 46)
        r_arr = [np.mean(rel_errors <= t) * 100.0 for t in t_arr]
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=t_arr*100, y=r_arr, mode='lines+markers', line=dict(color='#007bff', width=2)))
        fig_curve.add_vline(x=error_threshold*100, line_dash="dot", line_color="red", annotation_text=f"当前选定: {int(error_threshold*100)}%")
        fig_curve.update_layout(xaxis_title="允许相对误差 (%)", yaxis_title="模型总符合率 (%)", height=400)
        st.plotly_chart(fig_curve, use_container_width=True)

# ----------------- TAB 4: 单井现场试算 -----------------
with tab4:
    st.subheader("4.1 现场新井参数实时推演")
    st.write("基于当前最佳超参数配置及增强全样本训练出的统一模型进行试算：")

    # 全量拟合
    if use_augmentation:
        X_full_fit, y_full_fit = augment_training_data(X, y, multiplier=aug_ratio, noise=noise_level)
    else:
        X_full_fit, y_full_fit = X, y

    if use_log_transform:
        y_full_fit = np.log(np.clip(y_full_fit, a_min=1e-6, a_max=None))

    final_scaler = StandardScaler()
    X_full_sc = final_scaler.fit_transform(X_full_fit)

    final_m = create_model(model_name, hyperparams)
    final_m.fit(X_full_sc, y_full_fit)

    # 动态输入面板
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
    vec_scaled = final_scaler.transform(vec)

    pred_res = final_m.predict(vec_scaled)[0]
    if use_log_transform:
        pred_res = np.exp(pred_res)
    pred_res = max(0.0, float(pred_res))

    st.markdown("---")
    res_c1, res_c2, res_c3 = st.columns([1, 1, 1])
    with res_c2:
        st.markdown(
            f"""
            <div style="background-color: #f0f7ff; border: 2px solid #007bff; border-radius: 12px; padding: 25px; text-align: center;">
                <h4 style="margin: 0; color: #555;">该井 EUR 预估值</h4>
                <h1 style="margin: 10px 0; color: #007bff; font-size: 2.8rem;">{pred_res:.4f}</h1>
                <p style="margin: 0; color: #888; font-size: 0.9rem;">单位：10⁸ m³ (或相应储量单位)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
