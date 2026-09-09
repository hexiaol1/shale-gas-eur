import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(
    page_title="油气井 EUR 小样本高精度预测与符合率检验系统",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⛏️ 油气井 EUR 预测与符合率检验系统 (小样本专用强化版)")
st.caption("针对非常规油气少井工况：集成『对数物理变换』、『主成分降维 (PCA)』与『抗共线性正则化算法』提升小样本符合率。")

# ----------------- 侧边栏：文件上传与参数选择 -----------------
with st.sidebar:
    st.header("📂 1. 数据上传")
    uploaded_file = st.file_uploader("上传井位数据表 (支持 .csv / .xlsx / .xls)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is None:
        st.info("💡 请先上传数据文件开始分析。")
        st.stop()
        
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        st.stop()

    st.success(f"成功载入 {len(df)} 行数据")
    
    st.markdown("---")
    st.header("🎯 2. 列定义")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    if len(numeric_cols) < 2:
        st.error("表格中数值型列不足 2 列，无法提取特征与目标值。")
        st.stop()

    id_col = st.selectbox("井号 / ID 列", options=all_cols, index=0)
    
    # 智能推断 EUR 列
    eur_candidates = [c for c in numeric_cols if 'eur' in c.lower()]
    default_target_idx = numeric_cols.index(eur_candidates[0]) if eur_candidates else len(numeric_cols) - 1
    target_col = st.selectbox("目标变量 (EUR)", options=numeric_cols, index=default_target_idx)

    available_features = [c for c in numeric_cols if c != target_col]
    selected_features = st.multiselect("参与建模的地质/工程特征", options=available_features, default=available_features)

    st.markdown("---")
    st.header("⚙️ 3. 建模优化与验证")
    
    use_log_transform = st.checkbox(
        "启用目标值对数变换 (Log-Transform)", 
        value=True, 
        help="油气产能呈偏态分布，将 EUR 取对数后再拟合，可大幅降低由于极端高产井导致的整体误差。"
    )
    
    use_pca = st.checkbox(
        "启用特征主成分降维 (PCA)", 
        value=(len(selected_features) > 4 and len(df) < 30),
        help="样本数极少时，输入特征过多会导致维数灾难。PCA 可将多个高共线性参数压缩为少数几个综合指标。"
    )
    
    pca_n_components = 2
    if use_pca:
        max_c = min(len(selected_features), max(1, len(df) - 2))
        pca_n_components = st.slider("PCA 主成分保留数", min_value=1, max_value=max(1, max_c), value=min(2, max_c))

    cv_method = st.radio("交叉验证方式", ["留一交叉验证 (LOOCV - 小样本必备)", "K折交叉验证 (5-Fold)"])
    error_threshold = st.slider("合格符合率相对误差容限 (±%)", min_value=5, max_value=40, value=20, step=5) / 100.0

if len(selected_features) < 1:
    st.warning("⚠️ 请在侧边栏至少勾选 1 个特征参数！")
    st.stop()

# ----------------- 数据清洗 -----------------
clean_cols = selected_features + [target_col]
df_clean = df.dropna(subset=clean_cols).copy()
if len(df_clean) < len(df):
    st.warning(f"注意：排除了包含缺失值的 {len(df) - len(df_clean)} 条数据，当前可用样本数: {len(df_clean)}")

if len(df_clean) < 3:
    st.error("有效样本量不足 3 个，无法进行建模训练与交叉验证。")
    st.stop()

X = df_clean[selected_features].values
y = df_clean[target_col].values
well_ids = df_clean[id_col].astype(str).values
n_samples = len(df_clean)

# ----------------- 选项卡 -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 数据检查与相关性", 
    "🚀 小样本自适应建模与预测", 
    "🎯 符合率与误差诊断", 
    "🔮 单井新参数估算"
])

# ----------------- TAB 1: 数据检查 -----------------
with tab1:
    st.subheader("1.1 导入数据概览")
    st.dataframe(df_clean[[id_col] + clean_cols], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1.2 统计指标")
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

# ----------------- TAB 2: 模型训练与预测 -----------------
with tab2:
    st.subheader("2.1 算法选择")
    col_algo, col_info = st.columns([1, 2])
    
    with col_algo:
        model_name = st.selectbox(
            "选择适合少井的回归算法",
            [
                "Lasso 回归 (强稀疏降维，适合小样本冗余特征)",
                "岭回归 (RidgeCV - 抗多重共线性稳定型)",
                "弹性网络 (ElasticNetCV - 平衡综合型)",
                "支持向量回归 (SVR - RBF 核非线性映射)"
            ]
        )
    with col_info:
        if "Lasso" in model_name:
            st.info("📌 **Lasso 机制**：加入 L1 正则化惩罚，会自动将冗余和共线性特征的权重压缩至 0，相当于自主做特征筛选，防止过拟合。")
        elif "岭回归" in model_name:
            st.info("📌 **岭回归机制**：加入 L2 正则化惩罚，在特征强相关（例如游离气与总含气量）时保持极高稳定性，不会因少数异常点发生抖动。")
        elif "支持向量回归" in model_name:
            st.info("📌 **SVR 机制**：基于结构风险最小化，核函数映射使它在样本量甚至少于 20 个时，表现显著优于随机森林和深度学习。")
        else:
            st.info("📌 **弹性网络**：兼具 L1 特征选择与 L2 协方差控制，适合中等样本。")

    # 配置交叉验证
    if cv_method.startswith("留一"):
        cv_splitter = LeaveOneOut()
    else:
        k_val = min(5, n_samples)
        cv_splitter = KFold(n_splits=k_val, shuffle=True, random_state=42)

    y_preds = np.zeros(n_samples)

    # 交叉验证主循环
    for train_idx, test_idx in cv_splitter.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # 1. 目标值对数变换
        if use_log_transform:
            y_tr_fit = np.log(np.clip(y_tr, a_min=1e-6, a_max=None))
        else:
            y_tr_fit = y_tr

        # 2. 特征标准化
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        # 3. PCA 降维 (可选)
        if use_pca:
            pca = PCA(n_components=pca_n_components)
            X_tr_sc = pca.fit_transform(X_tr_sc)
            X_te_sc = pca.transform(X_te_sc)

        # 4. 模型适配
        inner_cv = min(3, len(X_tr))
        if "Lasso" in model_name:
            m = LassoCV(cv=inner_cv, random_state=42)
        elif "岭回归" in model_name:
            m = RidgeCV()
        elif "弹性网络" in model_name:
            m = ElasticNetCV(cv=inner_cv, random_state=42)
        else:
            m = SVR(kernel="rbf", C=5.0, epsilon=0.05)

        m.fit(X_tr_sc, y_tr_fit)
        pred_sub = m.predict(X_te_sc)

        # 反变换
        if use_log_transform:
            pred_sub = np.exp(pred_sub)

        y_preds[test_idx] = pred_sub

    # 截断非物理负值
    y_preds = np.clip(y_preds, a_min=1e-4, a_max=None)

    # 指标计算
    r2 = r2_score(y, y_preds)
    rmse = np.sqrt(mean_squared_error(y, y_preds))
    mae = mean_absolute_error(y, y_preds)
    rel_errors = np.abs(y_preds - y) / y
    mape = np.mean(rel_errors) * 100.0
    accuracy_rate = np.mean(rel_errors <= error_threshold) * 100.0

    st.markdown("---")
    st.write("##### 盲测指标概览")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("决定系数 (R²)", f"{r2:.3f}", help="越接近 1 越好，小于 0 说明模型泛化弱于均值猜测")
    m2.metric("均方根误差 (RMSE)", f"{rmse:.3f}")
    m3.metric("平均相对误差 (MAPE)", f"{mape:.1f}%")
    m4.metric(f"符合率 (≤±{int(error_threshold*100)}%)", f"{accuracy_rate:.1f}%")

    # 绘制实测-预测交会图
    cross_df = pd.DataFrame({
        "井号": well_ids,
        "实测 EUR": y,
        "预测 EUR": y_preds,
        "相对误差(%)": np.round(rel_errors * 100, 2),
        "检验结论": ["合格" if e <= error_threshold else "超标" for e in rel_errors]
    })

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
            mode='markers+text',
            text=sub_d["井号"],
            textposition="top center",
            name=f"{res}井位",
            marker=dict(size=11, color=c, opacity=0.85)
        ))

    fig_eval.update_layout(
        xaxis_title="实际 EUR",
        yaxis_title="交叉验证预测 EUR",
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
    st.subheader("4.1 现场新井测井参数快速反演 EUR")
    st.write("利用全量数据与当前优化方案（对数变换/PCA降维）进行参数推演：")

    # 全量拟合
    final_scaler = StandardScaler()
    X_full = final_scaler.fit_transform(X)
    
    if use_pca:
        final_pca = PCA(n_components=pca_n_components)
        X_full = final_pca.fit_transform(X_full)
    else:
        final_pca = None

    y_full_fit = np.log(np.clip(y, a_min=1e-6, a_max=None)) if use_log_transform else y

    if "Lasso" in model_name:
        final_m = LassoCV(cv=min(3, n_samples), random_state=42)
    elif "岭回归" in model_name:
        final_m = RidgeCV()
    elif "弹性网络" in model_name:
        final_m = ElasticNetCV(cv=min(3, n_samples), random_state=42)
    else:
        final_m = SVR(kernel="rbf", C=5.0, epsilon=0.05)

    final_m.fit(X_full, y_full_fit)

    # 动态输入表单
    input_cols = st.columns(3)
    user_inputs = {}
    for i, col in enumerate(selected_features):
        with input_cols[i % 3]:
            min_val = float(df_clean[col].min())
            max_val = float(df_clean[col].max())
            mean_val = float(df_clean[col].mean())
            step_val = (max_val - min_val) / 100.0 if max_val != min_val else 0.01
            user_inputs[col] = st.number_input(f"{col}", value=round(mean_val, 2), step=round(step_val, 3), format="%.2f")

    # 实时推演
    vec = np.array([[user_inputs[c] for c in selected_features]])
    vec_scaled = final_scaler.transform(vec)
    if final_pca:
        vec_scaled = final_pca.transform(vec_scaled)

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
                <p style="margin: 0; color: #888; font-size: 0.9rem;">单位：10⁸ m³ (或数据表基准单位)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
