import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import RidgeCV, ElasticNetCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 页面基础配置
st.set_page_config(
    page_title="页岩气井 EUR 预测与模型符合率检验系统",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- 内置演示数据（截图中7口井的数据） -----------------
DEFAULT_DATA = {
    "well_id": ["well_W202H10-1", "well_W204H10-2", "well_W204H45-1", "well_W208", "well_W209", "well_W216", "well_W217"],
    "Depth": [2557.13, 3338.48, 2844.59, 2732.64, 2929.72, 3438.99, 3664.14],
    "TOC": [3.20, 2.69, 3.44, 2.88, 2.72, 2.24, 1.87],
    "Brittle": [76.32, 66.05, 72.34, 59.36, 65.43, 58.31, 57.76],
    "Gas_satur": [58.93, 73.49, 70.83, 50.94, 54.34, 62.80, 57.76],
    "Gas_content": [3.84, 2.56, 1.95, 1.20, 1.80, 1.65, 1.58],
    "Porosity": [5.96, 5.45, 5.37, 6.66, 6.96, 5.75, 5.88],
    "Free_gas": [3.03, 5.13, 3.11, 3.24, 3.35, 2.39, 3.41],
    "Total_gas": [4.15, 6.04, 4.19, 4.18, 4.12, 3.88, 3.97],
    "Pressure": [1.40, 1.40, 1.40, 1.61, 1.80, 1.80, 2.15],
    "Adsorbed": [1.13, 0.91, 1.08, 0.94, 0.77, 1.49, 0.56],
    "eur": [0.0756, 0.2830, 1.2762, 0.3176, 0.4380, 0.7893, 0.3125]
}

# ----------------- 页面标题 -----------------
st.title("🛢️ 页岩气井 EUR 预测与模型符合率检验系统")
st.markdown("""
本系统针对非常规油气地质物性与产能数据，提供**相关性分析**、**多算法回归训练**、**交叉验证盲测**、**单井符合率检验**及**现场新井快速试算**。
""")

# ----------------- 侧边栏：数据配置 -----------------
with st.sidebar:
    st.header("⚙️ 1. 数据配置与输入")
    data_source = st.radio("选择数据源", ["使用示例数据（截图中7口井）", "上传自定义数据 (CSV/Excel)"])
    
    if data_source == "使用示例数据（截图中7口井）":
        df = pd.DataFrame(DEFAULT_DATA)
    else:
        uploaded_file = st.file_uploader("上传文件", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            st.info("请上传数据文件，暂自动加载示例数据。")
            df = pd.DataFrame(DEFAULT_DATA)

    st.markdown("---")
    st.header("🎯 2. 变量定义")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    
    default_id = "well_id" if "well_id" in all_cols else all_cols[0]
    id_col = st.selectbox("井号 / ID 列", options=all_cols, index=all_cols.index(default_id))
    
    default_target = "eur" if "eur" in numeric_cols else numeric_cols[-1]
    target_col = st.selectbox("预测目标列 (EUR)", options=numeric_cols, index=numeric_cols.index(default_target))
    
    available_features = [c for c in numeric_cols if c != target_col]
    selected_features = st.multiselect(
        "选择参与建模的特征",
        options=available_features,
        default=available_features
    )
    
    st.markdown("---")
    st.header("🧪 3. 验证与符合率阈值")
    cv_method = st.radio("交叉验证方式", ["留一交叉验证 (LOOCV - 小样本推荐)", "K折交叉验证 (5-Fold)"])
    error_threshold = st.slider("相对误差合格阈值 (±%)", min_value=5, max_value=40, value=20, step=5) / 100.0

if len(selected_features) < 1:
    st.warning("⚠️ 请在侧边栏至少勾选 1 个特征参数！")
    st.stop()

# ----------------- 标签页 -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 数据全景与相关性", 
    "🚀 模型训练与预测", 
    "🎯 符合率与误差深度检验", 
    "🔮 单井参数快速试算"
])

# ----------------- TAB 1: 数据全景与相关性 -----------------
with tab1:
    st.subheader("1.1 原始数据表")
    st.dataframe(df, use_container_width=True)
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.subheader("1.2 统计描述")
        st.dataframe(df[selected_features + [target_col]].describe().T.style.format("{:.3f}"), use_container_width=True)
        
    with col_stat2:
        st.subheader("1.3 特征与 EUR 皮尔逊相关性热图")
        corr = df[selected_features + [target_col]].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            labels=dict(color="Correlation")
        )
        fig_corr.update_layout(height=420, margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

# ----------------- TAB 2: 模型训练与预测 -----------------
with tab2:
    st.subheader("2.1 模型选择与配置")
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        model_type = st.selectbox(
            "选择回归算法",
            ["弹性网络回归 (ElasticNetCV - 推荐)", "岭回归 (RidgeCV)", "普通多元线性回归 (OLS)", "随机森林回归 (Random Forest)"]
        )
        use_scaling = st.checkbox("特征标准化 (StandardScaler)", value=True)
        
    X = df[selected_features].values
    y = df[target_col].values
    well_ids = df[id_col].values
    n_samples = len(df)
    
    # 交叉验证配置
    if cv_method.startswith("留一"):
        cv_splitter = LeaveOneOut()
    else:
        k_val = min(5, n_samples)
        cv_splitter = KFold(n_splits=k_val, shuffle=True, random_state=42)

    # 执行交叉验证盲测
    y_preds = np.zeros(n_samples)
    
    for train_idx, test_idx in cv_splitter.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        if use_scaling:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
        if model_type == "弹性网络回归 (ElasticNetCV - 推荐)":
            inner_cv = min(3, len(X_tr))
            m = ElasticNetCV(cv=inner_cv, random_state=42)
        elif model_type == "岭回归 (RidgeCV)":
            m = RidgeCV()
        elif model_type == "普通多元线性回归 (OLS)":
            m = LinearRegression()
        else:
            m = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
            
        m.fit(X_tr, y_tr)
        y_preds[test_idx] = m.predict(X_te)

    # 裁剪负值（产会不会为负）
    y_preds_clipped = np.clip(y_preds, a_min=0.001, a_max=None)
    
    r2 = r2_score(y, y_preds_clipped)
    rmse = np.sqrt(mean_squared_error(y, y_preds_clipped))
    mae = mean_absolute_error(y, y_preds_clipped)
    rel_errors = np.abs(y_preds_clipped - y) / y
    mape = np.mean(rel_errors) * 100.0
    accuracy_rate = np.mean(rel_errors <= error_threshold) * 100.0
    
    with col_m2:
        st.write("##### 交叉验证评估表现 (盲测指标)")
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("决定系数 (R²)", f"{r2:.3f}")
        m_c2.metric("均方根误差 (RMSE)", f"{rmse:.3f}")
        m_c3.metric("平均相对误差 (MAPE)", f"{mape:.1f}%")
        m_c4.metric(f"符合率 (≤±{int(error_threshold*100)}%)", f"{accuracy_rate:.1f}%")

    st.markdown("---")
    st.subheader("2.2 实测值 vs 预测值交会图")
    
    cross_df = pd.DataFrame({
        "井号": well_ids,
        "实测 EUR": y,
        "预测 EUR": y_preds_clipped,
        "相对误差(%)": np.round(rel_errors * 100, 2),
        "是否合格": ["合格" if e <= error_threshold else "超标" for e in rel_errors]
    })
    
    max_val = max(np.max(y), np.max(y_preds_clipped)) * 1.15
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', name='1:1 完全吻合线', line=dict(color='gray', dash='dash')))
    fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val*(1+error_threshold)], mode='lines', name=f'+{int(error_threshold*100)}% 误差线', line=dict(color='rgba(255,100,100,0.4)', dash='dot')))
    fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val*(1-error_threshold)], mode='lines', name=f'-{int(error_threshold*100)}% 误差线', line=dict(color='rgba(255,100,100,0.4)', dash='dot')))
    
    for status, color in [("合格", "green"), ("超标", "crimson")]:
        sub = cross_df[cross_df["是否合格"] == status]
        fig_scatter.add_trace(go.Scatter(
            x=sub["实测 EUR"],
            y=sub["预测 EUR"],
            mode='markers+text',
            text=sub["井号"],
            textposition="top center",
            name=f"预测点 ({status})",
            marker=dict(size=10, color=color, opacity=0.85)
        ))
        
    fig_scatter.update_layout(
        xaxis_title="实际 EUR",
        yaxis_title="预测 EUR",
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------- TAB 3: 符合率与误差深度检验 -----------------
with tab3:
    st.subheader("3.1 单井详细检验诊断表")
    
    def highlight_status(val):
        color = '#d4edda' if val == '合格' else '#f8d7da'
        return f'background-color: {color}'

    st.dataframe(
        cross_df.style.map(highlight_status, subset=['是否合格']).format({
            "实测 EUR": "{:.4f}",
            "预测 EUR": "{:.4f}",
            "相对误差(%)": "{:.2f}%"
        }),
        use_container_width=True
    )
    
    col_err1, col_err2 = st.columns(2)
    with col_err1:
        st.subheader("3.2 井位相对误差柱状图")
        fig_bar = px.bar(
            cross_df,
            x="井号",
            y="相对误差(%)",
            color="是否合格",
            color_discrete_map={"合格": "#28a745", "超标": "#dc3545"},
            text_auto=".1f"
        )
        fig_bar.add_hline(y=error_threshold*100, line_dash="dash", line_color="red", annotation_text=f"合格阈值 ({int(error_threshold*100)}%)")
        fig_bar.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_err2:
        st.subheader("3.3 符合率随误差容限阈值变化曲线")
        thresholds_scan = np.linspace(0.05, 0.50, 46)
        rates_scan = [np.mean(rel_errors <= t) * 100.0 for t in thresholds_scan]
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=thresholds_scan*100, y=rates_scan, mode='lines+markers', line=dict(color='#17a2b8', width=2)))
        fig_curve.add_vline(x=error_threshold*100, line_dash="dot", line_color="red", annotation_text=f"当前容限: {int(error_threshold*100)}%")
        fig_curve.update_layout(
            xaxis_title="允许相对误差容限 (%)",
            yaxis_title="模型总符合率 (%)",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_curve, use_container_width=True)

# ----------------- TAB 4: 单井参数快速试算 -----------------
with tab4:
    st.subheader("4.1 现场新井 EUR 快速估算")
    st.write("在下方调节或输入地质测井解释参数，系统将利用全量样本拟合的模型进行实时预测：")
    
    if use_scaling:
        final_scaler = StandardScaler()
        X_all_scaled = final_scaler.fit_transform(X)
    else:
        final_scaler = None
        X_all_scaled = X
        
    if model_type == "弹性网络回归 (ElasticNetCV - 推荐)":
        final_model = ElasticNetCV(cv=min(3, len(X)), random_state=42)
    elif model_type == "岭回归 (RidgeCV)":
        final_model = RidgeCV()
    elif model_type == "普通多元线性回归 (OLS)":
        final_model = LinearRegression()
    else:
        final_model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
        
    final_model.fit(X_all_scaled, y)
    
    col_inputs = st.columns(3)
    user_inputs = {}
    
    for idx, feature in enumerate(selected_features):
        with col_inputs[idx % 3]:
            min_v = float(df[feature].min())
            max_v = float(df[feature].max())
            mean_v = float(df[feature].mean())
            step_v = (max_v - min_v) / 100.0 if max_v != min_v else 0.01
            user_inputs[feature] = st.number_input(
                f"{feature}",
                value=round(mean_v, 2),
                step=round(step_v, 3),
                format="%.2f"
            )
            
    input_vector = np.array([[user_inputs[f] for f in selected_features]])
    if final_scaler:
        input_vector = final_scaler.transform(input_vector)
        
    eur_pred_single = max(0.0, float(final_model.predict(input_vector)[0]))
    
    st.markdown("---")
    res_c1, res_c2, res_c3 = st.columns([1, 1, 1])
    with res_c2:
        st.markdown(
            f"""
            <div style="background-color: #f0f7ff; border: 2px solid #0066cc; border-radius: 10px; padding: 20px; text-align: center;">
                <h3 style="margin: 0; color: #555;">该井 EUR 预测估算值</h3>
                <h1 style="margin: 10px 0; color: #0066cc; font-size: 2.8rem;">{eur_pred_single:.4f}</h1>
                <p style="margin: 0; color: #888; font-size: 0.9rem;">单位：10⁸ m³ (或相应储量单位)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
