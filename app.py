import os
import re
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import streamlit as st

# =========================================================================
# 1. 网页全局美化与学术风格配置
# =========================================================================
st.set_page_config(
    page_title="Anti-HIV 多模态智能预测平台",
    page_icon="🧬",
    layout="wide",  # 宽屏学术看板布局
    initial_sidebar_state="expanded"
)

# 注入自定义 CSS 样式美化网页前端 UI
st.markdown("""
    <style>
    .main-title { font-size: 34px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 25px; }
    .status-active { background-color: #FEE2E2; color: #DC2626; padding: 12px 20px; border-radius: 8px; font-weight: bold; text-align: center; border: 1px solid #FCA5A5;}
    .status-inactive { background-color: #DBEAFE; color: #2563EB; padding: 12px 20px; border-radius: 8px; font-weight: bold; text-align: center; border: 1px solid #93C5FD;}
    .report-box { background-color: #F9FAFB; padding: 20px; border-radius: 8px; border: 1px solid #E5E7EB; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)


# =========================================================================
# 2. 核心特征解析引擎与大模型缓存机制
# =========================================================================
def parse_fasta_content(fasta_text):
    """鲁棒的多序列 FASTA 格式解析器"""
    records = fasta_text.split('>')[1:]
    seqs, headers = [], []
    for r in records:
        lines = r.strip().split('\n')
        if len(lines) > 1:
            header = lines[0].split()[0]
            # 过滤特殊字符确保标识符安全
            safe_header = "".join([c for c in header if c.isalpha() or c.isdigit() or c in '_-'])
            seq = "".join(lines[1:]).strip().upper()
            seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)
            if seq:
                seqs.append(seq)
                headers.append(safe_header)
    return seqs, headers


def extract_12_descriptors_single(seq):
    """计算单条序列的氨基酸理化特征"""
    L = len(seq)
    if L == 0: return np.zeros(12, dtype=np.float32)
    counts = Counter(seq)
    f_A, f_C, f_G, f_I, f_L, f_V, f_Y = counts['A'] / L, counts['C'] / L, counts['G'] / L, counts['I'] / L, counts[
        'L'] / L, counts['V'] / L, counts['Y'] / L
    pos_charge = (counts['R'] + counts['K'] + counts['H']) / L
    neg_charge = (counts['D'] + counts['E']) / L
    aromatic = (counts['F'] + counts['W'] + counts['Y']) / L
    hydrophobic = (counts['A'] + counts['I'] + counts['L'] + counts['F'] + counts['V'] + counts['W'] + counts['M']) / L
    hydrophilic = (counts['R'] + counts['K'] + counts['D'] + counts['E'] + counts['N'] + counts['Q']) / L
    return np.array([f_A, f_C, f_G, f_I, f_L, f_V, f_Y, pos_charge, neg_charge, aromatic, hydrophobic, hydrophilic],
                    dtype=np.float32)


@st.cache_resource
def load_esm2_model(model_name="esm2_t33_650M_UR50D"):
    """安全缓存加载 ESM-2 蛋白质语言模型"""
    from transformers import AutoTokenizer, EsmModel
    import torch
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
    model = EsmModel.from_pretrained(f"facebook/{model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return tokenizer, model, device


def extract_esm2_embeddings_single(seq, tokenizer, model, device):
    """提取高维蛋白质表征向量"""
    import torch
    with torch.no_grad():
        inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        outputs = model(**inputs)
        mean_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze(0)
    return mean_embedding.cpu().numpy()


# =========================================================================
# 3. 侧边栏：集成模型架构、学术指标与核心调参展示
# =========================================================================
st.sidebar.markdown("## 📊 模型核心参数面板")

# 1. 静态指标看板 (你训练出来的 5折交叉验证最优指标)
st.sidebar.markdown("### 🏆 交叉验证对比指标 (5-Fold)")
metrics_data = {
    "Algorithm": ["XGBoost", "MLP", "RandomForest", "ExtraTrees", "SVM", "KNN", "Ensemble_Adaptive"],
    "ACC": [0.7696, 0.7726, 0.7621, 0.7452, 0.7487, 0.5893, 0.7836],
    "Recall": [0.5189, 0.4974, 0.5656, 0.5859, 0.6145, 0.8316, 0.5977],
    "Precision": [0.5489, 0.5595, 0.5331, 0.5022, 0.5030, 0.3652, 0.5822],
    "MCC": [0.3808, 0.3783, 0.3879, 0.3682, 0.3838, 0.2974, 0.4410]
}
st.sidebar.dataframe(pd.DataFrame(metrics_data), hide_index=True)

MODEL_PATH = "best_multimodal_pipeline.pkl"

if not os.path.exists(MODEL_PATH):
    st.sidebar.error("⚠️ 未在根目录检测到 `best_multimodal_pipeline.pkl` 模型文件！")
    pipeline = None
else:
    @st.cache_resource
    def load_pipeline():
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)


    pipeline = load_pipeline()

    # 2. 动态自适应权重矩阵展示
    st.sidebar.markdown("### ⚙️ 各基分类器集成权重分配")
    weights_list = [{"分类器算法": name, "权重贡献度": f"{w:.3f}"} for name, w in pipeline['weights'].items()]
    st.sidebar.table(pd.DataFrame(weights_list))

    # 3. 黄金决策阈值展示
    st.sidebar.markdown("### 🎯 动态决策核心截断点")
    st.sidebar.metric(label="最优寻优阈值 (Optimized Th)", value=f"{pipeline['optimized_threshold']:.4f}")

# =========================================================================
# 4. 主界面：流式交互决策系统
# =========================================================================
st.markdown('<div class="main-title">🧬 Anti-HIV 活性多模态智能预测平台</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">融合 12 维物理化学描述符与大规模自监督蛋白质语言模型特征的多分类集成预测系统</div>',
            unsafe_allow_html=True)

if pipeline is not None:
    # 建立多功能分流标签页
    tab1, tab2 = st.tabs(["🔤 单条氨基酸序列预测", "📂 FASTA 结构文件批量上传"])

    # --- TAB 1: 手动录入界面 ---
    with tab1:
        st.markdown("### 📥 输入单条待测序列")
        raw_seq_input = st.text_area("请粘贴纯氨基酸序列（系统将自动清洗非标残基）：", value="ACDEFGHIKLMNPQRSTVWY",
                                     height=100)

        if st.button("🚀 开始精准评估", type="primary"):
            clean_seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', raw_seq_input.strip().upper())
            if not clean_seq:
                st.warning("⚠️ 请输入有效的蛋白质或多肽氨基酸序列！")
            else:
                with st.spinner("机器学习流水线计算中... 请稍候..."):
                    try:
                        # 特征级联与标准化变换
                        desc_feat = extract_12_descriptors_single(clean_seq).reshape(1, -1)
                        desc_scaled = pipeline['scalers']['desc'].transform(desc_feat)

                        tokenizer, model, device = load_esm2_model()
                        plm_feat = extract_esm2_embeddings_single(clean_seq, tokenizer, model, device).reshape(1, -1)
                        plm_scaled = pipeline['scalers']['plm'].transform(plm_feat)
                        plm_selected = pipeline['selector'].transform(plm_scaled)

                        # 云端解耦兜底处理（用 0 矩阵模拟输入 PSSM 缩放器，防止云端找不到 BLAST 环境报错）
                        pssm_feat = np.zeros((1, 120), dtype=np.float32)
                        pssm_scaled = pipeline['scalers']['pssm'].transform(pssm_feat)

                        X_fused = np.hstack((desc_scaled, plm_selected, pssm_scaled))

                        # 自适应加权软投票概率融合
                        ensemble_proba = 0.0
                        for name, clf in pipeline['models'].items():
                            proba = clf.predict_proba(X_fused)[0, 1]
                            weight = pipeline['weights'].get(name, 1.0 / len(pipeline['models']))
                            ensemble_proba += proba * weight

                        th = pipeline['optimized_threshold']
                        is_active = ensemble_proba >= th

                        # 展示精美的前端评估卡片
                        st.markdown('<div class="report-box">', unsafe_allow_html=True)
                        st.subheader("📊 评估报告详情")
                        c1, c2 = st.columns(2)
                        with c1:
                            if is_active:
                                st.markdown('<div class="status-active">🔥 具备抗HIV活性 (Positive)</div>',
                                            unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="status-inactive">🌙 不具备明显活性 (Negative)</div>',
                                            unsafe_allow_html=True)
                        with c2:
                            st.metric(label="自适应集成概率分数 (Ensemble Score)", value=f"{ensemble_proba:.4f}",
                                      delta=f"临界线: {th:.4f}")

                        with st.expander("🔍 拆解各子分类器决策权重得分"):
                            for name, clf in pipeline['models'].items():
                                sub_p = clf.predict_proba(X_fused)[0, 1]
                                st.text(
                                    f"分类器: {name:15} | 给出概率值: {sub_p:.4f} | 分配权重: {pipeline['weights'].get(name, 0.0):.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"💥 运行错误: {str(e)}")

    # --- TAB 2: FASTA 批量上传预测与可视化看板 ---
    with tab2:
        st.markdown("### 📂 批量流式评估系统 (.fasta / .fa)")
        uploaded_file = st.file_uploader("点击选择或直接拖拽你的多序列 FASTA 文件至此处", type=["fasta", "fa"])

        st.markdown("**或者直接在下方文本框批量粘贴标准的 FASTA 段落：**")
        pasted_fasta = st.text_area("输入格式如：\n>seq_1\nACDEF...\n>seq_2\nWYVTS...", value="", height=130,
                                    placeholder=">Header_1\nACDEF...")

        if st.button("🔮 启动批量智能多模态评估", type="secondary"):
            content = ""
            if uploaded_file is not None:
                content = uploaded_file.read().decode("utf-8", errors="ignore")
            elif pasted_fasta.strip():
                content = pasted_fasta

            if not content:
                st.warning("⚠️ 检测到未提供任何数据源，请提供上传文件或粘贴内容。")
            else:
                seqs, headers = parse_fasta_content(content)
                if not seqs:
                    st.error("❌ 格式解析失败！未能找到带 '>' 表头开头的标准格式残基序列。")
                else:
                    st.success(f"📋 成功捕获并切分出 {len(seqs)} 条多肽序列！正在启动大模型批量流水线...")

                    p_bar = st.progress(0)
                    status_lbl = st.empty()
                    results_data = []

                    tokenizer, model, device = load_esm2_model()
                    th = pipeline['optimized_threshold']

                    # 批量循环流式特征提取与评估
                    for i, (seq, h) in enumerate(zip(seqs, headers)):
                        status_lbl.text(f"⏳ 正在编码并演算第 {i + 1}/{len(seqs)} 条: {h}")

                        desc_feat = extract_12_descriptors_single(seq).reshape(1, -1)
                        desc_scaled = pipeline['scalers']['desc'].transform(desc_feat)

                        plm_feat = extract_esm2_embeddings_single(seq, tokenizer, model, device).reshape(1, -1)
                        plm_scaled = pipeline['scalers']['plm'].transform(plm_feat)
                        plm_selected = pipeline['selector'].transform(plm_scaled)

                        pssm_feat = np.zeros((1, 120), dtype=np.float32)
                        pssm_scaled = pipeline['scalers']['pssm'].transform(pssm_feat)

                        X_fused = np.hstack((desc_scaled, plm_selected, pssm_scaled))

                        ensemble_proba = 0.0
                        for name, clf in pipeline['models'].items():
                            proba = clf.predict_proba(X_fused)[0, 1]
                            weight = pipeline['weights'].get(name, 1.0 / len(pipeline['models']))
                            ensemble_proba += proba * weight

                        conclusion = "活性 (Positive)" if ensemble_proba >= th else "非活性 (Negative)"

                        results_data.append({
                            "序列名称 (Header)": h,
                            "氨基酸长度 (Length)": len(seq),
                            "集成预测概率 (Probability)": round(float(ensemble_proba), 4),
                            "分类决策结论": conclusion
                        })
                        p_bar.progress((i + 1) / len(seqs))

                    status_lbl.text("🎉 恭喜！当前队列所有多模态批量预测任务已全部计算结束！")

                    # 1. 展现多肽数据结果表格
                    final_df = pd.DataFrame(results_data)
                    st.markdown("### 📊 批量评估结果汇总矩阵")
                    st.dataframe(final_df, use_container_width=True)

                    # 一键导出为学术通用的 CSV 数据报表文件
                    csv_bytes = final_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 下载预测结果报告 (.csv 文件)",
                        data=csv_bytes,
                        file_name="anti_hiv_predictions_matrix.csv",
                        mime="text/csv"
                    )

                    # 2. 动态图形化结果可视化看板
                    st.markdown("---")
                    st.markdown("### 📊 预测结果多维度可视化大屏")

                    col_chart1, col_chart2 = st.columns(2)

                    with col_chart1:
                        st.markdown("##### 📈 活性与非活性分类数量柱状图")
                        class_counts = final_df["分类决策结论"].value_counts().reset_index()
                        class_counts.columns = ["分类状态", "分子频数"]
                        st.bar_chart(
                            data=class_counts,
                            x="分类状态",
                            y="分子频数",
                            color="分类状态",
                            use_container_width=True
                        )

                    with col_chart2:
                        st.markdown("##### 🎯 氨基酸长度对活性的宏观交叉分布 (交互散点图)")
                        st.scatter_chart(
                            data=final_df,
                            x="氨基酸长度 (Length)",
                            y="集成预测概率 (Probability)",
                            color="分类决策结论",
                            size="氨基酸长度 (Length)",
                            use_container_width=True
                        )

                    # 整体概率区间分布直方走势
                    st.markdown("##### 🌊 集成决策概率总体密度分布区间走势 (Density)")
                    hist_values, _ = np.histogram(final_df["集成预测概率 (Probability)"], bins=10, range=(0, 1))
                    hist_df = pd.DataFrame({
                        "概率分箱区间": [f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)],
                        "分子样本数": hist_values
                    })
                    st.area_chart(data=hist_df, x="概率分箱区间", y="分子样本数", use_container_width=True)