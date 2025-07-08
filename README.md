# Care Home Analysis Dashboard

这是一个用于护理院数据分析的 Streamlit 应用套件，包含两个主要应用：

## 应用概览

### 1. 主分析应用 (`app.py`)
- **Step 1**: 数据上传和概览
- **Step 2**: 护理院级别分析，包括使用分析、健康洞察和预测

### 2. 批量预测应用 (`step3_batch_prediction.py`)
- **Step 3A**: 离线批量预测任务
- **Step 3B**: 预测结果可视化

## 安装依赖

```bash
pip install streamlit pandas plotly matplotlib scipy numpy
```

## 运行应用

### 主分析应用
```bash
streamlit run app.py
```
访问: http://localhost:8501

### 批量预测应用
```bash
streamlit run step3_batch_prediction.py --server.port 8502
```
访问: http://localhost:8502

## 功能说明

### 主分析应用功能

1. **数据上传**: 支持 Excel 文件上传
2. **数据概览**: 显示护理院统计信息
3. **使用分析**: 按时间粒度分析使用情况
4. **健康洞察**: 基于 NEWS2 分数的健康指标分析
5. **预测功能**: 使用简化模型预测下个月的数据

### 批量预测应用功能

1. **离线预测**: 批量处理多个护理院的预测任务
2. **进度显示**: 实时显示处理进度
3. **结果下载**: 生成 CSV 格式的预测结果
4. **结果可视化**: 上传预测结果进行可视化分析

## 数据要求

输入数据应包含以下列：
- `Care Home ID`: 护理院 ID
- `Care Home Name`: 护理院名称
- `Date/Time`: 观测时间
- `NEWS2 score`: NEWS2 评分
- `Clinical concern?`: 临床关注标记
- `No of Beds`: 床位数

## 技术说明

- 使用 Streamlit 构建用户界面
- 使用 Plotly 进行数据可视化
- 使用 NumPy 和 SciPy 进行统计分析
- 预测模型使用简化的统计方法，避免复杂的贝叶斯推断

## 故障排除

### 常见问题

1. **模块导入错误**: 确保所有依赖包已正确安装
2. **数据格式错误**: 检查输入数据的列名和格式
3. **内存不足**: 对于大数据集，可能需要增加系统内存

### 版本兼容性

- Python 3.9+
- Streamlit 1.45+
- Pandas 2.1+
- Plotly 6.1+

## 文件结构

```
├── app.py                          # 主分析应用
├── step3_batch_prediction.py       # 批量预测应用
├── data_processor_simple.py        # 简化的数据处理模块
├── data_processor.py               # 原始数据处理模块（包含 PyMC）
└── README.md                       # 说明文档
```

## 使用流程

1. 启动主分析应用
2. 上传数据文件
3. 查看数据概览
4. 选择护理院进行分析
5. 查看各种分析图表
6. 如需批量预测，启动批量预测应用
7. 上传数据并运行批量预测
8. 下载预测结果
9. 使用结果可视化功能查看预测效果

def predict_next_month_news2(df_carehome, window=2, sigma=0.5, score_list=range(0, 10)):
    """
    Bayesian prediction for next month's NEWS2 score distribution using PyMC.
    Args:
        df_carehome: DataFrame for a single care home, must have 'Date/Time', 'NEWS2 score'
        window: number of recent months to use for training
        sigma: prior uncertainty for LogNormal
        score_list: list of NEWS2 scores to predict
    Returns:
        result_df: DataFrame with columns ['NEWS2 Score', 'Predicted Mean', '95% Lower', '95% Upper', 'Actual']
        target_month: str, the month being predicted
    """
    import pymc as pm

    if df_carehome.empty or 'NEWS2 score' not in df_carehome.columns:
        return pd.DataFrame(), None

    df_carehome['Date/Time'] = pd.to_datetime(df_carehome['Date/Time'])
    df_carehome = df_carehome.dropna(subset=['Date/Time'])
    df_carehome['Month'] = df_carehome['Date/Time'].dt.to_period('M')
    monthly_counts = df_carehome.groupby(['Month', 'NEWS2 score']).size().unstack(fill_value=0)
    months = monthly_counts.index.astype(str)
    if len(monthly_counts) < window + 2:
        return pd.DataFrame(), None  # Not enough data

    target_month = (pd.Period(months[-1]) + 1).strftime('%Y-%m')
    preds, lowers, uppers = [], [], []

    for score in score_list:
        if score not in monthly_counts.columns:
            preds.append(np.nan)
            lowers.append(np.nan)
            uppers.append(np.nan)
            continue
        y_train = monthly_counts.iloc[-window:][score].values
        prior_mean = y_train.mean()
        prior_logmu = np.log(prior_mean + 1e-5)
        with pm.Model() as model:
            lam_pred = pm.Lognormal("lam_pred", mu=prior_logmu, sigma=sigma)
            obs = pm.Poisson("obs", mu=lam_pred, observed=y_train)
            trace = pm.sample(2000, tune=1000, target_accept=0.95, progressbar=False)
        lam_samples = trace.posterior["lam_pred"].values.flatten()
        pred_counts = np.random.poisson(lam_samples)
        preds.append(np.mean(pred_counts))
        lowers.append(np.percentile(pred_counts, 2.5))
        uppers.append(np.percentile(pred_counts, 97.5))

    # Build result DataFrame
    result_df = pd.DataFrame({
        'NEWS2 Score': list(score_list),
        'Predicted Mean': preds,
        '95% Lower': lowers,
        '95% Upper': uppers,
        'Actual': [monthly_counts.iloc[-1][score] if score in monthly_counts.columns else np.nan for score in score_list]
    })
    return result_df, target_month


