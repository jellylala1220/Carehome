import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pymc as pm
import arviz as az

# 你关心的生理参数字段
PHYSIO_COLS = [
    "O2_New", "Systolic_New", "Pulse_New", "Temperate_New",
    "Respiraties_New", "O2 Delivery_New", "Consciouness New"
]

def get_care_home_list(df):
    """获取所有 Care Home 列表"""
    return sorted(df['Care Home ID'].unique().tolist())

def get_care_home_info(df, care_home_id):
    """返回指定Care Home ID的基本信息"""
    care_home_id = str(care_home_id).strip()
    df['Care Home ID'] = df['Care Home ID'].astype(str).str.strip()
    rows = df[df['Care Home ID'] == care_home_id]
    
    if rows.empty:
        return {}
    
    row = rows.iloc[0]
    info = {
        'name': row.get('Care Home Name', ''),
        'beds': row.get('No of Beds', 10),
        'obs_count': len(rows),
        'date_range': f"{rows['Date/Time'].min().date()} to {rows['Date/Time'].max().date()}"
    }
    return info

def process_usage_data(df, care_home_id, beds, period):
    """处理使用数据"""
    care_home_id = str(care_home_id).strip()
    df['Care Home ID'] = df['Care Home ID'].astype(str).str.strip()
    care_home_data = df[df['Care Home ID'] == care_home_id]
    
    if care_home_data.empty:
        return pd.DataFrame()
    
    dt_col = pd.to_datetime(care_home_data['Date/Time'])
    
    if period == 'Daily':
        grouped = care_home_data.groupby(dt_col.dt.date).size()
        grouped.index = pd.to_datetime(grouped.index)
    elif period == 'Weekly':
        grouped = care_home_data.groupby(dt_col.dt.to_period('W')).size()
        grouped.index = grouped.index.to_timestamp()
    elif period == 'Monthly':
        grouped = care_home_data.groupby(dt_col.dt.to_period('M')).size()
        grouped.index = grouped.index.to_timestamp()
    else:  # Yearly
        grouped = care_home_data.groupby(dt_col.dt.to_period('Y')).size()
        grouped.index = grouped.index.to_timestamp()
    
    usage_df = grouped.reset_index()
    usage_df.columns = ['Date', 'Count']
    usage_df['Usage_per_bed'] = usage_df['Count'] / beds
    
    return usage_df.sort_values('Date')

def calculate_coverage_percentage(data):
    """计算覆盖率百分比"""
    if data is None or data.empty:
        return pd.DataFrame()
    
    dt_col = pd.to_datetime(data['Date/Time'])
    monthly = data.groupby(dt_col.dt.to_period('M'))
    coverage = []
    days_list = []
    total_days_list = []
    months = []
    
    for month, group in monthly:
        days_with_obs = group['Date/Time'].dt.date.nunique()
        total_days = pd.Period(month).days_in_month
        percent = days_with_obs / total_days  # 0-1比例
        coverage.append(percent)
        days_list.append(days_with_obs)
        total_days_list.append(total_days)
        months.append(month.to_timestamp())
    
    return pd.DataFrame({
        'Date': pd.to_datetime(months),
        'coverage': coverage,
        'days_with_obs': days_list,
        'total_days': total_days_list
    }).sort_values('Date')

def process_health_insights(df, care_home_id, period):
    """处理健康洞察数据"""
    care_home_id = str(care_home_id).strip()
    df['Care Home ID'] = df['Care Home ID'].astype(str).str.strip()
    care_home_data = df[df['Care Home ID'] == care_home_id]
    
    if care_home_data.empty or 'NEWS2 score' not in care_home_data.columns:
        return {}
    
    dt_col = pd.to_datetime(care_home_data['Date/Time'])
    
    if period == 'Daily':
        grouped = care_home_data.groupby(dt_col.dt.date)
    elif period == 'Weekly':
        grouped = care_home_data.groupby(dt_col.dt.to_period('W'))
    elif period == 'Monthly':
        grouped = care_home_data.groupby(dt_col.dt.to_period('M'))
    else:  # Yearly
        grouped = care_home_data.groupby(dt_col.dt.to_period('Y'))
    
    # 计算各种指标
    news2_counts = {}
    high_risk_prop = {}
    concern_prop = {}
    judgement_accuracy = {}
    
    for period_name, group in grouped:
        if period == 'Daily':
            period_key = pd.to_datetime(period_name)
        else:
            period_key = period_name.to_timestamp()
        
        # NEWS2 分数计数
        if 'NEWS2 score' in group.columns:
            news2_counts[period_key] = group['NEWS2 score'].value_counts().sort_index()
        
        # 高风险比例 (NEWS2 >= 6)
        if 'NEWS2 score' in group.columns:
            high_risk_prop[period_key] = (group['NEWS2 score'] >= 6).mean()
        
        # Clinical concern 比例
        if 'Clinical concern?' in group.columns:
            concern_prop[period_key] = (group['Clinical concern?'] == 'Yes').mean()
        
        # Clinical judgement 准确率
        if 'Clinical concern?' in group.columns and 'NEWS2 score' in group.columns:
            concern_data = group[group['Clinical concern?'] == 'Yes']
            if len(concern_data) > 0:
                judgement_accuracy[period_key] = (concern_data['NEWS2 score'] >= 6).mean()
            else:
                judgement_accuracy[period_key] = 0
    
    return {
        'news2_counts': pd.DataFrame(news2_counts).T.sort_index(),
        'high_risk_prop': pd.Series(high_risk_prop).sort_index(),
        'concern_prop': pd.Series(concern_prop).sort_index(),
        'judgement_accuracy': pd.Series(judgement_accuracy).sort_index()
    }

def predict_next_month_bayesian(df_carehome, window_length=2, sigma=0.5):
    """
    使用PyMC进行贝叶斯时间序列预测.
    :param df_carehome: 单个 care home 的 DataFrame.
    :param window_length: 用于计算先验的移动平均窗口.
    :param sigma: 先验的离散程度.
    :return: 包含预测结果的 DataFrame 和目标月份字符串.
    """
    if df_carehome.empty or 'NEWS2 score' not in df_carehome.columns:
        return pd.DataFrame(), None

    df_ch = df_carehome.copy()
    df_ch['Date/Time'] = pd.to_datetime(df_ch['Date/Time'])
    df_ch = df_ch.dropna(subset=['Date/Time'])
    df_ch['Month'] = df_ch['Date/Time'].dt.to_period('M')
    
    monthly_counts = df_ch.groupby(['Month', 'NEWS2 score']).size().unstack(fill_value=0)
    
    if len(monthly_counts) < window_length:
        # 如果数据不足，无法进行预测
        return pd.DataFrame(), None

    months = monthly_counts.index.to_timestamp()
    target_month_period = months[-1].to_period('M') + 1
    target_month_str = target_month_period.strftime('%Y-%m')
    
    # 使用最后 `window_length` 个月的数据作为训练集
    train_months = months[-(window_length):]
    train_counts = monthly_counts[monthly_counts.index.to_timestamp().isin(train_months)]

    results = []
    score_list = sorted(list(range(0, 11))) # NEWS2 scores 0-10

    for score in score_list:
        if score not in train_counts.columns:
            # 如果历史数据中没有这个分数，我们仍然可以基于0计数进行预测
            y_train = np.zeros(window_length)
        else:
            y_train = train_counts[score].values
        
        # 将移动平均值作为先验
        prior_mean = np.mean(y_train)
        prior_logmu = np.log(prior_mean + 1e-5) # 避免log(0)

        with pm.Model() as model:
            # 定义先验
            lam_pred = pm.Lognormal("lam_pred", mu=prior_logmu, sigma=sigma)
            # 定义似然
            pm.Poisson("obs", mu=lam_pred, observed=y_train)
            # 运行MCMC采样
            with open('/dev/null', 'w') as f: # Mute progress bar
                trace = pm.sample(2000, tune=1000, target_accept=0.95, progressbar=False, cores=1)

        # 从后验分布中获取预测
        posterior_lam = trace.posterior["lam_pred"].values
        
        # 使用泊松分布从后验lambda生成预测计数
        pred_counts = np.random.poisson(posterior_lam)

        results.append({
            'NEWS2 Score': score,
            'Predicted Mean': np.mean(pred_counts),
            '95% Lower': np.percentile(pred_counts, 2.5),
            '95% Upper': np.percentile(pred_counts, 97.5)
        })
        
    result_df = pd.DataFrame(results)
    return result_df, target_month_str

# 绘图函数
def plot_usage_counts(df, period):
    """绘制使用次数图"""
    if df.empty:
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['Count'],
        mode='lines+markers',
        name='Usage Count'
    ))
    fig.update_layout(
        title=f'Usage Count ({period})',
        xaxis_title='Date',
        yaxis_title='Count'
    )
    return fig

def plot_usage_per_bed(df, period):
    """绘制每床位使用率图"""
    if df.empty:
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['Usage_per_bed'],
        mode='lines+markers',
        name='Usage per Bed'
    ))
    fig.update_layout(
        title=f'Usage per Bed ({period})',
        xaxis_title='Date',
        yaxis_title='Usage per Bed'
    )
    return fig

def plot_coverage(df):
    """绘制覆盖率图"""
    if df.empty or 'coverage' not in df.columns:
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['coverage'] * 100,
        mode='lines+markers',
        name='Coverage %'
    ))
    fig.update_layout(
        title='Monthly Coverage %',
        xaxis_title='Date',
        yaxis_title='Coverage %'
    )
    return fig

def plot_news2_counts(hi_data, period):
    """绘制 NEWS2 计数图"""
    if 'news2_counts' not in hi_data or hi_data['news2_counts'].empty:
        return go.Figure()
    
    df = hi_data['news2_counts']
    fig = go.Figure()
    
    for score in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[score],
            mode='lines+markers',
            name=f'NEWS2 Score {score}'
        ))
    
    fig.update_layout(
        title=f'NEWS2 Score Counts ({period})',
        xaxis_title='Date',
        yaxis_title='Count'
    )
    return fig

def plot_high_risk_prop(hi_data, period):
    """绘制高风险比例图"""
    if 'high_risk_prop' not in hi_data or hi_data['high_risk_prop'].empty:
        return go.Figure()
    
    df = hi_data['high_risk_prop']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.values * 100,
        mode='lines+markers',
        name='High Risk %'
    ))
    fig.update_layout(
        title=f'High Risk Proportion ({period})',
        xaxis_title='Date',
        yaxis_title='Percentage'
    )
    return fig

def plot_concern_prop(hi_data, period):
    """绘制关注比例图"""
    if 'concern_prop' not in hi_data or hi_data['concern_prop'].empty:
        return go.Figure()
    
    df = hi_data['concern_prop']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.values * 100,
        mode='lines+markers',
        name='Clinical Concern %'
    ))
    fig.update_layout(
        title=f'Clinical Concern Proportion ({period})',
        xaxis_title='Date',
        yaxis_title='Percentage'
    )
    return fig

def plot_judgement_accuracy(hi_data, period):
    """绘制判断准确率图"""
    if 'judgement_accuracy' not in hi_data or hi_data['judgement_accuracy'].empty:
        return go.Figure()
    
    df = hi_data['judgement_accuracy']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.values * 100,
        mode='lines+markers',
        name='Judgement Accuracy %'
    ))
    fig.update_layout(
        title=f'Clinical Judgement Accuracy ({period})',
        xaxis_title='Date',
        yaxis_title='Percentage'
    )
    return fig

def plot_high_score_params(hi_data, period):
    """绘制高分参数图"""
    # 简化版本，返回空图
    fig = go.Figure()
    fig.update_layout(
        title=f'High Score Parameters ({period})',
        xaxis_title='Parameter',
        yaxis_title='Value'
    )
    return fig
