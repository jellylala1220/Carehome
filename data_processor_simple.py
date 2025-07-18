import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pymc as pm
import arviz as az
import pgeocode

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
    care_home_data = df[df['Care Home ID'] == care_home_id].copy()

    if care_home_data.empty or 'NEWS2 score' not in care_home_data.columns:
        return {}

    # 修正: 使用 PHYSIO_COLS 常量来确保所有相关参数都被包括
    physio_cols = [col for col in PHYSIO_COLS if col in care_home_data.columns]
    for col in physio_cols:
        care_home_data[col] = pd.to_numeric(care_home_data[col], errors='coerce')

    dt_col = pd.to_datetime(care_home_data['Date/Time'])
    
    if period == 'Daily':
        grouper = dt_col.dt.date
    elif period == 'Weekly':
        grouper = dt_col.dt.to_period('W')
    elif period == 'Monthly':
        grouper = dt_col.dt.to_period('M')
    else:  # Yearly
        grouper = dt_col.dt.to_period('Y')
    
    grouped = care_home_data.groupby(grouper)
    
    # 初始化结果存储
    news2_counts = {}
    high_risk_prop = {}
    concern_prop = {}
    judgement_accuracy = {}
    param_trigger_data = {}

    # --- 计算参数触发率 ---
    high_risk_data = care_home_data[care_home_data['NEWS2 score'] >= 6]
    if not high_risk_data.empty:
        high_risk_grouped = high_risk_data.groupby(grouper)
        for period_name, group in high_risk_grouped:
            # 修正: 正确处理不同时间粒度的索引
            if period == 'Daily':
                period_key = pd.to_datetime(period_name)
            else:
                period_key = period_name.to_timestamp()
            
            period_result = {}
            for col in physio_cols:
                if col in group.columns:
                    period_result[col] = (group[col] > 0).mean() # 计算比例
            param_trigger_data[period_key] = period_result

    param_trigger_df = pd.DataFrame(param_trigger_data).T.sort_index()
    if not param_trigger_df.empty:
        param_trigger_df = param_trigger_df.dropna(axis=1, how='all')

    # --- 原有计算 ---
    for period_name, group in grouped:
        # 修正: 正确处理不同时间粒度的索引
        if period == 'Daily':
            period_key = pd.to_datetime(period_name)
        else:
            period_key = period_name.to_timestamp()
        
        if 'NEWS2 score' in group.columns:
            news2_counts[period_key] = group['NEWS2 score'].value_counts().sort_index()
        
        if 'NEWS2 score' in group.columns:
            high_risk_prop[period_key] = (group['NEWS2 score'] >= 6).mean()
        
        if 'Clinical concern?' in group.columns:
            concern_prop[period_key] = (group['Clinical concern?'] == 'Yes').mean()
        
        if 'Clinical concern?' in group.columns and 'NEWS2 score' in group.columns:
            concern_data = group[group['Clinical concern?'] == 'Yes']
            if not concern_data.empty:
                judgement_accuracy[period_key] = (concern_data['NEWS2 score'] >= 6).mean()
            else:
                judgement_accuracy[period_key] = 0
    
    return {
        'news2_counts': pd.DataFrame(news2_counts).T.sort_index(),
        'high_risk_prop': pd.Series(high_risk_prop).sort_index(),
        'concern_prop': pd.Series(concern_prop).sort_index(),
        'judgement_accuracy': pd.Series(judgement_accuracy).sort_index(),
        'param_trigger': param_trigger_df
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

def get_news2_color(score):
    """
    为一个NEWS2分数返回一个包含背景色和适配文本色的字典。
    """
    try:
        score = int(score)
    except (ValueError, TypeError):
        # 为无效分数提供默认灰色
        return {'background': 'rgba(128,128,128,0.5)', 'text': '#FFFFFF'}

    r, g, b = 0, 0, 0
    alpha = 0.7
    if score <= 3:
        r, g, b = 0, 200, 0
        # 修正：反转alpha逻辑，让0分为深绿色，3分为浅绿色
        alpha = 0.9 - 0.2 * score
    elif 4 <= score <= 5:
        r, g, b = 255, 215, 0
        alpha = 0.5 + 0.2 * (score - 4)
    else:
        r, g, b = 220, 0, 0
        alpha = 0.5 + 0.1 * (score - 6)

    background_color = f'rgba({r},{g},{b},{alpha})'
    
    # 基于背景亮度计算，以确保文字可读性
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    text_color = '#FFFFFF' if brightness < 128 else '#000000'
    
    return {'background': background_color, 'text': text_color}

def plot_news2_counts(hi_data, period, selected_scores=None):
    df = hi_data.get('news2_counts')
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available for NEWS2 counts", xref="paper", yref="paper", showarrow=False, font=dict(size=20))
        return fig
    
    fig = go.Figure()
    scores_to_plot = selected_scores if selected_scores else df.columns

    for col in scores_to_plot:
        if col not in df.columns: continue
        
        color_details = get_news2_color(col) # 调用新函数
        color = color_details['background']
            
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], mode='lines+markers',
            name=f'NEWS2={col}', line=dict(color=color, width=3),
            marker=dict(color=color)
        ))
        
    fig.update_layout(
        title=f'NEWS2 Score Distribution ({period}) - Line Chart',
        xaxis_title='Time', yaxis_title='Count',
        hovermode='x unified'
    )
    return fig

def plot_news2_barchart(hi_data, period, selected_scores=None):
    """新增：绘制NEWS2分数分布的堆叠柱状图"""
    df = hi_data.get('news2_counts')
    if df is None or df.empty:
        return go.Figure()

    fig = go.Figure()
    scores_to_plot = selected_scores if selected_scores else df.columns

    for col in scores_to_plot:
        if col not in df.columns: continue
        
        color_details = get_news2_color(col) # 调用新函数
        color = color_details['background']
            
        fig.add_trace(go.Bar(
            x=df.index, 
            y=df[col],
            name=f'NEWS2={col}',
            marker_color=color
        ))
        
    fig.update_layout(
        barmode='stack',
        title=f'NEWS2 Score Distribution ({period}) - Stacked Bar Chart',
        xaxis_title='Time', yaxis_title='Total Count',
        hovermode='x unified'
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
    """绘制高分参数触发率的时间序列图"""
    df = hi_data.get('param_trigger')
    
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f'High NEWS2 Score Parameter Trigger Rate ({period})',
            annotations=[{
                "text": "No high-score (NEWS2 ≥ 6) events to analyze for the selected period.",
                "xref": "paper", "yref": "paper",
                "showarrow": False, "font": {"size": 16}
            }]
        )
        return fig

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df[col].fillna(0) * 100, # 填充NaN以连接线条，并转换为百分比
            mode='lines+markers', 
            name=col.replace('_New', '') # 清理图例标签
        ))
        
    fig.update_layout(
        title=f'High NEWS2 Score Parameter Trigger Rate ({period})',
        xaxis_title='Time',
        yaxis_title='Trigger Rate (%)',
        yaxis=dict(range=[0, 101], tickformat='.0f'),
        hovermode='x unified',
        legend_title_text='Parameter'
    )
    return fig

def calculate_benchmark_data(df):
    """
    计算所有护理院的每月每床使用量，进行基准分组，并计算地理分布统计。
    :param df: 包含所有观测数据的完整 DataFrame。
    :return: 一个包含基准分析结果的 DataFrame。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # 确保数据类型正确
    df_copy = df.copy()
    df_copy['Date/Time'] = pd.to_datetime(df_copy['Date/Time'])
    df_copy['Month'] = df_copy['Date/Time'].dt.strftime('%Y-%m')

    # 1. 获取每家护理院的床位数
    # 假设床位数在数据中对于每个护理院是固定的
    beds_info = df_copy.drop_duplicates(subset=['Care Home ID']).set_index('Care Home ID')['No of Beds']

    # 2. 计算每家护理院每月的观测总数
    monthly_counts = df_copy.groupby(['Care Home ID', 'Care Home Name', 'Month']).size().reset_index(name='Monthly Observations')

    # 3. 合并床位数信息
    benchmark_df = pd.merge(monthly_counts, beds_info, on='Care Home ID')
    
    # 过滤掉床位数为0或无效的情况，防止除零错误
    benchmark_df = benchmark_df[benchmark_df['No of Beds'] > 0]

    # 4. 计算核心指标：平均每床使用量
    benchmark_df['Usage per Bed'] = benchmark_df['Monthly Observations'] / benchmark_df['No of Beds']
    
    # 5. 计算每个月的四分位数 (Q1, Q3)
    # 我们使用 transform 将每个月的Q1, Q3值广播到该月的所有行
    quartiles = benchmark_df.groupby('Month')['Usage per Bed'].quantile([0.25, 0.75]).unstack()
    quartiles.columns = ['Q1', 'Q3']
    
    # 将分位数合并回主表
    benchmark_df = pd.merge(benchmark_df, quartiles, on='Month', how='left')

    # 6. 根据分位数进行分组
    conditions = [
        benchmark_df['Usage per Bed'] >= benchmark_df['Q3'],
        benchmark_df['Usage per Bed'] <= benchmark_df['Q1']
    ]
    choices = ['High', 'Low']
    benchmark_df['Group'] = np.select(conditions, choices, default='Medium')
    
    group_map = {'Low': 0, 'Medium': 1, 'High': 2}
    benchmark_df['Group Value'] = benchmark_df['Group'].map(group_map)

    # 7. 统计每个 care home 的高使用月次数 (ci) 和总有效月份数
    # is_high 是一个布尔序列，标记每个月是否为 'High'
    benchmark_df['is_high'] = (benchmark_df['Group'] == 'High').astype(int)
    
    # 按 care home 分组计算
    geo_stats = benchmark_df.groupby('Care Home ID').agg(
        ci=('is_high', 'sum'),
        total_months=('Month', 'count')
    ).reset_index()

    # 8. 计算高使用月占比 (pi)
    geo_stats['pi'] = geo_stats['ci'] / geo_stats['total_months']
    
    # 9. 计算排名 (Rank)
    geo_stats['Rank'] = geo_stats['pi'].rank(method='min', ascending=False).astype(int)
    
    # 10. 将地理统计数据合并回主 benchmark_df
    # 我们需要一个包含每个 care home 唯一信息的新表
    care_home_info = benchmark_df.drop_duplicates(subset='Care Home ID').copy()
    
    # 合并 ci, pi, Rank
    final_benchmark_df = pd.merge(care_home_info, geo_stats, on='Care Home ID')

    # 11. 如果存在经纬度，则一并处理
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        lat_lon = df.drop_duplicates(subset='Care Home ID')[['Care Home ID', 'Latitude', 'Longitude']]
        final_benchmark_df = pd.merge(final_benchmark_df, lat_lon, on='Care Home ID')

    return final_benchmark_df.sort_values(by='Rank').reset_index(drop=True)

def get_monthly_regional_benchmark_data(df):
    """
    为区域分析准备月度基准数据。
    此函数现在使用 'Area' 列进行分组。
    """
    if 'Date/Time' not in df.columns or 'No of Beds' not in df.columns or 'Area' not in df.columns:
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy['Date/Time'] = pd.to_datetime(df_copy['Date/Time'])
    df_copy['Month'] = df_copy['Date/Time'].dt.strftime('%Y-%m')

    # 1. 计算每个 care home 每月的每床使用量
    beds_info = df_copy.drop_duplicates(subset=['Care Home ID']).set_index('Care Home ID')['No of Beds']
    monthly_counts = df_copy.groupby(['Care Home ID', 'Care Home Name', 'Area', 'Month']).size().reset_index(name='Monthly Observations')
    monthly_df = pd.merge(monthly_counts, beds_info, on='Care Home ID')
    monthly_df = monthly_df[monthly_df['No of Beds'] > 0]
    monthly_df['Usage per Bed'] = monthly_df['Monthly Observations'] / monthly_df['No of Beds']
    
    # 2. 基于整体数据计算每月的Q1, Q3分位数
    quartiles = monthly_df.groupby('Month')['Usage per Bed'].quantile([0.25, 0.75]).unstack()
    quartiles.columns = ['Q1', 'Q3']
    
    # 3. 合并分位数并确定分组
    monthly_df = pd.merge(monthly_df, quartiles, on='Month', how='left')
    conditions = [
        monthly_df['Usage per Bed'] >= monthly_df['Q3'],
        monthly_df['Usage per Bed'] <= monthly_df['Q1']
    ]
    choices = ['High', 'Low']
    monthly_df['Group'] = np.select(conditions, choices, default='Medium')
    
    # 确保列存在
    final_cols = ['Month', 'Area', 'Care Home ID', 'Care Home Name', 'Usage per Bed', 'Group']
    for col in final_cols:
        if col not in monthly_df.columns:
            # 如果关键列丢失，返回空DF以避免下游错误
            return pd.DataFrame()
            
    return monthly_df[final_cols]

def geocode_uk_postcodes(df, postcode_column='Post Code'):
    """
    Generates 'Latitude' and 'Longitude' from a UK postcode column if they don't exist
    or are null. It operates on a copy and returns the modified DataFrame.
    """
    if postcode_column not in df.columns:
        return df

    df_copy = df.copy()

    # Determine which rows need geocoding
    if 'Latitude' in df_copy.columns and 'Longitude' in df_copy.columns:
        # Ensure lat/lon are numeric, coercing errors to NaN
        df_copy['Latitude'] = pd.to_numeric(df_copy['Latitude'], errors='coerce')
        df_copy['Longitude'] = pd.to_numeric(df_copy['Longitude'], errors='coerce')
        rows_to_geocode_mask = df_copy['Latitude'].isnull() | df_copy['Longitude'].isnull()
    else:
        rows_to_geocode_mask = pd.Series([True] * len(df_copy), index=df_copy.index)
        df_copy['Latitude'] = np.nan
        df_copy['Longitude'] = np.nan

    if not rows_to_geocode_mask.any():
        return df_copy # Nothing to do

    postcodes_to_query = df_copy.loc[rows_to_geocode_mask, postcode_column].astype(str).str.upper().str.strip()
    
    # Filter out empty or invalid postcode strings before querying
    valid_postcodes = postcodes_to_query.dropna()
    valid_postcodes = valid_postcodes[valid_postcodes.str.len() > 3] # Basic validation
    valid_postcodes = valid_postcodes[valid_postcodes != 'NAN']

    if valid_postcodes.empty:
        return df_copy

    nomi = pgeocode.Nominatim('gb')
    geo_data = nomi.query_postal_code(valid_postcodes.tolist())

    # Create Series for latitude and longitude with the correct index to align them
    latitudes = pd.Series(geo_data['latitude'].values, index=valid_postcodes.index)
    longitudes = pd.Series(geo_data['longitude'].values, index=valid_postcodes.index)
    
    # Use the generated coordinates to fill NaNs in the respective rows
    # The .loc accessor is important for safe assignment
    df_copy['Latitude'] = df_copy['Latitude'].fillna(latitudes)
    df_copy['Longitude'] = df_copy['Longitude'].fillna(longitudes)
    
    return df_copy

def calculate_correlation_data(df):

    """
    计算高NEWS数与每床使用量的相关性.
    此函数基于用户提供的逻辑实现。
    返回包含月度数据的DataFrame和包含相关系数的DataFrame.
    """
    if 'NEWS2 score' not in df.columns or 'No of Beds' not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    df_copy = df.copy()
    df_copy['Date/Time'] = pd.to_datetime(df_copy['Date/Time'])
    df_copy['Month'] = df_copy['Date/Time'].dt.to_period('M')

    # xij: 每家每月高NEWS数 (NEWS2 score >= 6)
    high_news = df_copy[df_copy['NEWS2 score'] >= 6]
    xij = high_news.groupby(['Care Home ID', 'Care Home Name', 'Month']).size().rename('High NEWS Count').reset_index()

    # yij: 每家每月usage per bed
    beds_info = df_copy.drop_duplicates('Care Home ID').set_index('Care Home ID')['No of Beds']
    # 修正：在 groupby 中加入 'Care Home Name'
    usage = df_copy.groupby(['Care Home ID', 'Care Home Name', 'Month']).size().rename('Obs Count').reset_index()
    yij = usage.merge(beds_info, on='Care Home ID', how='left')
    yij['Usage per Bed'] = yij['Obs Count'] / yij['No of Beds']
    yij.dropna(subset=['Usage per Bed'], inplace=True)

    # 合并 xij 和 yij
    # 修正：确保合并键在两个DataFrame中都存在
    full_df = pd.merge(yij, xij, on=['Care Home ID', 'Care Home Name', 'Month'], how='left').fillna({'High NEWS Count': 0})
    full_df['High NEWS Count'] = full_df['High NEWS Count'].astype(int)
    
    # 计算相关系数
    corrs = []
    for care_home_id in full_df['Care Home ID'].unique():
        sub = full_df[full_df['Care Home ID'] == care_home_id]
        
        if len(sub) >= 3: # 至少需要3个数据点来计算有意义的相关性
            care_home_name = sub['Care Home Name'].iloc[0]
            
            # 检查标准差是否为零，避免计算错误
            if np.isclose(sub['High NEWS Count'].std(), 0) or np.isclose(sub['Usage per Bed'].std(), 0):
                pr, pp, sr, sp = np.nan, np.nan, np.nan, np.nan
            else:
                pr, pp = stats.pearsonr(sub['High NEWS Count'], sub['Usage per Bed'])
                sr, sp = stats.spearmanr(sub['High NEWS Count'], sub['Usage per Bed'])

            corrs.append({
                'Care Home ID': care_home_id,
                'Care Home Name': care_home_name,
                'Pearson r': pr, 
                'Pearson p-value': pp,
                'Spearman r': sr,
                'Spearman p-value': sp,
                'Months': len(sub)
            })
            
    corr_df = pd.DataFrame(corrs)
    
    # 将月份转为字符串以便绘图
    full_df['Month'] = full_df['Month'].astype(str)

    return full_df, corr_df


