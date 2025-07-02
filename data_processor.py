import pandas as pd
import numpy as np
import pymc as pm
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# 你关心的生理参数字段
PHYSIO_COLS = [
    "O2_New", "Systolic_New", "Pulse_New", "Temperate_New",
    "Respiraties_New", "O2 Delivery_New", "Consciouness New"
]

class DataProcessor:
    def __init__(self):
        self.data = None
        self.physio_params = None
        
    def load_physio_params(self, physio_params_path):
        """加载生理参数表"""
        self.physio_params = pd.read_excel(physio_params_path)
        
    def load_data(self, data_path):
        """加载数据文件"""
        self.data = pd.read_excel(data_path)
        if 'Date/Time' in self.data.columns:
            self.data['Date/Time'] = pd.to_datetime(self.data['Date/Time'])
        else:
            raise ValueError("Data file does not contain 'Date/Time' column")
        if 'Care Home ID' not in self.data.columns:
            raise ValueError("Data file does not contain 'Care Home ID' column")
        
    def get_care_homes(self):
        if self.data is None:
            return []
        self.data.columns = [col.strip() for col in self.data.columns]
        if 'Care Home ID' not in self.data.columns:
            print('Data file does not contain Care Home ID column, actual columns:', self.data.columns)
            return []
        return sorted(self.data['Care Home ID'].dropna().unique().tolist())
    
    def get_care_home_info(self, care_home_id):
        """返回指定Care Home ID的基本信息"""
        if self.data is None:
            return {}
        # 强制类型一致
        care_home_id = str(care_home_id).strip()
        self.data['Care Home ID'] = self.data['Care Home ID'].astype(str).str.strip()
        rows = self.data[self.data['Care Home ID'] == care_home_id]
        if rows.empty:
            return {}
        row = rows.iloc[0]
        info = {
            'Care Home Name': row.get('Care Home Name', ''),
            'No of Beds': row.get('No of Beds', ''),
            'Area': row.get('Area', ''),
            'Type': row.get('Type', ''),
            'Amount of Asset': row.get('Amount of Asset', ''),
            'Provider company': row.get('Provider company', '')
        }
        return info
    
    def filter_by_care_home(self, care_home):
        if self.data is None:
            return None
        care_home = str(care_home).strip()
        self.data['Care Home ID'] = self.data['Care Home ID'].astype(str).str.strip()
        return self.data[self.data['Care Home ID'] == care_home]
    
    def calculate_usage(self, data, time_granularity):
        if data is None or data.empty:
            return pd.Series(dtype=int)
        dt_col = pd.to_datetime(data['Date/Time'])
        if time_granularity == 'Daily':
            grouped = data.groupby(dt_col.dt.date).size()
            grouped.index = pd.to_datetime(grouped.index)
        elif time_granularity == 'Weekly':
            grouped = data.groupby(dt_col.dt.to_period('W')).size()
            grouped.index = grouped.index.to_timestamp()
        elif time_granularity == 'Monthly':
            grouped = data.groupby(dt_col.dt.to_period('M')).size()
            grouped.index = grouped.index.to_timestamp()
        else:
            grouped = data.groupby(dt_col.dt.to_period('Y')).size()
            grouped.index = grouped.index.to_timestamp()
        return grouped.sort_index()
    
    def calculate_usage_per_bed(self, data, time_granularity, beds_count):
        usage = self.calculate_usage(data, time_granularity)
        return (usage / beds_count).sort_index()
    
    def calculate_coverage_percentage(self, data):
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
            'coverage': coverage,
            'days_with_obs': days_list,
            'total_days': total_days_list
        }, index=pd.to_datetime(months)).sort_index()
    
    def calculate_news2_counts(self, data):
        """按月统计每个NEWS2分数的数量"""
        if data is None or data.empty or 'NEWS2 score' not in data.columns:
            return pd.DataFrame()
        
        # 将日期转换为datetime并按月分组
        dt_col = pd.to_datetime(data['Date/Time'])
        monthly = data.groupby(dt_col.dt.to_period('M'))
        
        # 统计每个NEWS2分数的数量
        result = {}
        for month, group in monthly:
            month_result = group['NEWS2 score'].value_counts().sort_index()
            result[month.to_timestamp()] = month_result
        
        return pd.DataFrame(result).T.sort_index()
    
    def calculate_high_risk_proportion(self, data):
        """按月统计高风险分数比例（NEWS2≥6）"""
        if data is None or data.empty or 'NEWS2 score' not in data.columns:
            return pd.Series()
        
        # 将日期转换为datetime并按月分组
        dt_col = pd.to_datetime(data['Date/Time'])
        monthly = data.groupby(dt_col.dt.to_period('M'))
        
        # 计算每月高风险比例
        result = {}
        for month, group in monthly:
            high_risk = (group['NEWS2 score'] >= 6).mean()
            result[month.to_timestamp()] = high_risk
        
        return pd.Series(result).sort_index()
    
    def calculate_concern_proportion(self, data):
        """按月统计Clinical concern?=Yes的比例"""
        if data is None or data.empty or 'Clinical concern?' not in data.columns:
            return pd.Series()
        
        # 将日期转换为datetime并按月分组
        dt_col = pd.to_datetime(data['Date/Time'])
        monthly = data.groupby(dt_col.dt.to_period('M'))
        
        # 计算每月Concern比例
        result = {}
        for month, group in monthly:
            concern = (group['Clinical concern?'] == 'Yes').mean()
            result[month.to_timestamp()] = concern
        
        return pd.Series(result).sort_index()
    
    def calculate_clinical_judgement_accuracy(self, data):
        """计算staff clinical judgement准确率"""
        if data is None or data.empty or 'Clinical concern?' not in data.columns or 'NEWS2 score' not in data.columns:
            return pd.Series()
        
        # 将日期转换为datetime并按月分组
        dt_col = pd.to_datetime(data['Date/Time'])
        monthly = data.groupby(dt_col.dt.to_period('M'))
        
        # 计算每月准确率
        result = {}
        for month, group in monthly:
            # 筛选出Clinical concern?=Yes的记录
            concern_data = group[group['Clinical concern?'] == 'Yes']
            if len(concern_data) > 0:
                # 计算其中NEWS2≥6的比例
                accuracy = (concern_data['NEWS2 score'] >= 6).mean()
                result[month.to_timestamp()] = accuracy
            else:
                result[month.to_timestamp()] = 0
        
        return pd.Series(result).sort_index()
    
    def analyze_all_physio_parameters(self, data):
        """分析所有生理参数在高NEWS2分数中的触发比例"""
        if data is None or data.empty or 'NEWS2 score' not in data.columns:
            return pd.DataFrame()
        
        # 筛选出NEWS2≥6的数据
        high_risk_data = data[data['NEWS2 score'] >= 6]
        if high_risk_data.empty:
            return pd.DataFrame()
        
        # 获取所有*_New参数
        new_cols = [col for col in data.columns if col.endswith('_New')]
        
        # 将日期转换为datetime并按月分组
        dt_col = pd.to_datetime(high_risk_data['Date/Time'])
        monthly = high_risk_data.groupby(dt_col.dt.to_period('M'))
        
        # 统计每月每个参数的触发比例
        result = {}
        for month, group in monthly:
            month_result = {}
            for col in new_cols:
                if col in group.columns:
                    # 只要该参数>0就算触发
                    month_result[col] = (group[col] > 0).mean()
            result[month.to_timestamp()] = month_result
        
        return pd.DataFrame(result).T.sort_index()

def plot_usage_counts(df, period):
    if df is None or df.empty or 'usage_count' not in df.columns or df['usage_count'].dropna().empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据", xref="paper", yref="paper", showarrow=False, font=dict(size=20))
        return fig
    fig = px.line(df, y='usage_count', title=f'Usage Count ({period})')
    fig.update_layout(xaxis_title='Time', yaxis_title='Count')
    return fig

def plot_usage_per_bed(df, period):
    if df is None or df.empty or 'usage_per_bed' not in df.columns or df['usage_per_bed'].dropna().empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据", xref="paper", yref="paper", showarrow=False, font=dict(size=20))
        return fig
    fig = px.line(df, y='usage_per_bed', title=f'Usage per Bed ({period})')
    fig.update_layout(xaxis_title='Time', yaxis_title='Usage/Bed')
    return fig

def plot_coverage(df):
    if df is None or df.empty or 'coverage' not in df.columns or df['coverage'].dropna().empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据", xref="paper", yref="paper", showarrow=False, font=dict(size=20))
        return fig
    fig = px.line(df, y='coverage', title='Coverage Percentage (Monthly)')
    fig.update_layout(xaxis_title='Month', yaxis_title='Coverage %')
    return fig

def plot_news2_counts(hi_data, period):
    df = hi_data['news2_counts']
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据", xref="paper", yref="paper", showarrow=False, font=dict(size=20))
        return fig
    fig = go.Figure()
    for col in df.columns:
        score = int(col)
        # 颜色分配
        if score <= 3:
            # 绿色，分数越低越浅
            color = f'rgba(0,200,0,{0.3+0.2*score})'
        elif 4 <= score <= 5:
            # 黄色，分数越低越浅
            color = f'rgba(255,215,0,{0.5+0.2*(score-4)})'
        else:
            # 红色，分数越高越深
            color = f'rgba(220,0,0,{0.5+0.1*(score-6)})'
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], mode='lines+markers',
            name=f'NEWS2={col}', line=dict(color=color, width=3)
        ))
    fig.update_layout(
        title=f'NEWS2 Score Distribution ({period})',
        xaxis_title='Time', yaxis_title='Count',
        hovermode='x unified'
    )
    return fig

def plot_high_risk_prop(hi_data, period):
    s = hi_data['high_risk_prop']
    if s is None or s.empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据", xref="paper", yref="paper", showarrow=False, font=dict(size=20))
        return fig
    fig = px.line(x=s.index, y=s.values * 100, title=f'High Risk (NEWS2≥6) Proportion ({period})')
    fig.update_traces(mode='lines+markers')
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='High Risk %',
        yaxis=dict(range=[0, 100], tickformat='.0f')
    )
    return fig

def plot_concern_prop(hi_data, period):
    s = hi_data['concern_prop']
    if s is None or s.empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据", xref="paper", yref="paper", showarrow=False, font=dict(size=20))
        return fig
    fig = px.line(x=s.index, y=s.values * 100, title=f'Clinical Concern Proportion ({period})')
    fig.update_traces(mode='lines+markers')
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Concern %',
        yaxis=dict(range=[0, 100], tickformat='.0f')
    )
    return fig

def plot_judgement_accuracy(hi_data, period):
    s = hi_data['judgement_accuracy']
    if s is None or s.empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据", xref="paper", yref="paper", showarrow=False, font=dict(size=20))
        return fig
    fig = px.line(x=s.index, y=s.values * 100, title=f'Staff Clinical Judgement Accuracy ({period})')
    fig.update_traces(mode='lines+markers')
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Accuracy %',
        yaxis=dict(range=[0, 100], tickformat='.0f')
    )
    return fig

def plot_high_score_params(hi_data, period):
    df = hi_data['param_trigger']
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据", xref="paper", yref="paper", showarrow=False, font=dict(size=20))
        return fig
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col] * 100, mode='lines+markers', name=col))
    fig.update_layout(
        title=f'High NEWS2 Score Parameter Trigger Rate ({period})',
        xaxis_title='Time',
        yaxis_title='Trigger %',
        yaxis=dict(range=[0, 100], tickformat='.0f')
    )
    return fig
import pymc as pm

def predict_next_month_news2(df_carehome, window=2, sigma=0.5):
    # df_carehome: 某个care home的全部观测df
    if df_carehome.empty or 'NEWS2 score' not in df_carehome.columns or 'Date/Time' not in df_carehome.columns:
        return pd.DataFrame(), None

    df_carehome = df_carehome.copy()
    df_carehome['Date/Time'] = pd.to_datetime(df_carehome['Date/Time'])
    df_carehome['Month'] = df_carehome['Date/Time'].dt.to_period('M')
    monthly_counts = df_carehome.groupby(['Month', 'NEWS2 score']).size().unstack(fill_value=0)
    months = monthly_counts.index.astype(str)
    if len(monthly_counts) < window + 1:
        return pd.DataFrame(), None
    target_month = months[-1]
    moving_avg_months = months[-window:]
    moving_avg = monthly_counts.loc[moving_avg_months].mean()
    # 预测"下一个月"
    next_month = (pd.Period(target_month) + 1).strftime('%Y-%m')
    score_list = monthly_counts.columns.tolist()
    results = []
    for score in score_list:
        y = monthly_counts[score].values[-window:]
        prior_mean = moving_avg[score]
        prior_logmu = np.log(prior_mean + 1e-5)
        with pm.Model() as model:
            lam_pred = pm.Lognormal("lam_pred", mu=prior_logmu, sigma=sigma)
            obs = pm.Poisson("obs", mu=lam_pred, observed=y)
            trace = pm.sample(600, tune=300, target_accept=0.95, progressbar=False)
        lam_samples = trace.posterior["lam_pred"].values.flatten()
        pred_counts = np.random.poisson(lam_samples)
        pred_mean = np.mean(pred_counts)
        pred_lower = np.percentile(pred_counts, 2.5)
        pred_upper = np.percentile(pred_counts, 97.5)
        results.append({
            'NEWS2 Score': score,
            'Predicted Mean': pred_mean,
            '95% Lower': pred_lower,
            '95% Upper': pred_upper,
            'Actual': monthly_counts[score].values[-1]
        })
    result_df = pd.DataFrame(results)
    return result_df, next_month

def get_care_home_list(df):
    return sorted(df['Care Home ID'].dropna().unique())

def get_care_home_info(df, care_home_id):
    sub = df[df['Care Home ID'] == care_home_id]
    beds = sub['No of Beds'].iloc[0] if 'No of Beds' in sub.columns else 10
    obs_count = len(sub)
    date_range = f"{pd.to_datetime(sub['Date/Time']).min().date()} ~ {pd.to_datetime(sub['Date/Time']).max().date()}"
    return {
        'beds': beds,
        'obs_count': obs_count,
        'date_range': date_range
    }

def process_usage_data(df, care_home_id, beds, period):
    sub = df[df['Care Home ID'] == care_home_id].copy()
    sub['Date/Time'] = pd.to_datetime(sub['Date/Time'])
    if period == 'Daily':
        grp = sub.groupby(sub['Date/Time'].dt.date)
    elif period == 'Weekly':
        grp = sub.groupby(sub['Date/Time'].dt.to_period('W').apply(lambda r: r.start_time))
    elif period == 'Monthly':
        grp = sub.groupby(sub['Date/Time'].dt.to_period('M').apply(lambda r: r.start_time))
    else:
        grp = sub.groupby(sub['Date/Time'].dt.to_period('Y').apply(lambda r: r.start_time))
    usage = grp.size().rename('usage_count')
    usage_per_bed = usage / beds
    df_out = pd.DataFrame({
        'usage_count': usage,
        'usage_per_bed': usage_per_bed
    })
    if period == 'Monthly':
        # 计算每月覆盖率
        sub['Month'] = sub['Date/Time'].dt.to_period('M')
        days_with_obs = sub.groupby('Month')['Date/Time'].apply(lambda x: x.dt.date.nunique())
        months = sub['Month'].unique()
        total_days = {m: m.days_in_month for m in months}
        coverage = days_with_obs / days_with_obs.index.map(lambda m: total_days[m])
        coverage.index = coverage.index.to_timestamp()
        df_out['coverage'] = coverage
    return df_out

def process_health_insights(df, care_home_id, period):
    sub = df[df['Care Home ID'] == care_home_id].copy()
    sub['Date/Time'] = pd.to_datetime(sub['Date/Time'])
    if period == 'Daily':
        sub['period'] = sub['Date/Time'].dt.date
    elif period == 'Weekly':
        sub['period'] = sub['Date/Time'].dt.to_period('W').apply(lambda r: r.start_time)
    elif period == 'Monthly':
        sub['period'] = sub['Date/Time'].dt.to_period('M').apply(lambda r: r.start_time)
    else:
        sub['period'] = sub['Date/Time'].dt.to_period('Y').apply(lambda r: r.start_time)

    # 1. NEWS2分数分布
    if 'NEWS2 score' in sub.columns:
        news2_counts = sub.groupby(['period', 'NEWS2 score']).size().unstack(fill_value=0)
    else:
        news2_counts = pd.DataFrame()

    # 2. 高风险分数比例
    if 'NEWS2 score' in sub.columns:
        sub['high_risk'] = sub['NEWS2 score'] >= 6
        high_risk_prop = sub.groupby('period')['high_risk'].mean()  # 0-1比例
    else:
        high_risk_prop = pd.Series(dtype=float)

    # 3. Clinical concern?=Yes的比例
    if 'Clinical concern?' in sub.columns:
        sub['concern_flag'] = sub['Clinical concern?'].astype(str).str.lower().isin(['yes', 'y', 'true', '1'])
        concern_prop = sub.groupby('period')['concern_flag'].mean()  # 0-1比例
    else:
        concern_prop = pd.Series(dtype=float)

    # 4. staff clinical judgement accuracy
    if 'Clinical concern?' in sub.columns and 'NEWS2 score' in sub.columns:
        concern_data = sub[sub['concern_flag']]
        if not concern_data.empty:
            concern_data['high_risk'] = concern_data['NEWS2 score'] >= 6
            acc = concern_data.groupby('period')['high_risk'].mean()  # 0-1比例
        else:
            acc = pd.Series(dtype=float)
    else:
        acc = pd.Series(dtype=float)

    # 5. 生理参数触发比例
    if 'NEWS2 score' in sub.columns:
        high_news2 = sub[sub['NEWS2 score'] >= 6]
        param_cols = [col for col in PHYSIO_COLS if col in sub.columns]
        param_trigger = {}
        for col in param_cols:
            param_trigger[col] = high_news2.groupby('period')[col].apply(lambda x: (x > 0).mean() if len(x) > 0 else 0)
        param_trigger_df = pd.DataFrame(param_trigger)
    else:
        param_trigger_df = pd.DataFrame()

    return {
        'news2_counts': news2_counts,
        'high_risk_prop': high_risk_prop,
        'concern_prop': concern_prop,
        'judgement_accuracy': acc,
        'param_trigger': param_trigger_df
    }
