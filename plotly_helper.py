import plotly.graph_objects as go
import pandas as pd
import datetime
import webview
import plotly.io as pio

def _format_x(x):
    if isinstance(x, pd.DatetimeIndex):
        return x
    elif hasattr(x, 'to_timestamp'):
        return x.to_timestamp()
    elif hasattr(x, '__len__') and len(x) > 0:
        if isinstance(x[0], (pd.Timestamp, datetime.datetime)):
            return pd.to_datetime(x)
        elif isinstance(x[0], datetime.date):
            return pd.to_datetime(x)
        elif hasattr(x[0], 'strftime'):
            return [i.strftime('%Y-%m-%d') for i in x]
    return [str(i) for i in x]

def _empty_fig(msg='No data'):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref='paper', yref='paper', showarrow=False, font=dict(size=24, color='red'))
    fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False})
    return fig

def plot_usage_volume_over_time(df, care_home_id, tickformat='%Y-%m-%d'):
    if df.empty:
        return _empty_fig()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_format_x(df.index), y=df.values,
        mode='lines+markers', name='Observation Count'
    ))
    fig.update_layout(
        title=f'Care Home {care_home_id} Usage Volume Over Time',
        xaxis_title='Time', yaxis_title='Usage Volume Over Time',
        hovermode='x unified'
    )
    fig.update_xaxes(
        tickformat=tickformat,
        tickangle=45
    )
    return fig

def plot_usage_per_bed(df, care_home_id, tickformat='%Y-%m-%d'):
    if df.empty:
        return _empty_fig()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_format_x(df.index), y=df.values,
        mode='lines+markers', name='Usage Per Bed'
    ))
    fig.update_layout(
        title=f'Care Home {care_home_id} Usage Per Bed',
        xaxis_title='Time', yaxis_title='Usage Per Bed',
        hovermode='x unified'
    )
    fig.update_xaxes(
        tickformat=tickformat,
        tickangle=45
    )
    return fig

def plot_coverage_percentage(df, care_home_id, tickformat='%Y-%m'):
    if df.empty:
        return _empty_fig()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_format_x(df.index), y=df.values,
        mode='lines+markers', name='Coverage (%)'
    ))
    fig.update_layout(
        title=f'Care Home {care_home_id} Monthly Coverage Percentage',
        xaxis_title='Month', yaxis_title='Monthly Coverage Percentage(%)',
        hovermode='x unified'
    )
    fig.update_xaxes(
        tickformat=tickformat,
        tickangle=45
    )
    return fig

def plot_news2_score_category_counts(df, care_home_id, tickformat='%Y-%m-%d'):
    """绘制NEWS2分数分布时序图"""
    if df.empty:
        return _empty_fig()
    
    fig = go.Figure()
    
    # 为每个NEWS2分数添加一条线
    for score in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[score],
            mode='lines+markers',
            name=f'NEWS2 Score {score}',
            hovertemplate='%{x|%Y-%m-%d}<br>Count: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Monthly NEWS2 Score Distribution - Care Home {care_home_id}',
        xaxis_title='Month',
        yaxis_title='Count',
        xaxis=dict(tickformat=tickformat),
        hovermode='x unified',
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def plot_high_news2_score_proportion(series, care_home_id, tickformat='%Y-%m-%d'):
    """绘制高风险分数比例时序图"""
    if series.empty:
        return _empty_fig()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode='lines+markers',
        name=series.name,
        hovertemplate='%{x|%Y-%m-%d}<br>Percentage: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Monthly {series.name} - Care Home {care_home_id}',
        xaxis_title='Month',
        yaxis_title='Percentage (%)',
        xaxis=dict(tickformat=tickformat),
        hovermode='x unified',
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def plot_staff_judgement_accuracy(series, care_home_id, tickformat='%Y-%m-%d'):
    """绘制staff clinical judgement准确率时序图"""
    if series.empty:
        return _empty_fig()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode='lines+markers',
        name='Accuracy',
        hovertemplate='%{x|%Y-%m-%d}<br>Accuracy: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Monthly Staff Clinical Judgement Accuracy - Care Home {care_home_id}',
        xaxis_title='Month',
        yaxis_title='Accuracy (%)',
        xaxis=dict(tickformat=tickformat),
        hovermode='x unified',
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def plot_all_physio_parameters(df, care_home_id, tickformat='%Y-%m-%d'):
    """绘制所有生理参数触发比例时序图"""
    if df.empty:
        return _empty_fig()
    
    fig = go.Figure()
    
    # 为每个生理参数添加一条线
    for param in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[param],
            mode='lines+markers',
            name=param.replace('_New', ''),
            hovertemplate='%{x|%Y-%m-%d}<br>Trigger Rate: %{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Monthly Physiological Parameters Trigger Rate (NEWS2≥6) - Care Home {care_home_id}',
        xaxis_title='Month',
        yaxis_title='Trigger Rate (%)',
        xaxis=dict(tickformat=tickformat),
        hovermode='x unified',
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def plot_bar(df, care_home_id, title, x_title, y_title):
    if df.empty:
        return _empty_fig()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=_format_x(df.index), y=df.values,
        name='Value'
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_title, yaxis_title=y_title,
        hovermode='x unified'
    )
    return fig

def show_plotly_in_webview(fig):
    html = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
    webview.create_window('Analysis Chart', html=html)
    webview.start()