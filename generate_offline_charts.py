import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
from data_processor import (
    process_usage_data, process_health_insights,
    plot_usage_counts, plot_usage_per_bed, plot_coverage,
    plot_news2_counts, plot_high_risk_prop, plot_concern_prop,
    plot_judgement_accuracy, plot_high_score_params
)

def create_output_dir():
    """创建输出目录"""
    output_dir = "analysis_charts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def predict_news2_scores(df):
    """预测NEWS2分数"""
    # 准备特征
    feature_cols = [col for col in df.columns if col.endswith('_New')]
    if not feature_cols:
        return None, None
    
    # 准备数据
    X = df[feature_cols].fillna(0)
    y = df['NEWS2 score']
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, {
        'mse': mse,
        'r2': r2,
        'feature_importance': dict(zip(feature_cols, model.feature_importances_))
    }

def plot_prediction_analysis(df, model, metrics, output_dir):
    """生成预测分析相关的图表"""
    # 1. 特征重要性图
    importance_df = pd.DataFrame({
        'Feature': list(metrics['feature_importance'].keys()),
        'Importance': list(metrics['feature_importance'].values())
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(importance_df, x='Feature', y='Importance',
                 title='Feature Importance for NEWS2 Score Prediction')
    fig.write_image(f"{output_dir}/feature_importance.png")
    
    # 2. 预测vs实际值散点图
    feature_cols = [col for col in df.columns if col.endswith('_New')]
    X = df[feature_cols].fillna(0)
    y = df['NEWS2 score']
    y_pred = model.predict(X)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y, y=y_pred,
        mode='markers',
        name='Predicted vs Actual'
    ))
    fig.add_trace(go.Scatter(
        x=[y.min(), y.max()],
        y=[y.min(), y.max()],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))
    fig.update_layout(
        title='NEWS2 Score: Predicted vs Actual',
        xaxis_title='Actual NEWS2 Score',
        yaxis_title='Predicted NEWS2 Score'
    )
    fig.write_image(f"{output_dir}/prediction_vs_actual.png")
    
    # 3. 预测误差分布图
    errors = y - y_pred
    fig = px.histogram(errors, title='Distribution of Prediction Errors')
    fig.update_layout(
        xaxis_title='Prediction Error',
        yaxis_title='Count'
    )
    fig.write_image(f"{output_dir}/error_distribution.png")

def generate_all_charts(df, output_dir):
    """生成所有分析图表"""
    # 1. 获取所有Care Home
    care_homes = df['Care Home ID'].unique()
    
    # 2. 对每个Care Home生成图表
    for care_home in care_homes:
        care_home_dir = f"{output_dir}/{care_home}"
        if not os.path.exists(care_home_dir):
            os.makedirs(care_home_dir)
            
        # 获取Care Home信息
        care_home_info = get_care_home_info(df, care_home)
        beds = care_home_info.get('beds', 10)
        
        # 生成Usage Analysis图表
        usage_df = process_usage_data(df, care_home, beds, 'Monthly')
        fig = plot_usage_counts(usage_df, 'Monthly')
        fig.write_image(f"{care_home_dir}/usage_counts.png")
        
        fig = plot_usage_per_bed(usage_df, 'Monthly')
        fig.write_image(f"{care_home_dir}/usage_per_bed.png")
        
        if 'coverage' in usage_df.columns:
            fig = plot_coverage(usage_df)
            fig.write_image(f"{care_home_dir}/coverage.png")
        
        # 生成Health Insights图表
        hi_data = process_health_insights(df, care_home, 'Monthly')
        
        fig = plot_news2_counts(hi_data, 'Monthly')
        fig.write_image(f"{care_home_dir}/news2_counts.png")
        
        fig = plot_high_risk_prop(hi_data, 'Monthly')
        fig.write_image(f"{care_home_dir}/high_risk_prop.png")
        
        fig = plot_concern_prop(hi_data, 'Monthly')
        fig.write_image(f"{care_home_dir}/concern_prop.png")
        
        fig = plot_judgement_accuracy(hi_data, 'Monthly')
        fig.write_image(f"{care_home_dir}/judgement_accuracy.png")
        
        fig = plot_high_score_params(hi_data, 'Monthly')
        fig.write_image(f"{care_home_dir}/high_score_params.png")

def main():
    # 读取数据
    df = pd.read_excel("/Users/lvlei/PycharmProjects/pythonProject3/Data/New_Area_Working_Data.xlsx")  # 请替换为实际的数据文件路径
    
    # 创建输出目录
    output_dir = create_output_dir()
    
    # 预测NEWS2分数
    model, metrics = predict_news2_scores(df)
    if model is not None:
        plot_prediction_analysis(df, model, metrics, output_dir)
    
    # 生成所有分析图表
    generate_all_charts(df, output_dir)
    
    print(f"所有图表已生成到 {output_dir} 目录")

if __name__ == "__main__":
    main() 