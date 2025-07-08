import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Step 3: Batch Prediction", layout="wide")
st.title("Step 3: Batch Prediction (Offline)")

# 创建两个页面块
tab1, tab2 = st.tabs(["Step 3A: Offline Prediction", "Step 3B: Result Visualization"])

with tab1:
    st.header("Step 3A: Offline Prediction Task")
    
    # 上传主表格（原始观测数据）
    upload_file = st.file_uploader("Upload Observation Data for Batch Prediction (xlsx)", type=["xlsx"], key="predict_upload")
    predict_output_path = "prediction_results.csv"
    
    if upload_file:
        df = pd.read_excel(upload_file)
        st.success("Data uploaded. Ready for offline prediction.")
        
        # 显示数据概览
        st.subheader("Data Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Observations", len(df))
        col2.metric("Care Homes", df['Care Home ID'].nunique())
        col3.metric("Date Range", f"{df['Date/Time'].min().date()} to {df['Date/Time'].max().date()}")
        
        # 选择是否运行离线预测
        if st.button("Run Offline Prediction (observation > 50 only)", type="primary"):
            st.info("Offline batch prediction running, this may take a few minutes ...")
            
            # 进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 批量预测核心逻辑
                all_carehomes = df['Care Home Name'].value_counts()
                result_list = []
                total_carehomes = len(all_carehomes)
                processed = 0
                
                for care_home, cnt in all_carehomes.items():
                    if cnt <= 50: 
                        processed += 1
                        progress_bar.progress(processed / total_carehomes)
                        status_text.text(f"Processing {care_home} (skipped - insufficient data)")
                        continue
                        
                    df_ch = df[df['Care Home Name'] == care_home].copy()
                    df_ch['Date/Time'] = pd.to_datetime(df_ch['Date/Time'])
                    df_ch['Month'] = df_ch['Date/Time'].dt.to_period('M')
                    monthly_counts = df_ch.groupby(['Month', 'NEWS2 score']).size().unstack(fill_value=0)
                    months = monthly_counts.index.astype(str)
                    
                    if len(monthly_counts) < 3: 
                        processed += 1
                        progress_bar.progress(processed / total_carehomes)
                        status_text.text(f"Processing {care_home} (skipped - insufficient months)")
                        continue
                        
                    window = 2  # 固定2个月
                    target_month = months[-1]
                    moving_avg_months = months[-window:]
                    moving_avg = monthly_counts.loc[moving_avg_months].mean()
                    score_list = monthly_counts.columns.tolist()
                    
                    for score in score_list:
                        y = monthly_counts[score].values[-window:]
                        prior_mean = moving_avg[score]
                        prior_logmu = np.log(prior_mean + 1e-5)
                        
                        # 简化的贝叶斯预测（不使用 PyMC）
                        # 使用对数正态分布的近似
                        pred_mean = np.exp(prior_logmu)
                        pred_std = pred_mean * 0.5  # 假设标准差为均值的50%
                        
                        # 生成预测区间
                        pred_lower = max(0, pred_mean - 1.96 * pred_std)
                        pred_upper = pred_mean + 1.96 * pred_std
                        
                        result_list.append({
                            'Care Home Name': care_home,
                            'NEWS2 Score': score,
                            'Month': target_month,
                            'Predicted Mean': pred_mean,
                            '95% Lower': pred_lower,
                            '95% Upper': pred_upper,
                            'Actual': monthly_counts[score].values[-1] if len(monthly_counts[score].values) > 0 else 0
                        })
                    
                    processed += 1
                    progress_bar.progress(processed / total_carehomes)
                    status_text.text(f"Processed {care_home}")
                
                if result_list:
                    result_df = pd.DataFrame(result_list)
                    result_df.to_csv(predict_output_path, index=False)
                    
                    st.success(f"Offline prediction completed! {len(result_list)} predictions generated.")
                    st.download_button(
                        "Download Prediction Results", 
                        result_df.to_csv(index=False), 
                        file_name="prediction_results.csv",
                        mime="text/csv"
                    )
                    
                    # 显示结果概览
                    st.subheader("Prediction Summary")
                    st.write(f"Total predictions: {len(result_df)}")
                    st.write(f"Care homes processed: {result_df['Care Home Name'].nunique()}")
                    st.write(f"NEWS2 scores covered: {result_df['NEWS2 Score'].nunique()}")
                    
                else:
                    st.warning("No predictions generated. Check data requirements.")
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Please check your data format and try again.")

with tab2:
    st.header("Step 3B: Prediction Result Visualization")
    
    upload_pred = st.file_uploader("Upload Batch Prediction Result CSV", type=["csv"], key="result_upload")
    
    if upload_pred:
        try:
            pred_df = pd.read_csv(upload_pred)
            st.success("Prediction result loaded! Select care home and NEWS2 score to visualize.")
            
            # 显示结果概览
            st.subheader("Result Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Predictions", len(pred_df))
            col2.metric("Care Homes", pred_df['Care Home Name'].nunique())
            col3.metric("NEWS2 Scores", pred_df['NEWS2 Score'].nunique())
            
            carehome_list = sorted(pred_df['Care Home Name'].unique())
            care_home = st.selectbox("Select Care Home", carehome_list)
            score_list = sorted(pred_df['NEWS2 Score'].unique())
            selected_score = st.selectbox("Select NEWS2 Score", score_list)
            
            show_df = pred_df[(pred_df['Care Home Name'] == care_home) & (pred_df['NEWS2 Score'] == selected_score)]
            
            if not show_df.empty:
                st.subheader(f"Results for {care_home} - NEWS2 Score {selected_score}")
                st.dataframe(show_df)
                
                # 可视化
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # 创建时间序列图
                months = show_df['Month'].astype(str).tolist()
                y_actual = show_df['Actual'].values
                y_pred = show_df['Predicted Mean'].values
                y_low = show_df['95% Lower'].values
                y_up = show_df['95% Upper'].values
                
                # 绘制实际值和预测值
                ax.plot(range(len(months)), y_actual, marker='o', color='red', linewidth=2, markersize=8, label='Actual', alpha=0.7)
                ax.errorbar(range(len(months)), y_pred, yerr=[y_pred - y_low, y_up - y_pred], 
                           fmt='o', color='blue', capsize=6, label='Prediction (95% CI)', alpha=0.7)
                
                ax.set_xticks(range(len(months)))
                ax.set_xticklabels(months, rotation=45)
                ax.set_title(f"{care_home} - NEWS2 Score {selected_score} Prediction")
                ax.set_xlabel("Month")
                ax.set_ylabel("Monthly Count")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
                
                # 预测准确性统计
                st.subheader("Prediction Accuracy")
                if len(y_actual) > 0 and len(y_pred) > 0:
                    mae = np.mean(np.abs(y_actual - y_pred))
                    mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-8))) * 100
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Mean Absolute Error", f"{mae:.2f}")
                    col2.metric("Mean Absolute Percentage Error", f"{mape:.1f}%")
                
            else:
                st.warning("No data found for selected care home and NEWS2 score.")
                
        except Exception as e:
            st.error(f"Error loading prediction results: {str(e)}")
            st.info("Please check the CSV file format.") 