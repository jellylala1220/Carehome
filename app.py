import streamlit as st
import pandas as pd
from data_processor import predict_next_month_news2
from data_processor import (
    get_care_home_list, get_care_home_info, 
    process_usage_data, process_health_insights,
    plot_usage_counts, plot_usage_per_bed, plot_coverage,
    plot_news2_counts, plot_high_risk_prop, plot_concern_prop,
    plot_judgement_accuracy, plot_high_score_params
)

st.set_page_config(page_title="Care Home Analysis Dashboard", layout="wide")
st.title("Care Home Analysis Dashboard")

st.sidebar.header("Step 1: Upload Data File")
main_data_file = st.sidebar.file_uploader("Upload Observation Data (Excel)", type=["xlsx"])

# Use session_state to control whether to enter main analysis
if 'go_analysis' not in st.session_state:
    st.session_state['go_analysis'] = False

if main_data_file and not st.session_state['go_analysis']:
    df = pd.read_excel(main_data_file)
    st.session_state['df'] = df
    # Count observations for each Care Home
    carehome_counts = df['Care Home ID'].value_counts()
    total_count = carehome_counts.sum()
    # Care Home Name mapping
    id_to_name = df.drop_duplicates('Care Home ID').set_index('Care Home ID')['Care Home Name'].to_dict()
    # Construct DataFrame
    table = carehome_counts.reset_index()
    table.columns = ['Care Home ID', 'Count']
    table['Care Home Name'] = table['Care Home ID'].map(id_to_name)
    # Percentage column
    table['Percentage'] = (table['Count'] / total_count) * 100
    # Adjust column order
    table = table[['Care Home ID', 'Care Home Name', 'Count', 'Percentage']]
    # Sort by Count in descending order
    table = table.sort_values('Count', ascending=False).reset_index(drop=True)
    # Valid/Invalid
    valid_carehomes = table['Care Home ID'].tolist()
    all_carehomes = set(df['Care Home ID'].unique())
    invalid_carehomes = all_carehomes - set(valid_carehomes)
    valid_count_sum = table['Count'].sum()
    # Display cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Valid Care Homes", len(valid_carehomes))
    col2.metric("Number of Invalid Care Homes", len(invalid_carehomes))
    col3.metric("Total Valid Observations", valid_count_sum)
    # Display observation count table
    st.subheader("Care Home Observation Counts (Descending)")
    st.dataframe(
        table.style.format({'Percentage': '{:.1f}%'}),
        use_container_width=True
    )
    # Enter analysis button
    if st.button("Enter Analysis"):
        st.session_state['go_analysis'] = True
        st.rerun()

# Main analysis function (only shown after clicking "Enter Analysis")
if main_data_file and st.session_state['go_analysis']:
    if 'df' in st.session_state:
        df = st.session_state['df']
    else:
        st.error("Data not loaded. Please upload file again.")
        st.stop()
    # Your original analysis main interface code
    st.sidebar.header("Step 2: Select Analysis Type")
    analysis_mode = st.sidebar.radio("Analysis Level", options=["Carehome Level Analysis", "Regional Analysis"], index=0)
    
    if analysis_mode == "Carehome Level Analysis":
        care_home_list = get_care_home_list(df)
        care_home = st.sidebar.selectbox("Select Care Home", care_home_list)
        care_home_info = get_care_home_info(df, care_home)
        beds = care_home_info.get('beds', 10)
        
        with st.expander("Care Home Basic Information", expanded=True):
            st.markdown(f"**Name:** {care_home}")
            st.markdown(f"**Number of Beds:** {beds}")
            st.markdown(f"**Number of Observations:** {care_home_info.get('obs_count', 'N/A')}")
            st.markdown(f"**Data Time Range:** {care_home_info.get('date_range', 'N/A')}")
        
        tab1, tab2, tab3 = st.tabs(["Usage Analysis", "Health Insights", "Prediction"])
        
        with tab1:
            st.header("Usage Analysis")
            period = st.selectbox("Time Granularity", ["Daily", "Weekly", "Monthly", "Yearly"], index=2)
            usage_df = process_usage_data(df, care_home, beds, period)
            st.plotly_chart(plot_usage_counts(usage_df, period), use_container_width=True)
            st.plotly_chart(plot_usage_per_bed(usage_df, period), use_container_width=True)
            if period == "Monthly":
                st.plotly_chart(plot_coverage(usage_df), use_container_width=True)
            else:
                st.info("Coverage % is only displayed in Monthly mode.")
        
        with tab2:
            st.header("Health Insights (Based on NEWS2)")
            period2 = st.selectbox("Time Granularity (Health Insights)", ["Daily", "Weekly", "Monthly", "Yearly"], index=2, key="health_period")
            hi_data = process_health_insights(df, care_home, period2)
            st.plotly_chart(plot_news2_counts(hi_data, period2), use_container_width=True)
            st.plotly_chart(plot_high_risk_prop(hi_data, period2), use_container_width=True)
            st.plotly_chart(plot_concern_prop(hi_data, period2), use_container_width=True)
            st.plotly_chart(plot_judgement_accuracy(hi_data, period2), use_container_width=True)
            st.plotly_chart(plot_high_score_params(hi_data, period2), use_container_width=True)
        
        with tab3:
            st.header("Prediction (Next Month, Bayesian Poisson, 2-month Window)")
        #  取当前care home所有观测

            care_home_df = df[df['Care Home ID'] == care_home].copy()  # ID比Name稳妥
            if len(care_home_df) <= 100:
               st.warning("This care home does not have enough observations (>100 required) to run prediction.")
               st.stop()
            else:
                      # ==== 历史每月统计（必须有） ====
        care_home_df['Date/Time'] = pd.to_datetime(care_home_df['Date/Time'])
        care_home_df['Month'] = care_home_df['Date/Time'].dt.to_period('M')
        monthly_counts = care_home_df.groupby(['Month', 'NEWS2 score']).size().unstack(fill_value=0)
        history_months = monthly_counts.index.astype(str)

        # ==== 调用预测函数 ====
        result_df, next_month = predict_next_month_news2(care_home_df)
        if result_df.empty or next_month is None:
            st.warning("Not enough months of data to make prediction.")
        else:
            st.markdown(f"**Prediction for month:** {next_month}")
            st.dataframe(result_df.style.format({'Predicted Mean': '{:.2f}', '95% Lower': '{:.2f}', '95% Upper': '{:.2f}'}))

            # ==== 选择分数交互 ====
            score_list = sorted(result_df['NEWS2 Score'].unique())
            selected_score = st.selectbox("Select NEWS2 Score to view time series", score_list)

            # ==== 历史每月y值 ====
            if selected_score in monthly_counts.columns:
                actual_monthly = monthly_counts[selected_score].values
            else:
                actual_monthly = [0] * len(history_months)

            # ==== 预测结果 ====
            pred_row = result_df[result_df['NEWS2 Score'] == selected_score].iloc[0]
            # 预测月
            predict_month = next_month
            pred_mean = pred_row['Predicted Mean']
            pred_lower = pred_row['95% Lower']
            pred_upper = pred_row['95% Upper']

            # ==== 组装x轴 ====
            all_months = list(history_months) + [predict_month]
            # 历史y，预测y
            y_hist = list(actual_monthly) + [None]
            y_pred = [None]*len(history_months) + [pred_mean]
            yerr_low = [None]*len(history_months) + [pred_mean - pred_lower]
            yerr_up = [None]*len(history_months) + [pred_upper - pred_mean]

            # ==== 画时间序列图 ====
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(history_months, actual_monthly, marker='o', color='gray', label='Actual')
            # 预测点+error bar
            ax.errorbar([predict_month], [pred_mean],
                        yerr=[[pred_mean - pred_lower], [pred_upper - pred_mean]],
                        fmt='o', color='blue', capsize=6, label='Prediction (95% CI)')
            ax.set_xticks(all_months)
            ax.set_xticklabels(all_months, rotation=45)
            ax.set_title(f"NEWS2 Score {selected_score} Time Series & Next Month Prediction")
            ax.set_xlabel("Month")
            ax.set_ylabel("Monthly Count")
            ax.legend()
            st.pyplot(fig)
            plt.close()

            # ==== 下载预测结果表 ====
            st.download_button("Download Prediction Results", result_df.to_csv(index=False), file_name="prediction_results.csv")
    else:
        st.info("Regional Analysis is not implemented yet. Please select Carehome Level Analysis.")
elif not main_data_file:
    st.warning("Please upload the main data file to begin analysis.")



