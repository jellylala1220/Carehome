import streamlit as st
import pandas as pd
from data_processor_simple import (
    get_care_home_list, get_care_home_info, 
    process_usage_data, process_health_insights,
    plot_usage_counts, plot_usage_per_bed, plot_coverage,
    plot_news2_counts, plot_high_risk_prop, plot_concern_prop,
    plot_judgement_accuracy, plot_high_score_params,
    predict_next_month_bayesian,
    calculate_benchmark_data
)
import plotly.graph_objects as go
from io import StringIO
import numpy as np

st.set_page_config(page_title="Care Home Analysis Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.header("Navigation")
step = st.sidebar.radio(
    "Select Step",
    ["Step 1: Upload Data", 
     "Step 2: Care Home Analysis",
     "Step 3: Batch Prediction (Offline)", 
     "Step 4: Prediction Visualization",
     "Step 5: Overall Statistics/Benchmark Grouping"]
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'go_analysis' not in st.session_state:
    st.session_state['go_analysis'] = False
if 'prediction_df' not in st.session_state:
    st.session_state['prediction_df'] = None

# Step 1: Upload Data
if step == "Step 1: Upload Data":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 1: Upload Data")
    
    main_data_file = st.file_uploader("Upload Observation Data (Excel)", type=["xlsx"])
    
    if main_data_file and not st.session_state['go_analysis']:
        df = pd.read_excel(main_data_file)
        
        # Fix data type issues - ensure Care Home Name is string
        if 'Care Home Name' in df.columns:
            df['Care Home Name'] = df['Care Home Name'].astype(str)
        
        st.session_state['df'] = df
        
        # Count observations for each Care Home
        carehome_counts = df['Care Home ID'].value_counts()
        total_count = carehome_counts.sum()
        
        # Care Home Name mapping - ensure string type
        id_to_name = df.drop_duplicates('Care Home ID').set_index('Care Home ID')['Care Home Name'].astype(str).to_dict()
        
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
    
    elif not main_data_file:
        st.warning("Please upload the main data file to begin analysis.")

# Step 2: Care Home Analysis
elif step == "Step 2: Care Home Analysis":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 2: Care Home Analysis")
    
    if st.session_state['df'] is not None and st.session_state['go_analysis']:
        df = st.session_state['df']
        
        st.sidebar.header("Step 2: Select Analysis Type")
        analysis_mode = st.sidebar.radio("Analysis Level", options=["Care Home Level Analysis", "Regional Analysis"], index=0)
        
        if analysis_mode == "Care Home Level Analysis":
            # 1. Add display column
            df['Care Home Display'] = df['Care Home ID'].astype(str) + " | " + df['Care Home Name'].astype(str)

            # 2. Build ID to display string mapping
            care_home_map = (
                df[['Care Home ID', 'Care Home Display']]
                .drop_duplicates()
                .set_index('Care Home ID')['Care Home Display']
                .to_dict()
            )

            # 3. Dropdown: show "ID | Name", return ID
            care_home_id = st.sidebar.selectbox(
                "Select Care Home",
                options=list(care_home_map.keys()),
                format_func=lambda x: care_home_map[x]
            )
            care_home = care_home_id
            care_home_info = get_care_home_info(df, care_home)
            beds = care_home_info.get('beds', 10)
            
            with st.expander("Care Home Basic Information", expanded=True):
                st.markdown(f"**Name:** {care_home}")
                st.markdown(f"**Number of Beds:** {beds}")
                st.markdown(f"**Number of Observations:** {care_home_info.get('obs_count', 'N/A')}")
                st.markdown(f"**Data Time Range:** {care_home_info.get('date_range', 'N/A')}")
            
            # Only two tabs now
            tab1, tab2 = st.tabs(["Usage Analysis", "Health Insights"])

            with tab1:
                st.header("Usage Analysis")
                period = st.selectbox("Time Granularity", ["Daily", "Weekly", "Monthly", "Yearly"], index=2)
                usage_df = process_usage_data(df, care_home, beds, period)
                st.plotly_chart(plot_usage_counts(usage_df, period), use_container_width=True, key="usage_counts")
                st.plotly_chart(plot_usage_per_bed(usage_df, period), use_container_width=True, key="usage_per_bed")
                if period == "Monthly":
                    from data_processor_simple import calculate_coverage_percentage
                    coverage_df = calculate_coverage_percentage(df[df['Care Home ID'] == care_home])
                    st.plotly_chart(plot_coverage(coverage_df), use_container_width=True, key="coverage")
                else:
                    st.info("Coverage % is only displayed in Monthly mode.")

            with tab2:
                st.header("Health Insights (Based on NEWS2)")
                period2 = st.selectbox("Time Granularity (Health Insights)", ["Daily", "Weekly", "Monthly", "Yearly"], index=2, key="health_period")
                hi_data = process_health_insights(df, care_home, period2)
                st.plotly_chart(plot_news2_counts(hi_data, period2), use_container_width=True, key="news2_counts")
                st.plotly_chart(plot_high_risk_prop(hi_data, period2), use_container_width=True, key="high_risk_prop")
                st.plotly_chart(plot_concern_prop(hi_data, period2), use_container_width=True, key="concern_prop")
                st.plotly_chart(plot_judgement_accuracy(hi_data, period2), use_container_width=True, key="judgement_accuracy")
                st.plotly_chart(plot_high_score_params(hi_data, period2), use_container_width=True, key="high_score_params")
        
        else:
            st.info("Regional Analysis is not implemented yet. Please select Care Home Level Analysis.")
    else:
        st.warning("Please complete Step 1 first by uploading data and entering analysis.")

# Step 3: Batch Prediction (Offline)
elif step == "Step 3: Batch Prediction (Offline)":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 3: Batch Prediction (Offline)")

    if st.session_state['df'] is None:
        st.warning("Please upload data in Step 1 before running predictions.")
    else:
        st.info("This step will run predictions for all care homes with sufficient data (>50 observations) and generate a downloadable CSV file.")
        
        # Batch prediction parameters
        st.subheader("Prediction Parameters")
        min_obs = st.number_input("Minimum observations required per care home", min_value=1, value=50, step=10)
        window_length = st.slider("Moving average window (months)", min_value=1, max_value=12, value=2)
        sigma = st.slider("Prior belief variance (sigma)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

        if st.button("Start Batch Prediction"):
            df = st.session_state['df']
            
            # Filter care homes with enough data
            obs_counts = df['Care Home ID'].value_counts()
            valid_care_homes = obs_counts[obs_counts > min_obs].index.tolist()
            
            if not valid_care_homes:
                st.error(f"No care homes found with more than {min_obs} observations.")
            else:
                all_predictions = []
                id_to_name = df.drop_duplicates('Care Home ID').set_index('Care Home ID')['Care Home Name'].to_dict()
                
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, care_home_id in enumerate(valid_care_homes):
                    care_home_name = id_to_name.get(care_home_id, "Unknown")
                    status_text.text(f"Processing: {care_home_name} ({i+1}/{len(valid_care_homes)})...")
                    
                    df_carehome = df[df['Care Home ID'] == care_home_id]
                    
                    # Run Bayesian prediction
                    pred_df, target_month = predict_next_month_bayesian(df_carehome, window_length, sigma)
                    
                    if not pred_df.empty:
                        pred_df['Care Home ID'] = care_home_id
                        pred_df['Care Home Name'] = care_home_name
                        pred_df['Month'] = target_month
                        all_predictions.append(pred_df)
                    
                    progress_bar.progress((i + 1) / len(valid_care_homes))

                status_text.success("Batch prediction complete!")
                
                if all_predictions:
                    final_pred_df = pd.concat(all_predictions, ignore_index=True)
                    st.session_state['prediction_df'] = final_pred_df
                    st.subheader("Prediction Results Preview")
                    st.dataframe(final_pred_df.head())
                else:
                    st.warning("Prediction could not be generated for any care home. This might be due to insufficient historical data (e.g., less than the window length).")

    # Show download button if prediction results exist in session state
    if st.session_state['prediction_df'] is not None:
        csv = st.session_state['prediction_df'].to_csv(index=False)
        st.download_button(
            label="Download Prediction Results (.csv)",
            data=csv,
            file_name=f"batch_prediction_results_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# Step 4: Prediction Visualization
elif step == "Step 4: Prediction Visualization":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 4: Prediction Visualization")

    if st.session_state['df'] is None:
        st.warning("Please upload the historical data in Step 1 to compare with predictions.")
        
    st.info("Upload the prediction results file (generated in Step 3) to visualize.")
    upload_pred_file = st.file_uploader(
        "Upload Prediction Results (.csv)", 
        type=["csv"], 
        key="step4_upload"
    )

    if upload_pred_file and st.session_state['df'] is not None:
        pred_df = pd.read_csv(upload_pred_file)
        hist_df = st.session_state['df']

        # --- Data Merging Logic ---
        # 1. Prepare historical data: count monthly observations by NEWS2 score
        hist_df['Date/Time'] = pd.to_datetime(hist_df['Date/Time'])
        hist_df['Month'] = hist_df['Date/Time'].dt.strftime('%Y-%m')

        # Standardize column name to prevent KeyError. The original file might use 'NEWS2 score'.
        if 'NEWS2 score' in hist_df.columns and 'NEWS2 Score' not in hist_df.columns:
            hist_df.rename(columns={'NEWS2 score': 'NEWS2 Score'}, inplace=True)

        actual_counts = hist_df.groupby(['Care Home ID', 'Care Home Name', 'Month', 'NEWS2 Score']).size().reset_index(name='Actual')

        # 2. Merge historical actuals with predictions
        # Ensure correct types for merging
        pred_df['Care Home ID'] = pred_df['Care Home ID'].astype(str)
        pred_df['NEWS2 Score'] = pred_df['NEWS2 Score'].astype(int)
        actual_counts['Care Home ID'] = actual_counts['Care Home ID'].astype(str)
        actual_counts['NEWS2 Score'] = actual_counts['NEWS2 Score'].astype(int)
        
        # Merge based on prediction month
        merged_df = pd.merge(
            pred_df,
            actual_counts,
            how='left',
            on=['Care Home ID', 'Care Home Name', 'Month', 'NEWS2 Score']
        )
        # --- End of Merging Logic ---

        st.subheader("Combined Prediction and Actual Data")
        st.dataframe(merged_df, use_container_width=True)

        # ---- Batch Visualization ----
        st.subheader("Time Series Visualization (for all Care Homes)")
        
        # Add display column for dropdown
        merged_df['Care Home Display'] = merged_df['Care Home ID'].astype(str) + " | " + merged_df['Care Home Name'].astype(str)
        care_home_options = sorted(merged_df['Care Home Display'].unique())
        
        for care_home_display in care_home_options:
            st.markdown(f"---")
            st.markdown(f"### {care_home_display}")
            
            care_home_id = merged_df[merged_df['Care Home Display'] == care_home_display]['Care Home ID'].iloc[0]
            
            # Get all data for this care home for context
            full_hist_ch = actual_counts[actual_counts['Care Home ID'] == care_home_id]
            pred_ch = merged_df[merged_df['Care Home ID'] == care_home_id]
            
            score_list = sorted(pred_ch['NEWS2 Score'].unique())

            for score in score_list:
                fig = go.Figure()
                
                # Plot full historical data for this score
                hist_score_df = full_hist_ch[full_hist_ch['NEWS2 Score'] == score].sort_values('Month')
                if not hist_score_df.empty:
                    fig.add_trace(go.Scatter(
                        x=hist_score_df['Month'],
                        y=hist_score_df['Actual'],
                        mode='lines+markers',
                        name='Historical Actual',
                        line=dict(color='gray'),
                        marker=dict(symbol='circle')
                    ))

                # Plot prediction point with error bars
                pred_point = pred_ch[pred_ch['NEWS2 Score'] == score]
                if not pred_point.empty:
                    fig.add_trace(go.Scatter(
                        x=pred_point['Month'],
                        y=pred_point['Predicted Mean'],
                        mode='markers',
                        name='Prediction',
                        marker=dict(color='blue', size=12, symbol='diamond'),
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=pred_point['95% Upper'] - pred_point['Predicted Mean'],
                            arrayminus=pred_point['Predicted Mean'] - pred_point['95% Lower'],
                            visible=True
                        )
                    ))
                    # Plot the actual for the predicted month, if available
                    if pred_point['Actual'].notna().any():
                         fig.add_trace(go.Scatter(
                            x=pred_point['Month'],
                            y=pred_point['Actual'],
                            mode='markers',
                            name='Actual (at prediction)',
                            marker=dict(color='red', size=12, symbol='star')
                        ))


                fig.update_layout(
                    title=f'NEWS2 Score = {score}',
                    xaxis_title='Month',
                    yaxis_title='Monthly Count',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
    elif not upload_pred_file:
        st.info("Awaiting upload of prediction file.")

# Step 5: Overall Statistics/Benchmark Grouping
elif step == "Step 5: Overall Statistics/Benchmark Grouping":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 5: Overall Statistics & Benchmark Grouping")

    if st.session_state['df'] is None:
        st.warning("Please upload data in Step 1 to begin this analysis.")
    else:
        # 计算基准数据
        benchmark_df = calculate_benchmark_data(st.session_state['df'])

        if benchmark_df.empty:
            st.info("Not enough data to generate benchmark statistics.")
        else:
            # 1. 箱线图 (Boxplot)
            st.subheader("Monthly Distribution of Usage per Bed")
            st.markdown("This boxplot shows the distribution of 'average usage per bed' across all care homes for each month.")
            
            # 确保月份按时间顺序排列
            sorted_months = sorted(benchmark_df['Month'].unique())
            
            fig_box = px.box(
                benchmark_df,
                x='Month',
                y='Usage per Bed',
                points='all',
                category_orders={'Month': sorted_months},
                labels={'Usage per Bed': 'Average Usage per Bed', 'Month': 'Month'},
                title='Distribution of Monthly Usage per Bed Across All Care Homes'
            )
            st.plotly_chart(fig_box, use_container_width=True)

            # 2. Benchmark Grouping 热力图 (Heatmap)
            st.subheader("Benchmark Grouping Heatmap")
            st.markdown("This heatmap classifies each care home's monthly usage into three tiers based on the quartiles of that month's distribution.")
            st.markdown("- **<span style='color:green;'>High</span>**: Usage ≥ 75th percentile (Q3)\n"
                        "- **<span style='color:goldenrod;'>Medium</span>**: Usage between 25th (Q1) and 75th (Q3) percentile\n"
                        "- **<span style='color:red;'>Low</span>**: Usage ≤ 25th percentile (Q1)",
                        unsafe_allow_html=True)

            # 准备热力图数据
            heatmap_pivot = benchmark_df.pivot_table(
                index='Care Home Name',
                columns='Month',
                values='Group Value' # 使用我们创建的数值列
            )
            # 按月份排序
            heatmap_pivot = heatmap_pivot[sorted_months]
            
            # 自定义颜色方案
            colorscale = [
                [0, 'red'],       # Low
                [0.5, 'yellow'],  # Medium
                [1, 'green']      # High
            ]

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title='Benchmark Group',
                    tickvals=[0, 1, 2],
                    ticktext=['Low', 'Medium', 'High']
                )
            ))
            fig_heatmap.update_layout(
                title='Care Home Monthly Usage Benchmark',
                xaxis_title='Month',
                yaxis_title='Care Home',
                yaxis_autorange='reversed' # 让列表从上到下显示
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # 3. 明细表
            st.subheader("Detailed Benchmark Data")
            st.markdown("The complete data used for the charts above. You can download it as a CSV file.")
            
            # 为了显示，去掉一些计算用的列
            display_cols = ['Care Home Name', 'Month', 'Usage per Bed', 'Group']
            st.dataframe(benchmark_df[display_cols], use_container_width=True)

            # 下载按钮
            csv = benchmark_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Detailed Data (.csv)",
                data=csv,
                file_name="care_home_benchmark_data.csv",
                mime="text/csv",
            )



