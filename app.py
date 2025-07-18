import streamlit as st
import pandas as pd
from data_processor_simple import (
    get_care_home_list, get_care_home_info, 
    process_usage_data, process_health_insights,
    plot_usage_counts, plot_usage_per_bed, plot_coverage,
    plot_news2_counts, plot_high_risk_prop, plot_concern_prop,
    plot_judgement_accuracy, plot_high_score_params,
    plot_news2_barchart,  # <-- 新增导入
    predict_next_month_bayesian,
    calculate_benchmark_data,
    geocode_uk_postcodes,
    get_monthly_regional_benchmark_data,
    calculate_correlation_data,
    get_news2_color # <-- 新增导入
)
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Care Home Analysis Dashboard", layout="wide")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'go_analysis' not in st.session_state:
    st.session_state['go_analysis'] = False
if 'prediction_df' not in st.session_state:
    st.session_state['prediction_df'] = None
if 'processed_file_name' not in st.session_state:
    st.session_state['processed_file_name'] = None

# Sidebar navigation - 改用新的 option_menu
with st.sidebar:
    step_title = option_menu(
        menu_title="Navigation",  # 菜单标题
        options=[
            "Upload Data", 
            "Care Home Analysis",
            "Batch Prediction", 
            "Prediction Visualization",
            "Benchmark Grouping",
            "Regional Analysis",
            "Correlation Analysis"
        ],
        # 可选：为每个按钮添加图标
        icons=[
            "cloud-upload", 
            "house", 
            "cpu", 
            "graph-up-arrow", 
            "bar-chart-line",
            "globe-americas",
            "link-45deg"
        ],
        menu_icon="cast",  # 菜单图标
        default_index=0,  # 默认选中的按钮
    )

# Step 1: Upload Data
if step_title == "Upload Data":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 1: Upload Data")

    main_data_file = st.file_uploader("Upload Observation Data (Excel)", type=["xlsx"])

    if main_data_file:
        # 检查是否上传了一个新文件
        if st.session_state.get('processed_file_name') != main_data_file.name:
            try:
                with st.spinner("Processing new file..."):
                    df = pd.read_excel(main_data_file)
                    
                    # Clean column names
                    df.columns = [str(col).strip() for col in df.columns]

                    # Geocoding logic
                    needs_geocoding = ('Latitude' not in df.columns or 'Longitude' not in df.columns or
                                       df['Latitude'].isnull().any() or df['Longitude'].isnull().any())
                    
                    if needs_geocoding and 'Post Code' in df.columns:
                        lat_nan_before = df['Latitude'].isnull().sum() if 'Latitude' in df.columns else len(df)
                        
                        df = geocode_uk_postcodes(df, 'Post Code')
                        
                        if 'Latitude' in df.columns:
                            lat_nan_after = df['Latitude'].isnull().sum()
                            generated_count = lat_nan_before - lat_nan_after
                            if generated_count > 0:
                                st.success(f"Successfully generated coordinates for {generated_count} entries.")
                            if lat_nan_after > 0:
                                st.warning(f"Could not find coordinates for {lat_nan_after} entries. These will be excluded from the map.")
                        else:
                             st.error("Failed to create 'Latitude'/'Longitude' columns during geocoding.")

                    if 'Care Home Name' in df.columns:
                        df['Care Home Name'] = df['Care Home Name'].astype(str)

                    # 更新 session state
                    st.session_state['df'] = df
                    st.session_state['processed_file_name'] = main_data_file.name
                    st.session_state['go_analysis'] = False # 仅为新文件重置分析状态
                st.success("File uploaded and processed successfully!")
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
                # 出错时清空状态
                st.session_state['df'] = None
                st.session_state['processed_file_name'] = None

        # 只要 session state 中有数据，就显示概览和按钮
        if st.session_state.get('df') is not None:
            df = st.session_state.df
            # Data overview
            carehome_counts = df['Care Home ID'].value_counts()
            total_count = carehome_counts.sum()
            id_to_name = df.drop_duplicates('Care Home ID').set_index('Care Home ID')['Care Home Name'].astype(str).to_dict()
            table = carehome_counts.reset_index()
            table.columns = ['Care Home ID', 'Count']
            table['Care Home Name'] = table['Care Home ID'].map(id_to_name)
            table['Percentage'] = (table['Count'] / total_count) * 100
            table = table[['Care Home ID', 'Care Home Name', 'Count', 'Percentage']]
            table = table.sort_values('Count', ascending=False).reset_index(drop=True)
            valid_carehomes = table['Care Home ID'].tolist()
            all_carehomes = set(df['Care Home ID'].unique())
            invalid_carehomes = all_carehomes - set(valid_carehomes)
            valid_count_sum = table['Count'].sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Number of Valid Care Homes", len(valid_carehomes))
            col2.metric("Number of Invalid Care Homes", len(invalid_carehomes))
            col3.metric("Total Valid Observations", valid_count_sum)

            st.subheader("Care Home Observation Counts (Descending)")
            st.dataframe(
                table.style.format({'Percentage': '{:.1f}%'}),
                use_container_width=True
            )

            if st.button("Enter Analysis"):
                st.session_state['go_analysis'] = True
                st.rerun()

    else:
        # 如果用户清空了上传文件，则重置所有相关状态
        st.warning("Please upload the main data file to begin analysis.")
        st.session_state['df'] = None
        st.session_state['processed_file_name'] = None
        st.session_state['go_analysis'] = False

# Step 2: Care Home Analysis
elif step_title == "Care Home Analysis":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 2: Care Home Analysis")

    if st.session_state['df'] is not None and st.session_state['go_analysis']:
        df = st.session_state['df']

        # 移除侧边栏的分析类型选择
        # analysis_mode = st.sidebar.radio("Analysis Level", options=["Care Home Level Analysis", "Regional Analysis"], index=0)

        # 默认进入 "Care Home Level Analysis" 逻辑
        
        # --- 将护理院选择器从侧边栏移到主页面 ---
        st.subheader("Select Care Home to Analyze")
        df['Care Home Display'] = df['Care Home ID'].astype(str) + " | " + df['Care Home Name'].astype(str)
        care_home_map = (
            df[['Care Home ID', 'Care Home Display']]
            .drop_duplicates()
            .set_index('Care Home ID')['Care Home Display']
            .to_dict()
        )
        
        # 将 selectbox 放在主页面
        care_home_id = st.selectbox(
            "Select Care Home",
            options=sorted(list(care_home_map.keys())), 
            format_func=lambda x: care_home_map.get(x, x), # 使用 .get() 增加健壮性
            label_visibility="collapsed" # 隐藏标签，因为我们已经有了subheader
        )
        
        st.markdown("---") # 添加分割线

        care_home = care_home_id
        care_home_info = get_care_home_info(df, care_home)
        beds = care_home_info.get('beds', 10)
        
        with st.expander("Care Home Basic Information", expanded=True):
            st.markdown(f"**Name:** {care_home_map.get(care_home, 'N/A')}")
            st.markdown(f"**Number of Beds:** {beds}")
            st.markdown(f"**Number of Observations:** {care_home_info.get('obs_count', 'N/A')}")
            st.markdown(f"**Data Time Range:** {care_home_info.get('date_range', 'N/A')}")
        
        tab1, tab2 = st.tabs(["Usage Analysis", "Health Insights"])
    
        with tab1:
            st.header("Usage Analysis")
            period = st.selectbox("Time Granularity", ["Daily", "Weekly", "Monthly", "Yearly"], index=2, key="usage_period")
            usage_df = process_usage_data(df, care_home, beds, period)
            st.plotly_chart(plot_usage_counts(usage_df, period), use_container_width=True, key="usage_counts")
            st.plotly_chart(plot_usage_per_bed(usage_df, period), use_container_width=True, key="usage_per_bed")
            if period == "Monthly":
                # This import is fine here as it's specific to this block
                from data_processor_simple import calculate_coverage_percentage
                coverage_df = calculate_coverage_percentage(df[df['Care Home ID'] == care_home])
                st.plotly_chart(plot_coverage(coverage_df), use_container_width=True, key="coverage")
            else:
                st.info("Coverage % is only displayed in Monthly mode.")
    
        with tab2:
            st.header("Health Insights (Based on NEWS2)")
            period2 = st.selectbox("Time Granularity (Health Insights)", ["Daily", "Weekly", "Monthly", "Yearly"], index=2, key="health_period")
            
            hi_data = process_health_insights(df, care_home, period2)
            
            if hi_data.get('news2_counts') is not None and not hi_data['news2_counts'].empty:
                all_scores = sorted(hi_data['news2_counts'].columns)
                
                # --- 新增：动态生成颜色图例 ---
                legend_items = []
                for score in all_scores:
                    colors = get_news2_color(score)
                    legend_items.append(
                        f'<span style="background-color: {colors["background"]}; color: {colors["text"]}; padding: 3px 8px; margin: 2px; border-radius: 5px; font-weight: bold; display: inline-block;">'
                        f'{score}</span>'
                    )
                st.markdown("<b>NEWS2 Score Legend & Filter:</b><br>" + " ".join(legend_items), unsafe_allow_html=True)

                selected_scores = st.multiselect(
                    "You can hide scores by removing them below:", # 更新了提示语
                    options=all_scores,
                    default=all_scores,
                    key=f"news2_filter_{care_home}"
                )

                if selected_scores:
                    st.plotly_chart(plot_news2_counts(hi_data, period2, selected_scores=selected_scores), use_container_width=True, key="news2_counts")
                    st.plotly_chart(plot_news2_barchart(hi_data, period2, selected_scores=selected_scores), use_container_width=True, key="news2_barchart")
                else:
                    st.info("Please select at least one NEWS2 score from the filter to display the charts.")
            else:
                st.info("No NEWS2 score data available to display.")

            # --- 恢复其他图表 ---
            st.plotly_chart(plot_high_risk_prop(hi_data, period2), use_container_width=True, key="high_risk_prop")
            st.plotly_chart(plot_concern_prop(hi_data, period2), use_container_width=True, key="concern_prop")
            st.plotly_chart(plot_judgement_accuracy(hi_data, period2), use_container_width=True, key="judgement_accuracy")
            st.plotly_chart(plot_high_score_params(hi_data, period2), use_container_width=True, key="high_score_params")

    else:
        st.warning("Please complete Step 1 first by uploading data and entering analysis.")

# Step 3: Batch Prediction (Offline)
elif step_title == "Batch Prediction":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 3: Batch Prediction (Offline)")

    if st.session_state['df'] is None:
        st.warning("Please upload data in Step 1 before running predictions.")
    else:
        st.info("This step will run predictions for all care homes with sufficient data (>50 observations) and generate a downloadable CSV file.")

        st.subheader("Prediction Parameters")
        min_obs = st.number_input("Minimum observations required per care home", min_value=1, value=50, step=10)
        window_length = st.slider("Moving average window (months)", min_value=1, max_value=12, value=2)
        sigma = st.slider("Prior belief variance (sigma)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

        if st.button("Start Batch Prediction"):
            df = st.session_state['df']
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

    if st.session_state['prediction_df'] is not None:
        csv = st.session_state['prediction_df'].to_csv(index=False)
        st.download_button(
            label="Download Prediction Results (.csv)",
            data=csv,
            file_name=f"batch_prediction_results_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# Step 4: Prediction Visualization
elif step_title == "Prediction Visualization":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 4: Prediction Visualization")

    if st.session_state['df'] is None:
        st.warning("Please upload the historical data in Step 1 to compare with predictions.")
    else:
        st.info("Upload the prediction results file (generated in Step 3) to visualize.")
        upload_pred_file = st.file_uploader(
            "Upload Prediction Results (.csv)",
            type=["csv"],
            key="step4_upload"
        )

        if upload_pred_file and st.session_state['df'] is not None:
            pred_df = pd.read_csv(upload_pred_file)
            hist_df = st.session_state['df']

            hist_df['Date/Time'] = pd.to_datetime(hist_df['Date/Time'])
            hist_df['Month'] = hist_df['Date/Time'].dt.strftime('%Y-%m')

            if 'NEWS2 score' in hist_df.columns and 'NEWS2 Score' not in hist_df.columns:
                hist_df.rename(columns={'NEWS2 score': 'NEWS2 Score'}, inplace=True)

            actual_counts = hist_df.groupby(['Care Home ID', 'Care Home Name', 'Month', 'NEWS2 Score']).size().reset_index(name='Actual')

            pred_df['Care Home ID'] = pred_df['Care Home ID'].astype(str)
            pred_df['NEWS2 Score'] = pred_df['NEWS2 Score'].astype(int)
            actual_counts['Care Home ID'] = actual_counts['Care Home ID'].astype(str)
            actual_counts['NEWS2 Score'] = actual_counts['NEWS2 Score'].astype(int)

            merged_df = pd.merge(
                pred_df,
                actual_counts,
                how='left',
                on=['Care Home ID', 'Care Home Name', 'Month', 'NEWS2 Score']
            )
            
            merged_df['Care Home Display'] = merged_df['Care Home ID'].astype(str) + " | " + merged_df['Care Home Name'].astype(str)

            # --- 新增：护理院选择下拉菜单 ---
            st.subheader("Filter by Care Home")
            all_care_homes_option = "All Care Homes"
            care_home_options = [all_care_homes_option] + sorted(merged_df['Care Home Display'].unique())
            
            selected_care_home = st.selectbox(
                "Select a care home to view its specific data and charts:",
                options=care_home_options
            )

            # --- 根据选择筛选数据 ---
            if selected_care_home == all_care_homes_option:
                display_df = merged_df
                care_homes_to_plot = sorted(merged_df['Care Home Display'].unique())
            else:
                display_df = merged_df[merged_df['Care Home Display'] == selected_care_home]
                care_homes_to_plot = [selected_care_home]
            
            # --- 修改：显示重新排序和筛选后的表格 ---
            st.subheader("Combined Prediction and Actual Data")
            
            # 重新排序，将关键列放在前面
            front_cols = ['Care Home ID', 'Care Home Name', 'Month']
            other_cols = [col for col in display_df.columns if col not in front_cols and col != 'Care Home Display']
            display_df_ordered = display_df[front_cols + other_cols]
            
            st.dataframe(display_df_ordered, use_container_width=True)

            # --- 修改：根据筛选结果显示图表 ---
            st.subheader("Time Series Visualization")

            for care_home_display in care_homes_to_plot:
                st.markdown(f"---")
                st.markdown(f"### {care_home_display}")
                
                # 从已筛选的DataFrame中获取数据，而不是从原始的merged_df中获取
                care_home_data_to_plot = display_df[display_df['Care Home Display'] == care_home_display]
                care_home_id = care_home_data_to_plot['Care Home ID'].iloc[0]
                
                full_hist_ch = actual_counts[actual_counts['Care Home ID'] == care_home_id]
                pred_ch = care_home_data_to_plot
                score_list = sorted(pred_ch['NEWS2 Score'].unique())

                for score in score_list:
                    fig = go.Figure()
                    
                    # --- 新增：根据分数获取颜色 ---
                    color_details = get_news2_color(score)
                    score_color = color_details['background']

                    hist_score_df = full_hist_ch[full_hist_ch['NEWS2 Score'] == score].sort_values('Month')
                    if not hist_score_df.empty:
                        fig.add_trace(go.Scatter(
                            x=hist_score_df['Month'],
                            y=hist_score_df['Actual'],
                            mode='lines+markers',
                            name='Historical Actual',
                            line=dict(color=score_color), # 修改
                            marker=dict(symbol='circle', color=score_color) # 修改
                        ))
                    pred_point = pred_ch[pred_ch['NEWS2 Score'] == score]
                    if not pred_point.empty:
                        fig.add_trace(go.Scatter(
                            x=pred_point['Month'],
                            y=pred_point['Predicted Mean'],
                            mode='markers',
                            name='Prediction',
                            marker=dict(color=score_color, size=12, symbol='diamond'), # 修改
                            error_y=dict(
                                type='data',
                                symmetric=False,
                                array=pred_point['95% Upper'] - pred_point['Predicted Mean'],
                                arrayminus=pred_point['Predicted Mean'] - pred_point['95% Lower'],
                                visible=True,
                                color=score_color # 新增
                            )
                        ))
                        if pred_point['Actual'].notna().any():
                             fig.add_trace(go.Scatter(
                                x=pred_point['Month'],
                                y=pred_point['Actual'],
                                mode='markers',
                                name='Actual (at prediction)',
                                marker=dict(color='red', size=12, symbol='star') # 保持红色以突出显示
                            ))
                    fig.update_layout(
                        title=f'NEWS2 Score = {score}',
                        xaxis_title='Month',
                        yaxis_title='Monthly Count',
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"plot_{care_home_id}_{score}")
        elif not upload_pred_file:
            st.info("Awaiting upload of prediction file.")

# Step 5: Overall Statistics/Benchmark Grouping
elif step_title == "Benchmark Grouping":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 5: Overall Statistics & Benchmark Grouping")

    if st.session_state['df'] is None:
        st.warning("Please upload data in Step 1 to begin this analysis.")
    else:
        df_full = st.session_state['df']
        
        # 为了生成箱线图和热力图，我们需要原始的月度数据
        # 我们需要一个新的函数来只计算这部分
        # (为了快速实现，我们暂时在这里复制逻辑，理想状态下应重构 data_processor)
        
        # --- 为箱线图和热力图准备月度数据 ---
        df_copy = df_full.copy()
        df_copy['Date/Time'] = pd.to_datetime(df_copy['Date/Time'])
        df_copy['Month'] = df_copy['Date/Time'].dt.strftime('%Y-%m')
        if 'No of Beds' not in df_copy.columns:
            st.error("Source data must contain 'No of Beds' column for this analysis.")
            st.stop()
        
        beds_info = df_copy.drop_duplicates(subset=['Care Home ID']).set_index('Care Home ID')['No of Beds']
        monthly_counts = df_copy.groupby(['Care Home ID', 'Care Home Name', 'Month']).size().reset_index(name='Monthly Observations')
        monthly_benchmark_df = pd.merge(monthly_counts, beds_info, on='Care Home ID')
        monthly_benchmark_df = monthly_benchmark_df[monthly_benchmark_df['No of Beds'] > 0]
        monthly_benchmark_df['Usage per Bed'] = monthly_benchmark_df['Monthly Observations'] / monthly_benchmark_df['No of Beds']
        quartiles = monthly_benchmark_df.groupby('Month')['Usage per Bed'].quantile([0.25, 0.75]).unstack()
        quartiles.columns = ['Q1', 'Q3']
        monthly_benchmark_df = pd.merge(monthly_benchmark_df, quartiles, on='Month', how='left')
        conditions = [
            monthly_benchmark_df['Usage per Bed'] >= monthly_benchmark_df['Q3'],
            monthly_benchmark_df['Usage per Bed'] <= monthly_benchmark_df['Q1']
        ]
        choices = ['High', 'Low']
        monthly_benchmark_df['Group'] = np.select(conditions, choices, default='Medium')
        group_map = {'Low': 0, 'Medium': 1, 'High': 2}
        monthly_benchmark_df['Group Value'] = monthly_benchmark_df['Group'].map(group_map)
        
        # --- 现在开始计算地理分布数据 ---
        geospatial_df = calculate_benchmark_data(df_full)

        if monthly_benchmark_df.empty or geospatial_df.empty:
            st.info("Not enough data to generate benchmark statistics.")
        else:
            # --- 新增：带有 "All Years" 选项的年份选择功能 ---
            monthly_benchmark_df['Year'] = pd.to_datetime(monthly_benchmark_df['Month']).dt.year
            available_years = sorted(monthly_benchmark_df['Year'].unique(), reverse=True)
            year_options = ["All Years"] + available_years
            
            selected_year = st.selectbox(
                "Select Year to Display",
                options=year_options,
                index=0,
                key="benchmark_year_select"
            )
            
            # 根据选择的年份筛选数据
            if selected_year == "All Years":
                data_to_plot = monthly_benchmark_df
                title_suffix = "(All Years)"
            else:
                data_to_plot = monthly_benchmark_df[monthly_benchmark_df['Year'] == selected_year]
                title_suffix = f"for {selected_year}"

            # 1. 箱线图 (Boxplot) - 使用筛选后的数据
            st.subheader(f"Monthly Distribution of Usage per Bed {title_suffix}")
            st.markdown("This boxplot shows the distribution of 'average usage per bed' across all care homes for each month.")
            
            sorted_months = sorted(data_to_plot['Month'].unique())

            fig_box = px.box(
                data_to_plot,
                x='Month',
                y='Usage per Bed',
                points='all',
                category_orders={'Month': sorted_months},
                labels={'Usage per Bed': 'Average Usage per Bed', 'Month': 'Month'},
                title=f'Distribution of Monthly Usage per Bed {title_suffix}'
            )
            fig_box.update_traces(pointpos=0)
            st.plotly_chart(fig_box, use_container_width=True)

            # 2. Benchmark Grouping 热力图 (Heatmap) - 同样使用筛选后的数据
            st.subheader(f"Benchmark Grouping Heatmap {title_suffix}")
            st.markdown("This heatmap classifies each care home's monthly usage into three tiers based on the quartiles of that month's distribution.")
            st.markdown("- **<span style='color:green;'>High</span>**: Usage ≥ 75th percentile (Q3)\n"
                        "- **<span style='color:goldenrod;'>Medium</span>**: Usage between 25th (Q1) and 75th (Q3) percentile\n"
                        "- **<span style='color:red;'>Low</span>**: Usage ≤ 25th percentile (Q1)",
                        unsafe_allow_html=True)

            heatmap_pivot = data_to_plot.pivot_table(
                index='Care Home Name',
                columns='Month',
                values='Group Value'
            )
            if not heatmap_pivot.empty:
                heatmap_pivot = heatmap_pivot[sorted_months]
                colorscale = [
                    [0, 'red'],
                    [0.5, 'yellow'],
                    [1, 'green']
                ]

                num_care_homes = len(heatmap_pivot.index)
                heatmap_height = max(400, num_care_homes * 30)

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
                    title=f'Care Home Monthly Usage Benchmark {title_suffix}',
                    xaxis_title='Month',
                    yaxis_title='Care Home',
                    yaxis_autorange='reversed',
                    height=heatmap_height
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

            # 3. 新增：地理分布图
            st.subheader("Geospatial Distribution of High Usage Frequency")
            
            if 'Latitude' not in geospatial_df.columns or 'Longitude' not in geospatial_df.columns:
                st.warning("Geospatial map cannot be generated because 'Latitude' and/or 'Longitude' columns are missing in the source data.")
            else:
                st.markdown("This map shows each care home's location, colored by its frequency of being a 'High' usage facility (pi value). Size reflects the magnitude of this frequency.")

                # --- 改进：使用固定阈值对 pi 进行分组，以获得更一致的颜色 ---
                conditions = [
                    geospatial_df['pi'] == 0,
                    geospatial_df['pi'] >= 0.5
                ]
                choices = ['Low', 'High']
                geospatial_df['pi_group'] = np.select(conditions, choices, default='Medium')
                
                color_map = {'Low': 'red', 'Medium': 'yellow', 'High': 'green'}
                
                # --- 改进：创建一个新的列用于大小，确保 pi=0 的点可见且大小差异更明显 ---
                # 添加一个很小的基数，然后放大，使得大小差异更显著
                geospatial_df['size'] = (geospatial_df['pi'] * 20) + 5
                
                # --- 新增：为重叠点添加“抖动”以改善可见性 ---
                # 定义抖动幅度 (约等于 +/- 200米)
                jitter_amount = 0.002
                # 创建带有随机抖动的新坐标列
                geospatial_df['lat_jittered'] = geospatial_df['Latitude'] + np.random.uniform(-jitter_amount, jitter_amount, size=len(geospatial_df))
                geospatial_df['lon_jittered'] = geospatial_df['Longitude'] + np.random.uniform(-jitter_amount, jitter_amount, size=len(geospatial_df))

                # 创建地图
                fig_map = px.scatter_mapbox(
                    geospatial_df.dropna(subset=['Latitude', 'Longitude']), # 确保没有NaN的经纬度
                    lat="lat_jittered", # 使用抖动后的纬度
                    lon="lon_jittered", # 使用抖动后的经度
                    color="pi_group",
                    size="size",  # 使用新的 size 列
                    color_discrete_map=color_map,
                    category_orders={"pi_group": ["Low", "Medium", "High"]},
                    mapbox_style="open-street-map",
                    zoom=5,
                    center={"lat": 54.5, "lon": -2.0}, # 大致的英国中心
                    hover_name="Care Home Name",
                    hover_data={
                        "pi": ":.2f", # 格式化 pi 值为两位小数
                        "ci": True,
                        "total_months": True,
                        "Rank": True,
                        # 隐藏不需要的悬停信息
                        "Latitude": False,
                        "Longitude": False,
                        "pi_group": False,
                        "size": False # 不在悬停信息中显示内部的size列
                    },
                    size_max=30 # 限制点的最大尺寸
                )
                fig_map.update_layout(
                    legend_title_text='High Usage Frequency',
                    margin={"r":0,"t":0,"l":0,"b":0},
                    height=700 # 增加地图高度
                )
                st.plotly_chart(fig_map, use_container_width=True)

            # 4. 明细表 - 使用地理数据
            st.subheader("Detailed High Usage Frequency Ranking")
            display_cols = ['Rank', 'Care Home Name', 'pi', 'ci', 'total_months']
            st.dataframe(geospatial_df[display_cols], use_container_width=True)

            csv = geospatial_df[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Ranking Data (.csv)",
                data=csv,
                file_name="care_home_pi_ranking.csv",
                mime="text/csv",
            )

# Step 6: Regional Analysis
elif step_title == "Regional Analysis":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 6: Regional Analysis")

    if st.session_state['df'] is None:
        st.warning("Please upload data in Step 1 to begin this analysis.")
    else:
        df = st.session_state['df']
        
        if 'Area' not in df.columns or 'No of Beds' not in df.columns:
            st.error("Source data must contain 'Area' and 'No of Beds' columns for this analysis.")
        else:
            monthly_df_full = get_monthly_regional_benchmark_data(df)

            if monthly_df_full.empty:
                st.info("Not enough data to generate regional analysis.")
            else:
                # --- 新增：为区域分析图表添加年份选择器 ---
                monthly_df_full['Year'] = pd.to_datetime(monthly_df_full['Month']).dt.year
                regional_available_years = sorted(monthly_df_full['Year'].unique(), reverse=True)
                regional_year_options = ["All Years"] + regional_available_years
                
                selected_regional_year = st.selectbox(
                    "Select Year to Display",
                    options=regional_year_options,
                    index=0,
                    key="regional_year_select"
                )

                # 根据选择筛选数据
                if selected_regional_year == "All Years":
                    monthly_df = monthly_df_full
                    regional_title_suffix = "(All Years)"
                else:
                    monthly_df = monthly_df_full[monthly_df_full['Year'] == selected_regional_year]
                    regional_title_suffix = f"for {selected_regional_year}"
                
                sorted_months = sorted(monthly_df['Month'].unique())

                st.subheader(f"A. Monthly Usage per Bed by Area {regional_title_suffix}")
                st.markdown("This boxplot shows the distribution of 'average usage per bed' across all care homes within each area, for each month.")
                fig_box = px.box(
                    monthly_df,
                    x='Month', y='Usage per Bed', color='Area',
                    category_orders={'Month': sorted_months},
                    labels={'Usage per Bed': 'Average Usage per Bed', 'Month': 'Month', 'Area': 'Area'},
                    title=f'Distribution of Monthly Usage per Bed by Area {regional_title_suffix}',
                    points='all'
                )
                fig_box.update_traces(pointpos=0)
                st.plotly_chart(fig_box, use_container_width=True)

                st.markdown("---")

                st.subheader("B. Area Benchmark Grouping Percentage")
                st.markdown("This chart shows the percentage of care homes in each benchmark group (High/Medium/Low) for each area, on a monthly basis.")

                summary = monthly_df.groupby(['Month', 'Area', 'Group'])['Care Home ID'].nunique().reset_index()
                summary.rename(columns={'Care Home ID': 'Count'}, inplace=True)
                total_per_region_month = monthly_df.groupby(['Month', 'Area'])['Care Home ID'].nunique().reset_index().rename(columns={'Care Home ID':'Total'})
                summary = summary.merge(total_per_region_month, on=['Month', 'Area'])
                summary['Percentage'] = summary['Count'] / summary['Total']

                selected_month = st.selectbox(
                    "Select Month to View Benchmark Split", 
                    options=sorted_months, 
                    index=len(sorted_months)-1
                )

                if selected_month:
                    fig_bar = px.bar(
                        summary[summary['Month'] == selected_month],
                        x='Area', y='Percentage', color='Group',
                        title=f"Benchmark Group Split by Area - {selected_month}",
                        labels={'Percentage':'Percentage of Care Homes', 'Area':'Area', 'Group':'Usage Group'},
                        barmode='stack',
                        color_discrete_map={'High': 'green', 'Medium': 'yellow', 'Low': 'red'},
                        category_orders={"Group": ["Low", "Medium", "High"]}
                    )
                    fig_bar.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                st.markdown("---")

                st.subheader("C. Detailed Grouping Data")
                st.markdown("This table provides the detailed numbers and percentages used for the benchmark grouping chart above.")
                
                display_summary = summary[['Month', 'Area', 'Group', 'Count', 'Total', 'Percentage']].copy()
                display_summary['Percentage'] = (display_summary['Percentage'] * 100).map('{:.1f}%'.format)
                st.dataframe(display_summary, use_container_width=True)
                
                csv = summary.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Detailed Grouping Data (.csv)",
                    data=csv,
                    file_name="regional_benchmark_summary.csv",
                    mime="text/csv",
                )

# Step 7: Correlation Analysis
elif step_title == "Correlation Analysis":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 7: Correlation Analysis")
    st.markdown("Analysis of the correlation between monthly high NEWS scores (≥6) and average usage per bed.")

    if st.session_state['df'] is None:
        st.warning("Please upload data in Step 1 to begin this analysis.")
    else:
        df = st.session_state['df']

        if 'NEWS2 score' not in df.columns or 'No of Beds' not in df.columns:
            st.error("Source data must contain 'NEWS2 score' and 'No of Beds' columns for this analysis.")
        else:
            with st.spinner("Calculating monthly data and correlations..."):
                monthly_corr_df, corr_summary_df = calculate_correlation_data(df)

            if corr_summary_df.empty:
                st.info("Not enough data to generate correlation analysis. Each care home needs at least 3 months of data with valid observations.")
            else:
                # Part A: Correlation Analysis
                st.subheader("A. Correlation Coefficient Summary")
                st.markdown("This table shows the Pearson and Spearman correlation coefficients between 'High NEWS Count' and 'Usage per Bed' for each care home.")
                
                st.dataframe(corr_summary_df.style.format({
                    'Pearson r': '{:.3f}', 'Pearson p-value': '{:.3f}',
                    'Spearman r': '{:.3f}', 'Spearman p-value': '{:.3f}'
                }), use_container_width=True)

                csv = corr_summary_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Correlation Summary (.csv)",
                    data=csv,
                    file_name="correlation_summary.csv",
                    mime="text/csv",
                )

                st.markdown("---")

                # Part B: Trend Visualization
                st.subheader("B. Trend Visualization")
                st.markdown("Select a care home to visualize the monthly trend of 'High NEWS Count' and 'Usage per Bed'.")

                care_home_map = (
                    monthly_corr_df[['Care Home ID', 'Care Home Name']]
                    .drop_duplicates()
                    .set_index('Care Home ID')['Care Home Name']
                    .to_dict()
                )
                
                selected_care_home_id = st.selectbox(
                    "Select Care Home for Trend Analysis",
                    options=sorted(list(care_home_map.keys())),
                    format_func=lambda x: f"{x} | {care_home_map[x]}"
                )

                if selected_care_home_id:
                    sub = monthly_corr_df[monthly_corr_df['Care Home ID'] == selected_care_home_id].sort_values('Month')
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=sub['Month'], y=sub['High NEWS Count'], name='High NEWS (≥6) Count',
                        mode='lines+markers', yaxis='y1',
                        line=dict(color='blue'), marker=dict(color='blue')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=sub['Month'], y=sub['Usage per Bed'], name='Avg Usage per Bed',
                        mode='lines+markers', yaxis='y2',
                        line=dict(color='red'), marker=dict(color='red')
                    ))
                    
                    fig.update_layout(
                        title_text=f"<b>Care Home: {care_home_map[selected_care_home_id]}</b>",
                        xaxis_title="Month",
                        yaxis=dict(title="<b>High NEWS (≥6) Count</b>", side='left', color='blue'),
                        yaxis2=dict(title="<b>Avg Usage per Bed</b>", overlaying='y', side='right', showgrid=False, color='red'),
                        legend=dict(x=0.01, y=0.99, yanchor='top', xanchor='left', borderwidth=1),
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)