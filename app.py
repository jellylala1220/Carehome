import streamlit as st
import pandas as pd
from data_processor_simple import (
    get_care_home_list, get_care_home_info, 
    process_usage_data, process_health_insights,
    plot_usage_counts, plot_usage_per_bed, plot_coverage,
    plot_news2_counts, plot_high_risk_prop, plot_concern_prop,
    plot_judgement_accuracy, plot_high_score_params,
    predict_next_month_bayesian,
    calculate_benchmark_data,
    geocode_uk_postcodes,
    calculate_coverage_percentage
)
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Care Home Analysis Dashboard", layout="wide")

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'prediction_df' not in st.session_state:
    st.session_state['prediction_df'] = None

# --- Sidebar Navigation ---
with st.sidebar:
    default_page_index = 0
    if st.session_state.get('df') is not None:
        default_page_index = 1

    step_title = option_menu(
        menu_title="Navigation",
        options=[
            "Upload Data",
            "Care Home Analysis",
            "Batch Prediction",
            "Prediction Visualization",
            "Overall Statistics"
        ],
        icons=["cloud-upload", "house", "cpu", "graph-up-arrow", "bar-chart-line"],
        menu_icon="cast",
        default_index=default_page_index,
    )

# ==============================================================================
# Step 1: Upload Data
# ==============================================================================
if step_title == "Upload Data":
st.title("Care Home Analysis Dashboard")
    st.header("Step 1: Upload Data")

    main_data_file = st.file_uploader("Upload Observation Data (Excel)", type=["xlsx"])

    if main_data_file:
        try:
            df = pd.read_excel(main_data_file)
            df.columns = [str(col).strip() for col in df.columns]
            df = df.rename(columns={'NEWS2 score': 'NEWS2 Score'}, errors='ignore')

            needs_geocoding = ('Latitude' not in df.columns or 'Longitude' not in df.columns or
                               df['Latitude'].isnull().any() or df['Longitude'].isnull().any())
            
            if needs_geocoding and 'Post Code' in df.columns:
                with st.spinner("Generating coordinates from 'Post Code'..."):
                    df = geocode_uk_postcodes(df, 'Post Code')
            
    st.session_state['df'] = df
            st.success("File uploaded and processed successfully! You can now navigate to other sections.")
            
            # Data overview
            st.subheader("Data Overview")
            df = st.session_state.get('df')
    carehome_counts = df['Care Home ID'].value_counts()
            id_to_name = df.drop_duplicates('Care Home ID').set_index('Care Home ID')['Care Home Name'].astype(str).to_dict()
    table = carehome_counts.reset_index()
    table.columns = ['Care Home ID', 'Count']
    table['Care Home Name'] = table['Care Home ID'].map(id_to_name)
            table['Percentage'] = (table['Count'] / carehome_counts.sum()) * 100
            
            # 修复: 动态计算表格高度，使其自适应
            table_height = min(500, (len(table) + 1) * 35 + 3) # 每行35像素，最多500像素
    st.dataframe(
                table[['Care Home ID', 'Care Home Name', 'Count', 'Percentage']].style.format({'Percentage': '{:.1f}%'}), 
                use_container_width=True,
                height=table_height
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.session_state['df'] = None

# ==============================================================================
# Step 2: Care Home Analysis
# ==============================================================================
elif step_title == "Care Home Analysis":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 2: Care Home Analysis")

    if st.session_state.get('df') is not None:
        df = st.session_state['df']
        st.sidebar.header("Step 2: Analysis Options")
        
        df['Care Home Display'] = df['Care Home ID'].astype(str) + " | " + df['Care Home Name'].astype(str)
        care_home_map = df[['Care Home ID', 'Care Home Display']].drop_duplicates().set_index('Care Home ID')['Care Home Display'].to_dict()
        care_home_id = st.sidebar.selectbox("Select Care Home", options=sorted(list(care_home_map.keys())), format_func=lambda x: care_home_map.get(x, x))
        
        care_home_info = get_care_home_info(df, care_home_id)
        beds = care_home_info.get('beds', 10)
        
        with st.expander("Care Home Basic Information", expanded=True):
            st.markdown(f"**Name:** {care_home_info.get('name', 'N/A')}")
            st.markdown(f"**Number of Beds:** {beds}")
            st.markdown(f"**Number of Observations:** {care_home_info.get('obs_count', 'N/A')}")
            st.markdown(f"**Data Time Range:** {care_home_info.get('date_range', 'N/A')}")
        
        tab1, tab2 = st.tabs(["Usage Analysis", "Health Insights"])
        
        with tab1:
            period = st.selectbox("Time Granularity", ["Daily", "Weekly", "Monthly", "Yearly"], index=2, key="usage_period")
            usage_df = process_usage_data(df, care_home_id, beds, period)
            st.plotly_chart(plot_usage_counts(usage_df, period), use_container_width=True, key="usage_counts_chart")
            st.plotly_chart(plot_usage_per_bed(usage_df, period), use_container_width=True, key="usage_per_bed_chart")
            if period == "Monthly":
                coverage_df = calculate_coverage_percentage(df[df['Care Home ID'] == care_home_id])
                st.plotly_chart(plot_coverage(coverage_df), use_container_width=True, key="coverage_chart")
        
        with tab2:
            period2 = st.selectbox("Time Granularity (Health)", ["Daily", "Weekly", "Monthly", "Yearly"], index=2, key="health_period")
            hi_data = process_health_insights(df, care_home_id, period2)
            st.plotly_chart(plot_news2_counts(hi_data, period2), use_container_width=True, key="news2_counts_chart")
            st.plotly_chart(plot_high_risk_prop(hi_data, period2), use_container_width=True, key="high_risk_chart")
            st.plotly_chart(plot_concern_prop(hi_data, period2), use_container_width=True, key="concern_prop_chart")
            st.plotly_chart(plot_judgement_accuracy(hi_data, period2), use_container_width=True, key="judgement_accuracy_chart")
            st.plotly_chart(plot_high_score_params(hi_data, period2), use_container_width=True, key="high_score_params_chart")
    else:
        st.warning("Please upload data in Step 1 to begin.")

# ==============================================================================
# Step 3: Batch Prediction
# ==============================================================================
elif step_title == "Batch Prediction":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 3: Batch Prediction")

    if st.session_state.get('df') is not None:
        df = st.session_state.get('df')
        st.info("This step will run predictions for all care homes with sufficient data and generate a downloadable CSV file.")

        min_obs = st.number_input("Minimum observations required", value=50, min_value=1, step=10)
        window_length = st.slider("Moving average window (months)", value=2, min_value=1, max_value=12)
        sigma = st.slider("Prior belief variance (sigma)", value=0.5, min_value=0.1, max_value=2.0, step=0.1)

        if st.button("Start Batch Prediction"):
            obs_counts = df['Care Home ID'].value_counts()
            valid_care_homes = obs_counts[obs_counts >= min_obs].index.tolist()

            if not valid_care_homes:
                st.error(f"No care homes found with at least {min_obs} observations.")
            else:
                all_predictions = []
                id_to_name = df.drop_duplicates('Care Home ID').set_index('Care Home ID')['Care Home Name'].to_dict()
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, care_home_id in enumerate(valid_care_homes):
                    care_home_name = id_to_name.get(care_home_id, "Unknown")
                    status_text.text(f"Processing: {care_home_name} ({i+1}/{len(valid_care_homes)})...")
                    df_carehome = df[df['Care Home ID'] == care_home_id]
                    pred_df_single, target_month = predict_next_month_bayesian(df_carehome, window_length, sigma)
                    
                    if pred_df_single is not None and not pred_df_single.empty:
                        pred_df_single['Care Home ID'] = care_home_id
                        pred_df_single['Care Home Name'] = care_home_name
                        pred_df_single['Month'] = target_month
                        all_predictions.append(pred_df_single)
                    
                    progress_bar.progress((i + 1) / len(valid_care_homes))

                status_text.success("Batch prediction complete!")
                if all_predictions:
                    final_pred_df = pd.concat(all_predictions, ignore_index=True)
                    st.session_state['prediction_df'] = final_pred_df
                    st.dataframe(final_pred_df.head())

        if st.session_state.get('prediction_df') is not None:
            csv = st.session_state['prediction_df'].to_csv(index=False).encode('utf-8')
            st.download_button("Download Prediction Results", csv, f"preds_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", "text/csv")
    else:
        st.warning("Please upload data in Step 1.")

# ==============================================================================
# Step 4: Prediction Visualization
# ==============================================================================
elif step_title == "Prediction Visualization":
    st.title("Prediction Visualization")
    if st.session_state.get('df') is None:
        st.warning("Please upload historical data in Step 1.")
    else:
        upload_pred_file = st.file_uploader("Upload Prediction Results (.csv)", type=["csv"], key="pred_uploader")
        if upload_pred_file:
            try:
                pred_df = pd.read_csv(upload_pred_file)
                hist_df = st.session_state['df']
                hist_df['Month'] = pd.to_datetime(hist_df['Date/Time']).dt.to_period('M').astype(str)
                hist_counts = hist_df.groupby(['Care Home ID', 'Care Home Name', 'Month', 'NEWS2 Score']).size().reset_index(name='Count')
                ch_map = pred_df[['Care Home ID', 'Care Home Name']].drop_duplicates().set_index('Care Home ID')['Care Home Name'].to_dict()

                col1, col2 = st.columns(2)
                with col1:
                    selected_ch_id = st.selectbox(
                        "Filter by Care Home",
                        options=["All"] + sorted(list(ch_map.keys())),
                        format_func=lambda x: "All (for score comparison)" if x == "All" else f"{ch_map.get(x, 'Unknown')} ({x})"
                    )
                with col2:
                    selected_score = st.selectbox(
                        "Filter by NEWS2 Score (for 'All' view)",
                        options=list(range(11)), index=1,
                        disabled=(selected_ch_id != "All")
                    )

                if selected_ch_id != "All":
                    # --- Single Care Home Time Series View ---
                    st.info(f"Showing full history and prediction for: {ch_map.get(selected_ch_id)}")
                    ch_hist = hist_counts[hist_counts['Care Home ID'] == selected_ch_id]
                    ch_pred = pred_df[pred_df['Care Home ID'] == selected_ch_id]
                    if not ch_pred.empty:
                        fig = go.Figure()
                        for score_val in sorted(ch_hist['NEWS2 Score'].unique()):
                            d = ch_hist[ch_hist['NEWS2 Score'] == score_val]
                            fig.add_trace(go.Scatter(x=d['Month'], y=d['Count'], mode='lines+markers', name=f'History Score {score_val}'))
                        for _, row in ch_pred.iterrows():
                            fig.add_trace(go.Scatter(x=[row['Month']], y=[row['Predicted Mean']], mode='markers', error_y=dict(type='data', symmetric=False, array=[row['95% Upper'] - row['Predicted Mean']], arrayminus=[row['Predicted Mean'] - row['95% Lower']]), name=f"Prediction Score {int(row['NEWS2 Score'])}", marker=dict(size=10)))
                        fig.update_layout(title=f"History vs. Prediction for {ch_map.get(selected_ch_id)}", yaxis_title="Observation Count")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    # --- All Care Homes, Single Score Comparison View ---
                    st.subheader(f"Comparing predictions for NEWS2 Score = {selected_score}")
                    # Use .copy() to avoid SettingWithCopyWarning
                    pred_filtered = pred_df[pred_df['NEWS2 Score'] == selected_score].copy()
                    
                    if not hist_counts.empty:
                        last_month = hist_counts['Month'].max()
                        hist_filtered = hist_counts[(hist_counts['NEWS2 Score'] == selected_score) & (hist_counts['Month'] == last_month)].copy()
                        
                        # 修复: 在合并前，强制将ID列都转换为字符串类型，这是核心修复
                        pred_filtered['Care Home ID'] = pred_filtered['Care Home ID'].astype(str)
                        hist_filtered['Care Home ID'] = hist_filtered['Care Home ID'].astype(str)

                        # 安全地合并
                        comp_df = pd.merge(pred_filtered, hist_filtered[['Care Home ID', 'Count']], on="Care Home ID", how='left')
                        # 只对需要的列填充NA，避免污染其他列
                        comp_df['Count'] = comp_df['Count'].fillna(0)
                        comp_df.rename(columns={'Count': 'Last Month Actual'}, inplace=True)
                    else:
                        comp_df = pred_filtered
                        comp_df['Last Month Actual'] = 0
                        last_month = "N/A"
                    
                    if not comp_df.empty:
                        # 修复: 在绘图前，确保 Care Home Name 是字符串
                        comp_df['Care Home Name'] = comp_df['Care Home Name'].astype(str)
                        
                        fig_bar = go.Figure()
                        fig_bar.add_trace(go.Bar(
                            x=comp_df['Care Home Name'],
                            y=comp_df['Last Month Actual'],
                            name=f'Actual Count (Month: {last_month})'
                        ))
                        fig_bar.add_trace(go.Bar(
                            x=comp_df['Care Home Name'],
                            y=comp_df['Predicted Mean'],
                            name='Predicted Mean (Next Month)',
                            error_y=dict(type='data', symmetric=False,
                                         array=comp_df['95% Upper'] - comp_df['Predicted Mean'],
                                         arrayminus=comp_df['Predicted Mean'] - comp_df['95% Lower'])
                        ))
                        fig_bar.update_layout(barmode='group', title=f"Actual vs. Predicted Counts for Score {selected_score}")
                        st.plotly_chart(fig_bar, use_container_width=True)

            except Exception as e:
                st.error(f"Failed to process prediction file: {e}")

# ==============================================================================
# Step 5: Overall Statistics
# ==============================================================================
elif step_title == "Overall Statistics":
    st.title("Overall Statistics")
    st.header("High-Risk Event Analysis (NEWS2 Score >= 6)")

    if st.session_state.get('df') is not None:
        df = st.session_state.get('df')
        summary_df, details_df = calculate_benchmark_data(df)

        tab1, tab2 = st.tabs(["Monthly Heatmap & Details", "Geospatial Map & Summary"])

        with tab1:
            st.subheader("Monthly High-Risk Event Heatmap")
            if not details_df.empty:
                heatmap_data = details_df.pivot_table(index='Care Home Name', columns='Month', values='high_score_count', fill_value=0)
                heatmap_height = max(400, len(heatmap_data.index) * 25)
                fig_heatmap = px.imshow(heatmap_data, labels=dict(x="Month", y="Care Home", color="High-Risk Events"), color_continuous_scale=px.colors.sequential.Reds, height=heatmap_height)
                st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
                st.info("No high-risk events (NEWS2 >= 6) found to generate a heatmap.")

            st.subheader("Distribution of NEWS2 Scores by Care Home")
            if 'NEWS2 Score' in df.columns:
                fig_box = px.box(df.dropna(subset=['NEWS2 Score']), x="Care Home Name", y="NEWS2 Score", points="outliers")
                fig_box.update_layout(height=500)
                st.plotly_chart(fig_box, use_container_width=True)

            st.subheader("Detailed Monthly High-Score Events Table")
            st.dataframe(details_df, use_container_width=True)

        with tab2:
            st.subheader("Geospatial Distribution of High-Risk Proportions")
            if 'Latitude' in df.columns and 'Longitude' in df.columns and not summary_df.empty:
                geo_df = df[['Care Home ID', 'Care Home Name', 'Latitude', 'Longitude']].drop_duplicates(subset=['Care Home ID']).copy()
                geo_df['Latitude'] = pd.to_numeric(geo_df['Latitude'], errors='coerce')
                geo_df['Longitude'] = pd.to_numeric(geo_df['Longitude'], errors='coerce')
                geo_df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
                
                geospatial_df = pd.merge(geo_df, summary_df[['Care Home ID', 'pi']], on='Care Home ID', how='left').fillna(0)
                
                if not geospatial_df.empty:
                    geospatial_df['size'] = 10 if geospatial_df['pi'].max() == 0 else (geospatial_df['pi'] * 20 + 5)
                    fig_map = px.scatter_map(
                        geospatial_df, lat="Latitude", lon="Longitude",
                        hover_name="Care Home Name", hover_data={"pi": ":.2%", "size": False},
                        size="size", color="pi",
                        color_continuous_scale=px.colors.sequential.Reds,
                        zoom=5, height=600, map_style="open-street-map"
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
            
            st.subheader("High-Risk Event Summary Table")
            st.dataframe(summary_df.style.format({'pi': '{:.2%}'}), use_container_width=True)
    else:
        st.warning("Please upload data in Step 1 to perform statistical analysis.")
