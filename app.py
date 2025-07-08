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
    # Set default index based on whether data is loaded
    default_page_index = 0
    if st.session_state.df is not None:
        default_page_index = 1

    step_title = option_menu(
        menu_title="Navigation",
        options=[
            "Upload Data",
            "Care Home Analysis",
            "Batch Prediction",
            "Prediction Visualization",
            "Overall Statistics"  # Renamed from "Benchmark Grouping"
        ],
        icons=[
            "cloud-upload",
            "house",
            "cpu",
            "graph-up-arrow",
            "bar-chart-line"
        ],
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

            # --- Geocoding Logic ---
            needs_geocoding = ('Latitude' not in df.columns or 'Longitude' not in df.columns or
                               df['Latitude'].isnull().any() or df['Longitude'].isnull().any())
            
            if needs_geocoding and 'Post Code' in df.columns:
                with st.spinner("Geospatial data missing or incomplete. Attempting to generate from 'Post Code' column..."):
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

    st.session_state['df'] = df
            
            st.success("File uploaded and processed successfully! You can now navigate to other sections.")
            
            # --- Data Overview ---
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
            
            if st.button("Go to Analysis"):
                # This button is mostly for user guidance now.
                # The navigation is handled by the sidebar.
                st.info("Please use the navigation menu on the left to proceed.")


        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.session_state['df'] = None # Clear state on error

    else:
        st.warning("Please upload the main data file to begin analysis.")

# ==============================================================================
# Step 2: Care Home Analysis
# ==============================================================================
elif step_title == "Care Home Analysis":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 2: Care Home Analysis")

    if st.session_state['df'] is not None:
        df = st.session_state['df']

        st.sidebar.header("Step 2: Analysis Options")
        
        df['Care Home Display'] = df['Care Home ID'].astype(str) + " | " + df['Care Home Name'].astype(str)
        care_home_map = (
            df[['Care Home ID', 'Care Home Display']]
            .drop_duplicates()
            .set_index('Care Home ID')['Care Home Display']
            .to_dict()
        )
        care_home_id = st.sidebar.selectbox(
            "Select Care Home",
            options=sorted(list(care_home_map.keys())), # Sort for better UI
            format_func=lambda x: care_home_map.get(x, x)
        )
        
        care_home_info = get_care_home_info(df, care_home_id)
        beds = care_home_info.get('beds', 10)
        
        with st.expander("Care Home Basic Information", expanded=True):
            st.markdown(f"**Name:** {care_home_info.get('name', 'N/A')}")
            st.markdown(f"**Number of Beds:** {beds}")
            st.markdown(f"**Number of Observations:** {care_home_info.get('obs_count', 'N/A')}")
            st.markdown(f"**Data Time Range:** {care_home_info.get('date_range', 'N/A')}")
        
        tab1, tab2 = st.tabs(["Usage Analysis", "Health Insights"])
        
        with tab1:
            st.header("Usage Analysis")
            period = st.selectbox("Time Granularity", ["Daily", "Weekly", "Monthly", "Yearly"], index=2, key="usage_period")
            usage_df = process_usage_data(df, care_home_id, beds, period)
            st.plotly_chart(plot_usage_counts(usage_df, period), use_container_width=True, key="usage_counts_chart")
            st.plotly_chart(plot_usage_per_bed(usage_df, period), use_container_width=True, key="usage_per_bed_chart")
            
            if period == "Monthly":
                coverage_df = calculate_coverage_percentage(df[df['Care Home ID'] == care_home_id])
                st.plotly_chart(plot_coverage(coverage_df), use_container_width=True, key="coverage_chart")
            else:
                st.info("Coverage % is only displayed in Monthly mode.")
        
        with tab2:
            st.header("Health Insights (Based on NEWS2)")
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

    if st.session_state['df'] is None:
        st.warning("Please upload data in Step 1 before running predictions.")
    else:
        st.info("This step will run predictions for all care homes with sufficient data and generate a downloadable CSV file.")

        st.subheader("Prediction Parameters")
        min_obs = st.number_input("Minimum observations required per care home", min_value=1, value=50, step=10)
        window_length = st.slider("Moving average window (months)", min_value=1, max_value=12, value=2)
        sigma = st.slider("Prior belief variance (sigma)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

        if st.button("Start Batch Prediction"):
            df = st.session_state['df']
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
                    pred_df, target_month = predict_next_month_bayesian(df_carehome, window_length, sigma)

                    if pred_df is not None and not pred_df.empty:
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
                    st.warning("Prediction could not be generated. This might be due to insufficient historical data.")

    if st.session_state['prediction_df'] is not None:
        csv = st.session_state['prediction_df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Prediction Results (.csv)",
            data=csv,
            file_name=f"batch_prediction_results_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# ==============================================================================
# Step 4: Prediction Visualization
# ==============================================================================
elif step_title == "Prediction Visualization":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 4: Prediction Visualization")

    if st.session_state.get('df') is None:
        st.warning("Please upload the historical data in Step 1 to compare with predictions.")
    else:
        upload_pred_file = st.file_uploader("Upload Prediction Results (.csv)", type=["csv"], key="pred_uploader")

        if upload_pred_file:
            try:
                pred_df = pd.read_csv(upload_pred_file)
                hist_df = st.session_state['df']

                # --- Data Preparation ---
                hist_df = hist_df.rename(columns={'NEWS2 score': 'NEWS2 Score'}, errors='ignore')
                hist_df['Month'] = pd.to_datetime(hist_df['Date/Time']).dt.to_period('M').astype(str)
                hist_counts = hist_df.groupby(['Care Home ID', 'Care Home Name', 'Month', 'NEWS2 Score']).size().reset_index(name='Count')

                ch_map = pred_df[['Care Home ID', 'Care Home Name']].drop_duplicates().set_index('Care Home ID')['Care Home Name'].to_dict()
                
                # --- Filters ---
                col1, col2 = st.columns(2)
                with col1:
                    selected_ch_id = st.selectbox(
                        "Filter by Care Home",
                        options=["All"] + sorted(list(ch_map.keys())),
                        format_func=lambda x: "All Care Homes (for score comparison)" if x == "All" else f"{ch_map.get(x, 'Unknown')} ({x})"
                    )
                with col2:
                    score_options = list(range(11))
                    selected_score = st.selectbox(
                        "Filter by NEWS2 Score (for 'All' view)",
                        options=score_options,
                        index=1, # Default to score 1
                        disabled=(selected_ch_id != "All") # Disable if a specific home is selected
                    )

                # --- Visualization ---
                if selected_ch_id != "All":
                    # --- Single Care Home Time Series View ---
                    st.info(f"Showing full history and prediction for: {ch_map.get(selected_ch_id)}")
                    ch_hist = hist_counts[hist_counts['Care Home ID'] == selected_ch_id]
                    ch_pred = pred_df[pred_df['Care Home ID'] == selected_ch_id]
                    if not ch_pred.empty:
                        fig = go.Figure()
                        # Plot historical lines for each score
                        for score_val in sorted(ch_hist['NEWS2 Score'].unique()):
                            d = ch_hist[ch_hist['NEWS2 Score'] == score_val]
                            fig.add_trace(go.Scatter(x=d['Month'], y=d['Count'], mode='lines+markers', name=f'History Score {score_val}'))
                        # Plot prediction points for each score
                        for _, row in ch_pred.iterrows():
                            fig.add_trace(go.Scatter(
                                x=[row['Month']], y=[row['Predicted Mean']], mode='markers',
                                error_y=dict(type='data', symmetric=False, array=[row['95% Upper'] - row['Predicted Mean']], arrayminus=[row['Predicted Mean'] - row['95% Lower']]),
                                name=f"Prediction Score {int(row['NEWS2 Score'])}", marker=dict(size=10)
                            ))
                        fig.update_layout(title=f"History vs. Prediction for {ch_map.get(selected_ch_id)}", yaxis_title="Observation Count")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    # --- All Care Homes, Single Score Comparison View ---
                    st.subheader(f"Comparing predictions for NEWS2 Score = {selected_score}")
                    pred_filtered = pred_df[pred_df['NEWS2 Score'] == selected_score]
                    
                    if not hist_counts.empty:
                        last_month = hist_counts['Month'].max()
                        hist_filtered = hist_counts[(hist_counts['NEWS2 Score'] == selected_score) & (hist_counts['Month'] == last_month)]
                        # Use copy to avoid SettingWithCopyWarning
                        comp_df = pred_filtered.copy()
                        # Merge for comparison plot
                        comp_df = pd.merge(comp_df, hist_filtered[['Care Home ID', 'Count']], on="Care Home ID", how='left').fillna(0)
                        comp_df.rename(columns={'Count': 'Last Month Actual'}, inplace=True)
                    else:
                        comp_df = pred_filtered.copy()
                        comp_df['Last Month Actual'] = 0
                        last_month = "N/A"
                    
                    if not comp_df.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=comp_df['Care Home Name'],
                            y=comp_df['Last Month Actual'],
                            name=f'Actual Count (Month: {last_month})'
                        ))
                        fig.add_trace(go.Bar(
                            x=comp_df['Care Home Name'],
                            y=comp_df['Predicted Mean'],
                            name='Predicted Mean (Next Month)',
                            error_y=dict(type='data', symmetric=False,
                                         array=comp_df['95% Upper'] - comp_df['Predicted Mean'],
                                         arrayminus=comp_df['Predicted Mean'] - comp_df['95% Lower'])
                        ))
                        fig.update_layout(barmode='group', title=f"Actual vs. Predicted Counts for Score {selected_score}",
                                          xaxis_title="Care Home", yaxis_title="Count")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No prediction data available for NEWS2 Score = {selected_score}.")

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
        df = df.rename(columns={'NEWS2 score': 'NEWS2 Score'}, errors='ignore')

        summary_df, details_df = calculate_benchmark_data(df)

        tab1, tab2 = st.tabs(["Monthly Heatmap & Details", "Geospatial Map & Summary"])

        with tab1:
            st.subheader("Monthly High-Risk Event Heatmap")
            if not details_df.empty:
                heatmap_data = details_df.pivot_table(index='Care Home Name', columns='Month', values='high_score_count', fill_value=0)
                # 恢复: 热力图自适应高度
                heatmap_height = max(400, len(heatmap_data.index) * 25)
                fig_heatmap = px.imshow(heatmap_data, labels=dict(x="Month", y="Care Home", color="High-Risk Events"), color_continuous_scale=px.colors.sequential.Reds, height=heatmap_height)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("No high-risk events (NEWS2 >= 6) found to generate a heatmap.")

            # 恢复: 箱形图 (Boxplot)
            st.subheader("Distribution of NEWS2 Scores by Care Home")
            if 'NEWS2 Score' in df.columns:
                fig_box = px.box(df.dropna(subset=['NEWS2 Score']), x="Care Home Name", y="NEWS2 Score", points="outliers")
                fig_box.update_layout(xaxis_title="", yaxis_title="NEWS2 Score", height=500)
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

                geospatial_df = pd.merge(geo_df, summary_df[['Care Home ID', 'pi']], on='Care Home ID', how='left')
                geospatial_df['pi'] = geospatial_df['pi'].fillna(0)
                
                if not geospatial_df.empty:
                    # 修复: 确保当地图点pi值为0时，点仍然可见
                    if geospatial_df['pi'].max() > 0:
                        geospatial_df['size'] = geospatial_df['pi'] * 20 + 5 # 动态大小
                    else:
                        geospatial_df['size'] = 10 # 固定大小

                    # 修复: 使用 px.scatter_map 并修正参数，同时放大圆点
                    fig_map = px.scatter_map(
                        geospatial_df, lat="Latitude", lon="Longitude",
                        hover_name="Care Home Name", hover_data={"pi": ":.2%", "size": False},
                        size="size", # 使用新的 'size' 列
                        color="pi", # 颜色仍然基于 'pi'
                        color_continuous_scale=px.colors.sequential.Reds,
                        zoom=5, height=600, 
                        map_style="open-street-map" # 修复参数: mapbox_style -> map_style
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
            
            st.subheader("High-Risk Event Summary Table")
            st.dataframe(summary_df.style.format({'pi': '{:.2%}'}), use_container_width=True)

    else:
        st.warning("Please upload data in Step 1 to perform statistical analysis.")