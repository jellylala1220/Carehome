import streamlit as st
import pandas as pd
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
        
        tab1, tab2 = st.tabs(["Usage Analysis", "Health Insights"])
        
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
    else:
        st.info("Regional Analysis is not implemented yet. Please select Carehome Level Analysis.")
elif not main_data_file:
    st.warning("Please upload the main data file to begin analysis.")



