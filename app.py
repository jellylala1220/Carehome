import streamlit as st
import pandas as pd
import hashlib
from data_processor_simple import (
    get_care_home_list, get_care_home_info, 
    process_usage_data, process_health_insights,
    plot_usage_counts, plot_usage_per_bed, plot_coverage,
    plot_news2_counts, plot_high_risk_prop, plot_concern_prop,
    plot_judgement_accuracy, plot_high_score_params,
    plot_news2_barchart,
    predict_next_month_bayesian,
    calculate_benchmark_data,
    geocode_uk_postcodes,
    get_monthly_regional_benchmark_data,
    calculate_correlation_data,
    get_news2_color
)
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import numpy as np
from streamlit_option_menu import option_menu

NAV_OPTIONS = [
    "Upload Data",
    "Care Home Analysis",
    "Batch Prediction",
    "Prediction Visualization",
    "Benchmark Grouping",
    "Regional Analysis",
    "Correlation Analysis"
]

NAV_LABELS = [
    "Upload",
    "Care Home",
    "Predict",
    "Validate",
    "Benchmark",
    "Regional",
    "Correlation"
]

NAV_LABEL_TO_PAGE = dict(zip(NAV_LABELS, NAV_OPTIONS))

NAV_ICONS = [
    "cloud-upload",
    "house",
    "cpu",
    "graph-up-arrow",
    "bar-chart-line",
    "globe-americas",
    "link-45deg"
]

PHASE2_FLOW = [
    ("1", "Prepare working data", "Cleaned Phase 2 observations are uploaded here."),
    ("2", "Care home review", "Check usage, coverage, NEWS2 trends, and high-score drivers."),
    ("3", "Prediction run", "Generate next-month NEWS2 predictions for eligible care homes."),
    ("4", "Validation", "Compare predictions with actual monthly NEWS2 counts."),
    ("5", "Population analysis", "Benchmark care homes by usage, area, and high-NEWS correlation."),
]

# ----------- 统一美化所有 plotly 折线图 -----------
def beautify_line_chart(fig):
    if not isinstance(fig, go.Figure):
        return fig
    try:
        fig.update_traces(
            selector=dict(mode="lines+markers"),
            line=dict(width=3)
        )
        fig.update_layout(
            font=dict(size=22, family="Arial", color="black"),
            legend=dict(font=dict(size=20)),
            xaxis=dict(tickfont=dict(size=22)),
            yaxis=dict(tickfont=dict(size=20)),
            xaxis_title_font=dict(size=24),
            yaxis_title_font=dict(size=24),
        )
    except Exception as e:
        print(f"Beautify error: {e}")
    return fig

st.set_page_config(page_title="Care Home Analysis Dashboard", layout="wide")


def inject_ui_css():
    st.markdown(
        """
        <style>
        div.stButton > button,
        div.stDownloadButton > button {
            min-height: 44px;
            border-radius: 8px;
            font-weight: 700;
            letter-spacing: 0;
        }
        div.stButton > button[kind="primary"],
        div.stDownloadButton > button[kind="primary"] {
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12);
        }
        .stDataFrame tbody tr td {
            font-size: 18px !important;
        }
        .stDataFrame thead tr th {
            font-size: 18px !important;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 12px 14px;
        }
        div[data-testid="stAlert"] {
            border-radius: 8px;
        }
        .phase2-step {
            border-left: 4px solid #2563eb;
            padding: 8px 12px;
            margin: 6px 0;
            background: #f8fafc;
            border-radius: 0 8px 8px 0;
        }
        .phase2-step strong {
            color: #0f172a;
        }
        section[data-testid="stSidebar"] .stMarkdown {
            overflow-wrap: normal;
            word-break: normal;
        }
        .sidebar-status {
            font-size: 13px;
            line-height: 1.45;
            color: #334155;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 10px 12px;
            margin-top: 4px;
        }
        .sidebar-status strong {
            color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_ui_css()

# Initialize session state
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'go_analysis' not in st.session_state:
    st.session_state['go_analysis'] = False
if 'prediction_df' not in st.session_state:
    st.session_state['prediction_df'] = None
if 'processed_file_name' not in st.session_state:
    st.session_state['processed_file_name'] = None
if 'batch_prediction_authenticated' not in st.session_state:
    st.session_state['batch_prediction_authenticated'] = False
if 'nav_target' not in st.session_state:
    st.session_state['nav_target'] = None


def navigate_to(page_name):
    st.session_state['nav_target'] = page_name


def clear_loaded_data():
    st.session_state['df'] = None
    st.session_state['processed_file_name'] = None
    st.session_state['processed_file_signature'] = None
    st.session_state['go_analysis'] = False
    st.session_state['prediction_df'] = None


def file_signature(file_name, file_bytes):
    digest = hashlib.md5(file_bytes).hexdigest()
    return f"{file_name}:{len(file_bytes)}:{digest}"


@st.cache_data(show_spinner=False)
def load_observation_data(file_name, file_bytes):
    df = pd.read_excel(BytesIO(file_bytes))
    df.columns = [str(col).strip() for col in df.columns]

    if 'Date/Time' in df.columns:
        df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')
    if 'Care Home ID' in df.columns:
        df['Care Home ID'] = (
            df['Care Home ID']
            .astype(str)
            .str.replace(r'\.0$', '', regex=True)
            .str.strip()
        )
    if 'Care Home Name' in df.columns:
        df['Care Home Name'] = df['Care Home Name'].astype(str).str.strip()

    numeric_cols = [
        'No of Beds', 'NEWS2 score', 'New2 Score_New', 'O2', 'O2_New',
        'Systolic', 'Systolic_New', 'Diasolic', 'Pulse', 'Pulse_New',
        'Temperature', 'Temperate_New', 'Respiration rate',
        'Respiraties_New', 'O2 Delivery_New', 'Consciouness New',
        'Latitude', 'Longitude'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    report = {
        'coordinates_generated': 0,
        'coordinates_missing': None,
        'geocoding_attempted': False,
    }

    needs_geocoding = (
        'Latitude' not in df.columns or
        'Longitude' not in df.columns or
        df['Latitude'].isnull().any() or
        df['Longitude'].isnull().any()
    )

    if needs_geocoding and 'Post Code' in df.columns:
        report['geocoding_attempted'] = True
        missing_before = df['Latitude'].isnull().sum() if 'Latitude' in df.columns else len(df)
        df = geocode_uk_postcodes(df, 'Post Code')
        if 'Latitude' in df.columns:
            missing_after = df['Latitude'].isnull().sum()
            report['coordinates_generated'] = int(missing_before - missing_after)
            report['coordinates_missing'] = int(missing_after)

    return df, report


@st.cache_data(show_spinner=False)
def cached_health_insights(df, care_home_id, period):
    return process_health_insights(df, care_home_id, period)


@st.cache_data(show_spinner=False)
def cached_benchmark_data(df):
    return calculate_benchmark_data(df)


@st.cache_data(show_spinner=False)
def cached_monthly_benchmark_table(df):
    df_copy = df.copy()
    df_copy['Date/Time'] = pd.to_datetime(df_copy['Date/Time'])
    df_copy['Month'] = df_copy['Date/Time'].dt.strftime('%Y-%m')

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
    return monthly_benchmark_df


@st.cache_data(show_spinner=False)
def cached_monthly_regional_data(df):
    return get_monthly_regional_benchmark_data(df)


@st.cache_data(show_spinner=False)
def cached_correlation_data(df, min_months):
    return calculate_correlation_data(df, min_months=min_months)


def render_dataset_status(df):
    if df is None:
        st.warning("No data loaded. Upload the Phase 2 working data first.")
        return

    date_series = pd.to_datetime(df['Date/Time'], errors='coerce') if 'Date/Time' in df.columns else pd.Series(dtype='datetime64[ns]')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", f"{len(df):,}")
    col2.metric("Care Homes", f"{df['Care Home ID'].nunique():,}" if 'Care Home ID' in df.columns else "N/A")
    col3.metric("First Date", date_series.min().date().isoformat() if not date_series.dropna().empty else "N/A")
    col4.metric("Last Date", date_series.max().date().isoformat() if not date_series.dropna().empty else "N/A")


def render_sidebar_dataset_status(df):
    if df is None:
        st.info("No data loaded.")
        return

    date_series = pd.to_datetime(df['Date/Time'], errors='coerce') if 'Date/Time' in df.columns else pd.Series(dtype='datetime64[ns]')
    first_date = date_series.min().date().isoformat() if not date_series.dropna().empty else "N/A"
    last_date = date_series.max().date().isoformat() if not date_series.dropna().empty else "N/A"
    care_home_count = df['Care Home ID'].nunique() if 'Care Home ID' in df.columns else "N/A"

    st.markdown(
        f"""
        <div class="sidebar-status">
            <strong>Loaded data</strong><br>
            Observations: {len(df):,}<br>
            Care homes: {care_home_count}<br>
            Date: {first_date} to {last_date}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_need_data_actions():
    st.warning("Please upload the Phase 2 working data before using this page.")
    st.button(
        "Go to Upload Data",
        type="primary",
        icon=":material/upload_file:",
        use_container_width=True,
        on_click=navigate_to,
        args=("Upload Data",),
    )


def render_phase2_workflow():
    with st.expander("Phase 2 workflow", expanded=True):
        for number, title, detail in PHASE2_FLOW:
            st.markdown(
                f"""
                <div class="phase2-step">
                    <strong>{number}. {title}</strong><br>
                    <span>{detail}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

# Sidebar navigation - 改用新的 option_menu
with st.sidebar:
    nav_target = st.session_state.get('nav_target')
    manual_select = NAV_OPTIONS.index(nav_target) if nav_target in NAV_OPTIONS else None
    selected_nav_label = option_menu(
        menu_title="Navigation",  # 菜单标题
        options=NAV_LABELS,
        icons=NAV_ICONS,
        menu_icon="cast",  # 菜单图标
        default_index=0,  # 默认选中的按钮
        manual_select=manual_select,
        key="main_navigation",
        styles={
            "container": {"padding": "0!important", "background-color": "#ffffff"},
            "icon": {"color": "#2563eb", "font-size": "18px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "3px 0",
                "--hover-color": "#eff6ff",
                "border-radius": "8px",
                "font-weight": "600",
                "white-space": "nowrap",
            },
            "nav-link-selected": {"background-color": "#2563eb", "font-weight": "700"},
        },
    )
    step_title = NAV_LABEL_TO_PAGE[selected_nav_label]
    st.session_state['nav_target'] = None

    st.markdown("---")
    render_sidebar_dataset_status(st.session_state.get('df'))

    # 保证 copy right 在侧边栏最下方
    st.markdown("""
    <div style='flex:1'></div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-top: 40px; font-size: 13px; color: #888; text-align: center;'>
    © 2025 Inventors: Professor Diwei Zhou (Loughborough University) and Lei Lyu (PhD student). Contributor: Tara Marshall (These Hands Academy Ltd)
    </div>
    """, unsafe_allow_html=True)

# 在主页面顶部展示 logo（只展示一次，且放大，且更靠近）
st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)  # 顶部留白
col0, col1, col2, col3, col4 = st.columns([1, 2, 1, 2, 1])
with col1:
    st.image("loughborough_logo.png", width=220)
with col3:
    st.image("these_hands_academy_logo.png", width=220)

# Step 1: Upload Data
if step_title == "Upload Data":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 1: Upload Phase 2 Working Data")

    render_phase2_workflow()

    main_data_file = st.file_uploader(
        "Upload Observation Data (Excel)",
        type=["xlsx"],
        help="Use the cleaned Phase 2 working data file generated from the local pipeline.",
        width="stretch",
    )

    if main_data_file is not None:
        file_bytes = main_data_file.getvalue()
        current_signature = file_signature(main_data_file.name, file_bytes)

        if st.session_state.get('processed_file_signature') != current_signature:
            try:
                with st.spinner("Loading and preparing the uploaded workbook..."):
                    df, load_report = load_observation_data(main_data_file.name, file_bytes)
                    st.session_state['df'] = df
                    st.session_state['processed_file_name'] = main_data_file.name
                    st.session_state['processed_file_signature'] = current_signature
                    st.session_state['go_analysis'] = True
                    st.session_state['prediction_df'] = None

                st.success("Data loaded and ready for analysis.")
                if load_report.get('coordinates_generated', 0) > 0:
                    st.info(f"Generated coordinates for {load_report['coordinates_generated']} rows.")
                if load_report.get('coordinates_missing'):
                    st.warning(f"{load_report['coordinates_missing']} rows still have missing coordinates and may be excluded from map views.")

            except Exception as e:
                st.error(f"Error processing file: {e}")
                clear_loaded_data()
        else:
            st.info(f"Using cached data from `{main_data_file.name}`.")

    if st.session_state.get('df') is not None:
        df = st.session_state.df

        st.subheader("Loaded Dataset")
        render_dataset_status(df)

        carehome_counts = df['Care Home ID'].value_counts()
        total_count = carehome_counts.sum()
        id_to_name = df.drop_duplicates('Care Home ID').set_index('Care Home ID')['Care Home Name'].astype(str).to_dict()
        table = carehome_counts.reset_index()
        table.columns = ['Care Home ID', 'Count']
        table['Care Home Name'] = table['Care Home ID'].map(id_to_name)
        table['Percentage'] = (table['Count'] / total_count) * 100
        table = table[['Care Home ID', 'Care Home Name', 'Count', 'Percentage']]
        table = table.sort_values('Count', ascending=False).reset_index(drop=True)

        action_col1, action_col2, action_col3 = st.columns([1.3, 1.3, 1])
        with action_col1:
            if st.button(
                "Open Care Home Analysis",
                type="primary",
                icon=":material/analytics:",
                use_container_width=True,
            ):
                st.session_state['go_analysis'] = True
                navigate_to("Care Home Analysis")
                st.rerun()
        with action_col2:
            if st.button(
                "Go to Benchmarking",
                icon=":material/bar_chart:",
                use_container_width=True,
            ):
                navigate_to("Benchmark Grouping")
                st.rerun()
        with action_col3:
            if st.button(
                "Clear Data",
                icon=":material/delete:",
                use_container_width=True,
                help="Remove the loaded workbook from this session.",
            ):
                clear_loaded_data()
                st.rerun()

        with st.expander("Care Home Observation Counts", expanded=False):
            st.dataframe(
                table.style.format({'Percentage': '{:.1f}%'}),
                use_container_width=True,
                hide_index=True,
            )

    elif main_data_file is None and st.session_state.get('df') is None:
        st.warning("Please upload the main data file to begin analysis.")


# Step 2: Care Home Analysis
elif step_title == "Care Home Analysis":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 2: Care Home Analysis")

    if st.session_state['df'] is not None:
        df = st.session_state['df'].copy()
        render_dataset_status(df)

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
            placeholder="Choose a care home"
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
            with st.spinner("Preparing usage charts..."):
                usage_df = process_usage_data(df, care_home, beds, period)
            st.plotly_chart(plot_usage_counts(usage_df, period), use_container_width=True, key="usage_counts")
            st.plotly_chart(plot_usage_per_bed(usage_df, period), use_container_width=True, key="usage_per_bed")
            if period == "Monthly":
                from data_processor_simple import calculate_coverage_percentage
                coverage_df = calculate_coverage_percentage(df[df['Care Home ID'].astype(str) == str(care_home)])
                st.plotly_chart(plot_coverage(coverage_df), use_container_width=True, key="coverage")
            else:
                st.info("Coverage % is only displayed in Monthly mode.")
        
        with tab2:
            st.header("Health Insights (Based on NEWS2)")
            period2 = st.selectbox("Time Granularity (Health Insights)", ["Daily", "Weekly", "Monthly", "Yearly"], index=2, key="health_period")

            with st.spinner("Preparing NEWS2 insight charts..."):
                hi_data = cached_health_insights(df, care_home, period2)

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
        render_need_data_actions()

# Step 3: Batch Prediction (Offline)
elif step_title == "Batch Prediction":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 3: Batch Prediction (Offline)")

    def show_password_form():
        """显示密码输入表单"""
        st.warning("This module is password protected. Please enter the password to continue.")
        # 在实际应用中，应使用 st.secrets 来安全地存储密码
        CORRECT_PASSWORD = "admin"

        with st.form("password_form"):
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Authenticate",
                type="primary",
                icon=":material/login:",
                use_container_width=True,
            )

            if submitted:
                if password == CORRECT_PASSWORD:
                    st.session_state["batch_prediction_authenticated"] = True
                    st.rerun()
                else:
                    st.error("The password you entered is incorrect.")

    def show_batch_prediction_page():
        """显示批量预测页面的实际内容"""
        # 在侧边栏添加一个登出/锁定按钮
        st.sidebar.button(
            "Lock Batch Prediction Page",
            icon=":material/lock:",
            use_container_width=True,
            on_click=lambda: st.session_state.update(batch_prediction_authenticated=False)
        )

        if st.session_state['df'] is None:
            render_need_data_actions()
        else:
            render_dataset_status(st.session_state['df'])
            st.info("This step will run predictions for all care homes with sufficient data (>50 observations) and generate a downloadable CSV file.")

            st.subheader("Prediction Parameters")
            with st.form("batch_prediction_settings"):
                min_obs = st.number_input("Minimum observations required per care home", min_value=1, value=50, step=10)
                window_length = st.slider("Moving average window (months)", min_value=1, max_value=12, value=2)
                sigma = st.slider("Prior belief variance (sigma)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
                run_prediction = st.form_submit_button(
                    "Start Batch Prediction",
                    type="primary",
                    use_container_width=True,
                    icon=":material/play_arrow:",
                )

            if run_prediction:
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
                type="primary",
                icon=":material/download:",
                use_container_width=True,
                on_click="ignore",
            )

    # --- 主逻辑：根据认证状态显示密码表单或页面内容 ---
    if st.session_state.get("batch_prediction_authenticated", False):
        show_batch_prediction_page()
    else:
        show_password_form()


# Step 4: Prediction Visualization
elif step_title == "Prediction Visualization":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 4: Prediction Visualization")

    if st.session_state['df'] is None:
        render_need_data_actions()
    else:
        render_dataset_status(st.session_state['df'])
        st.info("Upload the prediction results file (generated in Step 3) to visualize.")
        upload_pred_file = st.file_uploader(
            "Upload Prediction Results (.csv)",
            type=["csv"],
            key="step4_upload",
            width="stretch",
        )

        if upload_pred_file and st.session_state['df'] is not None:
            with st.spinner("Combining prediction and actual data..."):
                pred_df = pd.read_csv(upload_pred_file)
                hist_df = st.session_state['df'].copy()

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
            all_care_homes_option = "All Care Homes (slower)"
            care_home_options = sorted(merged_df['Care Home Display'].unique()) + [all_care_homes_option]

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
            actual_cols = display_df.columns.tolist()
            front_cols_in_df = [col for col in front_cols if col in actual_cols]
            other_cols = [col for col in display_df.columns if col not in front_cols and col != 'Care Home Display']
            other_cols_in_df = [col for col in other_cols if col in actual_cols and col not in front_cols_in_df]
            display_df_ordered = display_df[front_cols_in_df + other_cols_in_df]

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

                # --- 新增：为当前护理院的所有图表计算一个统一的Y轴范围 ---
                max_y_hist = full_hist_ch['Actual'].max()
                max_y_pred = pred_ch['95% Upper'].max()
                max_y_hist = 0 if pd.isna(max_y_hist) else max_y_hist
                max_y_pred = 0 if pd.isna(max_y_pred) else max_y_pred
                overall_max_y = max(max_y_hist, max_y_pred)
                # 设置一个最小范围5，并增加15%的顶部空间，确保图表不会被压扁
                yaxis_range = [0, max(5, overall_max_y * 1.15)]

                score_list = sorted(pred_ch['NEWS2 Score'].unique())

                for score in score_list:
                    # --- 改进：在绘图前先检查历史数据 ---
                    hist_score_df = full_hist_ch[full_hist_ch['NEWS2 Score'] == score].sort_values('Month')

                    # 如果该分数没有任何历史数据，则显示一条信息并跳过
                    if hist_score_df.empty:
                        st.markdown(f"**NEWS2 Score = {score}:** No historical data available, so no chart is generated.")
                        continue

                    # --- 如果有数据，则继续绘图 ---
                    fig = go.Figure()

                    # --- 根据分数获取颜色 ---
                    color_details = get_news2_color(score)
                    score_color = color_details['background']

                    # 因为已经确认 hist_score_df 非空，所以可以直接添加 trace
                    fig.add_trace(go.Scatter(
                        x=hist_score_df['Month'],
                        y=hist_score_df['Actual'],
                        mode='lines+markers',
                        name='Historical Actual',
                        line=dict(color=score_color),
                        marker=dict(symbol='circle', color=score_color)
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
                        showlegend=True,
                        yaxis=dict(range=yaxis_range) # 应用固定的Y轴范围
                    )
                    st.plotly_chart(beautify_line_chart(fig), use_container_width=True, key=f"plot_{care_home_id}_{score}")
        elif not upload_pred_file:
            st.info("Awaiting upload of prediction file.")

# Step 5: Overall Statistics/Benchmark Grouping
elif step_title == "Benchmark Grouping":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 5: Overall Statistics & Benchmark Grouping")

    if st.session_state['df'] is None:
        render_need_data_actions()
    else:
        df_full = st.session_state['df']
        render_dataset_status(df_full)

        # --- 为箱线图和热力图准备月度数据 ---
        if 'No of Beds' not in df_full.columns:
            st.error("Source data must contain 'No of Beds' column for this analysis.")
            st.stop()

        with st.spinner("Preparing benchmark data..."):
            monthly_benchmark_df = cached_monthly_benchmark_table(df_full)
            geospatial_df = cached_benchmark_data(df_full)

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
            st.plotly_chart(beautify_line_chart(fig_box), use_container_width=True)

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
                st.plotly_chart(beautify_line_chart(fig_heatmap), use_container_width=True)

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
                rng = np.random.default_rng(42)
                geospatial_df['lat_jittered'] = geospatial_df['Latitude'] + rng.uniform(-jitter_amount, jitter_amount, size=len(geospatial_df))
                geospatial_df['lon_jittered'] = geospatial_df['Longitude'] + rng.uniform(-jitter_amount, jitter_amount, size=len(geospatial_df))

                # 创建地图
                fig_map = px.scatter_map(
                    geospatial_df.dropna(subset=['Latitude', 'Longitude']), # 确保没有NaN的经纬度
                    lat="lat_jittered", # 使用抖动后的纬度
                    lon="lon_jittered", # 使用抖动后的经度
                    color="pi_group",
                    size="size",  # 使用新的 size 列
                    color_discrete_map=color_map,
                    category_orders={"pi_group": ["Low", "Medium", "High"]},
                    map_style="open-street-map",
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
                    height=1000 # 增加地图高度
                )
                st.plotly_chart(beautify_line_chart(fig_map), use_container_width=True)

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
                icon=":material/download:",
                use_container_width=True,
                on_click="ignore",
            )

# Step 6: Regional Analysis
elif step_title == "Regional Analysis":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 6: Regional Analysis")

    if st.session_state['df'] is None:
        render_need_data_actions()
    else:
        df = st.session_state['df']
        render_dataset_status(df)

        if 'Area' not in df.columns or 'No of Beds' not in df.columns:
            st.error("Source data must contain 'Area' and 'No of Beds' columns for this analysis.")
        else:
            with st.spinner("Preparing regional benchmark data..."):
                monthly_df_full = cached_monthly_regional_data(df)

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

                st.subheader(f"Part 1. Monthly Usage per Bed by Area {regional_title_suffix}")
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
                st.plotly_chart(beautify_line_chart(fig_box), use_container_width=True)

                st.markdown("---")

                st.subheader("Part 2. Area Benchmark Grouping Percentage")
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
                    st.plotly_chart(beautify_line_chart(fig_bar), use_container_width=True)

                st.markdown("---")

                st.subheader("Part 3. Detailed Grouping Data")
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
                    icon=":material/download:",
                    use_container_width=True,
                    on_click="ignore",
                )

# Step 7: Correlation Analysis
elif step_title == "Correlation Analysis":
    st.title("Care Home Analysis Dashboard")
    st.header("Step 7: Correlation Analysis")
    st.markdown("Analysis of the correlation between monthly high NEWS scores (≥6) and average usage per bed.")

    if st.session_state['df'] is None:
        render_need_data_actions()
    else:
        df = st.session_state['df']
        render_dataset_status(df)

        if 'NEWS2 score' not in df.columns or 'No of Beds' not in df.columns:
            st.error("Source data must contain 'NEWS2 score' and 'No of Beds' columns for this analysis.")
        else:
            # --- 新增：允许用户设置最小月份数 ---
            st.sidebar.subheader("Correlation Settings")
            min_months_for_corr = st.sidebar.number_input(
                "Minimum months of data required per care home",
                min_value=2,
                max_value=24,
                value=3,
                step=1,
                help="Only care homes with at least this many months of data will be included in the correlation analysis."
            )

            with st.spinner("Calculating monthly data and correlations..."):
                monthly_corr_df, corr_summary_df, overall_stats = cached_correlation_data(df, min_months_for_corr)

            if corr_summary_df.empty:
                st.info(f"Not enough data to generate correlation analysis. No care homes found with at least {min_months_for_corr} months of data.")
            else:
                # --- 新增：总体相关性分析 ---
                st.subheader("Section 1. Overall Correlation Analysis")
                if overall_stats:
                    col1, col2 = st.columns(2)
                    col1.metric("Overall Pearson's r", f"{overall_stats['Pearson r']:.3f}")
                    col2.metric("p-value", f"{overall_stats['Pearson p-value']:.3f}")

                    fig_scatter = px.scatter(
                        monthly_corr_df,
                        x="Usage per Bed",
                        y="High NEWS Count",
                        hover_name="Care Home Name",
                        hover_data=["Month"],
                        title="High NEWS Count vs. Usage Per Bed (All Care Homes)",
                        labels={"Usage per Bed": "Average Usage per Bed", "High NEWS Count": "High NEWS (≥6) Count"},
                        trendline="ols", # 添加普通最小二乘趋势线
                        trendline_color_override="red"
                    )
                    st.plotly_chart(beautify_line_chart(fig_scatter), use_container_width=True)
                else:
                    st.info("Could not calculate overall correlation due to insufficient data or lack of variance.")

                st.markdown("---")

                # Part C: Per-Care-Home Analysis
                st.subheader("Section 2. Correlation Coefficient Summary (Per Care Home)")
                st.markdown("This table shows the Pearson and Spearman correlation coefficients between 'High NEWS Count' and 'Usage per Bed' for each care home.")

                st.dataframe(corr_summary_df.style.format({
                    'Pearson r': '{:.3f}', 'Pearson p-value': '{:.3f}',
                    'Spearman r': '{:.3f}', 'Spearman p-value': '{:.3f}'
                }), use_container_width=True)

                # --- 新增：p-value 解释 ---
                st.markdown(
                    """
                    <div style="font-size: 0.9em; margin-top: 1em;">
                    <strong>Note on p-value interpretation:</strong>
                    <ul>
                    <li><b>p-value < 0.05</b>: The correlation is statistically significant.</li>
                    <li><b>p-value ≥ 0.05</b>: The correlation is not statistically significant.</li>
                    </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                csv = corr_summary_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Correlation Summary (.csv)",
                    data=csv,
                    file_name="correlation_summary.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    use_container_width=True,
                    on_click="ignore",
                )

                st.markdown("---")

                # Part D: Trend Visualization
                st.subheader("Section 3. Trend Visualization (Per Care Home)")
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
                    st.plotly_chart(beautify_line_chart(fig), use_container_width=True)
