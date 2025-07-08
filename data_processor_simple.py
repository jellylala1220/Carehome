import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pymc as pm
import arviz as az
import pgeocode

def geocode_uk_postcodes(df, postcode_column='Post Code'):
    if postcode_column not in df.columns: return df
    df_copy = df.copy()
    if 'Latitude' in df_copy.columns and 'Longitude' in df_copy.columns:
        df_copy['Latitude'] = pd.to_numeric(df_copy['Latitude'], errors='coerce')
        df_copy['Longitude'] = pd.to_numeric(df_copy['Longitude'], errors='coerce')
        rows_to_geocode_mask = df_copy['Latitude'].isnull() | df_copy['Longitude'].isnull()
    else:
        rows_to_geocode_mask = pd.Series([True] * len(df_copy), index=df_copy.index)
        df_copy['Latitude'], df_copy['Longitude'] = np.nan, np.nan
    if not rows_to_geocode_mask.any(): return df_copy
    postcodes_to_query = df_copy.loc[rows_to_geocode_mask, postcode_column].astype(str).str.upper().str.strip()
    valid_postcodes = postcodes_to_query.dropna()[lambda x: (x.str.len() > 3) & (x != 'NAN')]
    if valid_postcodes.empty: return df_copy
    geo_data = pgeocode.Nominatim('gb').query_postal_code(valid_postcodes.tolist())
    latitudes = pd.Series(geo_data['latitude'].values, index=valid_postcodes.index)
    longitudes = pd.Series(geo_data['longitude'].values, index=valid_postcodes.index)
    df_copy['Latitude'] = df_copy['Latitude'].fillna(latitudes)
    df_copy['Longitude'] = df_copy['Longitude'].fillna(longitudes)
    return df_copy

def calculate_benchmark_data(df):
    if df.empty or 'NEWS2 Score' not in df.columns: return pd.DataFrame(), pd.DataFrame()
    df_copy = df.copy()
    df_copy['NEWS2 Score'] = pd.to_numeric(df_copy['NEWS2 Score'], errors='coerce').dropna()
    df_copy['Month'] = pd.to_datetime(df_copy['Date/Time']).dt.to_period('M')
    high_scores_mask = df_copy['NEWS2 Score'] >= 6
    monthly_high_scores = df_copy[high_scores_mask].groupby(['Care Home ID', 'Care Home Name', 'Month']).size().reset_index(name='high_score_count')
    if not monthly_high_scores.empty:
        monthly_high_scores = monthly_high_scores.sort_values(['Care Home Name', 'Month'])
        monthly_high_scores['Month'] = monthly_high_scores['Month'].astype(str)
    all_homes = df_copy[['Care Home ID', 'Care Home Name']].drop_duplicates()
    total_months = df_copy.groupby('Care Home ID')['Month'].nunique().reset_index(name='tm')
    summary_df = pd.merge(all_homes, total_months, on='Care Home ID', how='left')
    if not monthly_high_scores.empty:
        high_risk_months_count = monthly_high_scores.groupby('Care Home ID').size().reset_index(name='ci')
        summary_df = pd.merge(summary_df, high_risk_months_count, on='Care Home ID', how='left')
    else:
        summary_df['ci'] = 0
    summary_df['ci'] = summary_df['ci'].fillna(0).astype(int)
    summary_df['tm'] = summary_df['tm'].fillna(0).astype(int)
    summary_df['pi'] = np.where(summary_df['tm'] > 0, summary_df['ci'] / summary_df['tm'], 0)
    summary_df['rank'] = summary_df['pi'].rank(method='min', ascending=False).astype(int)
    summary_df = summary_df[['Care Home ID', 'Care Home Name', 'ci', 'tm', 'pi', 'rank']].sort_values('rank')
    return summary_df, monthly_high_scores

def get_care_home_list(df):
    return sorted(df['Care Home ID'].unique().tolist())

def get_care_home_info(df, care_home_id):
    rows = df[df['Care Home ID'].astype(str) == str(care_home_id).strip()]
    if rows.empty: return {}
    row = rows.iloc[0]
    return {'name': row.get('Care Home Name', ''), 'beds': row.get('No of Beds', 10), 'obs_count': len(rows), 'date_range': f"{rows['Date/Time'].min().date()} to {rows['Date/Time'].max().date()}"}

def process_usage_data(df, care_home_id, beds, period):
    care_home_data = df[df['Care Home ID'].astype(str) == str(care_home_id).strip()]
    if care_home_data.empty: return pd.DataFrame()
    dt_col = pd.to_datetime(care_home_data['Date/Time'])
    grouped = care_home_data.groupby(dt_col.dt.to_period(period[0])).size()
    usage_df = grouped.reset_index(name='Count')
    usage_df.columns = ['Date', 'Count']
    usage_df['Date'] = pd.to_datetime(usage_df['Date'].astype(str))
    usage_df['Usage_per_bed'] = usage_df['Count'] / beds if beds > 0 else 0
    return usage_df.sort_values('Date')

def calculate_coverage_percentage(data):
    if data is None or data.empty: return pd.DataFrame()
    dt_col = pd.to_datetime(data['Date/Time'])
    monthly = data.groupby(dt_col.dt.to_period('M'))
    coverage = [{'Date': month.to_timestamp(), 'coverage': group['Date/Time'].dt.date.nunique() / pd.Period(month).days_in_month} for month, group in monthly]
    return pd.DataFrame(coverage).sort_values('Date') if coverage else pd.DataFrame()

def process_health_insights(df, care_home_id, period):
    care_home_data = df[df['Care Home ID'].astype(str) == str(care_home_id).strip()]
    if care_home_data.empty or 'NEWS2 Score' not in care_home_data.columns: return {}
    dt_col = pd.to_datetime(care_home_data['Date/Time'])
    grouped = care_home_data.groupby(dt_col.dt.to_period(period[0]))
    insights = {}
    for period_name, group in grouped:
        key = pd.to_datetime(str(period_name))
        scores = group['NEWS2 Score']
        insights[key] = {'news2_counts': scores.value_counts(), 'high_risk_prop': (scores >= 6).mean()}
    if not insights: return {}
    result = {}
    for data_key in insights[next(iter(insights))]:
        series_dict = {period_key: data[data_key] for period_key, data in insights.items()}
        result[data_key] = pd.DataFrame(series_dict) if isinstance(insights[next(iter(insights))][data_key], pd.Series) else pd.Series(series_dict)
        result[data_key] = result[data_key].sort_index().T
    return result

def predict_next_month_bayesian(df_carehome, window_length=2, sigma=0.5):
    if df_carehome.empty or 'NEWS2 Score' not in df_carehome.columns: return pd.DataFrame(), None
    df_ch = df_carehome.copy()
    df_ch['Month'] = pd.to_datetime(df_ch['Date/Time']).dt.to_period('M')
    monthly_counts = df_ch.groupby(['Month', 'NEWS2 Score']).size().unstack(fill_value=0)
    if len(monthly_counts) < window_length: return pd.DataFrame(), None
    target_month_str = (monthly_counts.index.to_timestamp()[-1].to_period('M') + 1).strftime('%Y-%m')
    train_counts = monthly_counts.iloc[-window_length:]
    results = []
    for score in range(11):
        y_train = train_counts.get(score, 0)
        prior_logmu = np.log(np.mean(y_train) + 1e-5)
        with pm.Model() as model:
            lam_pred = pm.Lognormal("lam_pred", mu=prior_logmu, sigma=sigma)
            pm.Poisson("obs", mu=lam_pred, observed=y_train)
            trace = pm.sample(1000, tune=500, progressbar=False, cores=1)
        pred_counts = np.random.poisson(trace.posterior["lam_pred"].values)
        results.append({'NEWS2 Score': score, 'Predicted Mean': np.mean(pred_counts), '95% Lower': np.percentile(pred_counts, 2.5), '95% Upper': np.percentile(pred_counts, 97.5)})
    return pd.DataFrame(results), target_month_str

def plot_usage_counts(df, period):
    if df.empty: return go.Figure()
    return px.line(df, x='Date', y='Count', title=f'Usage Count ({period})', markers=True)

def plot_news2_counts(hi_data, period):
    if 'news2_counts' not in hi_data or hi_data['news2_counts'].empty: return go.Figure()
    df_to_plot = hi_data['news2_counts'].reset_index().rename(columns={'index': 'Date'}).melt(id_vars=['Date'], var_name='NEWS2 Score', value_name='Count')
    return px.bar(df_to_plot, x='Date', y='Count', color='NEWS2 Score', title=f'NEWS2 Score Counts ({period})', barmode='stack')

def plot_high_risk_prop(hi_data, period):
    if 'high_risk_prop' not in hi_data or hi_data['high_risk_prop'].empty: return go.Figure()
    return px.line(hi_data['high_risk_prop'].T, title=f'High-Risk Proportion (NEWS2 >= 6) ({period})', markers=True)