# NOTE: This file uses 4 spaces for indentation. Please ensure no tabs are used.
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
    """
    Generates 'Latitude' and 'Longitude' from a UK postcode column if they don't exist
    or are null. It operates on a copy and returns the modified DataFrame and a report.
    Uses the latest pgeocode API.
    """
    if postcode_column not in df.columns:
        return df, {"added": 0, "updated": 0, "failed": 0}

    df_copy = df.copy()

    if 'Latitude' not in df_copy.columns:
        df_copy['Latitude'] = np.nan
    if 'Longitude' not in df_copy.columns:
        df_copy['Longitude'] = np.nan
        
    df_copy['Latitude'] = pd.to_numeric(df_copy['Latitude'], errors='coerce')
    df_copy['Longitude'] = pd.to_numeric(df_copy['Longitude'], errors='coerce')

    # --- FIX: Use the latest pgeocode API ---
    # We directly query all unique non-null postcodes at once
    unique_postcodes = df_copy[postcode_column].dropna().unique()
    
    if len(unique_postcodes) > 0:
        nomi = pgeocode.Nominatim('gb')
        # The query_postal_code method handles lists of postcodes
        geo_data = nomi.query_postal_code(list(unique_postcodes))
        
        # We only need the columns for merging
        geo_data = geo_data[['postal_code', 'latitude', 'longitude']].set_index('postal_code')
        
        # Map the results back to the original dataframe
        df_copy = df_copy.set_index(postcode_column)
        
        # Use combine_first to fill NaNs in original with new data
        df_copy['Latitude'] = df_copy['Latitude'].combine_first(geo_data['latitude'])
        df_copy['Longitude'] = df_copy['Longitude'].combine_first(geo_data['longitude'])
        
        df_copy = df_copy.reset_index()

    # Reporting logic can be simplified as we process all at once
    original_nan_count = df['Latitude'].isnull().sum()
    final_nan_count = df_copy['Latitude'].isnull().sum()
    
    report = {
        "added": int(original_nan_count - final_nan_count),
        "updated": 0, # This logic is simpler now, focusing on adding missing data
        "failed": int(final_nan_count)
    }
    # --- END FIX ---

    return df_copy, report
    
def get_care_home_list(df):
    if 'Care Home ID' not in df.columns:
        return []
    return sorted(df['Care Home ID'].unique())

def get_care_home_info(df, care_home_id):
    home_df = df[df['Care Home ID'] == care_home_id]
    if not home_df.empty:
        info = home_df.iloc[0]
        return {
            'id': care_home_id,
            'name': info.get('Care Home Name', 'N/A'),
            'postcode': info.get('Post Code', 'N/A'),
            'beds': info.get('Number of Beds', 'N/A')
        }
    return {}

def process_usage_data(df, care_home_id, period):
    ch_df = df[df['Care Home ID'] == care_home_id].copy()
    ch_df['Date'] = pd.to_datetime(ch_df['Date'])
    
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
    freq = freq_map.get(period, "D")
    
    usage = ch_df.groupby(pd.Grouper(key='Date', freq=freq)).size().reset_index(name='count')
    beds = get_care_home_info(df, care_home_id).get('beds', np.nan)
    if pd.isna(beds) or beds == 0:
        usage['usage_per_bed'] = 0
    else:
        usage['usage_per_bed'] = usage['count'] / beds
    
    return usage

def process_health_insights(df, care_home_id, period):
    ch_df = df[df['Care Home ID'] == care_home_id].copy()
    ch_df['Date'] = pd.to_datetime(ch_df['Date'])
    ch_df['NEWS2 Score'] = pd.to_numeric(ch_df['NEWS2 Score'], errors='coerce')
    
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
    freq = freq_map.get(period, "D")
    
    grouped = ch_df.groupby(pd.Grouper(key='Date', freq=freq))
    
    result = grouped['NEWS2 Score'].agg(['count', lambda x: (x >= 6).sum(), lambda x: ((x >= 3) & (x < 6)).sum()]).reset_index()
    result.columns = ['Date', 'total_readings', 'high_risk_count', 'concern_count']
    
    result['high_risk_prop'] = result['high_risk_count'] / result['total_readings']
    result['concern_prop'] = result['concern_count'] / result['total_readings']
    
    judgement_df = ch_df.dropna(subset=['Clinical Judgement', 'NEWS2 Score'])
    judgement_grouped = judgement_df.groupby(pd.Grouper(key='Date', freq=freq))
    accuracy = judgement_grouped.apply(lambda x: ((x['Clinical Judgement'] == 'Normal') & (x['NEWS2 Score'] < 3) | (x['Clinical Judgement'] != 'Normal') & (x['NEWS2 Score'] >= 3)).mean()).reset_index(name='accuracy')
    
    result = pd.merge(result, accuracy, on='Date', how='left')
    
    high_score_df = ch_df[ch_df['NEWS2 Score'] >= 6]
    param_cols = ['Respiration', 'Oxygen Saturation', 'Supplemental Oxygen', 'Temperature', 'Blood Pressure', 'Heart Rate', 'Consciousness']
    
    param_counts_list = []
    if not high_score_df.empty:
        for name, group in high_score_df.groupby(pd.Grouper(key='Date', freq=freq)):
            date_counts = {'Date': name}
            for col in param_cols:
                if col in group:
                    date_counts[col] = (group[col] > 0).sum()
            param_counts_list.append(date_counts)

    if param_counts_list:
        param_counts = pd.DataFrame(param_counts_list)
        result = pd.merge(result, param_counts, on='Date', how='left')
    
    return result.fillna(0)

def calculate_coverage_percentage(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['month'] = df['Date'].dt.to_period('M')
    
    num_beds = df['Number of Beds'].iloc[0] if not df.empty else 0
    if num_beds == 0:
        return pd.DataFrame({'month': [], 'coverage': []})
        
    monthly_coverage = df.groupby('month')['Resident ID'].nunique().reset_index()
    monthly_coverage.columns = ['month', 'active_residents']
    monthly_coverage['coverage'] = (monthly_coverage['active_residents'] / num_beds) * 100
    monthly_coverage['month'] = monthly_coverage['month'].dt.to_timestamp()
    return monthly_coverage

def plot_usage_counts(df, period):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Date'], y=df['count'], name='Device Usage'))
    fig.update_layout(title=f'{period} Device Usage Counts', xaxis_title='Date', yaxis_title='Count')
    return fig

def plot_usage_per_bed(df, period):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['usage_per_bed'], mode='lines+markers', name='Usage per Bed'))
    fig.update_layout(title=f'{period} Device Usage per Bed', xaxis_title='Date', yaxis_title='Usage / Bed')
    return fig

def plot_coverage(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['month'], y=df['coverage'], mode='lines+markers', name='Coverage %'))
    fig.update_layout(title='Monthly Bed Coverage Percentage', xaxis_title='Month', yaxis_title='Coverage (%)', yaxis=dict(range=[0, 100]))
    return fig

def plot_news2_counts(df, period):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Date'], y=df['total_readings'], name='Total Readings'))
    fig.update_layout(title=f'{period} NEWS2 Score Readings', xaxis_title='Date', yaxis_title='Count')
    return fig

def plot_high_risk_prop(df, period):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['high_risk_prop'], mode='lines', name='High Risk Proportion'))
    fig.update_layout(title=f'{period} Proportion of High Risk Readings (Score >= 6)', xaxis_title='Date', yaxis_title='Proportion', yaxis=dict(tickformat=".0%"))
    return fig

def plot_concern_prop(df, period):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['concern_prop'], mode='lines', name='Concern Proportion'))
    fig.update_layout(title=f'{period} Proportion of Concern Readings (Score 3-5)', xaxis_title='Date', yaxis_title='Proportion', yaxis=dict(tickformat=".0%"))
    return fig

def plot_judgement_accuracy(df, period):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['accuracy'], mode='lines+markers', name='Judgement Accuracy'))
    fig.update_layout(title=f'{period} Clinical Judgement Accuracy', xaxis_title='Date', yaxis_title='Accuracy', yaxis=dict(tickformat=".0%"))
    return fig

def plot_high_score_params(df, period):
    param_cols = ['Respiration', 'Oxygen Saturation', 'Supplemental Oxygen', 'Temperature', 'Blood Pressure', 'Heart Rate', 'Consciousness']
    fig = go.Figure()
    for col in param_cols:
        if col in df.columns:
            fig.add_trace(go.Bar(x=df['Date'], y=df[col], name=col))
    fig.update_layout(barmode='stack', title=f'{period} Contributing Parameters for High Scores', xaxis_title='Date', yaxis_title='Count of Abnormal Readings')
    return fig

def predict_next_month_bayesian(df, care_home_id):
    ch_df = df[df['Care Home ID'] == care_home_id].copy()
    ch_df['Date'] = pd.to_datetime(ch_df['Date'])
    
    if len(ch_df) < 3:
        return None

    monthly_counts = ch_df.groupby([pd.Grouper(key='Date', freq='M'), 'NEWS2 Score']).size().unstack(fill_value=0)
    
    if len(monthly_counts) < 3:
        return None

    predictions = []
    for score_col in monthly_counts.columns:
        y = monthly_counts[score_col].values
        
        with pm.Model() as model:
            alpha = pm.HalfCauchy('alpha', beta=10)
            mu = pm.Gamma('mu', alpha=1.0, beta=1.0)
            
            y_pred = pm.NegativeBinomial('y_pred', mu=mu, alpha=alpha, observed=y)
            
            trace = pm.sample(2000, tune=1000, cores=1, progressbar=False, return_inferencedata=True)
            
            with model:
                posterior_pred = pm.sample_posterior_predictive(trace, var_names=['y_pred'], random_seed=42)

            next_month_pred_dist = posterior_pred.posterior_predictive['y_pred'].sel(chain=0)
            
            mean_pred = next_month_pred_dist.mean().item()
            hdi = az.hdi(next_month_pred_dist, hdi_prob=0.94)

            predictions.append({
                'NEWS2 Score': score_col,
                'predicted_mean': mean_pred,
                'hdi_94%_lower': hdi.x.data[0],
                'hdi_94%_upper': hdi.x.data[1]
            })
            
    return pd.DataFrame(predictions)

def calculate_benchmark_data(df):
    """
    Calculates high-risk event statistics for all care homes.
    Handles cases with zero high-risk events gracefully.
    """
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy['Month'] = df_copy['Date'].dt.to_period('M').astype(str)

    details_df = df_copy[df_copy['NEWS2 Score'] >= 6].copy()
    if not details_df.empty:
        details_df = details_df.groupby(['Care Home Name', 'Month', 'NEWS2 Score']).size().reset_index(name='count')
    else:
        details_df = pd.DataFrame(columns=['Care Home Name', 'Month', 'NEWS2 Score', 'count'])

    all_homes = df_copy[['Care Home ID', 'Care Home Name', 'Latitude', 'Longitude']].drop_duplicates().set_index('Care Home Name')

    if not details_df.empty:
        ci = details_df.groupby('Care Home Name')['Month'].nunique().reset_index(name='ci')
        total_months = df_copy.groupby('Care Home Name')['Month'].nunique().reset_index(name='total_months')
        summary_df = pd.merge(ci, total_months, on='Care Home Name', how='right')
        summary_df['ci'] = summary_df['ci'].fillna(0)
        summary_df['pi'] = (summary_df['ci'] / summary_df['total_months']).fillna(0)
    else:
        total_months = df_copy.groupby('Care Home Name')['Month'].nunique().reset_index(name='total_months')
        summary_df = total_months
        summary_df['ci'] = 0
        summary_df['pi'] = 0.0

    summary_df['rank'] = summary_df['pi'].rank(method='min', ascending=False).astype(int)
    
    summary_df = summary_df.set_index('Care Home Name').join(all_homes).reset_index()

    return summary_df.sort_values('rank'), details_df