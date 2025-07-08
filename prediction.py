import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==== Step 1: 数据加载 ====
file_path = "/Users/lvlei/PycharmProjects/pythonProject3/Data/New_Area_Working_Data.xlsx"
df = pd.read_excel(file_path)

window_lengths = 2
sigma = 0.5  # 可以调整
score_list = list(range(0, 10))  # NEWS2 score 0~9

care_home_names = df['Care Home Name'].dropna().unique()
final_results = []

output_dir = "all_care_home_time_series"
os.makedirs(output_dir, exist_ok=True)

for care_home in care_home_names:
    df_ch = df[df['Care Home Name'] == care_home].copy()
    if len(df_ch) == 0:
        continue
    df_ch['Date/Time'] = pd.to_datetime(df_ch['Date/Time'])
    df_ch = df_ch.dropna(subset=['Date/Time'])
    df_ch['Month'] = df_ch['Date/Time'].dt.to_period('M')
    monthly_counts = df_ch.groupby(['Month', 'NEWS2 score']).size().unstack(fill_value=0)
    months = monthly_counts.index.astype(str)
    if len(monthly_counts) < 4:
        continue  # 至少需要2个月数据

    target_month = (pd.Period(months[-1]) + 1).strftime('%Y-%m')  # 最后一个月作为预测目标

    for window_length in window_lengths:
        if len(monthly_counts) < window_length + 1:
            continue  # 数据不够长
        moving_avg_months = months[-window_length:]
        moving_avg = monthly_counts.loc[moving_avg_months].mean()
        actual_counts = monthly_counts.loc[target_month]
        preds = []
        lowers = []
        uppers = []

        for score in score_list:
            if score not in monthly_counts.columns:
                preds.append(np.nan)
                lowers.append(np.nan)
                uppers.append(np.nan)
                continue
            y = monthly_counts.loc[list(moving_avg_months) + [target_month], score].values
            y_train = monthly_counts.loc[moving_avg_months, score].values
            prior_mean = moving_avg.get(score, 0)
            prior_logmu = np.log(prior_mean + 1e-5)
            with pm.Model() as model:
                lam_pred = pm.Lognormal("lam_pred", mu=prior_logmu, sigma=sigma)
                obs = pm.Poisson("obs", mu=lam_pred, observed=y_train)
                trace = pm.sample(2000, tune=1000, target_accept=0.95, progressbar=False)
            lam_samples = trace.posterior["lam_pred"].values.flatten()
            pred_counts = np.random.poisson(lam_samples)
            preds.append(np.mean(pred_counts))
            lowers.append(np.percentile(pred_counts, 2.5))
            uppers.append(np.percentile(pred_counts, 97.5))

        # 画时间序列对比图（所有score都画一条线，最后一个月显示预测）
        plt.figure(figsize=(12, 6))
        for score in score_list:
            plt.plot(months, monthly_counts.get(score, pd.Series(0, index=months)), marker='o', label=f'NEWS2={score}')
            plt.scatter(months[-1], actual_counts.get(score, 0), color='red', marker='*', s=100)
            err_low = np.maximum(preds[score_list.index(score)] - lowers[score_list.index(score)], 0)
            err_up = np.maximum(uppers[score_list.index(score)] - preds[score_list.index(score)], 0)
            plt.errorbar(months[-1], preds[score_list.index(score)],
              yerr=[[err_low], [err_up]],
              fmt='D', color='tab:green', capsize=6)

        plt.title(f"{care_home} - Window={window_length}mo (Target: {target_month})")
        plt.xlabel("Month")
        plt.ylabel("Monthly Count")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{care_home}_window{window_length}mo.png")
        plt.savefig(fname, dpi=200)
        plt.close()

        # 计算误差指标
        mask = ~np.isnan(preds)
        try:
            actual_arr = np.array([actual_counts.get(score, np.nan) for score in score_list])[mask]
            pred_arr = np.array(preds)[mask]
            mae = mean_absolute_error(actual_arr, pred_arr)
            rmse = np.sqrt(mean_squared_error(actual_arr, pred_arr))
            pred_width = np.nanmean(np.array(uppers)[mask] - np.array(lowers)[mask])
        except:
            mae, rmse, pred_width = np.nan, np.nan, np.nan

        final_results.append({
            'Care Home': care_home,
            'Window Length': window_length,
            'Target Month': target_month,
            'MAE': mae,
            'RMSE': rmse,
            'Width of 95% Interval': pred_width
        })

# 总表格输出
results_df = pd.DataFrame(final_results)
results_df = results_df.sort_values(['Care Home', 'Window Length'])
results_df.to_excel('all_care_home_bayes_performance_summary.xlsx', index=False)
print(results_df)
