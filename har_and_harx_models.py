#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antoniomega


What this file does:
    
    
    - Compute HAR and HARX coefficients and show them in the console, along with statistical tests
    - Do predictions
    - Save predictions in a .xlsx file


How to use:
    
    - Run 1 time for each index


"""""""""""""""




import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import t
import collections
from arch.bootstrap import MCS

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def qlike(y_true, y_pred):
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)





ticker = "^GSPC"   # change between "^GSPC, ^DJI" or "^RUT" - or
file_path = "Data Indexes.xlsx"


# Read sheets
df_vol = pd.read_excel(file_path, sheet_name="YZ_volatility")
df_wass = pd.read_excel(file_path, sheet_name="wasserstein_dists")

# Convert dates
df_vol['Date'] = pd.to_datetime(df_vol['Date'])
df_wass['Date'] = pd.to_datetime(df_wass['Date'])

# keep only Date + ticker
df_vol = df_vol[['Date', ticker]].rename(columns={ticker: 'RV'})
df_wass = df_wass[['Date', 'wasserstein_dists']].rename(columns={'wasserstein_dists': 'PH'})

# Merge on Date
df = pd.merge(df_vol, df_wass, on='Date', how='inner')




# choose date interval
start_date = "2000-02-01"
end_date   = "2022-03-02"

# sort
df = df.sort_values('Date').set_index('Date')

# filter interval
df = df.loc[start_date:end_date]

# HAR regressors
df['RV_d'] = df['RV'].shift(1)
df['RV_w'] = df['RV'].shift(1).rolling(window=5).mean()
df['RV_m'] = df['RV'].shift(1).rolling(window=22).mean()

# PH with lag
df['PH_lag1'] = df['PH'].shift(1)

# valid samples
cols_har  = ['RV', 'RV_d', 'RV_w', 'RV_m']
cols_harx = cols_har + ['PH_lag1']

df_har  = df.dropna(subset=cols_har).copy()
df_harx = df.dropna(subset=cols_harx).copy()

#  use 80% for training
n_har  = len(df_har)
n_harx = len(df_harx)
split_har  = int(0.8 * n_har)
split_harx = int(0.8 * n_harx)

df_train_har  = df_har.iloc[:split_har]
df_train_harx = df_harx.iloc[:split_harx]

# HAR (without PH)
Y_har = df_train_har['RV']
X_har = df_train_har[['RV_d', 'RV_w', 'RV_m']]
X_har = sm.add_constant(X_har)

res_har = sm.OLS(Y_har, X_har).fit()
print("=== HAR (treino, sem PH) ===")
print(res_har.summary())
print("\nBetas HAR:\n", res_har.params)

# HARX (with PH)
Y_harx = df_train_harx['RV']
X_harx = df_train_harx[['RV_d', 'RV_w', 'RV_m', 'PH_lag1']]
X_harx = sm.add_constant(X_harx)

res_harx = sm.OLS(Y_harx, X_harx).fit()
print("\n=== HARX (treino, com PH) ===")
print(res_harx.summary())
print("\nBetas HARX:\n", res_harx.params)

# OOS: HAR and HARX test
# HAR test
df_test_har = df_har.iloc[split_har:]
X_test_har = sm.add_constant(df_test_har[['RV_d', 'RV_w', 'RV_m']])
y_test_har = df_test_har['RV']
pred_har = res_har.predict(X_test_har)

# HARX test
df_test_harx = df_harx.iloc[split_harx:]
X_test_harx = sm.add_constant(df_test_harx[['RV_d', 'RV_w', 'RV_m', 'PH_lag1']])
y_test_harx = df_test_harx['RV']
pred_harx = res_harx.predict(X_test_harx)



# error measures
errors_har = {
    'RMSE': rmse(y_test_har, pred_har),
    'MAE': mae(y_test_har, pred_har),
    'QLIKE': qlike(y_test_har, pred_har)
}
errors_harx = {
    'RMSE': rmse(y_test_harx, pred_harx),
    'MAE': mae(y_test_harx, pred_harx),
    'QLIKE': qlike(y_test_harx, pred_harx)
}

print("HAR:   RMSE={rmse:.5f}, MAE={mae:.5f}, QLIKE={qlike:.5f}".format(rmse=errors_har['RMSE'], mae=errors_har['MAE'], qlike=errors_har['QLIKE']))
print("HARX:  RMSE={rmse:.5f}, MAE={mae:.5f}, QLIKE={qlike:.5f}".format(rmse=errors_harx['RMSE'], mae=errors_harx['MAE'], qlike=errors_harx['QLIKE']))



actuals = y_test_har.values  # RV test
har_pred = pred_har          # HAR forecasts
harx_pred = pred_harx        # HARX forecasts

# adjust lengths if needed
min_len = min(len(actuals), len(har_pred), len(harx_pred))
actuals = actuals[:min_len]
har_pred = har_pred[:min_len]
harx_pred = harx_pred[:min_len]

print(f"OOS sample: {min_len} obs")


test_dates = df_test_har.index[:min_len]  # dates from the test period

Data = {'Date': test_dates,
        'Forecast without PH': har_pred,
        'Forecast with PH': harx_pred
        }


df1=pd.DataFrame(data=Data)
df1.to_excel("HAR & HARX forecasts " + ticker + " " + start_date + "-" + end_date + ".xlsx", index=False)
