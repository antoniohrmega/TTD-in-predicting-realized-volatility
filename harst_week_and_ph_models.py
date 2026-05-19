#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antoniomega


What this file does:
    
    
    - Compute HARST-Week and HARST-PH coefficients and show them in the console
    - Do predictions
    - Save predictions in a .xlsx file


How to use:
    
    - Run 1 time for each index


"""""""""""""""




import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize


 
# Error functions
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def qlike(y_true, y_pred):
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)



# HARST function with generic transition variable
def HARST_estimation_generic(data, z_var, phi):
    """
    data : vetor 1D de RV (numpy array)
    z_var: vetor 1D com variável de transição (mesmo comprimento que data)
    phi  : vetor de parâmetros (8x1) do modelo HARST para 2 regimes com c fixo
    """
    RVs = np.asarray(data, dtype=float)
    Z = np.asarray(z_var, dtype=float)

    # build HAR components with pandas to ensure alignment
    # RV_{t-1}, 5-day mean, 22-day mean
    tmp = pd.DataFrame({'RV': RVs, 'Z_raw': Z})
    tmp['RV_d'] = tmp['RV'].shift(1)
    tmp['RV_w'] = tmp['RV'].shift(1).rolling(window=5).mean()
    tmp['RV_m'] = tmp['RV'].shift(1).rolling(window=22).mean()

    # chosen transition variable Zt 
    tmp['Zt'] = tmp['Z_raw']

    # Remove inicial NaN (precisamos de pelo menos 22 observações)
    tmp = tmp.dropna()

    RV_actuals = tmp['RV'].values
    RV_d = tmp['RV_d'].values
    RV_w = tmp['RV_w'].values
    RV_m = tmp['RV_m'].values
    Zt   = tmp['Zt'].values

    # Threshold (75th percentile of Zt)
    Threshold = np.percentile(Zt, 75)

    RV_hat = np.zeros_like(RV_actuals)

    # Smooth Transition: 4 linear + 3 non-linear + 1 gamma)
    b0  = phi[0]
    bD  = phi[1]
    bW  = phi[2]
    bM  = phi[3]

    aD1 = phi[4]
    aW1 = phi[5]
    aM1 = phi[6]
    g1  = phi[7]

    for i in range(len(RV_hat)):
        # Transition occurs smoothly through logistic function based on the threshold
        logistic_1 = 1.0 / (1.0 + np.exp(-g1 * np.clip(Zt[i] - Threshold, -100, 100)))
        
        RV_hat[i] = (b0 + bD*RV_d[i] + bW*RV_w[i] + bM*RV_m[i] +
                     aD1*RV_d[i]*logistic_1 +
                     aW1*RV_w[i]*logistic_1 +
                     aM1*RV_m[i]*logistic_1)
        
    # force positive predictions
    RV_hat = np.maximum(RV_hat, 1e-8)
        
    return RV_hat, RV_actuals

def QMLE_generic(phi, data, z_var):
    RV_hat, RV_actuals = HARST_estimation_generic(data, z_var, phi)
    return qlike(RV_actuals, RV_hat)



ticker = "^GSPC"   # change to "^GSPC, ^DJI" or "^RUT"
file_path = "Data Indexes.xlsx"


df_vol = pd.read_excel(file_path, sheet_name="YZ_volatility")
df_wass = pd.read_excel(file_path, sheet_name="wasserstein_dists")

# read sheets
df_vol = pd.read_excel(file_path, sheet_name="YZ_volatility")
df_wass = pd.read_excel(file_path, sheet_name="wasserstein_dists")

# Convert dates
df_vol['Date'] = pd.to_datetime(df_vol['Date'])
df_wass['Date'] = pd.to_datetime(df_wass['Date'])

# keep only Date + ticker
df_vol = df_vol[['Date', ticker]].rename(columns={ticker: 'RV'})
df_wass = df_wass[['Date', 'wasserstein_dists']].rename(columns={'wasserstein_dists': 'PH'})

# Merge on date
df = pd.merge(df_vol, df_wass, on='Date', how='inner')




# choose date interval
start_date = "2000-02-01"
end_date   = "2022-03-02"

# sort
df = df.sort_values('Date').set_index('Date')

# filter interval
df = df.loc[start_date:end_date]

df['PH_lag1']= df['PH'].shift(1).values


# Split train / test (80% / 20%)

n = len(df)
split = int(0.8 * n)

df_train = df.iloc[:split].copy()
df_test = df.iloc[split:].copy()

# Arrays with RV 
RVs_train = df_train['RV'].values
RVs_test  = df_test['RV'].values

# Zt = weekly RV (5-day rolling of the series)
Z_week_train = pd.Series(RVs_train).shift(1).rolling(window=5).mean().values
Z_week_test  = pd.Series(RVs_test).shift(1).rolling(window=5).mean().values





# initialize array for HARST-week parameters 
phi0 = np.ones(8) * 0.1

opt = {'maxiter': 2000, 'maxfev': 2000, 'disp': True}
opt_out_week = minimize(QMLE_generic, phi0, args=(RVs_train,Z_week_train),
                   method='Nelder-Mead', options=opt)

phi_hat_week = opt_out_week.x
print("phi_hat_week:", phi_hat_week)

Y_hat_test_week, RV_actuals_test = HARST_estimation_generic(RVs_test, Z_week_test, phi_hat_week)





# OOS error measures

rmse_val_week  = rmse(RV_actuals_test, Y_hat_test_week)
mae_val_week   = mae(RV_actuals_test, Y_hat_test_week)
qlike_val_week = qlike(RV_actuals_test, Y_hat_test_week)

print(f"RMSE:  {rmse_val_week:.6f}")
print(f"MAE:   {mae_val_week:.6f}")
print(f"QLIKE: {qlike_val_week:.6f}")


# with PH now 

PH_train = df_train['PH_lag1'].values
PH_test  = df_test['PH_lag1'].values

Z_ph_train = pd.Series(PH_train)
Z_ph_test = pd.Series(PH_test)


# initialize array for HARST-PH parameters 
phi1 = np.ones(8) * 0.1

opt = {'maxiter': 2000, 'maxfev': 2000, 'disp': True}
opt_out_ph = minimize(QMLE_generic, phi1, args=(RVs_train,Z_ph_train),
                   method='Nelder-Mead', options=opt)

phi_hat_ph = opt_out_ph.x
print("phi_hat_ph:", phi_hat_ph)

Y_hat_test_ph, RV_actuals_test = HARST_estimation_generic(RVs_test, Z_ph_test, phi_hat_ph)


# OOS error measures

rmse_val_ph  = rmse(RV_actuals_test, Y_hat_test_ph)
mae_val_ph   = mae(RV_actuals_test, Y_hat_test_ph)
qlike_val_ph = qlike(RV_actuals_test, Y_hat_test_ph)

print(f"RMSE:  {rmse_val_ph:.6f}")
print(f"MAE:   {mae_val_ph:.6f}")
print(f"QLIKE: {qlike_val_ph:.6f}")


actuals = RV_actuals_test  # RV teste
harst_week_pred = Y_hat_test_week         # HARST-Week forecasts
harst_ph_pred = Y_hat_test_ph       # HARST-PH forecasts


# Adjust lenghts
min_len = min(len(actuals), len(harst_week_pred), len(harst_ph_pred))
actuals = actuals[:min_len]
harst_week_pred = harst_week_pred[:min_len]
harst_ph_pred = harst_ph_pred[:min_len]

print(f"OOS sample: {min_len} obs")



lag = len(df_test) - len(RV_actuals_test)
test_dates = df_test.index[lag:lag + len(harst_week_pred)]

Data = {'Date': test_dates,
        'Forecast without PH': harst_week_pred,
        'Forecast with PH': harst_ph_pred
        }


df1=pd.DataFrame(data=Data)
df1.to_excel("HARST Week & PH forecasts " + ticker + " " + start_date + "-" + end_date + ".xlsx", index=False)