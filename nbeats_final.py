#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antoniomega


What this file does:
    
    
    - Compute NBEATS
    - Do predictions
    - Save predictions in a .xlsx file


How to use:
    
    - Run 1 time for each index


"""""""""""""""

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
#from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import MQLoss, MSE, MAE
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import random
#from neuralforecast.utils import AirPassengers, AirPassengersPanel, AirPassengersStatic

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def qlike(y_true, y_pred):
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)


# global seed
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


ticker = "^RUT"   # change to "^GSPC, ^DJI" or "^RUT"
file_path = "Data Indexes.xlsx"




# read sheets
df_vol = pd.read_excel(file_path, sheet_name="YZ_volatility")
df_wass = pd.read_excel(file_path, sheet_name="wasserstein_dists")

# Convert dates
df_vol['Date'] = pd.to_datetime(df_vol['Date'])
df_wass['Date'] = pd.to_datetime(df_wass['Date'])

# keep Date + ticker
df_vol = df_vol[['Date', ticker]].rename(columns={ticker: 'RV'})

# Merge on date
df = pd.merge(df_vol, df_wass, on='Date', how='inner')




# choose date interval
start_date = "2000-02-01"
end_date   = "2022-03-02"

# sort
df = df.sort_values('Date').set_index('Date')

# filter interval
df = df.loc[start_date:end_date]


windows_batch_size_param=64   # 64, 256, 512 - use 64 to find params and 60 for predictions
val_check_steps_param=100      # 25, 50,... - use an inferior nr than epoch

df.reset_index(inplace=True)
df.rename(columns={'Date': 'ds', 'RV': 'y', 'wasserstein_dists_2D': 'wasserstein_dists'}, inplace=True)

df["unique_id"] = "S&P500"

df = df.dropna(subset=["y", "wasserstein_dists"])



# 80/20 split - 25% of the training sample used for validation (60-80% of the entire sample)
n_total = len(df)
n_80 = int(n_total * 0.80)
n_60 = int(n_total * 0.60)

validation_length = n_80 - n_60
test_length = n_total - n_80

hp_search_data = df.iloc[:n_80] 

print(f"Total: {n_total} | Treino(60%): {n_60} | Validação(20%): {validation_length} | Teste Final(20%): {test_length}")

# HYPERPARAMETER SPACE
n_inputs_space          = [5, 10, 21, 63, 84, 189, 252]
#n_inputs_space          = [63]
mlp_units_space         = [[[712, 712], [712, 712]], [[512, 512], [512, 512]], [[250, 250], [250, 250]], [[100, 100], [100, 100]]]
epochs_space            = [25, 50, 100, 150, 250, 350, 450, 550, 750]
lr_space                = [0.0005, 0.0001, 0.00005, 0.00001]
num_lr_decays_space     = [5, 3, 2, 1]
dropout_space           = [0, 0.2, 0.3, 0.4, 0.5]
scaler_space            = ["robust", "standard", "minmax"]
stack_space            = [['identity', 'identity']]
#stack_space             = [['identity', 'identity'], ['trend', 'identity'], ['seasonality', 'identity'], ['trend', 'seasonality']]
n_harmonics_space       = [0, 0, 1, 1]
n_blocks_space          = [[1,1], [2,2], [3,3], [5,5]]
n_polynomials_space     = [0, 1, 0, 1]
loss_space              = [MSE(), MAE(), MQLoss(level=[90]), MQLoss(level=[80,90]), MQLoss(level=[95]), MQLoss(level=[75])]





trials = 45

results_nbeats = []


for trial in range(trials):
    i = random.randrange(len(n_inputs_space))
    h = random.randrange(len(mlp_units_space))
    a = random.randrange(len(n_blocks_space))
    k = random.randrange(len(epochs_space))
    l = random.randrange(len(stack_space))
    n = random.randrange(len(lr_space))
    m = random.randrange(len(loss_space))
    o = random.randrange(len(scaler_space))
    p = random.randrange(len(num_lr_decays_space))
    q = random.randrange(len(n_harmonics_space))
    r = random.randrange(len(n_polynomials_space))
    seed_val = random.randint(1, 129228148)

    model = NBEATS(
        h=5,
        input_size=n_inputs_space[i],

        loss=loss_space[m],
        scaler_type=scaler_space[o],

        learning_rate=lr_space[n],
        num_lr_decays=num_lr_decays_space[p],

        stack_types=stack_space[l],
        n_blocks=n_blocks_space[a],
        mlp_units=mlp_units_space[h],

        n_harmonics=n_harmonics_space[q],
        n_polynomials=n_polynomials_space[r],

        windows_batch_size=windows_batch_size_param,
        #windows_batch_size=256,
        #windows_batch_size=512,
        val_check_steps=val_check_steps_param,

        max_steps=epochs_space[k],

        random_seed=seed_val
    )


    fcst = NeuralForecast(models=[model], freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
    forecasts = fcst.cross_validation(
        df=hp_search_data,  
        val_size=5,
        #static_df=AirPassengersStatic,
        n_windows=None,
        test_size=validation_length - (validation_length % 5), 
        step_size=5
    )
    forecasts = forecasts.dropna()
    if "NBEATS-median" not in forecasts.columns:
        Y_hat = forecasts["NBEATS"].values
    else:
        Y_hat = forecasts["NBEATS-median"].values
    Y_true = forecasts["y"].values

    # Y_hat = np.maximum(Y_hat, 1e-8)
    
    RMSE = rmse(Y_true, Y_hat)
    QLIKE = qlike(Y_true, Y_hat)

    results_nbeats.append({
        "trial": m,
        "input_size": n_inputs_space[i],
        "mlp_units": mlp_units_space[h],
        "stack_types": stack_space[l],
        "n_blocks": n_blocks_space[a],
        "n_harmonics": n_harmonics_space[q],
        "n_polynomials": n_polynomials_space[r],
        "learning_rate": lr_space[n],
        "num_lr_decays": num_lr_decays_space[p],
        "epochs": epochs_space[k],
        "loss": m,
        "scaler_type": scaler_space[o],
        "dropout": dropout_space[p],
        "seed": seed_val,
        "RMSE": RMSE,
        "QLIKE": QLIKE
    })


# BEST HYPERPARAMETERS
results_nbeats_sorted = sorted(results_nbeats, key=lambda x: (x["QLIKE"], x["RMSE"]))
best = results_nbeats_sorted[0]

print("\n" + "="*70)
print("HIPERPARÂMETROS ÓTIMOS (25% AMOSTRA, H=5):")
print("="*70)
print(f"Trial {best['trial']}: input_size={best['input_size']}, loss={best['loss']}, scaler={best['scaler_type']}")
print(f"RMSE: {best['RMSE']:.6f}, QLIKE: {best['QLIKE']:.6f}")



# 20 trials just for the seed, using the optimal parameters until now
n_seeds = 20
seeds = [random.randint(1, 129228148) for _ in range(n_seeds)]
seed_results = []

for seed_val in seeds:
    model = NBEATS(
        h=5,
        input_size=best["input_size"],
        loss=loss_space[best["loss"]],
        scaler_type=best["scaler_type"],
        learning_rate=best["learning_rate"],
        stack_types=best["stack_types"],
        n_blocks=best["n_blocks"],
        mlp_units=best["mlp_units"],
        windows_batch_size=windows_batch_size_param,
        #windows_batch_size=256,
        #windows_batch_size=512,
        num_lr_decays=best["num_lr_decays"],
        val_check_steps=val_check_steps_param,
        n_harmonics=best["n_harmonics"],
        n_polynomials=best["n_polynomials"],
        max_steps=best["epochs"],
        random_seed=seed_val,
    )

    fcst = NeuralForecast(models=[model], freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
    forecasts = fcst.cross_validation(
        df=hp_search_data,  
        val_size=5,
        #static_df=AirPassengersStatic,
        n_windows=None,
        test_size=validation_length - (validation_length % 5),
        step_size=5
    )
    forecasts = forecasts.dropna()
    if "NBEATS-median" not in forecasts.columns:
        Y_hat = forecasts["NBEATS"].values
    else:
        Y_hat = forecasts["NBEATS-median"].values
    Y_true = forecasts["y"].values

   # Y_hat = np.maximum(Y_hat, 1e-8)
    RMSE = rmse(Y_true, Y_hat)
    QLIKE = qlike(Y_true, Y_hat)

    seed_results.append({"seed": seed_val, "RMSE": RMSE, "QLIKE": QLIKE})

# Best seed
seed_sorted = sorted(seed_results, key=lambda x: (x["QLIKE"], x["RMSE"]))
best_seed = seed_sorted[0]
print(f"\nBest seed out of 20: {best_seed['seed']} (QLIKE={best_seed['QLIKE']:.6f})")




# FINAL MODEL
final_model = NBEATS(
    h=1,  # ou h=5 
    input_size=best["input_size"],
    loss=loss_space[best["loss"]],
    scaler_type=best["scaler_type"],
    learning_rate=best["learning_rate"],
    #stack_types=['identity','identity'],
    stack_types=best["stack_types"],
    n_blocks=best["n_blocks"],
    mlp_units=best["mlp_units"],
    windows_batch_size=60,
    num_lr_decays=best["num_lr_decays"],
    val_check_steps=val_check_steps_param,
    n_harmonics=best["n_harmonics"],
    n_polynomials=best["n_polynomials"],
    max_steps=best["epochs"],
    random_seed=best_seed["seed"],
)

fcst_final = NeuralForecast(models=[final_model], freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
forecasts_final = fcst_final.cross_validation(
    df=df,   # use the whole series
    val_size=5,
    #static_df=AirPassengersStatic,
    n_windows=None,
    test_size=test_length - (test_length % 1), 
    step_size=1                                # onde day prediction
)
forecasts_final.dropna(inplace=True)

if "NBEATS-median" not in forecasts_final.columns:
    Y_hat_f = forecasts_final["NBEATS"].values
else:
    Y_hat_f = forecasts_final["NBEATS-median"].values
Y_true_f = forecasts_final["y"].values


#Y_hat_f = np.maximum(Y_hat_f, 1e-8)

rmse_final = rmse(Y_true_f, Y_hat_f)
ql_final = qlike(Y_true_f, Y_hat_f)
mae_final = mae(Y_true_f, Y_hat_f)

print("\n" + "="*70)
print("FINAL RESULTS (80% training, 20% test, H=1):")
print("="*70)
print(f"RMSE: {rmse_final:.6f} ({rmse_final*100:.3f}%)")
print(f"QLIKE: {ql_final:.6f} ({ql_final*100:.2f}%)")
print(f"MAE: {mae_final:.6f} ({mae_final*100:.3f}%)")


print("\n\nOPTIMAL HYPERPARAMETERS NBEATS (25% sample, H=1):")
print(f"input_size={best['input_size']}, loss={best['loss']}, scaler={best['scaler_type']}, learning_rate={best['learning_rate']}, stack_types=[identity,identity], n_blocks={best['n_blocks']}, mlp_units={best['mlp_units']}, windows_batch_size=60, num_lr_decays={best['num_lr_decays']}, val_check_steps={val_check_steps_param}, n_harmonics={best['n_harmonics']}, n_polynomials={best['n_polynomials']}, max_steps={best['epochs']}, random_seed={best_seed['seed']}")

print(forecasts_final["ds"].min())
print(forecasts_final["ds"].max())


Data = {'Date': forecasts_final["ds"],
        'Forecast without PH': Y_hat_f,
        }

#print(f"\nTotal OOS sample: {len(Y_hat_f)}") 1112


df1=pd.DataFrame(data=Data)
df1.to_excel("NBEATS forecasts (author) " + ticker + " " + start_date + "-" + end_date + ".xlsx", index=False)
