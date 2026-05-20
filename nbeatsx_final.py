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
#from neuralforecast.models import NBEATS
from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import MQLoss, MSE, MAE
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import random

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





ticker = "^RUT"   # change to "^GSPC", "^DJI" or "^RUT"
file_path = "Data Indexes.xlsx"

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







windows_batch_size_param=64   # 64, 256, 512 - use 64 to find params and 60 to do predictions
val_check_steps_param=100

df.reset_index(inplace=True)
df.rename(columns={'Date': 'ds', 'RV': 'y', 'wasserstein_dists_2D': 'wasserstein_dists'}, inplace=True)
df["unique_id"] = "RUT"

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
mlp_units_space         = [[[712, 712], [712, 712]], [[512, 512], [512, 512]], [[250, 250], [250, 250]], [[100, 100], [100, 100]]]
epochs_space            = [25, 50, 100, 150, 250, 350, 450, 550, 750]
lr_space                = [0.0005, 0.0001, 0.00005, 0.00001]
num_lr_decays_space     = [5, 3, 2, 1]
dropout_space           = [0, 0.2, 0.3, 0.4, 0.5]
scaler_space            = ["robust", "standard", "minmax"]
stack_space            = [['identity', 'identity']]
#stack_space             = [['identity', 'identity'], ['trend', 'identity'], ['seasonality', 'identity'], ['trend', 'seasonality']]
n_harmonics_space       = [0, 0, 1, 1]
#n_harmonics_space       = [0]
n_blocks_space          = [[1,1], [2,2], [3,3], [5,5]]
n_polynomials_space     = [0, 1, 0, 1]
#n_polynomials_space     = [0]
loss_space              = [MSE(), MAE(), MQLoss(level=[90]), MQLoss(level=[80,90]), MQLoss(level=[95]), MQLoss(level=[75])]






trials = 45

# With PH 


results_nbeatsx = []


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

    model = NBEATSx(
        h=5,
        input_size=n_inputs_space[i],
        
        hist_exog_list=['wasserstein_dists'],

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
        n_windows=None,
        test_size=validation_length - (validation_length % 5),
        step_size=5
    )
    forecasts = forecasts.dropna()
    if "NBEATSx-median" not in forecasts.columns:
        Y_hat = forecasts["NBEATSx"].values
    else:
        Y_hat = forecasts["NBEATSx-median"].values
    Y_true = forecasts["y"].values

    #Y_hat = np.maximum(Y_hat, 1e-8)
    
    RMSE = rmse(Y_true, Y_hat)
    QLIKE = qlike(Y_true, Y_hat)

    results_nbeatsx.append({
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
results_nbeatsx_sorted = sorted(results_nbeatsx, key=lambda x: (x["QLIKE"], x["RMSE"]))
best_ph = results_nbeatsx_sorted[0]

print("\n" + "="*70)
print("OPTIMAL HYPERPARAMETERS (25% of the sample, H=5):")
print("="*70)
print(f"Trial {best_ph['trial']}: input_size={best_ph['input_size']}, loss={best_ph['loss']}, scaler={best_ph['scaler_type']}")
print(f"RMSE: {best_ph['RMSE']:.6f}, QLIKE: {best_ph['QLIKE']:.6f}")



# 20 trials just for the seed, using the optimal parameters until now
n_seeds = 20
seeds = [random.randint(1, 129228148) for _ in range(n_seeds)]
seed_results_ph = []

for seed_val in seeds:
    model = NBEATSx(
        h=5,
        input_size=best_ph["input_size"],
        hist_exog_list=['wasserstein_dists'],
        loss=loss_space[best_ph["loss"]],
        scaler_type=best_ph["scaler_type"],
        learning_rate=best_ph["learning_rate"],
        stack_types=best_ph["stack_types"],
        n_blocks=best_ph["n_blocks"],
        mlp_units=best_ph["mlp_units"],
        windows_batch_size=windows_batch_size_param,
        #windows_batch_size=256,
        #windows_batch_size=512,
        num_lr_decays=best_ph["num_lr_decays"],
        val_check_steps=val_check_steps_param,
        n_harmonics=best_ph["n_harmonics"],
        n_polynomials=best_ph["n_polynomials"],
        max_steps=best_ph["epochs"],
        random_seed=seed_val,
    )

    fcst = NeuralForecast(models=[model], freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
    forecasts = fcst.cross_validation(
        df=hp_search_data, 
        val_size=5,
        n_windows=None,
        test_size=validation_length - (validation_length % 5),
        step_size=5
    )
    forecasts = forecasts.dropna()
    if "NBEATSx-median" not in forecasts.columns:
        Y_hat = forecasts["NBEATSx"].values
    else:
        Y_hat = forecasts["NBEATSx-median"].values
    Y_true = forecasts["y"].values

    #Y_hat = np.maximum(Y_hat, 1e-8)
    RMSE = rmse(Y_true, Y_hat)
    QLIKE = qlike(Y_true, Y_hat)

    seed_results_ph.append({"seed": seed_val, "RMSE": RMSE, "QLIKE": QLIKE})

# Melhor seed
seed_sorted_ph = sorted(seed_results_ph, key=lambda x: (x["QLIKE"], x["RMSE"]))
best_seed_ph = seed_sorted_ph[0]
print(f"\nBest seed out of 20: {best_seed_ph['seed']} (QLIKE={best_seed_ph['QLIKE']:.6f})")




# FINAL MODEL
final_model = NBEATSx(
    h=1,  # ou h=5 
    input_size=best_ph["input_size"], 
    hist_exog_list=['wasserstein_dists'],
    loss=loss_space[best_ph["loss"]],
    scaler_type=best_ph["scaler_type"],
    learning_rate=best_ph["learning_rate"],
    #stack_types=['identity','identity'],
    stack_types=best_ph["stack_types"],
    n_blocks=best_ph["n_blocks"],
    mlp_units=best_ph["mlp_units"],
    windows_batch_size=60,
    num_lr_decays=best_ph["num_lr_decays"],
    val_check_steps=val_check_steps_param,
    n_harmonics=best_ph["n_harmonics"],
    n_polynomials=best_ph["n_polynomials"],
    max_steps=best_ph["epochs"],
    random_seed=best_seed_ph["seed"],
)

fcst_final_ph = NeuralForecast(models=[final_model], freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
forecasts_final_ph = fcst_final_ph.cross_validation(
    df=df,   # usar a série completa
    val_size=5,
    n_windows=None,
    test_size=test_length - (test_length % 1),
    step_size=1                                # one day prediction
)
forecasts_final_ph.dropna(inplace=True)

if "NBEATSx-median" not in forecasts_final_ph.columns:
    Y_hat_f_ph = forecasts_final_ph["NBEATSx"].values
else:
    Y_hat_f_ph = forecasts_final_ph["NBEATSx-median"].values
Y_true_f_ph = forecasts_final_ph["y"].values


# Y_hat_f_ph = np.maximum(Y_hat_f_ph, 1e-8)

#epsilon = np.percentile(Y_true_f_ph, 1) * 0.1
#Y_hat_f_ph = np.maximum(Y_hat_f_ph, epsilon)


rmse_final_ph = rmse(Y_true_f_ph, Y_hat_f_ph)
ql_final_ph = qlike(Y_true_f_ph, Y_hat_f_ph)
mae_final_ph = mae(Y_true_f_ph, Y_hat_f_ph)



print("\n" + "="*70)
print("FINAL RESULTS PH (80% training, 20% test, H=1):")
print("="*70)
print(f"RMSE: {rmse_final_ph:.6f} ({rmse_final_ph*100:.3f}%)")
print(f"QLIKE: {ql_final_ph:.6f} ({ql_final_ph*100:.2f}%)")
print(f"MAE: {mae_final_ph:.6f} ({mae_final_ph*100:.3f}%)")


print("\n\nOPTIMAL HYPERPARAMETERS NBEATSx-PH (25% sample, H=1):")
print(f"input_size={best_ph['input_size']}, loss={best_ph['loss']}, scaler={best_ph['scaler_type']}, learning_rate={best_ph['learning_rate']}, stack_types=[identity,identity], n_blocks={best_ph['n_blocks']}, mlp_units={best_ph['mlp_units']}, windows_batch_size=60, num_lr_decays={best_ph['num_lr_decays']}, val_check_steps=100, n_harmonics={best_ph['n_harmonics']}, n_polynomials={best_ph['n_polynomials']}, max_steps={best_ph['epochs']}, random_seed={best_seed_ph['seed']}")


print(forecasts_final_ph["ds"].min())
print(forecasts_final_ph["ds"].max())


Data = {'Date': forecasts_final_ph["ds"],
        'Forecast with PH': Y_hat_f_ph
        }


df1=pd.DataFrame(data=Data)
df1.to_excel("NBEATSx-PH forecasts " + ticker + " " + start_date + "-" + end_date + ".xlsx", index=False)


