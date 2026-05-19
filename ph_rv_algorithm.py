#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antoniomega


What this file does:
    
    
    - Retrieve index data from yahoo finance
    - Compute log returns
    - Compute wasserstein dists
    - Save everything in a .xlsx file


How to use:
    
    - Run 1 time for each basket of indexes only. The Date will later be filtered in other files


"""""""""""""""
# This algorithm requires the installment of ripster library
# One can do so by !pip install ripser

#import yfinance as yf
import numpy as np
import pandas as pd

from ripser import Rips
import persim

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#warnings.filterwarnings("ignore", message=".*non-finite death times.*")

def yang_zhang_rv_daily(data, window=5):
    
    df = data.copy()
    
    # --- log components ---
    df["oi"] = np.log(df["Open"] / df["Close"].shift(1))
    df["ci"] = np.log(df["Close"] / df["Open"])
    df["ui"] = np.log(df["High"] / df["Open"])
    df["di"] = np.log(df["Low"] / df["Open"])
    
    df = df.dropna()
    
    # Rogers-Satchell
    df["RS"] = df["ui"]*(df["ui"] - df["ci"]) + df["di"]*(df["di"] - df["ci"])
    
    # rolling variances
    sigma_o2 = df["oi"].rolling(window).var()
    sigma_c2 = df["ci"].rolling(window).var()
    sigma_rs = df["RS"].rolling(window).mean()
    
    # coeficiente k
    n = window
    k = 0.34 / (1.34 + (n+1)/(n-1))
    
    # Yang-Zhang variance
    yz_var = sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs
    
    yz_vol = np.sqrt(yz_var)
    
    # Shift to use the center of the window
    yz_vol = yz_vol.shift(-window//2)
    
    return pd.DataFrame({
        "YZ_vol": yz_vol
    })




import yfinance as yf

# Using Yahoo Finance's API
# define index names: E.g. ^GSPC = S&P 500, ^DJI = DOW Jones, ^RUT = Russell 2000
index_names = ['^GSPC', '^DJI', '^RUT']

# define date range: E.g from 2000-01-01 until 2022-03-30
start_date_string = "2000-01-01"
end_date_string = "2026-04-17"

# pull data from yahoo finance
raw_data = yf.download(index_names, start=start_date_string, end=end_date_string)
print(raw_data.columns)
print(raw_data.head())

raw_data.to_excel("Raw Data Indexes.xlsx")




raw_data = pd.read_excel(
    "Raw Data Indexes.xlsx",
    header=[0,1],      # two heather lines
    index_col=0        # first row is date
)

raw_data.index = pd.to_datetime(raw_data.index)

print(raw_data.columns)
print(raw_data.head())

#df_close = raw_data['Close'].dropna(axis='rows')

df_close = raw_data['Close'][['^GSPC', '^DJI', '^RUT']]
df_close = df_close.dropna().sort_index()

# define array of adjusted closing prices
P = df_close.to_numpy()
# define array of log-returns defined as the log of the ratio between closing values of two subsequent days
r = np.log(np.divide(P[1:],P[:len(P)-1]))

# Instantiate Vietoris-Rips solver
#rips = Rips(maxdim = 2)
rips = Rips(maxdim=2, n_perm=0)

# some parameters
w = 22 # time window size - use 22 to align with 22 chosen for a month in HAR and HARST models
n = len(df_close)-(2*w) + 1 # number of time segments. Here a whole business month was chosen
print(len(df_close))
wasserstein_dists = np.zeros((n,1)) # initialize array for wasserstein distances

# compute wasserstein distances between persistence diagrams for subsequent time windows
for i in range(n):

    # Compute persistence diagrams for adjacent time windows
    dgm1 = rips.fit_transform(r[i:i+w])
    #if i==0:
    #    print(r[i:i+w])
    #    print(dgm1)
    dgm2 = rips.fit_transform(r[i+w:i+(2*w)])
    
    # Compute wasserstein distance between diagrams
    wasserstein_dists[i] = persim.wasserstein(dgm1[0], dgm2[0], matching=False)
    
# plot wasserstein distances over time if you wish. Here they are plotted together with S&P 500 scaled prices
plt.figure(figsize=(18, 8), dpi=80)
plt.rcParams.update({'font.size': 16})

plt.plot(raw_data.index[w:n+w],wasserstein_dists)
plt.plot(raw_data.index[w:n+w],df_close.iloc[w:n+w,0]/max(df_close.iloc[w:n+w,0]))
plt.legend(['wasserstein distances', 'S&P 500 (scaled)', 'Crash of 2020'])
plt.xlabel('Date')
plt.title('Homology Changes')
plt.show()

# Returns
log_returns = np.log(df_close / df_close.shift(1))
log_returns = log_returns.dropna()

# Save data
Data = {
    "Date": df_close.index[w:n+w],
    "wasserstein_dists": wasserstein_dists.reshape(len(wasserstein_dists))
}
wasserstein_df = pd.DataFrame(data=Data)

with pd.ExcelWriter("Data Indexes.xlsx") as writer:
    
    # sheet 1: log returns
    log_returns.to_excel(writer, sheet_name="log_returns")
    
    # sheet 2: wasserstein dists
    wasserstein_df.to_excel(writer, sheet_name="wasserstein_dists", index=False)
    
  
indexes = ['^GSPC', '^DJI', '^RUT']

yz_results = {}

for idx in indexes:
    
    df_idx = pd.DataFrame({
        "Open": raw_data['Open'][idx],
        "High": raw_data['High'][idx],
        "Low": raw_data['Low'][idx],
        "Close": raw_data['Close'][idx]
    }).dropna()
    
    yz_results[idx] = yang_zhang_rv_daily(df_idx)["YZ_vol"]

yz_df = pd.DataFrame(yz_results)

with pd.ExcelWriter("Data Indexes.xlsx", mode="a", engine="openpyxl") as writer:
    yz_df.to_excel(writer, sheet_name="YZ_volatility")
    


