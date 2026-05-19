"""
@author: antoniomega


What this file does:
    
    
    - Read all the files .xlsx with the predictions of the 6 models for a chosen - caution
      here when choosing the variables' names in order to read the right files
    - Run DM Tests in pairs for the models of the same family
    - Run MCS Test including all models
    - Print error measures for all models
    - Plot forecasts vs RV value for all 6 models, for a chosen index - for both the whole 
      test sample and COVID crash


How to use:
    
    - Run 1 time for each index. It is also possible to run everything but for the COVID
      crash period


"""""""""""""""

import pandas as pd
import numpy as np
from arch.bootstrap import MCS
import matplotlib.pyplot as plt


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def qlike(y_true, y_pred):
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

def run_mcs(losses, label):
    
    np.random.seed(42)
    print(f"\n=== MCS RESULTS ({label}) ===")
    
    block_size = int(len(losses)**(1/3))
    
    mcs = MCS(losses, size=0.05, method="R", block_size=block_size, reps=5000)
    mcs.compute()

    print("\nP-values:")
    print(mcs.pvalues.round(5))
    
    #pvals = pd.DataFrame(mcs.pvalues)
    #pvals = pvals.applymap(lambda x: f"{float(x):.4f}")
    
    #print(pvals)


    print("\nIncluded models:")
    print(mcs.included)

    print("\nExcluded models:")
    print(mcs.excluded)

def dm_test(actual_lst, pred1_lst, pred2_lst, h = 1, crit="MSE", power = 2):
    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt,msg)
        # Check the range of h
        if (h < 1):
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt,msg)
        len_act = len(actual_lst)
        len_p1  = len(pred1_lst)
        len_p2  = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return (rt,msg)
        # Check range of h
        if (h >= len_act):
            rt = -1
            msg = "The number of steps ahead is too large."
            return (rt,msg)
        # Check if criterion supported
        if (crit != "MSE" and crit != "QLIKE" and crit != "MAD" and crit != "poly"):
            rt = -1
            msg = "The criterion is not supported."
            return (rt,msg)
        # Check if every value of the input lists are numerical values
        from re import compile as re_compile
        comp = re_compile("^\d+?\.\d+?$")
        def compiled_regex(s):
            """ Returns True is string is a number. """
            if comp.match(s) is None:
                return s.isdigit()
            return True
        for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
            is_actual_ok = compiled_regex(str(abs(actual)))
            is_pred1_ok = compiled_regex(str(abs(pred1)))
            is_pred2_ok = compiled_regex(str(abs(pred2)))
            if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):
                msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                rt = -1
                return (rt,msg)
        return (rt,msg)

    # Error check
    error_code = error_check()
    # Raise error if cannot pass error check
    if (error_code[0] == -1):
        raise SyntaxError(error_code[1])
        return
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np

    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst  = []

    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()

    # Length of lists (as real numbers)
    T = float(len(actual_lst))

    # construct d according to crit
    if (crit == "MSE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((actual - p1)**2)
            e2_lst.append((actual - p2)**2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "QLIKE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((actual/p1-np.log(actual/p1)-1))
            e2_lst.append((actual/p2-np.log(actual/p2)-1))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(((actual - p1))**(power))
            e2_lst.append(((actual - p2))**(power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)

    # Mean of d
    mean_d = pd.Series(d_lst).mean()

    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0,h):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    DM_stat=V_d**(-0.5)*mean_d
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)

    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')

    rt = dm_return(DM = DM_stat, p_value = p_value)

    return rt

def avaliar_modelo(df, nome_coluna, label):

    temp = df[['RV', nome_coluna]].dropna()
    act = temp['RV'].values
    pred = temp[nome_coluna].values
    
    print(f"Medidas de erro {label} (N={len(temp)})")
    print(f"RMSE:  {rmse(act, pred):.6f}")
    print(f"MAE:   {mae(act, pred):.6f}")
    print(f"QLIKE: {qlike(act, pred):.6f}\n")


ticker = "^GSPC"   # change to "^GSPC, ^DJI" or "^RUT"

# choose date interval
start_date = "2000-02-01"
end_date   = "2022-03-02"


file_har_and_harx = "HAR & HARX forecasts " + ticker + " " + start_date + "-" + end_date + ".xlsx"
file_harst_week_ph = "HARST Week & PH forecasts " + ticker + " " + start_date + "-" + end_date + ".xlsx"
file_nbeats = "NBEATS forecasts " + ticker + " " + start_date + "-" + end_date + ".xlsx"
file_nbeatsx = "NBEATSx-PH forecasts " + ticker + " " + start_date + "-" + end_date + ".xlsx"
file_actuals = "Dados indices.xlsx"


df_har = pd.read_excel(file_har_and_harx)
df_har = df_har.rename(columns={
    "Forecast without PH": "HAR",
    "Forecast with PH": "HARX"
})

df_harst = pd.read_excel(file_harst_week_ph)
df_harst = df_harst.rename(columns={
    "Forecast without PH": "HARST_week",
    "Forecast with PH": "HARST_PH"
})

df_nbeats = pd.read_excel(file_nbeats)
df_nbeats = df_nbeats.rename(columns={"Forecast without PH": "NBEATS"})


df_nbeatsx = pd.read_excel(file_nbeatsx)
df_nbeatsx = df_nbeatsx.rename(columns={"Forecast with PH": "NBEATSx"})

df_actuals = pd.read_excel(file_actuals, sheet_name="YZ_volatility")
df_actuals = df_actuals[['Date', ticker]].rename(columns={ticker: 'RV'})


df = df_actuals.copy()

for d in [df_nbeats, df_nbeatsx, df_har, df_harst]:
    df = df.merge(d, on="Date", how="left")
    
    
# COVID crash period evaluation- coment to evaluate full sample
#df["Date"] = pd.to_datetime(df["Date"])

#df = df[
#    (df["Date"] >= "2020-02-20") &
#    (df["Date"] <= "2020-05-21")
#].copy()    
    
    
    
    
# DM tests - done in pairs   
    
df_pair = df[['RV', 'HAR', 'HARX']].dropna()    
    
    
print("\n DM TESTS (HAR vs HARX) " + ticker + " " + start_date + "-" + end_date)
for crit in ["MSE", "MAD", "QLIKE"]:
    dm = dm_test(df_pair['RV'], df_pair['HAR'], df_pair['HARX'], h=1, crit=crit)
    print(f"DM {crit}: statistic={dm.DM:.4f}, p-value={dm.p_value:.4f} ({'***' if dm.p_value<0.01 else '**' if dm.p_value<0.05 else '*' if dm.p_value<0.10 else ''})")

    
df_pair = df[['RV', 'HARST_week', 'HARST_PH']].dropna()    
    
    
print("\n DM TESTS (HARST-Week vs HARST-PH) " + ticker + " " + start_date + "-" + end_date)
for crit in ["MSE", "MAD", "QLIKE"]:
    dm = dm_test(df_pair['RV'], df_pair['HARST_week'], df_pair['HARST_PH'], h=1, crit=crit)
    print(f"DM {crit}: statistic={dm.DM:.4f}, p-value={dm.p_value:.4f} ({'***' if dm.p_value<0.01 else '**' if dm.p_value<0.05 else '*' if dm.p_value<0.10 else ''})")


df_pair = df[['RV', 'NBEATS', 'NBEATSx']].dropna()    
    
    
print("\n DM TESTS (NBEATS vs NBEATSx-PH) " + ticker + " " + start_date + "-" + end_date)
for crit in ["MSE", "MAD", "QLIKE"]:
    dm = dm_test(df_pair['RV'], df_pair['NBEATS'], df_pair['NBEATSx'], h=1, crit=crit)
    print(f"DM {crit}: statistic={dm.DM:.4f}, p-value={dm.p_value:.4f} ({'***' if dm.p_value<0.01 else '**' if dm.p_value<0.05 else '*' if dm.p_value<0.10 else ''})")
    



# MCS - all together
  
    
print("\n MCS TESTS (HAR, HARST, NBEATS) " + ticker + " " + start_date + "-" + end_date)

df_mcs = df.dropna(subset=["HAR", "HARX", "HARST_week", "HARST_PH", "NBEATS", "NBEATSx"])  


losses_mse = pd.DataFrame()

losses_mse["HAR"] = (df_mcs["RV"] - df_mcs["HAR"])**2
losses_mse["HARX"] = (df_mcs["RV"] - df_mcs["HARX"])**2
losses_mse["HARST_week"] = (df_mcs["RV"] - df_mcs["HARST_week"])**2
losses_mse["HARST_PH"] = (df_mcs["RV"] - df_mcs["HARST_PH"])**2
losses_mse["NBEATS"] = (df_mcs["RV"] - df_mcs["NBEATS"])**2
losses_mse["NBEATSx"] = (df_mcs["RV"] - df_mcs["NBEATSx"])**2

run_mcs(losses_mse, "MSE")

losses_mae = pd.DataFrame()

losses_mae["HAR"] = np.abs(df_mcs["RV"] - df_mcs["HAR"])
losses_mae["HARX"] = np.abs(df_mcs["RV"] - df_mcs["HARX"])
losses_mae["HARST_week"] = np.abs(df_mcs["RV"] - df_mcs["HARST_week"])
losses_mae["HARST_PH"] = np.abs(df_mcs["RV"] - df_mcs["HARST_PH"])
losses_mae["NBEATS"] = np.abs(df_mcs["RV"] - df_mcs["NBEATS"])
losses_mae["NBEATSx"] = np.abs(df_mcs["RV"] - df_mcs["NBEATSx"])

run_mcs(losses_mae, "MAE")





df_ql = df_mcs.copy()


cols = ["RV","HAR","HARX","HARST_week","HARST_PH","NBEATS","NBEATSx"]
for c in cols:
    df_ql = df_ql[df_ql[c] > 0]

losses_qlike = pd.DataFrame()

losses_qlike["HAR"] = df_ql["RV"]/df_ql["HAR"] - np.log(df_ql["RV"]/df_ql["HAR"]) - 1
losses_qlike["HARX"] = df_ql["RV"]/df_ql["HARX"] - np.log(df_ql["RV"]/df_ql["HARX"]) - 1
losses_qlike["HARST_week"] = df_ql["RV"]/df_ql["HARST_week"] - np.log(df_ql["RV"]/df_ql["HARST_week"]) - 1
losses_qlike["HARST_PH"] = df_ql["RV"]/df_ql["HARST_PH"] - np.log(df_ql["RV"]/df_ql["HARST_PH"]) - 1
losses_qlike["NBEATS"] = df_ql["RV"]/df_ql["NBEATS"] - np.log(df_ql["RV"]/df_ql["NBEATS"]) - 1
losses_qlike["NBEATSx"] = df_ql["RV"]/df_ql["NBEATSx"] - np.log(df_ql["RV"]/df_ql["NBEATSx"]) - 1

run_mcs(losses_qlike, "QLIKE")    
    
    
    
# Error measures

avaliar_modelo(df, "HAR", "HAR")
avaliar_modelo(df, "HARX", "HARX")
avaliar_modelo(df, "HARST_week", "HARST-Week")
avaliar_modelo(df, "HARST_PH", "HARST-PH")
avaliar_modelo(df, "NBEATS", "NBEATS")
avaliar_modelo(df, "NBEATSx", "NBEATSx-PH")




# Plots 

df['Date'] = pd.to_datetime(df['Date'])

models_config = [
    {"column": "HAR",        "title": "HAR"},
    {"column": "HARX",       "title": "HARX"},
    {"column": "HARST_week", "title": "HARST-week"},
    {"column": "HARST_PH",   "title": "HARST-PH"},
    {"column": "NBEATS",     "title": "NBEATS"},
    {"column": "NBEATSx",    "title": "NBEATSx-PH"}
]

# Loop to generate and open each plot in a completely separate window
for model in models_config:
    # Filter out NaNs for the specific model to keep the plot clean and continuous
    df_plot = df.sort_values('Date').dropna(subset=['RV', model['column']])
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(df_plot['Date'], df_plot['RV'], label='RV', color='gray', alpha=0.5, linewidth=1.5)
    
    plt.plot(df_plot['Date'], df_plot[model['column']], label='Forecast', color='#1f77b4', linewidth=1.5)
    
    plt.title(model['title'], fontsize=14, fontweight='bold') 
    plt.ylabel("RV", fontsize=12)                             
    plt.xlabel("")
    

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.xticks(rotation=45) # Rotate date labels slightly so they don't overlap
    
    # Adjust padding so labels, legends, or titles don't get clipped when saving
    plt.tight_layout()

# Render all 6 distinct windows on the screen simultaneously so you can save them one by one
plt.show()


    
    