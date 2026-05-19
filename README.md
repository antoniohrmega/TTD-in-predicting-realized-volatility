# Using Topological Tail Dependency in predicting Realized Volatility

This is a reproduction of Hugo Souto's work ("Topological tail dependence: Evidence from forecasting realized volatility").
The files in this repository allow you to see how TDA (Topological Data Analysis) affects the prediction of volatility. I recommend you do it for a basket of at least 3 indexes. I personally used Standard & Poor's 500 (^GSPC), Dow Jones Industrial Average (^DJI) and Russell 2000 (^RUT).



How to use:

After choosing the indexes and time period, you should run the files in the following order:

- ph_rv_algorythm.py
- har_and_harx_models.py
- harst_week_and_ph_models.py
- nbeats_final.py
- nbeatsx_final.py
- tests_clean.py

You only need the output of the last file in order to compare the models. I recommend paying attention to the names given to each file by the codes; also check in tests_clean.py what files you're choosing to read the forecasts from.
Each .py has a short description of what it does in the beggining.

Attention:

Due to lack of data (only daily was available), a modification was made to Yang and Zhang's Realized Volatility proxy. In case you have access to higher frequency data, I also leave in this repository a RV_functions.py with functions (H. Souto and Amir Moradi's work), choose which one's best for you.
