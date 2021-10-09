import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from scipy import interpolate  #For interpolation
import calendar
import statsmodels.tsa.stattools as smt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime

import matplotlib.dates as mdate
# ############get the data and clean them##################
file_directory = "F:/mit-micromaster/machinelearning/release_time_series_report_data/"
df_co2 = pd.read_csv((file_directory + "CO2.csv"),header=[0, 1, 2], comment='"')  #skip the comment part in csv file,
# print(df_co2.head())  #default 5 rows
# print(df_co2.columns)  # hearder by 3 rows
# print(type(df_co2.columns))  #<class 'pandas.core.indexes.multi.MultiIndex'>
# print(df_co2.columns[4])   #('     CO2', '        ', '   [ppm]')
# print(df_co2.columns[1])   #(' Mn', '   ', '   ')
# print(df_co2.columns[0])   #('  Yr', '    ', '    ')
# print(df_co2[('     CO2', '        ', '   [ppm]')])   #show column value
df_co2_clean = df_co2.drop(df_co2[df_co2[('     CO2', '        ', '   [ppm]')]==-99.99].index)  #clean the data
# print(df_co2_clean)

##############split data into train data and eveluate data################
split_data = 4/5
# perm = np.random.permutation(df_co2_clean.size)
n_train= int(split_data * df_co2_clean.shape[0])   #734*4/5=587
# print(n_train)
F_train_data = df_co2_clean.iloc[0:n_train,4].values   # get the co2 value
t_train_year = df_co2_clean.iloc[0:n_train,0].values  #get the year
t_train_month = df_co2_clean.iloc[0:n_train,1].values

t_train_data = (t_train_year-1958) + t_train_month/12.0

#######fit model##################
reg_linear = LinearRegression().fit(t_train_data.reshape(-1,1), F_train_data) #fitting a model requires requires a 2D arra
# print(reg_linear.get_params())
# print(reg_linear.coef_)
# print(reg_linear.intercept_)
#
# plt.plot(t_train_data,F_train_data-reg_linear.intercept_- reg_linear.coef_* t_train_data)
# plt.show()
######predict data############
F_test_data = df_co2_clean.iloc[n_train:,4].values   # get the co2 value
t_test_year = df_co2_clean.iloc[n_train:,0].values   #get the month
t_test_month = df_co2_clean.iloc[n_train:,1].values

t_test_data = (t_test_year-1958)  + t_test_month/12.0

# F_predict_data = reg_linear.predict(t_test_data.reshape(-1,1))
# MEAN = mean_squared_error(F_test_data,F_predict_data)
# print(MEAN)
#
# MAPE = mean_absolute_percentage_error(F_test_data, F_predict_data)
# print(MAPE)
#
# #####################
# #quadratic model
model = np.poly1d(np.polyfit(t_train_data,F_train_data,2))
# print(model)
# plt.plot(t_train_data,F_train_data- model(t_train_data))
# plt.show()

F_predict_quad = model(t_test_data)
MEAN_quad = mean_squared_error(F_test_data,F_predict_quad)
print(np.sqrt(MEAN_quad))
MAPE_quad = mean_absolute_percentage_error(F_test_data,F_predict_quad)
print(MAPE_quad)
#
# ############find periodic element###########
# # by training data
df_train_data = df_co2_clean.iloc[0:n_train,:]
F_train_predict = model(t_train_data)
df_train_data['residual'] = F_train_data-F_train_predict    #add a column residual
# print(df_train_data)
residual_month = df_train_data.groupby((' Mn', '   ', '   '))['residual'].mean() #get average by month
print(residual_month)    #-0.012919,0.646407

# df_train_data = df_co2_clean.iloc[0:n_train,:]
# #
# model = np.poly1d(np.polyfit(df_train_data.index/12.0,F_train_data,2))
# # print(model)
# F_train_predict = model(df_train_data.index/12.0)
# df_train_data['residual'] = F_train_data-F_train_predict    #add a column residual
# # # print(df_train_data)
# residual_month = df_train_data.groupby((' Mn', '   ', '   '))['residual'].mean() #get average by month
# print(residual_month)    #

###################
#plot periodic signals for evey month

# x = np.arange(1, 13,1)
# y = residual_month
# f = interpolate.interp1d(x, y,kind='cubic')
# plt.title(' periodic signal Pi',fontsize=12)
# plt.plot(x, y, color='r',markerfacecolor='blue',marker='o')
# plt.xlabel('month',fontsize=10)
# plt.ylabel('P',fontsize=10)
# new_ticks = np.linspace(1, 12, 12)
# plt.xticks(new_ticks)
# plt.show()

# x = np.arange(1, 13,1)
# y = residual_month
# f = interpolate.interp1d(x, y,kind='quadratic')
# x_new = list(np.arange(1,12,0.1))
# y_new = f(x_new)
# plt.title(' periodic signal Pi',fontsize=12)
# plt.plot(x, y, 'ro',x_new,y_new,'b-')
# plt.xlabel('month',fontsize=10)
# plt.ylabel('P',fontsize=10)
# new_ticks = np.linspace(1, 12, 12)
# plt.xticks(new_ticks)
# plt.xticks(x, calendar.month_name[1:13],color='blue',rotation=60)
# plt.show()
##################################
#plot Fi

#########################
#plot Fi + Pi
# print(type(residual_month))  #<class 'pandas.core.series.Series'>
residual = residual_month.values  #get series value,switch to numpy array
print(residual[0])   #-0.012919218906268511
######
#test data
F_test_predict = model(t_test_data)
F_predict = np.concatenate((F_train_predict,F_test_predict))
Ci_train =  np.zeros(t_train_data.shape[0]+t_test_data.shape[0])
#
for i in range(F_predict.shape[0]):
    Ci_train[i] = F_predict[i] + residual[i%12]
#
plt.title(' final model  on top of the entire time series',fontsize=12)
plt.xlabel('time series ti',fontsize=10)
plt.ylabel(' Fn(ti)+Pi',fontsize=10)
split = t_train_data.shape[0]
plt.plot(t_train_data,Ci_train[0:split],label='train data')
plt.plot(t_test_data,Ci_train[split:],label='test data')
plt.legend(loc='best')
plt.show()

##smoothing the curve???
# f = interpolate.interp1d((t_train_data+t_test_data), Ci_train,kind='quadratic')
# x_new = np.arange(0,t_train_data+t_test_data,0.1)
# y_new = f(x_new)
# plt.plot(t_train_data+t_test_data,Ci_train,x_new,y_new)
#
# plt.show()

#####RMSE & MAPE
# F_original = np.concatenate((F_train_data,F_test_data))  #ALL TIME SERIES
# MEAN_quad_all = mean_squared_error(F_original,Ci_train)
# print(np.sqrt(MEAN_quad_all))
# MAPE_quad_all = mean_absolute_percentage_error(F_original,Ci_train)
# print(MAPE_quad_all)

 #TEST DATA
MEAN_quad_all = mean_squared_error(F_test_data,Ci_train[split:])
print(np.sqrt(MEAN_quad_all))
MAPE_quad_all = mean_absolute_percentage_error(F_test_data,Ci_train[split:])
print(MAPE_quad_all)

###ratio
# F_max = np.max(F_predict)
# F_min = np.min(F_predict)
# F_range = F_max - F_min
#
# P_max = np.max(residual)
# P_min = np.min(residual)
# P_amplitude = (P_max - P_min)/2.0
#
# F_P_ratio = F_range/P_amplitude
# print(F_P_ratio)

# R_residual = np.zeros(F_train_data.shape[0])
# for i in range(F_train_data.shape[0]):
#     R_residual[i] = F_train_data[i] - (F_train_predict[i]+residual[i%12])
#
# R_max = np.max(R_residual)
# R_min = np.min(R_residual)
# R_range = R_max - R_min
#
# P_R_ratio = P_amplitude/R_range
# print(P_R_ratio)
# plt.plot(t_train_data,R_residual)
# plt.title(' Residual in time series',fontsize=12)
# plt.xlabel('time series ti',fontsize=10)
# plt.ylabel('R',fontsize=10)
# plt.show()
# #####################
# #cubic model
# # model = np.poly1d(np.polyfit(t_train_data,F_train_data,3))
# # print(model)
# # plt.plot(t_train_data,F_train_data- model(t_train_data))
# # plt.show()
# #
# # F_predict_quad = model(t_test_data)
# # MEAN_quad = mean_squared_error(F_test_data,F_predict_quad)
# # print(MEAN_quad)
# # MAPE_quad = mean_absolute_percentage_error(F_test_data,F_predict_quad)
# # print(MAPE_quad)


######################################
#######BPP data analysis，question 2
##################################
# df_CPI = pd.read_csv((file_directory + "PriceStats_CPI.csv"))  #skip the comment part in csv file,
# # print(df_CPI)
# df_CPI['month'] =  pd.DatetimeIndex(df_CPI['date']).month    #add two columns ,year,month
# df_CPI['year'] =  pd.DatetimeIndex(df_CPI['date']).year
# print(df_CPI)
#
# ####################################
# ############all the data#########
# df_CPI_AVE = df_CPI.groupby(['year','month'])  #combine year and month to get the first data point
# df_CPI_select  = df_CPI_AVE.first()
# # print(df_CPI_select)
# t_data_time = np.arange(0,df_CPI_select.shape[0],1)  #len(df) to get the rows,len(df.columns) to get column
# CPI_data= df_CPI_select['CPI']
#
# plt.plot(t_data_time,CPI_data)
# plt.show()
#
# ##################################
# ####train data
# #
# df_CPI_train = df_CPI[(df_CPI['date'] <'2013-09-01')]
# df_CPI_train_combine = df_CPI_train.groupby(['year','month'])  #combine year and month to get the first data point
# df_CPI_train_select  = df_CPI_train_combine.first()
# t_train_time = np.arange(len(df_CPI_train_select))
# CPI_train_data = df_CPI_train_select['CPI'].values
# # print(df_CPI_train)
# # print(df_CPI_train_select)
# ##validation data
# # df_CPI_validation = df_CPI[(df_CPI['date'] >='2013-09-01')]
# # df_CPI_validation_combine = df_CPI_validation.groupby(['year','month'])  #combine year and month to get the first data point
# # df_CPI_validation_select  = df_CPI_validation_combine.first()
# # CPI_validation_data = df_CPI_validation_select['CPI'].values
# # t_validation_time = np.arange(len(df_CPI_train_select),(len(df_CPI_validation_select)+len(df_CPI_train_select)))
#
# #linear regression
# reg_linear = LinearRegression().fit(t_train_time.reshape(-1,1),CPI_train_data) #fitting a model requires requires a 2D arra
# # print(reg_linear.get_params())
# # print(reg_linear.coef_)
# # print(reg_linear.intercept_)
#
# # plt.plot(t_train_time,CPI_train_data)
# # plt.show()
#
# CPI_train_predict = reg_linear.predict(t_train_time.reshape((-1,1)))
# R_train = CPI_train_data - CPI_train_predict
# plt.title('Detrend train data', fontsize=14)
# plt.xlabel('t_train', fontsize=12)
# plt.ylabel('detrend data', fontsize=12)
# plt.plot(t_train_time,R_train)
# plt.show()
# # R_train = CPI_train_data - reg_linear.coef_*t_train_time -reg_linear.intercept_
# print(np.max(np.abs(R_train)))

####AR model##################
#
#find the order P for AR model by autocorrelation
# plt.rc("figure", figsize=(10, 7))
# # Plot ACF of CPI
# sm.graphics.tsa.plot_acf(CPI_train_data, lags=20)
# plt.xlabel('Lags', fontsize=12)
# plt.ylabel('Autocorrelation', fontsize=12)
# plt.title('Autocorrelation ACF', fontsize=14)
# plt.show()
#
# plt.rc("figure", figsize=(10, 7))
# # Plot PACF of CPI
# sm.graphics.tsa.plot_pacf(CPI_train_data, lags=20)
# plt.xlabel('Lags', fontsize=12)
# plt.ylabel('Autocorrelation', fontsize=12)
# plt.title('Autocorrelation PACF', fontsize=14)
# plt.show()

###find the parameters
# AR_model_R = AutoReg((R_train),lags=2,trend='c')
# AR_model_fit_R = AR_model_R.fit()
# print('Coefficients R: %s' % AR_model_fit_R.params)
#
# # ####predict
# df_CPI_test = df_CPI[('2019-10-02'> df_CPI['date']) &(df_CPI['date'] >'2013-08-31')]
# df_CPI_test_combine = df_CPI_test.groupby(['year','month'])  #combine year and month to get the first data point
# df_CPI_test_select  = df_CPI_test_combine.first()
# t_test_time = np.arange(len(df_CPI_train_select),len(df_CPI_train_select)+len(df_CPI_test_select))
# CPI_test_data = df_CPI_test_select['CPI'].values
# T_predictions= reg_linear.predict(t_test_time.reshape(-1,1))  #get linear trend prediction
# R_test = CPI_test_data - T_predictions
# R_entire = np.concatenate((R_train,R_test))  #combine twonumpy array, DO NOT FORGET ((),INDEX=0/1/NONE)# print(df_CPI_test)
# AR_model = AutoReg(R_entire,lags=2,trend='c')
# # print(df_CPI_test_select)
# #initialize model on entire residuals
# # AR_model_fit = AR_model.fit()
# # print('Coefficients: %s' % AR_model_fit.params)
# # # print(AR_model_fit.summary())
#
# AR_predictions = AR_model.predict(AR_model_fit_R.params,start=len(df_CPI_train_select),end=len(df_CPI_train_select)+len(df_CPI_test_select)-1,dynamic=False)
# test_predictions = AR_predictions + T_predictions
# rmse = np.sqrt(mean_squared_error(CPI_test_data,test_predictions))
# print('test RMSE:%.8f'% rmse)
#
# ##############plot predict and original data in different P
# train_predictions = CPI_train_predict + AR_model_fit_R.predict(start=0,end=len(CPI_train_predict)+1,dynamic=False)
# t_time = np.concatenate((t_train_time,t_test_time))  #all data x
# CPI_data = np.concatenate((CPI_train_data,CPI_test_data))
# CPI_predict = np.concatenate((train_predictions,test_predictions))
#
# plt.title('The CPI plot in lags=2', fontsize=14)
# plt.plot(t_time,CPI_data,'b',label='real')
# plt.plot(t_time,CPI_predict,'r',label ='prediction')
# plt.legend()
# plt.show()
#
# plt.title('The validation set plot in lags=2', fontsize=14)
# plt.plot(t_test_time,CPI_test_data,'b',label='real')
# plt.plot(t_test_time,test_predictions,'r',label ='prediction')
# plt.legend()
# plt.show()

######lag=1
# AR_model_R_lag1 = AutoReg((R_train),lags=1,trend='c')
# AR_model_fit_R_lag1 = AR_model_R_lag1.fit()
# print('Coefficients R of lag1: %s' % AR_model_fit_R_lag1.params)
# AR_model_lag1 = AutoReg(R_entire,lags=1,trend='c')
#
# AR_predictions_lag1 = AR_model_lag1.predict(AR_model_fit_R_lag1.params,start=len(df_CPI_train_select),end=len(df_CPI_train_select)+len(df_CPI_test_select)-1,dynamic=False)
# test_predictions_lag1 = AR_predictions_lag1 + T_predictions
# rmse_lag1 = np.sqrt(mean_squared_error(CPI_test_data,test_predictions_lag1))
# print('test RMSE of lag1:%.8f'% rmse_lag1)
#
# AR_model_R_lag3 = AutoReg((R_train),lags=3,trend='c')
# AR_model_fit_R_lag3 = AR_model_R_lag3.fit()
# print('Coefficients R of lag3: %s' % AR_model_fit_R_lag3.params)
# AR_model_lag3 = AutoReg(R_entire,lags=3,trend='c')
#
# AR_predictions_lag3 = AR_model_lag3.predict(AR_model_fit_R_lag3.params,start=len(df_CPI_train_select),end=len(df_CPI_train_select)+len(df_CPI_test_select)-1,dynamic=False)
# test_predictions_lag3 = AR_predictions_lag3 + T_predictions
# rmse_lag3 = np.sqrt(mean_squared_error(CPI_test_data,test_predictions_lag3))
# print('test RMSE of lag3:%.8f'% rmse_lag3)

# plt.title('The validation set plot in lags=1', fontsize=14)
# plt.plot(t_test_time,CPI_test_data,'b',label='real')
# plt.plot(t_test_time,test_predictions_lag1,'r',label ='prediction')
# plt.legend()
# plt.show()

#######test lags and rmse
# n=10
# rmse_lag = []
# test_predictions_lag = np.zeros((n,len(df_CPI_test_select)))
#
# for i in range(0,n):
#     AR_model_R_lag = AutoReg((R_train), lags=i+1, trend='c')
#     AR_model_fit_R_lag = AR_model_R_lag.fit()
#     print('Coefficients R of lag1: %s' % AR_model_fit_R_lag.params)
#     AR_model_lag = AutoReg(R_entire, lags=i+1, trend='c')
#
#     AR_predictions_lag = AR_model_lag.predict(AR_model_fit_R_lag.params, start=len(df_CPI_train_select),
#                                                 end=len(df_CPI_train_select) + len(df_CPI_test_select) - 1,
#                                                 dynamic=False)
#     test_predictions_lag[i][:] = AR_predictions_lag + T_predictions
#
#     rmse_lag.append(np.sqrt(mean_squared_error(CPI_test_data, test_predictions_lag[i][:])))

# for i in range(0,n):
#     print('test RMSE of lag %d is %.8f' %(i+1,rmse_lag[i]))
#
# plt.title('The validation set plot in different lags', fontsize=14)
# plt.plot(t_test_time,CPI_test_data,'y-',label=' true value')
# plt.plot(t_test_time,test_predictions_lag1,'r',label ='lag1')
# plt.plot(t_test_time,test_predictions,'g',label ='lag2')
# plt.plot(t_test_time,test_predictions_lag3,'b',label ='lag3')
# plt.legend()
# plt.show()

#######plot lags and rmse
# plt.title('rmse in different lags', fontsize=14)
# plt.plot(np.arange(1,n+1),rmse_lag,'ro',np.arange(1,n+1),rmse_lag,'b')
# plt.xlabel('lags',fontsize=10)
# plt.ylabel('rmse',fontsize=10)
# plt.show()

##########################################
########Converting to Inflation Rates, 3 WAYS
# CPI_Jan_2013 = df_CPI['CPI'][df_CPI['date'] == '2013-01-01'].values
# CPI_Feb_2013 = df_CPI['CPI'][df_CPI['date'] == '2013-02-01'].values
#
# IR_Feb_2013 = (CPI_Feb_2013-CPI_Jan_2013)/CPI_Jan_2013
# print(IR_Feb_2013)
#
# IR_Feb = np.log(CPI_Feb_2013) - np.log(CPI_Jan_2013)
# print(IR_Feb)

# ####IR array for all of the data
# IR_CPI = np.zeros(CPI_data.shape[0])
# IR_AR = np.zeros(CPI_data.shape[0])
# CPI_log = np.log(CPI_data)
# IR_PS = np.zeros(CPI_data.shape[0])
# IR_BER = np.zeros(CPI_data.shape[0])
# PS_train = df_CPI_train_combine.mean()['PriceStats']
# PS_validation = df_CPI_test_combine.mean()['PriceStats']
# PS_data = np.concatenate((PS_train,PS_validation))
# # print(PS_data.shape[0])
#
# #####BER inflation rate
# df_BER_year_day = pd.read_csv((file_directory + "T10YIE.csv"))  #BER file
# df_BER_year_day['month'] =  pd.DatetimeIndex(df_BER_year_day['DATE']).month    #add two columns ,year,month
# df_BER_year_day['year'] =  pd.DatetimeIndex(df_BER_year_day['DATE']).year
# df_BER_year_month = df_BER_year_day.groupby(['year','month']).mean()  # index by year and month, value is mean
# df_BER_year_month['BER_month'] = np.power((df_BER_year_month['T10YIE']/100.0 +1),1/12) -1   #dateframe has 1 column
# # print(df_BER_year_month.columns.values)  #T10YIE
# # print(df_BER_year_month)
# # print(df_BER_year_month.loc[(2013,2):(2019,7)]['BER_month'])  #get the column's value by index
#
#
# for i in range(1,CPI_data.shape[0]):        #CPI
#     IR_CPI[i] = CPI_log[i]-CPI_log[i-1]
#     IR_AR[i] = (CPI_predict[i] - CPI_predict[i-1])/CPI_predict[i-1]
#     IR_PS[i] = (PS_data[i] - PS_data[i-1])/PS_data[i-1]
# start = t_train_time.shape[0]
# plt.title('monthly inflation rate', fontsize=14)
# plt.plot(t_test_time,IR_CPI[start:]*100,'b',label='CPI inflation')
# plt.plot(t_test_time,IR_AR[start:]*100,'r',label='AR predcit inflation')
# plt.plot(t_test_time,IR_PS[start:]*100,'g',label='PriceStats inflation')
# plt.plot(t_test_time,df_BER_year_month.loc[(2013,9):(2019,10)]['BER_month']*100,'black',label='BER inflation')
# plt.legend()
# plt.xlabel('month',fontsize=10)
# plt.ylabel('inflation rate %',fontsize=10)
# plt.show()

# plt.title('monthly inflation rate by log', fontsize=14)
# plt.plot(t_time,IR,'b')
# plt.xlabel('month',fontsize=10)
# plt.ylabel('inflation rate',fontsize=10)
# plt.show()

#################################
#get FEB,2013 IR

# PS_Jan_2013 = df_CPI['PriceStats'][df_CPI['date'] == '2013-01-31'].values
# PS_Feb_2013 = df_CPI['PriceStats'][df_CPI['date'] == '2013-02-28'].values
#
# IR_Feb_2013_PS = (PS_Feb_2013-PS_Jan_2013)/PS_Jan_2013
# print('IR last day:%f'%IR_Feb_2013_PS)
#
# PS_Feb_2013_Ave = df_CPI['PriceStats'][df_CPI['date'].str.contains('2013-02-')].values
# PS_Jan_2013_Ave = df_CPI['PriceStats'][df_CPI['date'].str.contains('2013-01-')].values
#
# IR_Feb_2013_AVE = (np.mean(PS_Feb_2013_Ave) - np.mean(PS_Jan_2013_Ave))/np.mean(PS_Jan_2013_Ave)
# print('IR average month:%f'%IR_Feb_2013_AVE)
# ############
# def AVE_RATE(a):
#     size = a.shape[0]
#     ave_rate_freq = 0
#     for i in range(0,size-1):
#         if a[i] != 0:
#             ave_rate_freq += (a[i+1] - a[i]) /a[i]
#
#     return ave_rate_freq/(size-1)
# print(np.insert(PS_Feb_2013_Ave,0,106.3605))
#
# Daily_Feb_2013 = AVE_RATE(np.insert(PS_Feb_2013_Ave,0,106.3605))
# print('IR daily:%f'% (Daily_Feb_2013*30))
#
# CPI_PS = smt.ccf(IR_CPI,IR_PS,adjusted=False)
# print(np.argmax(CPI_PS))
# print(np.argmax(np.abs(CPI_PS)))
#
# plt.plot(smt.ccf(IR_CPI,IR_PS,adjusted=False))
# plt.title('CPI versus PriceStats inflation rate', fontsize=14)
# plt.xlabel('lag',fontsize=10)
# plt.ylabel('CCF',fontsize=10)
# plt.show()
# #
# plt.bar(range(len(CPI_PS)), CPI_PS, fc="blue")
# plt.show()
#
# plt.title("CPI versus PriceStats inflation rate")
# plt.xcorr(IR_CPI[start:], IR_PS[start:], normed=True, usevlines=True, maxlags=73)
# plt.show()
#
# CPI_BER = smt.ccf(IR_CPI,df_BER_year_month.loc[(2008,7):(2019,10)]['BER_month'],adjusted=False)
# print(np.argmax(CPI_BER))
# plt.plot(smt.ccf(IR_CPI,df_BER_year_month.loc[(2008,7):(2019,10)]['BER_month'],adjusted=False))
# plt.title('CPI versus BER inflation rate', fontsize=14)
# plt.xlabel('lag',fontsize=10)
# plt.ylabel('CCF',fontsize=10)
# plt.show()
# #
# plt.xcorr(IR_CPI, df_BER_year_month.loc[(2008,7):(2019,10)]['BER_month'], normed=True, usevlines=True, maxlags=73)
# plt.show()

########it a new AR model to the CPI inflation rate
#let's check order(p,d,q)
# original series
# fig, axes = plt.subplots(3, 2)
# axes[0, 0].plot(IR_CPI[0:start])
# axes[0, 0].set_title('Original Series')
# plot_acf(IR_CPI[0:start], ax=axes[0, 1])
# # one derivative
# axes[1, 0].plot(np.diff(IR_CPI[0:start]))
# axes[1, 0].set_title('1st Order Differencing')
# IR_CPI_train_1st = np.diff(IR_CPI[0:start],n=1)
# IR_CPI_train_1st_NA = IR_CPI_train_1st[np.logical_not(np.isnan(IR_CPI_train_1st))]
# plot_acf(IR_CPI_train_1st_NA, ax=axes[1, 1])
# # second derivative
# axes[2, 0].plot(np.diff(IR_CPI[0:start],n=2))
# axes[2, 0].set_title('2nd Order Differencing')
# IR_CPI_train_2nd = np.diff(IR_CPI[0:start],n=2)
# IR_CPI_train_2nd_NA = IR_CPI_train_2nd[np.logical_not(np.isnan(IR_CPI_train_2nd))]
# plot_acf(IR_CPI_train_2nd_NA, ax=axes[2, 1])
# plt.show()
#
# fig, axes = plt.subplots(1, 2)
# plot_pacf(IR_CPI[0:start],ax=axes[0],title='partial autocorrelation of original series')
# plot_pacf(IR_CPI_train_1st_NA, ax=axes[1],title='partial autocorrelation of 1st Order Differencing')
# plt.show()
#TRAIN SET FOR IR_CPI
#d=0,p=1,q=0,1,(p,d,q)

# train_size = t_train_time.shape[0]
# whole_size = IR_CPI.shape[0]
#
# index = pd.period_range(start='2008-07-01', periods=IR_CPI.shape[0], freq='M')
# all_observation = pd.Series(IR_CPI[:]*100,index=index)
# train_observation = all_observation[:train_size]
# model_ARIMA_0 = SARIMAX(train_observation,trend='c',order=(1,0,1),seasonal_order=(1,0,1,4))
# model_ARIMA_fit_0 = model_ARIMA_0.fit(disp=False)
# print(type(index)) #[2008-07,2008-08,....]
# print(model_ARIMA_fit_0.params)
# print(model_ARIMA_fit_0.summary())
# forecast_month_ahead = np.zeros(whole_size-train_size)
# test_predictions_IR = np.zeros(whole_size-train_size)
# test_predictions_IR_all = np.zeros(whole_size)
# forecast_month_ahead_all = np.zeros(whole_size)
# count=0
#
# for i in range(train_size,whole_size):
#     # print(all_observation.iloc[i:i+1])
#     updated_observation = all_observation.iloc[i:i + 1]
#     model_ARIMA_fit_0 = model_ARIMA_fit_0.extend(updated_observation)
#     # # print(model_ARIMA_fit_0.params)
#     forecast_month_ahead[count] = (model_ARIMA_fit_0.forecast(1))
#     test_predictions_IR[count] = (model_ARIMA_fit_0.fittedvalues)
#     count+=1
# print(count)
#
# rmse_CPI_0 = np.sqrt(mean_squared_error(IR_CPI[start:],test_predictions_IR))
# print('test RMSE:%.8f'% rmse_CPI_0)
#
# rmse_CPI_predict = np.sqrt(mean_squared_error(IR_CPI[start:],forecast_month_ahead))
# print('month ahead test RMSE:%.8f'% rmse_CPI_0)
#
# print(all_observation[train_size:whole_size].index[1])
#
# plt.title('monthly inflation rate', fontsize=14)
# plt.plot(IR_CPI[start:]*100,'b',label='CPI inflation')
# plt.plot(test_predictions_IR[1:whole_size-train_size],'r',label='predict of validation set')
# plt.legend()
# plt.ylabel('inflation rate %',fontsize=10)
# plt.show()
#
# #improve
# count=0
# for i in range(train_size,whole_size):
#     history = all_observation[:i]
#     model_ARIMA_0 = SARIMAX(history, trend='c', order=(1, 0, 1))
#     model_ARIMA_fit_0 = model_ARIMA_0.fit(disp=False)
#     print(model_ARIMA_fit_0.params)
#
#     forecast_month_ahead[count] = (model_ARIMA_fit_0.forecast(1))
#     count+=1
#
# rmse_CPI_predict = np.sqrt(mean_squared_error(IR_CPI[start:],forecast_month_ahead))
# print('month ahead test RMSE:%.8f'% rmse_CPI_0)