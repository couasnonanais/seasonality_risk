# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:07:20 2020

This scripts tests for trends in monthly skew surge and correct

@author: acn980
"""

import glob, os, sys
import pandas as pd
import numpy as np
import warnings
import pymannkendall as pmk
import matplotlib.pyplot as plt

sys.path.insert(0,r'E:\github\seasonality_risk\Functions')
from Functions_HCMC import remove_NaN_skew, detrend_fft, select_epoch_pairs

warnings.filterwarnings("ignore")
#%% Setting the files and folder correctly
save = True
fn_trunk = 'E:/surfdrive/Documents'
fn_files = 'Master2019/Thomas/data'
fn = os.path.join(fn_trunk,fn_files)
#%%
fn_skew  = os.path.join(fn,'matlab_csv','skew_WACC_VungTau_Cleaned_Detrended_Strict_sel_const.csv')
date_parser = lambda x: pd.datetime.strptime(x, "%d-%m-%Y %H:%M:%S")
skew = pd.read_csv(fn_skew, parse_dates = True, date_parser= date_parser, index_col = 'Date')
skew.rename(columns = {skew.columns[0]:'skew'}, inplace = True)
skew = remove_NaN_skew(skew)
month_skew = skew.resample('M').max()
month_skew.rename(columns={month_skew.columns[0]:'current_skew'}, inplace = True) ######

skew_day = skew.resample('D').max()
skew_detrend = detrend_fft(skew_day, fillnavalue=0, frequency = 1. /(2*365), figure_plotting = 1)
month_skew_detrend = skew_detrend.resample('M').max()
month_skew_detrend.rename(columns={month_skew_detrend.columns[0]:'skew_fft'}, inplace = True) #######

##Exporting the dates of the monthly maxima
sel_sel = pd.concat([skew_detrend,skew_detrend], axis = 1)
dates_rainfall_value_raw = select_epoch_pairs(sel_sel, epoch = 'MS', nbofdays = 5, nbofrepet = 500, test_ind = False)
dates_rainfall_value_raw = dates_rainfall_value_raw[0].reset_index(drop=True).iloc[:,[0,-1]]
if save == True:
    fn_out_ori = os.path.join(fn,'NewSurge','TRENDS','DATES_MONTH_RAW', "skew_fft.csv")
    dates_rainfall_value_raw.to_csv(fn_out_ori, index_label = 'index')

fn_skew_gesla  = os.path.join(fn,'matlab_csv','skew_max_gesla.csv')
date_parser = lambda x: pd.datetime.strptime(x, "%d-%m-%Y %H:%M:%S")
skew_gesla = pd.read_csv(fn_skew_gesla, parse_dates = True, date_parser= date_parser, index_col = 'Date')
skew_gesla.rename(columns = {skew_gesla.columns[0]:'skew_gesla'}, inplace = True)
skew_gesla = remove_NaN_skew(skew_gesla)

month_skew_gesla = skew_gesla.resample('M').max()
month_skew_gesla.rename(columns={month_skew_gesla.columns[0]:'skew_gesla'}, inplace = True) #######

plt.figure()
plt.plot(month_skew_gesla.index, month_skew_gesla)
plt.plot(month_skew.index, month_skew)
plt.plot(month_skew_detrend.index, month_skew_detrend)
plt.show()

res_trend = pd.DataFrame(index = ['Trend','Tau','intercept','slope','p-value'])
res_trend2 = pd.DataFrame(index = ['Trend','Tau','intercept','slope','p-value'])
res_trend3 = pd.DataFrame(index = ['Trend','Tau','intercept','slope','p-value'])
res_trend4 = pd.DataFrame(index = ['Trend','Tau','intercept','slope','p-value'])

month_rain_max = month_skew_detrend.copy()
data = month_rain_max.columns[0]
print(data)

res_trend = pd.DataFrame(index = ['Trend','Tau','intercept','slope','p-value'])
res_trend2 = pd.DataFrame(index = ['Trend','Tau','intercept','slope','p-value'])
res_trend3 = pd.DataFrame(index = ['Trend','Tau','intercept','slope','p-value'])
res_trend4 = pd.DataFrame(index = ['Trend','Tau','intercept','slope','p-value'])   
for month, monthdat in month_rain_max.groupby(by=[month_rain_max.index.month]):
    print(month)
    
    #Save the data
    if save == True:
        fn_out_ori = os.path.join(fn,'NewSurge','TRENDS','MONTH_RAW', data+'_'+str(month)+".csv")
        monthdat.to_csv(fn_out_ori, index = True, index_label = 'Year')
          
    res1 = pmk.hamed_rao_modification_test(monthdat.values, alpha=0.05)       #We apply the Hamed and Rao modified test
    res2 = pmk.yue_wang_modification_test(monthdat.values, alpha=0.05)        #We apply the Yue and Wang modified test
    res3 = pmk.trend_free_pre_whitening_modification_test(monthdat.values, alpha=0.05) 
    res4 = pmk.pre_whitening_modification_test(monthdat.values, alpha=0.05)
      
    res_trend.loc['slope',month] = res1.slope
    res_trend.loc['intercept',month] = res1.intercept
    res_trend.loc['Trend',month] = res1.trend
    res_trend.loc['Tau',month] = res1.Tau
    res_trend.loc['p-value',month] = res1.p
    
    res_trend2.loc['slope',month] = res2.slope
    res_trend2.loc['intercept',month] = res2.intercept
    res_trend2.loc['Trend',month] = res2.trend
    res_trend2.loc['Tau',month] = res2.Tau
    res_trend2.loc['p-value',month] = res2.p
    
    res_trend3.loc['slope',month] = res3.slope
    res_trend3.loc['intercept',month] = res3.intercept
    res_trend3.loc['Trend',month] = res3.trend
    res_trend3.loc['Tau',month] = res3.Tau
    res_trend3.loc['p-value',month] = res3.p
    
    res_trend4.loc['slope',month] = res4.slope
    res_trend4.loc['intercept',month] = res4.intercept
    res_trend4.loc['Trend',month] = res4.trend
    res_trend4.loc['Tau',month] = res4.Tau
    res_trend4.loc['p-value',month] = res4.p
           
    if save == True:
        fn_out = os.path.join(fn,'NewSurge','TRENDS','MONTH', data+"_max_MKmod_Hamed_Rao.csv")
        res_trend.to_csv(fn_out, index= True, index_label = 'index')

        fn_out = os.path.join(fn,'NewSurge','TRENDS','MONTH', data+"_max_MKmod_Yue_Wang.csv")
        res_trend2.to_csv(fn_out, index= True, index_label = 'index')
        
        fn_out = os.path.join(fn,'NewSurge','TRENDS','MONTH', data+"_max_MKmod_Yue_Wang2002.csv")
        res_trend3.to_csv(fn_out, index= True, index_label = 'index')
        
        fn_out = os.path.join(fn,'NewSurge','TRENDS','MONTH', data+"_max_MKmod_Yue_Pilon.csv")
        res_trend4.to_csv(fn_out, index= True, index_label = 'index')

                   
#%% ANALYZING THE RESULTS
allfiles = glob.glob(os.path.join(fn,'NewSurge','TRENDS','MONTH','*.csv'))
res = pd.DataFrame(data=None)
#case = 'year_max'
res = {}
name_stations = list()
name_cases = list()
for files in allfiles:
    print(files)
    station = files.split('MONTH\\')[-1].split('_max')[0]
    name_stations.append(station)
    
    MK_type = files.split('MONTH\\')[-1].split('_')[-1].split('.csv')[0]
    name_cases.append(MK_type)
    
name_stations = set(name_stations)
name_cases = set(name_cases)
for station in name_stations:
    res[station] = {}
    res[station]['is_trend'] = pd.DataFrame(data=None, index = np.arange(1,13,1), columns = ['Rao','Pilon','Wang2002','Wang'] )
    res[station]['slope'] = pd.DataFrame(data=None, index = np.arange(1,13,1), columns = ['Rao','Pilon','Wang2002','Wang'] )
    res[station]['tau'] = pd.DataFrame(data=None, index = np.arange(1,13,1), columns = ['Rao','Pilon','Wang2002','Wang'] )
    res[station]['p-value'] = pd.DataFrame(data=None, index = np.arange(1,13,1), columns = ['Rao','Pilon','Wang2002','Wang'] )
    res[station]['intercept'] = pd.DataFrame(data=None, index = np.arange(1,13,1), columns = ['Rao','Pilon','Wang2002','Wang'] )


for files in allfiles:
    print(files)
    station = files.split('MONTH\\')[-1].split('_max')[0]    
    MK_type = files.split('MONTH\\')[-1].split('_')[-1].split('.csv')[0]

    data = pd.read_csv(files, index_col = 'index')
    res[station]['is_trend'].loc[:,MK_type] = data.iloc[0,:].values
    res[station]['slope'].loc[:,MK_type] = data.loc['slope',:].astype(float).values
    res[station]['tau'].loc[:,MK_type] = data.loc['Tau',:].astype(float).values
    res[station]['p-value'].loc[:,MK_type] = data.loc['p-value',:].astype(float).values
    res[station]['intercept'].loc[:,MK_type] = data.loc['intercept',:].astype(float).values

#%% Detrend time series

#We base our analysis on Yue and Wang (2004)
fn2 = os.path.join(fn,'NewSurge')
allfiles = glob.glob(os.path.join(fn,'NewSurge','TRENDS','MONTH_RAW', "*.csv"))

test_name = 'Wang2002'
for file in allfiles:
    print(file)
    month = int(file.split('MONTH_RAW\\')[-1].split('_')[-1].split('.csv')[0])
    station = file.split('MONTH_RAW\\')[-1].split('.csv')[0].split(f'_{month}')[0]
    
    data = pd.read_csv(file, index_col = 'Year', parse_dates=True)                     

    fn_out_corr = os.path.join(fn,'NewSurge','TRENDS','MONTH_CORRECTED', station+'_'+str(month)+"_corr.csv")
    if res[station]['is_trend'].loc[month, test_name] == 'no trend':
        if save == True:
            data.to_csv(fn_out_corr, index = True, index_label = 'Year')
    else:
        trend_line = np.arange(len(data)) * res[station]['slope'].loc[month, 'Wang'] + res[station]['intercept'].loc[month, 'Wang']
        trend_line = pd.Series(trend_line)
        rawdata = data.reset_index(drop = True).iloc[:,0]
        corr_monthdat = rawdata - trend_line + trend_line.iloc[-1] #Corrected to last year            
        corrected = pd.DataFrame(data = corr_monthdat.values, index = data.index, columns = [station])
                    
        f = plt.figure()
        plt.plot(data.index.year, data, 'ok')
        plt.plot(data.index.year, trend_line, '-b')
        plt.plot(data.index.year, corr_monthdat, 'ob')
        plt.annotate('Slope = {:.2f}'.format(res[station]['slope'].loc[month, 'Wang']), xy = (0.85, 0.95), xycoords = 'axes fraction', size=8)
        plt.annotate('p-value = {:.4f}'.format(res[station]['p-value'].loc[month, 'Wang']), xy = (0.85, 0.90), xycoords = 'axes fraction', size=8)
        plt.xlabel('Year')
        plt.ylabel('Monthly daily maxima - mm/day')
        plt.title('{} - Month {}'.format(station, str(month)))
        
        if save == True:
            fn_out = os.path.join(fn_trunk, 'Paper\Paper5\FIGURES', 'Univariate','Trend', test_name+'_'+station+'_'+str(month)+'.png')
            f.savefig(fn_out, bbox_inches = 'tight', frameon=False, dpi = 300) #transparent=True, 
            plt.close()
            
            corrected.to_csv(fn_out_corr, index = True, index_label = 'Year')