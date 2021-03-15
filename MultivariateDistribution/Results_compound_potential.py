# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:38:59 2020

This script plots the co-occurence rates

@author: acn980
"""
import calendar
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import os,sys,glob
sys.path.insert(0,r'E:\github\seasonality_risk\Functions')
from Functions_HCMC import detrend_fft, remove_NaN_skew, ax_joint_mm, extract_MM, joint_mm_all_cooc, plot_cooc_CI
np.set_printoptions(precision=15)
#%%
save = False

fn_trunk = 'E:/surfdrive/Documents'
fn = os.path.join(fn_trunk, 'Master2019\Thomas\data\matlab_csv')

fn_files = 'Master2019/Thomas/data'
fn2 = os.path.join(fn_trunk,fn_files)

lag_joint = 0 #days
#%% We import the monthly data AND the dates
fn_tide = os.path.join(fn,'Tide_WACC_VungTau_Cleaned_Detrended_Strict_sel_const.csv')
date_parser = lambda x: pd.datetime.strptime(x, "%d-%m-%Y %H:%M:%S")
tide = pd.read_csv(fn_tide, parse_dates = True, date_parser= date_parser, index_col = 'Date')
tide.rename(columns = {tide.columns[0]:'tide'}, inplace = True)
all_tide = tide.resample('M').max()
tide_day = tide.resample('D').max()

# Importing monthly data - rainfall
allfiles = glob.glob(os.path.join(fn2, 'NewRain\TRENDS\MONTH_CORRECTED', 'Thiessen_*.csv'))
all_rain = pd.DataFrame(data=None)
for file in allfiles:
    month = pd.read_csv(file, index_col = 'Year', parse_dates=True)  
    month.rename(columns={month.columns[0]:'Thiessen'}, inplace = True)
    all_rain = pd.concat([all_rain, month], axis = 0)

#Importing the monthly data surge
allfiles = glob.glob(os.path.join(fn2, 'NewSurge\TRENDS\MONTH_RAW', 'skew_fft_*.csv'))
all_skew = pd.DataFrame(data=None)
for file in allfiles:
    month = pd.read_csv(file, index_col = 'Year', parse_dates=True)  
    all_skew = pd.concat([all_skew, month], axis = 0) 

fn_skew  = os.path.join(fn,'skew_WACC_VungTau_Cleaned_Detrended_Strict_sel_const.csv')
skew = pd.read_csv(fn_skew, parse_dates = True, date_parser= date_parser, index_col = 'Date')
skew.rename(columns = {skew.columns[0]:'skew'}, inplace = True)
skew = remove_NaN_skew(skew)
skew_day = skew.resample('D').max()
skew_detrend = detrend_fft(skew_day, fillnavalue=0, frequency = 1./(2*365), figure_plotting =0)
skew_detrend_day = skew_detrend.resample('D').max()

sealevel = pd.concat([tide_day, skew_detrend_day], axis = 1)
sealevel['sealevel'] = sealevel.iloc[:,0]+sealevel.iloc[:,1]
sealevel = pd.DataFrame(sealevel['sealevel'])
dates_MM_sl = extract_MM(sealevel, freq='MS', label='sealevel')
all_seasevel = dates_MM_sl.drop('sealevel_date', axis = 1)

fn_out_ori = os.path.join(fn2,'NewSurge','TRENDS','DATES_MONTH_RAW', "skew_fft.csv")
dates_MM_skew = pd.read_csv(fn_out_ori, index_col='index')
dates_MM_skew.dropna(inplace = True)
dates_MM_skew['skew_date'] = [pd.datetime.strptime(x, "%Y-%m-%d") for x in dates_MM_skew['skew_date']]
dates_MM_skew['index'] = [pd.to_datetime(date(d.year, d.month, calendar.monthrange(d.year, d.month)[-1])) for d in dates_MM_skew['skew_date']]
dates_MM_skew.set_index('index',inplace = True)
dates_MM_skew = pd.concat([dates_MM_skew, all_skew], axis = 1).drop('skew', axis=1).copy()
dates_MM_skew.rename(columns={'skew_fft':'skew'}, inplace = True)

fn_out_ori = os.path.join(fn2,'NewRain','TRENDS','DATES_MONTH_RAW',"Thiessen_sum.csv")
dates_MM_rain = pd.read_csv(fn_out_ori, index_col='index')
dates_MM_rain.dropna(inplace = True)
dates_MM_rain['Thiessen_sum_date'] = [pd.datetime.strptime(x, "%Y-%m-%d") for x in dates_MM_rain['Thiessen_sum_date']]
for i in dates_MM_rain.index:
    if dates_MM_rain.loc[i,'Thiessen_sum'] == 0 and dates_MM_rain.loc[i,'Thiessen_sum_date'].day == 1:
        try:
            new_date = pd.to_datetime(date(dates_MM_rain.loc[i,'Thiessen_sum_date'].year, dates_MM_rain.loc[i,'Thiessen_sum_date'].month, np.random.randint(1,32)))
        except:
            new_date = pd.to_datetime(date(dates_MM_rain.loc[i,'Thiessen_sum_date'].year, dates_MM_rain.loc[i,'Thiessen_sum_date'].month, np.random.randint(1,32)))    
        dates_MM_rain.loc[i,'Thiessen_sum_date'] = new_date
dates_MM_rain['index'] = [pd.to_datetime(date(d.year, d.month, calendar.monthrange(d.year, d.month)[-1])) for d in dates_MM_rain['Thiessen_sum_date']]
dates_MM_rain.set_index('index',inplace = True)
dates_MM_rain = pd.concat([dates_MM_rain, all_rain], axis = 1).drop('Thiessen_sum', axis=1).copy()
dates_MM_rain.rename(columns={'Thiessen_sum_date':'Thiessen_date'}, inplace = True)
res = dates_MM_rain[~(dates_MM_rain['Thiessen_date'] < '1980-01-01')]
res2 = res[~(res['Thiessen_date'] > '2017-12-31')]
dates_MM_rain = res2.copy()
del res, res2

dates_MM_tide = extract_MM(tide_day, freq='MS', label='tide')
res = dates_MM_tide[~(dates_MM_tide['tide_date'] < '1980-01-01')]
res2 = res[~(res['tide_date'] > '2017-12-31')]
dates_MM_tide = res2.copy()
del res, res2
#%% We calculate the number of co-occurence per month
f,ax = plt.subplots(nrows=1, ncols = 3, figsize=(8,3), sharey=True) #, wspace=0.40)
ax = ax.reshape(-1)

# SKEW-RAIN
Joint_MM = ax_joint_mm(dates_MM_skew, dates_MM_rain, ax=None, label='_date', lag_joint=lag_joint, ls=7, formatting = False, plotting=False)
Joint_MM = Joint_MM.where(Joint_MM.Thiessen>0).dropna()
result_pair = joint_mm_all_cooc(Joint_MM, max_lag = 7, label = '_date')
plot_cooc_CI(result_pair, ax[0], lag_joint = lag_joint, c = 'g', size = 7, label = '(a) skew surge and rainfall', background = True)

# TIDE-RAIN
Joint_MM2 = ax_joint_mm(dates_MM_tide, dates_MM_rain, ax=None, label='_date', lag_joint=lag_joint, ls=7, formatting = False, plotting=False)
Joint_MM2 = Joint_MM2.where(Joint_MM2.Thiessen>0).dropna()
result_pair2 = joint_mm_all_cooc(Joint_MM2, max_lag = 7, label = '_date')
plot_cooc_CI(result_pair2, ax[1], lag_joint =lag_joint, c = 'r', size = 7, label = '(b) tide and rainfall', background = True)

# SL-RAIN
Joint_MM3 = ax_joint_mm(dates_MM_sl, dates_MM_rain, ax=None, label='_date', lag_joint=lag_joint, ls=7, formatting = False, plotting=False)
Joint_MM3 = Joint_MM3.where(Joint_MM3.Thiessen>0).dropna()
result_pair3 = joint_mm_all_cooc(Joint_MM3, max_lag = 7, label = '_date')
plot_cooc_CI(result_pair3, ax[2], lag_joint =lag_joint, c = 'y', size = 7, label = '(c) sea level and rainfall', background = True)

ax[0].set_ylabel('Nb. of co-occurring monthly maxima', fontsize = 8)
#plt.grid()
if save == True:
    fn_out = os.path.join(fn_trunk, 'Paper\Paper5\FIGURES', 'Bivariate', 'Cooc_MM_lag_{}.png'.format(lag_joint))
    f.savefig(fn_out, frameon=False, dpi = 300,papertype= 'a4', orientation='portrait', constrained_layout=True) #transparent=True, 
    plt.close()