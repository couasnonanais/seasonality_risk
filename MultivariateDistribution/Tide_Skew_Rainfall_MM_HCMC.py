# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:52:53 2020

This script plots the boxplots of the distributions 

@author: acn980
"""
import os, glob, sys
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
sys.path.insert(0,r'E:\github\seasonality_risk\Functions')
from Functions_HCMC import detrend_fft, remove_NaN_skew, extract_MM
from matplotlib.ticker import AutoMinorLocator
warnings.filterwarnings("ignore")

#%%
save = False

fn_trunk = 'E:/surfdrive/Documents'
fn = os.path.join(fn_trunk, 'Master2019\Thomas\data\matlab_csv')

fn_files = 'Master2019/Thomas/data'
fn2 = os.path.join(fn_trunk,fn_files)

lag_joint = 0 #days

#%% GESLA OR WACC
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

#%% Calculating range 
def month_statistics(tide_day):
    list_plot = []
    for month in tide_day.groupby(tide_day.index.month):
        list_plot.append(np.array(month[1].dropna().values).flatten())
    
    max_range = pd.Series(index = np.arange(1,13,1))
    coef_var = pd.Series(index = np.arange(1,13,1))
    medians = pd.Series(index = np.arange(1,13,1))
    i=1
    for month in list_plot:
        max_range[i] = np.quantile(month, 0.95) - np.quantile(month, 0.05)
        coef_var[i] = np.std(month)/np.mean(month)
        medians[i] = np.median(month)
        i +=1
    return max_range, coef_var, medians 

#%% Quantile plots of monthly maxima
month_label = ['Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 

fig, ax1 = plt.subplots(nrows=2, ncols = 2, gridspec_kw={'hspace':0.05}, sharex=True, figsize=(8, 4)) #height_ratios': [1,1,1,1]
ax1 = ax1.reshape(-1)

list_plot = []
for month in tide_day.groupby(tide_day.index.month):
    list_plot.append(np.array(month[1].dropna().values).flatten())
c = 'r'
ax1[0].boxplot(list_plot, whis=[5,95], showfliers=False, #sym = '.g',
   patch_artist=True, boxprops=dict(facecolor=c, color=c, alpha = 0.6), capprops=dict(color=c),
   whiskerprops=dict(color=c), flierprops=dict(color=c, markeredgecolor=c), medianprops=dict(color='k'))
ax1[0].set_ylabel('High tide (m)', size = 7, labelpad=0)
ax1[0].set_ylim(0.2,1.6)

#Adding the tide
list_plot = []
for month in all_skew.groupby(all_skew.index.month):
    list_plot.append(np.array(month[1].dropna().values).flatten())
c = 'g'
ax1[2].boxplot(list_plot, whis=[5,95], showfliers=False, #sym = '.g',
   patch_artist=True, boxprops=dict(facecolor=c, color=c, alpha = 0.6), capprops=dict(color=c),
   whiskerprops=dict(color=c), flierprops=dict(color=c, markeredgecolor=c), medianprops=dict(color='k'))
ax1[2].set_ylabel('Skew surge (m)', size = 7, labelpad=0)
 
list_plot = []
for month in all_seasevel.groupby(all_seasevel.index.month):
    list_plot.append(np.array(month[1].dropna().values).flatten())
c = 'y'
ax1[1].boxplot(list_plot, whis=[5,95], showfliers=False, #sym = '.r', 
   patch_artist=True, boxprops=dict(facecolor=c, color=c, alpha = 0.6), capprops=dict(color=c),
   whiskerprops=dict(color=c), flierprops=dict(color=c, markeredgecolor=c), medianprops=dict(color='k'))
ax1[1].set_ylabel('Sea level (m)', size = 7, labelpad=0)
ax1[1].set_ylim(0.8,1.6)

list_plot = []
for month in all_rain.groupby(all_rain.index.month):
    list_plot.append(np.array(month[1].dropna().values).flatten())
c = 'b'
ax1[3].boxplot(list_plot, whis=[5,95], showfliers=False, #sym = '.b',
   patch_artist=True, boxprops=dict(facecolor=c, color=c, alpha = 0.6), capprops=dict(color=c),
   whiskerprops=dict(color=c), flierprops=dict(color=c, markeredgecolor=c), medianprops=dict(color='k'))
ax1[3].yaxis.set_minor_locator(AutoMinorLocator())
ax1[3].tick_params(axis='y', labelsize=8)
ax1[3].set_ylabel('Rainfall (mm/day)', size = 7, labelpad=0)

ax1[0].yaxis.set_minor_locator(AutoMinorLocator())
ax1[1].yaxis.set_minor_locator(AutoMinorLocator())
ax1[2].yaxis.set_minor_locator(AutoMinorLocator())
ax1[0].tick_params(axis='y', labelsize=8)
ax1[1].tick_params(axis='y', labelsize=8)
ax1[2].tick_params(axis='y', labelsize=8)
#ax1[2].set_xticks(np.arange(0,14,1))
ax1[2].set_xticklabels(month_label, fontsize = 7)
ax1[3].set_xticklabels(month_label, fontsize = 7)
fig.text(0.15, 0.83, '(a)', ha='center', size = 10, fontweight='bold')
fig.text(0.14, 0.43, '(b)', va='center', size = 10, fontweight='bold')
fig.text(0.57, 0.83, '(c)', ha='center', size = 10, fontweight='bold')
fig.text(0.56, 0.43, '(d)', va='center', size = 10, fontweight='bold')
#plt.subplots_adjust(left = 0.15, right = 0.95, top = 0.98, bottom= 0.1, hspace = 0.15)
plt.show()

if save == True:
    fn_out = os.path.join(fn_trunk, 'Paper\Paper5\FIGURES', 'Univariate', 'rainfall_tide_skew_SL_monthly_maxima_boxplot.png')
    fig.savefig(fn_out, frameon=False, dpi = 300,papertype= 'a4', orientation='portrait') #transparent=True, 
    plt.close()

tide_range, tide_coef, tide_medians = month_statistics(tide_day)
surge_range, surge_coef, surge_medians = month_statistics(all_skew)
sl_range, sl_coef, sl_medians = month_statistics(all_seasevel)
rain_range, rain_coef, rain_medians = month_statistics(all_rain)