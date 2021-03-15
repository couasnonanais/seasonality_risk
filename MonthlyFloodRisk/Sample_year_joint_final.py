# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:30:13 2020

This script calculate damage for the yearly basis

@author: acn980
"""

import os, sys, glob
import pandas as pd
import numpy as np
import warnings
import scipy
import matplotlib.pyplot as plt
import subprocess
warnings.filterwarnings("ignore")
sys.path.insert(0,r'E:\github\seasonality_risk\Functions')
from Functions_HCMC import remove_NaN_skew, detrend_fft, collect_rainfall, calc_avg_max_min_rainfall, thiessen_rain
from Functions_HCMC import damage_surface_coord_z, load_damage, plot_damage_grid
from Functions_HCMC import simulate_rain, simulate_skew, pairs_cooc, pairs_rain, pairs_sl

#%% Setting the files and folder correctly
fn_trunk = 'E:/surfdrive/Documents'
fn_files = 'Paper/Paper5/Hydrodynamic_runs/RISK_maskTelemac'
fn = os.path.join(fn_trunk,fn_files)
#os.chdir(fn)
#%% We plot the drivers - damage curve
damage_grid = load_damage(fn_trunk, fn_files, max_rain=1000, max_sl=3000, thr_rain=50, thr_sl=1030) #max_rain, max_sl, thr_rain, thr_sl
coords, dam = damage_surface_coord_z(damage_grid)

#Plotting damage
f = plt.figure(figsize=(8,4))
ax = f.add_subplot(111, projection='3d', azim=-60, elev=30)
#Plotting damage
plot_damage_grid(damage_grid, ax = ax, rstride=1, ctride=1)  # damage_grid.drop(3000,axis=1).drop(1000, axis = 0)
plt.show()

#Plotting damage
f = plt.figure(figsize=(8,4))
ax = f.add_subplot(111, projection='3d', azim=-60, elev=30)
plot_damage_grid(damage_grid.drop([50,1000], axis=0).drop([1030,3000], axis = 1), ax = ax, rstride=1, ctride=1) 
plt.show()

xv, yv = np.meshgrid(damage_grid.index.values, damage_grid.columns.values, indexing = 'ij')
Z = damage_grid.to_numpy()
plt.figure()
plt.contour(xv, yv, Z, levels=np.arange(0,4e9, 5e7), c='k') # cmap=plt.cm.Reds)
plt.show()
#%%
varname1 = 'Thiessen'
varname2 = 'skew'
lag_joint = 0
n=50000
dep_type= 'copula' #'copula' #'full corr'
figure_joint = False
cop_sim_R = False

#%% We simulate based on monthly maxima of copula
if cop_sim_R:
    output=subprocess.run(["C:/ProgramData/Anaconda3/envs/r_env/Scripts/Rscript.exe",  "E:/github/HoChiMinh/YearlyCopFit.R", str(n), str(varname1), str(varname2)],
                           shell=True, stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE).stderr

#%% We simulate events based on their co-occurence and dependence
fn_rain = os.path.join(fn_trunk, 'Master2019/Thomas/data/NewRain/TRENDS/MONTH_CORRECTED/yearly_EVA')
param_rain = pd.read_csv(os.path.join(fn_rain,'all_params.csv'), index_col = 'distribution')
best_fit_rain = pd.read_csv(os.path.join(fn_rain,'best_fit_AIC.csv'))  #, index_col = 'month')
best_fit_rain.set_index('month', inplace = True)
del fn_rain

#Calculate montly skew surge mean
fn_skew  = os.path.join(fn_trunk, 'Master2019\Thomas\data\matlab_csv','skew_WACC_VungTau_Cleaned_Detrended_Strict_sel_const.csv')
date_parser = lambda x: pd.datetime.strptime(x, "%d-%m-%Y %H:%M:%S")
skew = pd.read_csv(fn_skew, parse_dates = True, date_parser= date_parser, index_col = 'Date')
skew.rename(columns = {skew.columns[0]:'skew'}, inplace = True)
skew = remove_NaN_skew(skew)
skew_day = skew.resample('D').max()
skew_detrend = detrend_fft(skew_day, fillnavalue=0, frequency = 1./(2*365), figure_plotting =0)
skew_detrend_day = skew_detrend.resample('D').max()
skew_month_avg =  skew_detrend_day.groupby([skew_detrend_day.index.month]).mean()
skew_month_avg.loc['year','skew']=skew_detrend_day.groupby([skew_detrend_day.index.month]).mean().mean().values 
del fn_skew, skew, skew_day, skew_detrend, skew_detrend_day

fn_copula = os.path.join(fn_trunk, 'Master2019/Thomas/data/NewBivariate/Simulated')

#We calculate the mean rainfall - below 50 mm/day
fn2 = os.path.join(fn_trunk, 'Master2019/Thomas/data/NewRain')
rainfall = collect_rainfall(fn2, figure_plotting = False)
rainfall_years = calc_avg_max_min_rainfall(rainfall, threshold=40)
fn_thiessen = os.path.join(fn_trunk, 'Master2019/Thomas/data/Thiessen_stations_weights.csv')
thies_rainfall = thiessen_rain(fn_thiessen, rainfall)
rainfall_month_avg =  thies_rainfall.groupby([thies_rainfall.index.month]).mean()
rainfall_month_avg.loc['year','Thiessen_sum']=rainfall_month_avg.mean().values 
del fn2, rainfall, rainfall_years, thies_rainfall, fn_thiessen

fn_skew = os.path.join(fn_trunk, 'Master2019/Thomas/data/NewSurge/TRENDS/MONTH_RAW','yearly_EVA')
param_skew = pd.read_csv(os.path.join(fn_skew,'all_params.csv'), index_col = 'distribution')
best_fit_skew = pd.read_csv(os.path.join(fn_skew,'best_fit_AIC.csv'))  #, index_col = 'month')
best_fit_skew.set_index('month', inplace = True)
del fn_skew

fn_tide = os.path.join(fn_trunk, 'Master2019/Thomas/data/NewTide')
all_tide = pd.DataFrame()
for month in np.arange(1,13,1):
    tide_sim = pd.read_csv(os.path.join(fn_tide, 'samples_tide_month_{}.csv'.format(str(month))), usecols=['tide'])
    all_tide = pd.concat([all_tide,tide_sim], axis=0, ignore_index=True)
all_tide.reset_index(drop = True, inplace = True)
all_tide=all_tide.sample(frac=1).reset_index(drop = True)*1000

#Storing results
damage_mod = pd.DataFrame(data = None, index = ['year'], columns = ['simulated_highest', 'full_dep', 'ind_highest', 'exclusive_highest'])

#Select marginal distribution for rain
dist_type = best_fit_rain['AIC'].idxmin()  
list_params = param_rain.loc[dist_type,:]
params = [float(e) for e in list_params]

dist_type_skew = best_fit_skew['AIC'].idxmin()
list_params_skew = param_skew.loc[dist_type_skew,:].dropna()
params_skew = [float(e) for e in list_params_skew]       

all_events_sampled = pd.DataFrame(data=None, columns=['rain','sealevel','month'])
all_events_sampled_dep = pd.DataFrame(data=None, columns=['rain','sealevel', 'month'])
all_events_sampled_ind = pd.DataFrame(data=None, columns=['rain','sealevel', 'month'])
all_events_sampled_excl = pd.DataFrame(data=None, columns=['rain','sealevel', 'month'])
month = 'year'

if dep_type == 'copula':
    rain_simcdf = pd.read_csv(os.path.join(fn_copula, 'New{}_New{}_data_year.csv'.format(str(varname1),str(varname2))), usecols=['V1'])
    rain_rvs = simulate_rain(rain_simcdf, params, dist_type)

    cdf_swl_rvs = pd.read_csv(os.path.join(fn_copula, 'New{}_New{}_data_year.csv'.format(str(varname1),str(varname2))), usecols=['V2']).values
    if varname2 == 'skew':
        skew_rvs = simulate_skew(cdf_swl_rvs, params_skew, dist_type_skew)
        tide_rvs = all_tide.sample(n=n).reset_index(drop = True)
        tide_rvs = np.reshape(tide_rvs.values, tide_rvs.shape[0]) 
    del cdf_swl_rvs, rain_simcdf

##### FULL DEPENDENCE ######
cooc_events = pairs_cooc(rain_rvs, skew_rvs, tide_rvs)

sampled_month_dep = pd.DataFrame(data=cooc_events, columns=['rain', 'sealevel'])
sampled_month_dep['month'] = month    

dam_full_dep = scipy.interpolate.griddata(coords, dam, cooc_events.values, method = 'linear')
dam_full = np.sum(dam_full_dep, dtype = float)
damage_mod.loc[month, 'full_dep'] = dam_full/n 
sampled_month_dep['cooc_damage'] = dam_full_dep

all_events_sampled_dep = pd.concat([all_events_sampled_dep, sampled_month_dep], axis = 0, ignore_index=True)
del dam_full_dep, dam_full, sampled_month_dep, tide_rvs, cooc_events

##### EXCLUSIVE ######
tide_rvs = all_tide.sample(n=n).reset_index(drop = True)
tide_rvs = np.reshape(tide_rvs.values, tide_rvs.shape[0]) 
rain_events = pairs_rain(rain_rvs, tide_rvs, skew_month_avg, month)
del tide_rvs    

tide_rvs = all_tide.sample(n=n).reset_index(drop = True)
tide_rvs = np.reshape(tide_rvs.values, tide_rvs.shape[0]) 
sealevel_events = pairs_sl(skew_rvs, tide_rvs, rainfall_month_avg, month)

dam_excl_rain = scipy.interpolate.griddata(coords, dam, (rain_events.values), method = 'linear') #
dam_excl_sl = scipy.interpolate.griddata(coords, dam, (sealevel_events.values), method = 'linear') #np.zeros(events_month[:,1].shape)

dam_excl_highest = pd.DataFrame(data=np.concatenate((dam_excl_rain, dam_excl_sl), axis=1), columns = ['rain_damage', 'sealevel_damage'])
dam_highest = dam_excl_highest.max(axis=1)
damage_mod.loc[month, 'exclusive_highest'] = (np.sum(dam_highest.values, dtype = float))/n      

dam_highest_type = dam_excl_highest.idxmax(axis=1)
sampled_month_excl = pd.concat([pd.concat([rain_events[dam_highest_type=='rain_damage'], dam_excl_highest[dam_highest_type=='rain_damage']['rain_damage']], axis = 1),
                                pd.concat([sealevel_events[dam_highest_type=='sealevel_damage'], dam_excl_highest[dam_highest_type=='sealevel_damage']['sealevel_damage']], axis = 1)
    ], axis = 0, ignore_index=True)
sampled_month_excl['month'] = month 
all_events_sampled_excl = pd.concat([all_events_sampled_excl, sampled_month_excl], axis = 0, ignore_index=True)    
del rain_events, sealevel_events, tide_rvs, dam_highest, dam_highest_type, dam_excl_rain, dam_excl_sl, sampled_month_excl, dam_excl_highest

#### INDEPENDENCE #### OUR CASE HERE
n_cooc_ind = int((1/365) * n) 
i_cooc_ind = np.random.choice(np.arange(0, n), n_cooc_ind, replace = False)
i_ind = np.delete(np.arange(0, n), i_cooc_ind)   

tide_rvs = all_tide.sample(n=len(i_cooc_ind)).reset_index(drop = True)
tide_rvs = np.reshape(tide_rvs.values, tide_rvs.shape[0]) 
cooc_events = pairs_cooc(rain_rvs[i_cooc_ind], skew_rvs[i_cooc_ind], tide_rvs)

tide_rvs = all_tide.sample(n=len(i_ind)).reset_index(drop = True)
tide_rvs = np.reshape(tide_rvs.values, tide_rvs.shape[0]) 
rain_events = pairs_rain(rain_rvs[i_ind], tide_rvs, skew_month_avg, month)

tide_rvs = all_tide.sample(n=len(i_ind)).reset_index(drop = True)
tide_rvs = np.reshape(tide_rvs.values, tide_rvs.shape[0]) 
sealevel_events = pairs_sl(skew_rvs[i_ind], tide_rvs, rainfall_month_avg, month)

dam_excl_rain = scipy.interpolate.griddata(coords, dam, (rain_events.values), method = 'linear') #
dam_excl_sl = scipy.interpolate.griddata(coords, dam, (sealevel_events.values), method = 'linear') #np.zeros(events_month[:,1].shape)
dam_cooc = scipy.interpolate.griddata(coords, dam, (cooc_events.values), method = 'linear') #np.zeros(events_month[:,1].shape)

dam_excl_highest = pd.DataFrame(data=np.concatenate((dam_excl_rain, dam_excl_sl), axis=1), columns = ['rain_damage', 'sealevel_damage'])
dam_highest = dam_excl_highest.max(axis=1)
dam_highest_type = dam_excl_highest.idxmax(axis=1)

damage_mod.loc[month, 'ind_highest'] = (np.sum(dam_highest.values, dtype = float) + np.sum(dam_cooc, dtype = float))/n  

sampled_month_ind = pd.concat([pd.concat([rain_events[dam_highest_type=='rain_damage'], dam_excl_highest[dam_highest_type=='rain_damage']['rain_damage']], axis = 1),
                                pd.concat([sealevel_events[dam_highest_type=='sealevel_damage'], dam_excl_highest[dam_highest_type=='sealevel_damage']['sealevel_damage']], axis = 1),
                                pd.concat([cooc_events, pd.DataFrame(dam_cooc, columns = ['cooc_damage'])], axis = 1)
    ], axis = 0, ignore_index=True)
sampled_month_ind['month'] = month 
all_events_sampled_ind = pd.concat([all_events_sampled_ind, sampled_month_ind], axis = 0, ignore_index=True)    
del n_cooc_ind, i_cooc_ind, i_ind, cooc_events, rain_events, sealevel_events, tide_rvs, dam_highest, dam_highest_type, dam_excl_rain, dam_excl_sl, sampled_month_ind, dam_cooc, dam_excl_highest

total_risk = damage_mod.sum()
print(total_risk)