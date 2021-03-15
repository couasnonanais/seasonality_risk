# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:39:07 2021

This scripts tests for the (in)dependence between tide and skew surge

@author: acn980
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os,sys,glob
import scipy.stats as sp
import statsmodels.api as sm

sys.path.insert(0,r'E:\github\seasonality_risk\Functions')
from Functions_HCMC import get_skew_surge

#%%
save = False

fn_trunk = 'E:/surfdrive/Documents'
fn = os.path.join(fn_trunk, 'Master2019\Thomas\data\matlab_csv')

fn_files = 'Master2019/Thomas/data'
fn2 = os.path.join(fn_trunk,fn_files)
#%% We import the total water level and tide to obtain the high tide and skew surge
#We import the tide
fn_tide = os.path.join(fn,'Tide_WACC_VungTau_Cleaned_Detrended_Strict_sel_const.csv')
date_parser = lambda x: pd.datetime.strptime(x, "%d-%m-%Y %H:%M:%S")
tide = pd.read_csv(fn_tide, parse_dates = True, date_parser= date_parser, index_col = 'Date')
tide.rename(columns = {tide.columns[0]:'waterlevel'}, inplace = True)

#We import the linearly detrended sea level
fn_sl = r'E:\surfdrive\Documents\Master2019\Thomas\data\hourly_VungTau_swl_cleaned_detrended_strict.csv'
date_parser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
sl = pd.read_csv(fn_sl, parse_dates = True, date_parser= date_parser, index_col = 'date')
sl.rename(columns = {sl.columns[0]:'waterlevel'}, inplace = True)

#we select the sealevel for the time where we both have data 
tide = tide.loc[sl.index,:].copy()

#We calculate the correct skew surge
skew_surge, high_tide = get_skew_surge(sl,tide,distance=4)
skew_surge.set_index(high_tide.index, drop=True, inplace = True)
skew_surge[skew_surge<=-0.8]=np.nan

#Plotting results
both = pd.concat([skew_surge, high_tide], axis = 1)
recons_sl = both['skew_surge'] + both['high_tide'] #> If this is not shuffled, then it contains the dependence between tide and skew surge

both = pd.concat([pd.DataFrame(recons_sl, columns=['total']), both], axis = 1)
both.dropna(inplace = True)
#%% Using Williams et al. (2016) to look at independence between skew surge and tide
def fit_kernel(X_fit, bw='normal_reference'):
    kde = sm.nonparametric.KDEUnivariate(X_fit)
    kde.fit(kernel='gau', bw=bw) # Estimate the densities
    return kde

nb_1 = int(len(both['skew_surge'].dropna())*0.01)
skew_1 = both.sort_values('skew_surge', axis = 0, ascending = False, inplace = False)
skew_1 = skew_1.iloc[:nb_1,:]

plt.figure()
plt.plot(skew_1['skew_surge'],skew_1['high_tide'], '.k')
plt.show()

#Testing correlation between skew surge and predicted HW
corr = sp.kendalltau(skew_1['skew_surge'].values,skew_1['high_tide'].values, nan_policy='omit') #Correlation between skew surge and tide is very low!

#Distribution of Predicted HW Associated With Extreme Skew Surges
plt.figure()
plt.hist(both.dropna()['high_tide'], bins=np.arange(-0.3,1.5,0.1), alpha=0.2, histtype='bar', color = 'black', density= True)
dens_all = fit_kernel(both.dropna()['high_tide'], bw=2.44/20)
plt.plot(dens_all.support, dens_all.density, color='black', lw=1, linestyle='-')

plt.hist(skew_1.dropna()['high_tide'], bins=np.arange(-0.3,1.5,0.1), alpha=0.2, histtype='bar', color = 'red', density= True)
dens_1 = fit_kernel(skew_1.dropna()['high_tide'], bw=2.44/20)
plt.plot(dens_1.support, dens_1.density, color='red', lw=1, linestyle='-')
plt.show()

#Using the K-Sample Anderson-Darling Tests to know if they come from the same distribution
result = sp.anderson_ksamp([both.dropna()['high_tide'].values, skew_1.dropna()['high_tide'].values]) #Better to use Anderson than KS
print('result: ', result[2])
#result2 = sp.ks_2samp(both.dropna()['high_tide'].values, skew_1.dropna()['high_tide'].values, alternative='two-sided', mode='auto')
#%%Doing in now on a monthly basis
skew_1['month'] = skew_1.index.month
both['month'] = both.index.month

f, ax = plt.subplots(nrows=6, ncols=2, sharex=True, sharey=False, gridspec_kw={'width_ratios': [1,1], 'height_ratios':[1,1,1,1,1,1]}, figsize=(8,8))
ax = ax.reshape(-1)

f2, ax2 = plt.subplots(nrows=6, ncols=2, sharex=False, sharey=False, gridspec_kw={'width_ratios': [1,1], 'height_ratios':[1,1,1,1,1,1]}, figsize=(8,8))
ax2 = ax2.reshape(-1)
for m in np.arange(1,13,1):
    both_sel = both.where(both['month'] == m).dropna()
    skew_sel = skew_1.where(both['month'] == m).dropna()    

    ax[m-1].hist(both_sel['high_tide'], bins=np.arange(-0.3,1.5,0.1), alpha=0.2, histtype='bar', color = 'black', density= True)
    dens_all = fit_kernel(both_sel['high_tide'], bw=2.44/20)
    ax[m-1].plot(dens_all.support, dens_all.density, color='black', lw=1, linestyle='-')
    if len(skew_sel)>0:
        ax[m-1].hist(skew_sel['high_tide'], bins=np.arange(-0.3,1.5,0.1), alpha=0.2, histtype='bar', color = 'red', density= True)
        dens_1 = fit_kernel(skew_sel['high_tide'], bw=2.44/20)
        ax[m-1].plot(dens_1.support, dens_1.density, color='red', lw=1, linestyle='-')
        
        # result = sp.ranksums(both_sel['high_tide'].dropna().values, skew_sel['high_tide'].dropna().values)#sp.anderson_ksamp([both_sel['high_tide'].values, skew_sel['high_tide'].values])
        # print('Month {}, sig. lev. {} \n'.format(str(m), result[1]))

        # result = sp.anderson_ksamp([both_sel['high_tide'].values, skew_sel['high_tide'].values])
        # print('Month {}, sig. lev. {} \n'.format(str(m), result[2]))
        # #print(sp.ks_2samp(both_sel['high_tide'].values, skew_sel['high_tide'].values))
        
        corr = sp.kendalltau(skew_sel['skew_surge'].values,skew_sel['high_tide'].values, nan_policy='omit') 
        print('Month {}, Kendall \n'.format(str(m)), corr)
        
        ax2[m-1].plot(skew_sel['skew_surge'],skew_sel['high_tide'], 'ok')
        
    ax[m-1].title.set_text('Month {}'.format(str(m)))

f.tight_layout()
fn_out = r'E:\surfdrive\Documents\Paper\Paper5\FIGURES\Bivariate\distribution_tide_skew.png'
f.savefig(fn_out, frameon=False, dpi = 300,papertype= 'a4', orientation='portrait', bbox_inches='tight') #transparent=True, 

#Doing it weighted average 
prop = skew_1.groupby('month').count()/nb_1
prop = prop * len(both)

#We sample the tide accordingly
tide_sampled = pd.DataFrame(data=None)
for m in np.arange(1,13,1):
    both_sel = both.where(both['month'] == m).dropna()
    try:
        r_tide_m = both_sel['high_tide'].sample(n=int(prop.loc[m,'high_tide']), replace = True)
        tide_sampled = pd.concat([tide_sampled, r_tide_m], axis = 0)
    except:
        continue
tide_sampled.rename(columns={0:'high_tide'}, inplace = True)

plt.figure()
plt.hist(tide_sampled['high_tide'], bins=np.arange(-0.3,1.5,0.1), alpha=0.2, histtype='bar', color = 'black', density= True)
dens_all = fit_kernel(tide_sampled['high_tide'], bw=2.44/20)
plt.plot(dens_all.support, dens_all.density, color='black', lw=1, linestyle='-')

plt.hist(skew_1.dropna()['high_tide'], bins=np.arange(-0.3,1.5,0.1), alpha=0.2, histtype='bar', color = 'red', density= True)
dens_1 = fit_kernel(skew_1.dropna()['high_tide'], bw=2.44/20)
plt.plot(dens_1.support, dens_1.density, color='red', lw=1, linestyle='-')

#Using the K-Sample Anderson-Darling Tests to know if they come from the same distribution
result = sp.anderson_ksamp([tide_sampled['high_tide'].values, skew_1['high_tide'].values])