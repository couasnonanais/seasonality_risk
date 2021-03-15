# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 07:31:20 2020

This scripts fits the marginal distributions (mix of Python 2 and 3 (!))

@author: acn980
"""
from fitter import Fitter
import os
import pandas as pd
import glob
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

#%%
def get_RPs(return_periods, params, dist_type):
    return_periods_col = [str(i) for i in return_periods]
    
    dist = getattr(sp, dist_type)
    if len(params) == 3:
        a = dist.isf(1./return_periods, params[0], params[1], params[2])
    elif len(params) == 2:
        a = dist.isf(1./return_periods, params[0], params[1])
    elif len(params) == 1:
        a = dist.isf(1./return_periods, params[0])
    else:
        print('Something is wrong!')

    RP_EVAs = pd.Series(a, index = return_periods_col)
    return RP_EVAs

def get_line_pt_RP_fit(exc_prob_x, data, params, dist_type):
    dist = getattr(sp, dist_type)
    if len(params) == 3: #shape, loc and scale
        #print('Skew param ', f.fitted_param[dist_type][0])
        print('Check param ', params)
        inv_cdf_dist = dist.sf(data, params[0], params[1], params[2])
        rp_y = dist.isf(exc_prob_x, params[0], params[1], params[2])
    elif len(params) == 2:#loc and scale
        inv_cdf_dist = dist.sf(data, params[0], params[1])
        rp_y = dist.isf(exc_prob_x, params[0], params[1])
    elif len(params) == 1:
        inv_cdf_dist = dist.sf(data, params[0])
        rp_y = dist.isf(exc_prob_x, params[0])
    else:
        print('Something is wrong!')
    
    return inv_cdf_dist, rp_y

def update_fitted_params_gev(data, param_start):
    '''uses the location from gumbel to update the GEV parameters'''
    dist_gev = getattr(sp, 'genextreme')
    param_gev = dist_gev.fit(f=data, floc=param_start[1], fscale = param_start[2])  #shape, loc and scale, floc=param_start[1]
    
    print('Old param: ', f.fitted_param['genextreme'])
    print('New param: ', param_gev)
    return param_gev

#%%
if __name__ == '__main__':
    import lmoments3 as lm
    from lmoments3 import distr, stats

#%% Setting the files and folder correctly
home = False
save = False
var = 'surge'

if home ==True:
    fn_trunk = 'D:/surfdrive/Documents'
else:
    fn_trunk = 'E:/surfdrive/Documents'
    
fn_files = 'Master2019/Thomas/data'
fn = os.path.join(fn_trunk,fn_files)

#Load the file
if var == 'rain':
    allfiles = glob.glob(os.path.join(fn, 'NewRain\TRENDS\MONTH_CORRECTED', 'Thiessen_*.csv'))
if var == 'surge':
    allfiles = glob.glob(os.path.join(fn, 'NewSurge\TRENDS\MONTH_RAW', 'skew_fft_*.csv'))

all_month = pd.DataFrame()
for file in allfiles:
    print(file)
    sel = pd.read_csv(file, index_col = 'Year', parse_dates=True)  
    if var == 'rain':
        sel.rename(columns={sel.columns[0]:'Thiessen'}, inplace = True)
    all_month = pd.concat([all_month, sel], axis = 0)

year_data = all_month.resample('Y').max()
date_ymax = all_month.groupby(all_month.index.year).idxmax()
date_ymax['month'] = date_ymax['skew_fft'].dt.month
#%% Fitting distribution on monthly maxima data - PYTHON 2
res = {}
AIC = pd.DataFrame(index=['genextreme', 'gumbel_r', 'pearson3', 'weibull_min', 'expon'], columns = np.arange(1,13,1))
all_params = pd.DataFrame(index=['genextreme', 'gumbel_r', 'pearson3', 'weibull_min', 'expon'], columns = np.arange(1,13,1))
res['data'] = {}

for file in allfiles:
    print(file)
    if var == 'rain':
        station = file.split('MONTH_CORRECTED\\')[-1].split('_')[0]
        month = int(file.split('MONTH_CORRECTED\\')[-1].split('_')[1])
    
    if var == 'surge':
        month = int(file.split('MONTH_RAW\\')[-1].split('_')[-1].split('.csv')[0])
        station = file.split('MONTH_RAW\\')[-1].split('_{}.csv'.format(str(month)))[0]
    

    sel = pd.read_csv(file, index_col = 'Year', parse_dates=True)  
    
    if var=='rain': #ADD NOISE TO 0 for rain
        sel_zeros = sel.where(sel == 0).dropna()
        if len(sel_zeros)>0:
            noise = np.random.normal(0,0.5,100)
            noise = noise[noise>0]
            sel.loc[sel_zeros.index] = np.reshape(noise[0:len(sel_zeros)], (len(sel_zeros), 1))
    
    if var=='surge': #dropna
        sel = sel.dropna()
    
    data = np.reshape(np.array(sel.values), (len(sel)))
    print(data)
        
    #Fitting gev 
    paras = distr.gev.lmom_fit(data)
    AIC_gev = stats.AIC(data, 'gev', paras)
    paras_gev = (paras['c'], paras['loc'], paras['scale']) 
    AIC.loc['genextreme', month] = AIC_gev
    all_params.loc['genextreme', month] = paras_gev
    
    #Fitting gumbel
    paras = distr.gum.lmom_fit(data)
    AIC_gum = stats.AIC(data, 'gum', paras)
    paras_gum = (paras['loc'], paras['scale'])
    AIC.loc['gumbel_r', month] = AIC_gum
    all_params.loc['gumbel_r', month] = paras_gum
    
    #Fitting pearsonIII 
    paras = distr.pe3.lmom_fit(data)
    AIC_pe3 = stats.AIC(data, 'pe3', paras)
    paras_pe3 = (paras['skew'], paras['loc'], paras['scale'])
    AIC.loc['pearson3', month] = AIC_pe3
    all_params.loc['pearson3', month] = paras_pe3
    
    #Fitting weibull
    paras = distr.wei.lmom_fit(data)
    AIC_wei = stats.AIC(data, 'wei', paras)
    paras_wei = (paras['c'], paras['loc'], paras['scale'])
    AIC.loc['weibull_min', month] = AIC_wei
    all_params.loc['weibull_min', month] = paras_wei
    
    #Fitting exponential
    paras = distr.exp.lmom_fit(data)
    AIC_exp = stats.AIC(data, 'exp', paras)
    paras_exp = (paras['loc'], paras['scale'])
    AIC.loc['expon', month] = AIC_exp
    all_params.loc['expon', month] = paras_exp
    
    res['data'][month] = data 
    
    del paras_exp, paras_wei, paras_pe3, paras_gum, paras_gev, AIC_gev, AIC_gum, AIC_pe3, AIC_wei, AIC_exp, paras

AIC.loc['weibull_min', :] = np.nan
AIC.loc['pearson3', :] = np.nan
res['AIC'] = AIC
res['dist_param'] = all_params
AIC = AIC.astype(np.float32)
best_fit = AIC.idxmin()

if save == True:
    if var == 'rain':
        fn_out = os.path.join(fn, 'NewRain\TRENDS\MONTH_CORRECTED', 'monthly_EVA')
        best_fit.to_csv(os.path.join(fn_out,'best_fit_AIC.csv'), index_label = 'month')
        all_params.to_csv(os.path.join(fn_out,'all_params_lmoments.csv'), index_label = 'distribution')
    if var == 'surge':
        fn_out = os.path.join(fn, 'NewSurge\TRENDS\MONTH_RAW', 'monthly_EVA')
        best_fit.to_csv(os.path.join(fn_out,'best_fit_AIC.csv'), index_label = 'month')
        all_params.to_csv(os.path.join(fn_out,'all_params_lmoments.csv'), index_label = 'distribution')
    
#%% Fitting distribution on yearly maxima data - PYTHON 3
return_periods = np.array([2,5,10,20,25,50,100,500,1000,5000])
cum_prob_x = np.arange(0.001,1,0.001)
exc_prob_x = 1 - cum_prob_x
rp_x=1/(exc_prob_x)

#Calculating empirical
f = Fitter(year_data.dropna().values,  distributions=['genextreme', 'gumbel_r'])
f.fit()      

AIC_year = pd.DataFrame.from_dict(f._aic, orient='index')
AIC_year.rename(columns={AIC_year.columns[0]:'AIC'}, inplace = True)

BIC_year = pd.DataFrame.from_dict(f._bic, orient='index')
BIC_year.rename(columns={BIC_year.columns[0]:'BIC'}, inplace = True)

emp_p = pd.DataFrame(data=f._data)
emp_p['rank'] = emp_p.iloc[:,0].rank(axis=0, ascending=False)
emp_p['exc_prob'] = emp_p['rank']/(emp_p['rank'].size+1) #change this line with what Anaïs sends to me, but is already correct
emp_p['cum_prob'] = 1 - emp_p['exc_prob']
emp_p['emp_rp'] = 1/emp_p['exc_prob']


inv_cdf_dist, rp_y_GEV = get_line_pt_RP_fit(exc_prob_x, f._data, f.fitted_param['genextreme'], 'genextreme')
b_GEV = get_RPs(return_periods, f.fitted_param['genextreme'], 'genextreme')

inv_cdf_dist, rp_y_Gumbel = get_line_pt_RP_fit(exc_prob_x, f._data, f.fitted_param['gumbel_r'], 'gumbel_r')
b_Gumbel = get_RPs(return_periods, f.fitted_param['gumbel_r'], 'gumbel_r')

fig = plt.figure()
plt.plot(emp_p['emp_rp'], emp_p.iloc[:,0], color='k', marker='o', ls = '', ms = 2)    
#plt.plot(rp_x, rp_y_GEV ,color = 'r', label = 'GEV')
plt.plot(rp_x, rp_y_Gumbel, color = 'g', label = 'Gumbel')
plt.show()
plt.grid(which='both', axis='x')
plt.grid(which='major', axis='y')
plt.xscale('log')
plt.legend()
plt.xlabel('Return Period (yrs)')
#plt.ylabel('Rainfall (mm/day)')
plt.ylabel('Storm surge (m)')

all_params = pd.DataFrame.from_dict(f.fitted_param, orient='index')
best_fit = AIC_year.copy()

if save == True:
    if var == 'surge':
        fn_out = os.path.join(fn, 'NewSurge\TRENDS\MONTH_RAW', 'yearly_EVA')
        best_fit.to_csv(os.path.join(fn_out,'best_fit_AIC.csv'), index_label = 'month')
        all_params.to_csv(os.path.join(fn_out,'all_params.csv'), index_label = 'distribution')
        
        fn_out = os.path.join(fn_trunk, 'Paper\Paper5\FIGURES', 'Univariate', '{}_returnperiods_yearly.png'.format(var))
        fig.savefig(fn_out, frameon=False, dpi = 300,papertype= 'a4', orientation='portrait') #transparent=True, 
        plt.close()
 
    
    if var == 'rain':
        fn_out = os.path.join(fn, 'NewRain\TRENDS\MONTH_CORRECTED', 'yearly_EVA')
        best_fit.to_csv(os.path.join(fn_out,'best_fit_AIC.csv'), index_label = 'month')
        all_params.to_csv(os.path.join(fn_out,'all_params.csv'), index_label = 'distribution')

        fn_out = os.path.join(fn_trunk, 'Paper\Paper5\FIGURES', 'Univariate', '{}_returnperiods_yearly.png'.format(var))
        fig.savefig(fn_out, frameon=False, dpi = 300,papertype= 'a4', orientation='portrait') #transparent=True, 
        plt.close()
    
    
