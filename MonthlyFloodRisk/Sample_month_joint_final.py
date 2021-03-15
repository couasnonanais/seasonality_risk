# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:30:13 2020

This script plots the damage surface and calculate damage for various scenarios

@author: acn980
"""

import os, sys, glob
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import subprocess
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")
sys.path.insert(0,r'E:\github\seasonality_risk\Functions')
from Functions_HCMC import remove_NaN_skew, detrend_fft, collect_rainfall, calc_avg_max_min_rainfall, thiessen_rain
from Functions_HCMC import damage_surface_coord_z, load_damage,import_monthly_rain, import_monthly_skew
from Functions_HCMC import  calculate_monthly_damage, calc_density, numfmt, plot_damage_grid,construct_final_set
import matplotlib.ticker as tkr 
import matplotlib.colors as colors
from matplotlib import gridspec

def func(x):
    s = "%.2g" % x
    if "e" in s:
        tup = s.split('e')
        significand = tup[0]#.rstrip('0') #.rstrip('.')
        sign = tup[1][0].replace('+', '')
        exponent = tup[1][1:].lstrip('0')
        s = ('%se%s%s' % (significand, sign, exponent)).rstrip('e')    
    return s

class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

#%% Setting the files and folder correctly
fn_trunk = 'E:/surfdrive/Documents'
fn_files = 'Paper/Paper5/Hydrodynamic_runs/RISK_maskTelemac'
fn = os.path.join(fn_trunk,fn_files)
#os.chdir(fn)
#%% We plot the drivers - damage curve
damage_grid = load_damage(fn_trunk, fn_files, max_rain=1000, max_sl=3000, thr_rain=50, thr_sl=1030) #max_rain, max_sl, thr_rain, thr_sl
coords, dam = damage_surface_coord_z(damage_grid)

damage_figure = damage_grid.drop([0,1030,3000],axis=1).drop([50,1000], axis = 0)

bounds = np.arange(0, 350, 50)#np.arange(0, 1.5e+09, 0.2e+09)
cmap = 'cool' #'copper'#'Spectral' 
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
alphas = np.linspace(0.2,1,len(damage_figure.columns))

fig = plt.figure(figsize=[8.5, 4])
gs = GridSpec(2, 2, left=0.05, bottom=0.1, right=0.95, top=0.90, width_ratios=[1,1], height_ratios=[1,1], wspace=0.40, hspace=0.50)#, width_ratios=None, height_ratios=[0.9,0.9,0.9,0.9,0.9,0.9])
ax1 = fig.add_subplot(gs[:, 0], projection='3d', azim=-60, elev=25)

plot_damage_grid(damage_grid.drop([50,1000], axis=0).drop([1030,3000], axis = 1), alphas=alphas, ax = ax1, rstride=1, ctride=1, cmap=cmap, norm=norm)

ax1.tick_params(axis='both', labelsize=7)

isel_index = [0,3,5]

ax2 = fig.add_subplot(gs[0, 1])
j=0
for i in np.arange(len(damage_figure.columns)):
    print(i)
    ax2.scatter(np.repeat(damage_figure.columns[i]/1000,len(isel_index)), damage_figure.iloc[isel_index,i]/1e6, c= damage_figure.index[isel_index], alpha=alphas[j], edgecolors='k', linewidths=0.5, cmap=cmap, norm=norm, zorder=10)#color='lightsteelblue', marker='o', linestyle='-', linewidth=1, markersize=4, label = '0 mm/day')
    ax2.scatter(np.repeat(damage_figure.columns[i]/1000,len(isel_index)), damage_figure.iloc[isel_index,i]/1e6, c= 'None', alpha=1, edgecolors='k', linewidths=0.5, zorder=15)#color='lightsteelblue', marker='o', linestyle='-', linewidth=1, markersize=4, label = '0 mm/day')
    j+=1
ax2.plot(damage_figure.columns/1000, damage_figure.iloc[0,:]/1e6, ls=':', c='k', lw=0.5, label = '0 mm/day')#color='lightsteelblue', marker='o', linestyle='-', linewidth=1, markersize=4, label = '0 mm/day')
ax2.plot(damage_figure.columns/1000, damage_figure.iloc[3,:]/1e6, ls='-', c='k', lw=0.5, label = '180 mm/day')#color= 'mediumblue', marker='o', linestyle='-', linewidth=1, markersize=4, label = '180 mm/day')
ax2.plot(damage_figure.columns/1000, damage_figure.iloc[5,:]/1e6, ls='--', c= 'k', lw=0.5, label = '300 mm/day')#color='midnightblue', marker='o', linestyle='-', linewidth=1, markersize=4, label = '300 mm/day')
ax2.legend(prop={'size': 6}, title="Rainfall", fontsize='small', title_fontsize='x-small')
ax2.set_ylim(-50,1600)
ax2.set_xlabel('Sea level (m)')
ax2.set_ylabel('Damage (M$)')

ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(damage_figure.index, damage_figure.iloc[:,0]/1e6, c= damage_figure.index, alpha=alphas[0], edgecolors='k', linewidths=0.5, cmap=cmap, norm=norm, zorder=10)#, color='khaki', marker='o', linestyle='-', linewidth=1, markersize=4,  label = '0.61 m')
ax3.scatter(damage_figure.index, damage_figure.iloc[:,2]/1e6, c= damage_figure.index, alpha=alphas[2], edgecolors='k', linewidths=0.5, cmap=cmap, norm=norm, zorder=10)#, color='gold', marker='o', linestyle='-', linewidth=1, markersize=4,  label = '1.11 m')
ax3.scatter(damage_figure.index, damage_figure.iloc[:,5]/1e6, c= damage_figure.index, alpha=alphas[-1], edgecolors='k', linewidths=0.5, cmap=cmap, norm=norm, zorder=10)#, color='darkgoldenrod', marker='o', linestyle='-', linewidth=1, markersize=4,  label = '1.86 m')
ax3.scatter(damage_figure.index, damage_figure.iloc[:,0]/1e6, c= 'None', alpha=1, edgecolors='k', linewidths=0.5, zorder=15)#, color='khaki', marker='o', linestyle='-', linewidth=1, markersize=4,  label = '0.61 m')
ax3.scatter(damage_figure.index, damage_figure.iloc[:,2]/1e6, c= 'None', alpha=1, edgecolors='k', linewidths=0.5, zorder=15)#, color='gold', marker='o', linestyle='-', linewidth=1, markersize=4,  label = '1.11 m')
ax3.scatter(damage_figure.index, damage_figure.iloc[:,5]/1e6, c= 'None', alpha=1, edgecolors='k', linewidths=0.5, zorder=15)#, color='darkgoldenrod', marker='o', linestyle='-', linewidth=1, markersize=4,  label = '1.86 m')
ax3.plot(damage_figure.index, damage_figure.iloc[:,0]/1e6, ls=':', c='k', lw=0.5, label = '0.61m')#, color='khaki', marker='o', linestyle='-', linewidth=1, markersize=4,  label = '0.61 m')
ax3.plot(damage_figure.index, damage_figure.iloc[:,2]/1e6, ls='-', c='k', lw=0.5, label = '1.11m')#, color='gold', marker='o', linestyle='-', linewidth=1, markersize=4,  label = '1.11 m')
ax3.plot(damage_figure.index, damage_figure.iloc[:,5]/1e6, ls='--', c='k', lw=0.5, label = '1.86m')#, color='darkgoldenrod', marker='o', linestyle='-', linewidth=1, markersize=4,  label = '1.86 m')
ax3.legend(prop={'size': 6}, title="Sea level", fontsize='small', title_fontsize='x-small')
ax3.set_ylim(-50,1600)
ax3.set_ylabel('Damage (M$)')
ax3.set_xlabel('Rainfall (mm/day)')

fig.text(0.42, 0.11, '(a)', ha='center', size = 10, fontweight='bold')
fig.text(0.92, 0.62, '(b)', ha='center', size = 10, fontweight='bold')
fig.text(0.92, 0.13, '(c)', ha='center', size = 10, fontweight='bold')

fn_out = os.path.join(fn_trunk, 'Paper\Paper5\FIGURES', 'Univariate', 'Damage_surface.png')
fig.savefig(fn_out, frameon=False, dpi = 300,papertype= 'a4', orientation='portrait', bbox_inches='tight') #transparent=True, 
plt.close()
#%%
xv, yv = np.meshgrid(damage_grid.index.values, damage_grid.columns.values, indexing = 'ij')
Z = damage_grid.to_numpy()
plt.figure()
plt.contour(xv, yv, Z, levels=np.arange(0,4e9, 5e7), c='k') # cmap=plt.cm.Reds)
plt.show()
plt.close()

#%%
varname1 = 'Thiessen'
varname2 = 'skew'
lag_joint = 0
n=10000
dep_type= 'copula' #'copula' #'full corr'
figure_joint = True
cop_sim_R = False

#%% We simulate based on monthly maxima of copula
if cop_sim_R == True:
    for month in np.arange(1,13,1):
        print('Month :', month)
        output=subprocess.run(["C:/ProgramData/Anaconda3/envs/r_env/Scripts/Rscript.exe",  "E:/github/HoChiMinh/MonthlyCopFit.R", str(month), str(n), str(varname1), str(varname2)],
                               shell=True, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE).stderr

#%% We simulate events based on their co-occurence and dependence
#Summarizing the fit of the copula
total_kend = pd.DataFrame(index=['family_name','family_nb', 'taildep_upper', 'taildep_lower', 'indeptest', 'tau'], data = None, columns=np.arange(1,13,1))
for month in np.arange(1,13,1):
    kend = pd.read_csv(os.path.join(fn_trunk, 'Master2019/Thomas/data/NewBivariate/Simulated', 'New{}_New{}_copulatype_month_{}.csv'.format(varname1, varname2, str(month))), index_col = 0)
    kend.rename(columns={kend.columns[0]:month}, inplace = True)
    total_kend.loc[:,month] = kend.values
total_kend = total_kend.transpose(copy=True)
total_kend['indeptest'] = pd.to_numeric(total_kend['indeptest'])
#selected_copulas = total_kend['family_name']
selected_copulas = total_kend[['family_name','indeptest']].where(total_kend['indeptest']<0.05, other = 'Independence')['family_name']

fn_rain = os.path.join(fn_trunk, 'Master2019/Thomas/data/NewRain/TRENDS/MONTH_CORRECTED/monthly_EVA')
param_rain = pd.read_csv(os.path.join(fn_rain,'all_params_lmoments.csv'), index_col = 'distribution')
best_fit_rain = pd.read_csv(os.path.join(fn_rain,'best_fit_AIC.csv'), header=None, names=['month', 'distribution'])  #, index_col = 'month')
best_fit_rain.set_index('month', inplace = True)
del fn_rain

#Combined skew and tide
fn_swl = os.path.join(fn_trunk, 'Master2019/Thomas/data/NewSurge/TRENDS/MONTH_RAW','monthly_EVA', 'SWL_gumbel_skew_fft_CDF.csv')
CDF_swl = pd.read_csv(fn_swl, index_col = 'index')
del fn_swl

#Co-occurence data
fn_cooc = os.path.join(fn_trunk, 'Master2019/Thomas/data/NewBivariate', 'New{}_New{}_cooccur_lag_{}.csv'.format(str(varname1),str(varname2),str(lag_joint)))
cooc = pd.read_csv(fn_cooc, index_col = 'month')
del fn_cooc

#Simulated copula data
fn_copula = os.path.join(fn_trunk, 'Master2019/Thomas/data/NewBivariate/Simulated')

#We calculate the mean rainfall - below 50 mm/day
fn2 = os.path.join(fn_trunk, 'Master2019/Thomas/data/NewRain')
rainfall = collect_rainfall(fn2, figure_plotting = False)
rainfall_years = calc_avg_max_min_rainfall(rainfall, threshold=40)
fn_thiessen = os.path.join(fn_trunk, 'Master2019/Thomas/data/Thiessen_stations_weights.csv')
thies_rainfall = thiessen_rain(fn_thiessen, rainfall)
rainfall_month_avg =  thies_rainfall.groupby([thies_rainfall.index.month]).mean()
del fn2, rainfall, rainfall_years, thies_rainfall, fn_thiessen

#EVA for monthly skew surge
fn_skew = os.path.join(fn_trunk, 'Master2019/Thomas/data/NewSurge/TRENDS/MONTH_RAW','monthly_EVA')
param_skew = pd.read_csv(os.path.join(fn_skew,'all_params_lmoments.csv'), index_col = 'distribution')
best_fit_skew = pd.read_csv(os.path.join(fn_skew,'best_fit_AIC.csv'), header=None, names=['month', 'distribution'])  #, index_col = 'month')
best_fit_skew.set_index('month', inplace = True)
del fn_skew

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
del fn_skew, skew, skew_day, skew_detrend, skew_detrend_day

monthly_rain = import_monthly_rain(os.path.join(fn_trunk,'Master2019/Thomas/data'))
monthly_rain['month'] = monthly_rain.index.month
monthly_skew = import_monthly_skew(os.path.join(fn_trunk, 'Master2019\Thomas\data\matlab_csv'))
monthly_skew['month'] = monthly_skew.index.month
monthly_skew.dropna(inplace = True)
monthly_data = pd.concat([monthly_rain, monthly_skew[['skew']]], axis = 1).dropna()

#Tide
fn_tide = os.path.join(fn_trunk, 'Master2019/Thomas/data/NewTide')

#Probability of co-occurence under independence
fn_prob =  os.path.join(fn_trunk, 'Master2019/Thomas/data/Binomial')
l_list = [28, 30, 31]
p_month = pd.DataFrame()
for l in l_list:
    p_data = pd.read_csv(os.path.join(fn_prob,'{}days'.format(str(l)),'Independent_prob_occurence.csv'), index_col=[0])
    p_data.rename(index={'prob_occurence':l}, inplace = True)
    p_month = pd.concat([p_month, p_data], axis = 0)
del l, l_list
month_duration = pd.DataFrame(data=[31,28,31,30,31,30,31,31,30,31,30,31], index=np.arange(1,13,1), columns=['length'])
#%%
#all_events_sampled_dep
damage_mod, all_events_sampled, all_events_sampled_dep, all_events_sampled_excl, all_events_sampled_ind = calculate_monthly_damage(best_fit_rain, param_rain, best_fit_skew, param_skew, n, monthly_data, coords, dam, skew_month_avg, rainfall_month_avg,
                             p_month, month_duration, cooc, lag_joint, selected_copulas, fn_tide, fn_copula, fn_trunk, varname1='Thiessen', varname2='skew', dep_type='copula', figure_joint=True)
total_risk = damage_mod.sum()
print(total_risk)

#We construc the final sample based on the dependence measured and calculate damage
final_events = construct_final_set(selected_copulas, all_events_sampled, all_events_sampled_ind)
print(final_events[['rain_damage','sealevel_damage','cooc_damage']].sum().sum()/n)
final_damage = pd.DataFrame(final_events['month'])

final_damage['damage'] = final_events[['rain_damage','sealevel_damage','cooc_damage']].sum(axis=1)
print(final_damage['damage'].sum()/n)

final_damage_dep = pd.DataFrame(all_events_sampled_dep['month'])
final_damage_dep['damage'] = all_events_sampled_dep[['cooc_damage']].sum(axis=1)
print(final_damage_dep['damage'].sum()/n/1e9)

#%%
damage_mod_full, all_events_sampled_corr, all_events_sampled_fulldep, _, _ = calculate_monthly_damage(best_fit_rain, param_rain, best_fit_skew, param_skew, n, monthly_data, coords, dam, skew_month_avg, rainfall_month_avg,
                                                                                p_month, month_duration, cooc, lag_joint, selected_copulas, fn_tide, fn_copula, fn_trunk, varname1='Thiessen', varname2='skew', dep_type='full corr', figure_joint=True)
total_risk_full = damage_mod_full.sum()
print(total_risk_full)

#We calculate damage
final_damage_dep_full = pd.DataFrame(all_events_sampled_fulldep['month'])
final_damage_dep_full['damage'] = all_events_sampled_fulldep[['cooc_damage']].sum(axis=1)
print(final_damage_dep_full['damage'].sum()/n/1e9)
#%% We compare the cases
case_M = final_damage.groupby('month').mean()
case_M.rename(columns={'damage':'case_M'}, inplace = True)

case_O = final_damage_dep.groupby('month').mean()
case_O.rename(columns={'damage':'case_O'}, inplace = True)

case_OC = final_damage_dep_full.groupby('month').mean()
case_OC.rename(columns={'damage':'case_OC'}, inplace = True)

all_cases = pd.concat([case_M,case_O,case_OC], axis = 1)
all_cases['M_OC'] = all_cases['case_M'] - all_cases['case_OC']
all_cases['M_O'] = all_cases['case_M'] - all_cases['case_O']

#%%
nbins=101
xedges, yedges, Damage_reldensity_meas =  calc_density(final_events, coords, dam, n, nbins, fn_trunk, fig_plot = False, title = 'damage_rainfall_skew_grid.png')

yfmt = tkr.FuncFormatter(numfmt)
month_label = ['Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 

fig = plt.figure(figsize=(8,4))
gs = gridspec.GridSpec(2, 2) 
ax1 = plt.subplot(gs[:,0])

list_plot = []
for month in np.arange(1,13,1):
    print(month)
    list_plot.append(np.array(final_damage.where(final_damage['month']==month).dropna()['damage'].values.flatten())/1e6)
c = 'k'
ax1.boxplot(list_plot, whis=[5,95], showfliers=False, #sym = '.g',
   patch_artist=True, boxprops=dict(facecolor=c, color=c, alpha = 0.6), capprops=dict(color=c),
   whiskerprops=dict(color=c), flierprops=dict(color=c, markeredgecolor=c), medianprops=dict(color='k'), widths=0.7)
ax1.plot(final_damage.groupby('month').mean().index, final_damage.groupby('month').mean()/1e6, 'ok')
ax1.set_ylabel('Damage (M$)', size = 8, labelpad=1)
ax1.set_xticklabels(month_label, fontsize = 7)
ax1.tick_params(axis='y', labelsize=7)

#We plot the yearly damage
ax1.axhline(y = (1.774916e+08/1e6)/12, c='k') 

ax2 = plt.subplot(gs[0,1])
ax2.set_facecolor('whitesmoke')
ax2.scatter(final_events.loc[:, 'rain'], final_events.loc[:, 'sealevel'], s=3, c='k',marker='.', alpha=0.7) 
CS = ax2.contour(xv, yv, Z, levels=np.arange(0,4e9, 2e8), colors='firebrick') #cmap=plt.cm.Reds
ax2.clabel(CS, inline=1, fontsize=7, inline_spacing=0, fmt = func)#'%.2e')#, manual=manual_locations) #inline=1, 
ax2.set_xlim(0, 500)
ax2.set_ylim(0,2000)
ax2.set_ylabel('Sea Level (m)', fontsize = 8)
ax2.tick_params(axis='both', labelsize=7)
ax2.yaxis.set_major_formatter(yfmt)

ax3 = plt.subplot(gs[1,1])
myextent  = [xedges[0],xedges[-1],yedges[0],yedges[-1]]
sc = ax3.imshow(Damage_reldensity_meas.T ,origin='low', extent=myextent, interpolation=None, aspect='auto', norm=colors.LogNorm(vmin=None, vmax=Damage_reldensity_meas.max()),cmap='Reds') #, extent=[80,120,32,0]
ax3.set_xlabel('Rainfall (mm/day)', fontsize = 8)
ax3.set_ylabel('Sea Level (m)', fontsize = 8)
ax3.yaxis.set_major_formatter(yfmt)
ax3.tick_params(axis='both', labelsize=7)
ax3.set_xlim(0,500)  
ax3.set_ylim(0,2000)

cbaxes = fig.add_axes([0.69, 0.435, 0.2, 0.02], frameon= False)  #[left, bottom, width, height] 
cbar = fig.colorbar(sc, ax=ax3, cax = cbaxes, orientation='horizontal', extend ='max') 
cbar.ax.tick_params(labelsize=6)
cbar.ax.tick_params(axis='both', labelsize=6)
cbar.set_label('Damage ($)', labelpad=0, fontsize = 6)

fig.text(0.15, 0.83, '(a)', ha='center', size = 10, fontweight='bold')
fig.text(0.58, 0.84, '(b)', ha='center', size = 10, fontweight='bold')
fig.text(0.58, 0.42, '(c)', ha='center', size = 10, fontweight='bold')
       
fn_out = os.path.join(fn_trunk, 'Paper\Paper5\FIGURES', 'Bivariate', 'Damage_boxplot_density_rainfall_skew.png')
fig.savefig(fn_out, frameon=False, dpi = 300,papertype= 'a4', orientation='portrait', bbox_inches='tight') #transparent=True, 
plt.close()

#%%
yfmt = tkr.FuncFormatter(numfmt)

nbins = 101
xedges, yedges, Damage_reldensity_dep_full =  calc_density(all_events_sampled_fulldep, coords, dam, n, nbins, fn_trunk, fig_plot = False, title = 'damage_rainfall_skew_grid_fullcorr.png')
xedges, yedges, Damage_reldensity_meas =  calc_density(final_events, coords, dam, n, nbins, fn_trunk, fig_plot = False, title = 'damage_rainfall_skew_grid.png')
xedges, yedges, Damage_reldensity_dep =  calc_density(all_events_sampled_dep, coords, dam, n, nbins, fn_trunk, fig_plot = False, title = 'damage_rainfall_skew_grid.png')
# xedges, yedges, Damage_reldensity_excl =  calc_density(all_events_sampled_excl, coords, dam, n, nbins, fn_trunk, fig_plot = False, title = 'damage_rainfall_skew_grid.png')

#%% MAKING A FIGURE OF THE RESULTS
month_label = ['Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
offset = 0.25

fig = plt.figure(figsize=(8,4))
gs = gridspec.GridSpec(2, 3) 
gs.update(wspace=0.35, hspace = 0.3) # set the spacing between axes. 
ax1 = plt.subplot(gs[:,0:2])

list_plot = []
for month in np.arange(1,13,1):
    print(month)
    list_plot.append(np.array(final_damage.where(final_damage['month']==month).dropna()['damage'].values.flatten())/1e6)
c = 'k'
pos = np.arange(len(list_plot))+1+ (offset*(-1)) 
bp1= ax1.boxplot(list_plot, positions= pos, widths = offset, whis=[5,95], showfliers=False, #sym = '.g',
   patch_artist=True, boxprops=dict(facecolor=c, color=c, alpha = 0.6), capprops=dict(color=c),
   whiskerprops=dict(color=c), flierprops=dict(color=c, markeredgecolor=c), medianprops=dict(color=c)) 
#   showmeans=True, meanprops=dict(mec=c, mfc=c, marker='+', markersize=4))
ax1.plot(pos, final_damage.groupby('month').mean()/1e6, 'o', color=c, ms = 5, mec='k', mew=1, zorder = 10)

list_plot = []
for month in np.arange(1,13,1):
    list_plot.append(np.array(final_damage_dep.where(final_damage_dep['month']==month).dropna()['damage'].values.flatten())/1e6)
c = '#ff7f00'
pos = np.arange(len(list_plot))+1+ (offset*(0)) 
bp2= ax1.boxplot(list_plot, positions= pos, widths = offset, whis=[5,95], showfliers=False, #sym = '.g',
   patch_artist=True, boxprops=dict(facecolor=c, color=c, alpha = 0.6), capprops=dict(color=c),
   whiskerprops=dict(color=c), flierprops=dict(color=c, markeredgecolor=c), medianprops=dict(color=c)) 
#   showmeans=True, meanprops=dict(mec=c, mfc=c, marker='+', markersize=4))
ax1.plot(pos, final_damage_dep.groupby('month').mean()/1e6, 'o', color=c, ms=5, mec='k', mew=1, zorder = 10)

list_plot = []
for month in np.arange(1,13,1):
    list_plot.append(np.array(final_damage_dep_full.where(final_damage_dep_full['month']==month).dropna()['damage'].values.flatten())/1e6)
c = '#e41a1c'
pos = np.arange(len(list_plot))+1+ (offset*(+1)) 
bp4= ax1.boxplot(list_plot, positions= pos, widths = offset, whis=[5,95], showfliers=False, #sym = '.g',
   patch_artist=True, boxprops=dict(facecolor=c, color=c, alpha = 0.6), capprops=dict(color=c),
   whiskerprops=dict(color=c), flierprops=dict(color=c, markeredgecolor=c), medianprops=dict(color=c)) 
#   showmeans=True, meanprops=dict(mec=c, mfc=c, marker='+', markersize=4))
ax1.plot(pos, final_damage_dep_full.groupby('month').mean()/1e6, 'o', color=c, ms=5, mec='k', mew=1, zorder = 10)

ax1.set_xticks(np.arange(1,13,1))
ax1.set_xticklabels(month_label, fontsize = 7)
ax1.tick_params(axis='y', labelsize=7)
ax1.set_ylabel('Damage (M$)', size = 8, labelpad=1)
ax1.legend([bp1["boxes"][0], bp2["boxes"][0], bp4["boxes"][0]], ['Modelled, M', 'All co-occurring, $O$', 'All co-occurring and fully correlated, $OC$'], loc='upper left', fontsize=7)

ax2 = plt.subplot(gs[1,2])
res = (Damage_reldensity_dep_full.T) - (Damage_reldensity_meas.T)
bounds = np.linspace(-30, 20, 21)
myextent  = [xedges[0],xedges[-1],yedges[0],yedges[-1]]
sc=ax2.imshow(res/1e6,origin='low',extent=myextent,interpolation=None,aspect='auto', cmap = plt.cm.seismic, norm=MidpointNormalize(midpoint=0,vmin=-30, vmax=10))
ax2.set_xlim(0,300)
ax2.set_ylim(0,2000)

ax2.set_xlabel('Rainfall (mm/day)', fontsize = 8, labelpad = 0)
ax2.set_ylabel('Sea Level (m)', fontsize = 8)
#ax2.xaxis.tick_top()
ax2.yaxis.set_major_formatter(yfmt)
ax2.tick_params(axis='both', labelsize=7)
#ax2.set_xticklabels([])

ax3 = plt.subplot(gs[0,2])
res = (Damage_reldensity_dep.T) - (Damage_reldensity_meas.T)
myextent  = [xedges[0],xedges[-1],yedges[0],yedges[-1]]
ax3.imshow(res/1e6,origin='low',extent=myextent,interpolation=None,aspect='auto', cmap = plt.cm.seismic, norm=MidpointNormalize(midpoint=0,vmin=-30, vmax=10))
ax3.set_xlim(0,300)
ax3.set_ylim(0,2000)

ax3.set_xlabel('Rainfall (mm/day)', fontsize = 8, labelpad = 0)
ax3.set_ylabel('Sea Level (m)', fontsize = 8)
ax3.yaxis.set_major_formatter(yfmt)
ax3.tick_params(axis='both', labelsize=7)

cbaxes = fig.add_axes([0.92, 0.35, 0.01,0.30], frameon= False)  #[left, bottom, width, height]  #[0.70, 0.90, 0.20, 0.02]
cbar = fig.colorbar(sc, ax=ax2, cax = cbaxes, orientation='vertical', extend ='both') 
cbar.ax.tick_params(labelsize=6)
cbar.ax.tick_params(axis='both', labelsize=6)
cbar.set_label('Change in damage (M$)', labelpad=2, fontsize = 7)

fig.text(0.58, 0.81, '(a)', ha='center', size = 10, fontweight='bold')
fig.text(0.84, 0.83, '(b) O - M', ha='center', size = 10, fontweight='bold')
fig.text(0.84, 0.39, '(c) OC - M', ha='center', size = 10, fontweight='bold')
fn_out = os.path.join(fn_trunk, 'Paper\Paper5\FIGURES', 'Univariate', 'boxplot_damage_sensitivity_dependence.png')

fig.savefig(fn_out, frameon=False, dpi = 300,papertype= 'a4', orientation='portrait', bbox_inches='tight') #transparent=True, 
plt.close()