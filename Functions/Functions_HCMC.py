 # -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:35:14 2019

@author: ACN980
"""

import os, glob, sys
import calendar
import pandas as pd
import numpy as np
import math
import warnings
import scipy
import scipy.stats as sp
import scipy.signal as ss
from sklearn.linear_model import LinearRegression
from datetime import date
import matplotlib.pyplot as plt
import itertools
from scipy.interpolate import Rbf
import matplotlib as mpl
warnings.filterwarnings("ignore")


def make_pseudo_obs(var1, var2):
    pseudo1 = var1.rank(method='first', ascending = True)/(len(var1)+1)
    pseudo2 = var2.rank(method='first', ascending = True)/(len(var2)+1)
    return pseudo1, pseudo2

def day_to_month_rad_year(data):
    """ Converts the Julian day of a date to radian (to perform directional statistics).
    input: data is  a univariate series with Timestamp index
    output: return a DataFrame with the angle in rad and corresponding x and y coordinate"""
    day_of_year =  data.apply(lambda x: x.timetuple().tm_yday)
    day_of_year.name = 'day_of_yr'
    
    month_of_year =  data.apply(lambda x: x.timetuple().tm_mon)
    month_of_year.name = 'month_of_yr'

    leap_year =  data.apply(lambda x: x.is_leap_year)
    length_year = data.apply(lambda x: 365)
    length_year[leap_year] = 366
    length_year.name = 'length_of_yr'
    
    output = pd.concat([data,day_of_year,length_year, month_of_year], axis = 1)
    output['angle_rad'] = output['day_of_yr']*2*math.pi/output['length_of_yr']
    output = output.assign(**{'x': output.angle_rad.apply(lambda x: math.cos(x))})
    output = output.assign(**{'y': output.angle_rad.apply(lambda x: math.sin(x))})   
    return output

def select_epoch_pairs(cont_data, epoch = 'AS', nbofdays = 5, nbofrepet = 500, test_ind = False):
    #epoch = 'AS'  #weekly = 'W', daily = 'D', monthly = 'MS' 

    result_max = pd.DataFrame(data = None)
    for window_time in cont_data.groupby(pd.Grouper(freq=epoch)): 
        if window_time[1].empty: continue
        ts_data = pd.DataFrame(window_time[1]) #Just selecting the data
        max_pairs = ts_data.max(axis=0).to_frame().transpose()
        max_time = ts_data.idxmax(axis = 0).to_frame().transpose() 
        max_time.rename(columns={max_time.columns[0]: max_time.columns[0]+'_date', max_time.columns[1]: max_time.columns[1]+'_date'}, inplace = True) 
        result = pd.concat([max_pairs, max_time], axis = 1)
        result_max = pd.concat([result_max, result], axis = 0, sort = False)
    
    if test_ind == True:
        result_ind_final = pd.DataFrame(data = None, index = np.arange(result_max.shape[0]))
        #Random interactions
        for j in np.arange(nbofrepet):
            date1_2 = np.random.randint(1, nbofdays+1, size = (result_max.shape[0],2))
            result_ind = pd.DataFrame(data = abs(date1_2[:,0]-date1_2[:,1]))
            result_ind_final = pd.concat([result_ind_final, result_ind], axis = 1)
    else:
        result_ind_final = []
        
    return (result_max, result_ind_final)

def import_skew(fn2):
    dateparse = lambda x: pd.datetime.strptime(x, '%d-%m-%Y %H:%M:%S')
    skew = pd.read_csv(fn2, parse_dates = True, date_parser=dateparse, index_col = 'Date', usecols = ['Date','skew '])
    skew.rename(columns = {skew.columns[0]:'skew'}, inplace = True)
    
    skew2 = skew.reset_index()
    ind_null = skew2[skew2['skew'].isnull()].index.tolist()
    for i in ind_null:
        skew2.loc[i-1,'skew'] = np.nan
        skew2.loc[i+1,'skew'] = np.nan
    skew2.set_index('Date', inplace = True)
    return skew2

def get_skew_surge(pandas_twl,pandas_tide,distance=6):
    '''
    Function from Job Dullaart
    The goal of this function is to compute annual maximum skew surge levels
    Input variables: 
        pandas_twl: total water levels time series provided as a pandas dataframe
        pandas_tide: tidal levels time series provided as a pandas dataframe
        distance: minimum number of timesteps between two tidal minima's.
        If not defined, set to 36.
        
    Return:
        skew_surge_yearmax: pandas dataframe with annual maximum skew surge levels, sorted by height
    '''    

    #1. reverse tidal levels and find indexes of tide minima's
    tide_array_inverse = pandas_tide.waterlevel.values*-1
    tide_minima_index, tide_minima_values = ss.find_peaks(tide_array_inverse, distance=distance, height = -10)
    tide_time_array = pandas_tide.index.values
    peaks_tide_time = tide_time_array[tide_minima_index.tolist()]
    
    #2. find maximum total water level and maximum tidal level between each two tidal minima's
    skew_surges=[]
    skew_surge_dates=[]
    max_tides=[]
    high_tide_dates=[]
    print('number of timesteps to be processed: ',len(peaks_tide_time)-1)
    print('number of timesteps processed: ')
    for ii in range(len(peaks_tide_time)-1):
        if ii%1000==0:
            print(ii)
        
        t1 = peaks_tide_time[ii]
        t2 = peaks_tide_time[ii+1]
        max_twl = pandas_twl[t1:t2].waterlevel.max()
        max_tide = pandas_tide[t1:t2].waterlevel.max()
        skew_surges.append(max_twl-max_tide)
        max_tides.append(max_tide)
        skew_surge_dates.append(pandas_twl[t1:t2].waterlevel.idxmax())
        high_tide_dates.append(pandas_tide[t1:t2].waterlevel.idxmax())
    
    #3. create a dataframe of the annual maximum skew surge levels together with the timestamp of the maximum total water level
    df = pd.DataFrame(data={'skew_surge':skew_surges},index=skew_surge_dates)
    df2 = pd.DataFrame(data={'high_tide':max_tides},index=high_tide_dates)   
    return df, df2

def collect_rainfall(fn, figure_plotting = False): #####

    all_files = glob.glob(os.path.join(fn,'daily_*_rainfall_cleaned.csv'))
    
    result = pd.DataFrame(data = None, index = pd.date_range(start = pd.datetime(1978,1,1), end = pd.datetime(2018,12,31), freq = 'D'))
    for file in all_files:
        print(file)
        rain = pd.read_csv(file, index_col = 'date', dtype={'value':np.float32}, parse_dates = True)
        name = file.split('_')[1]
        rain.rename(columns={'value':name}, inplace = True)
        if figure_plotting == True:
            plt.figure()
            plt.plot(rain.index, rain[name])
            plt.show()
            plt.title(name)
            plt.ylim(0, 250)
        result = pd.merge(result, rain, how = 'outer', left_index = True, right_index = True, sort = True)
    result = result.loc[result.index.isin(pd.date_range(start = pd.datetime(1978,1,1), end = pd.datetime(2018,12,31), freq = 'H')),:].copy()
    if figure_plotting == True:
        result.plot()
        plt.show()
        
        cmap = plt.cm.seismic
        bounds = np.linspace(-1,1,21)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        names = result.columns
        correlations = result.corr()
        # plot correlation matrix
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1, cmap=cmap, norm=norm)
        fig.colorbar(cax)
        ticks = np.arange(0,len(names),1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        plt.show()
    
    return result

def thiessen_rain(fn_thiessen, rainfall):
    weights = pd.read_csv(fn_thiessen, usecols = ['Station','Weight'])
    for i in weights.index:
        weights.loc[i,'Station'] = weights.loc[i,'Station'].replace(" ", "") 
    weights = weights.set_index('Station').transpose()
    
    sel_rainfall = rainfall.loc[:,weights.columns]
    for col in sel_rainfall.columns:
#        print(col)
        sel_rainfall[col] = sel_rainfall[col].apply(lambda x: x*weights.loc[weights.index[0],col])
    thiessen_rainfall = pd.DataFrame(sel_rainfall.sum(axis = 1))
    thiessen_rainfall.rename(columns={thiessen_rainfall.columns[0]:'Thiessen_sum'}, inplace = True)
    return thiessen_rainfall


def calc_avg_max_min_rainfall(result, threshold=40): ####
    rainfall_years = pd.DataFrame(data = None, index = pd.date_range(start = pd.datetime(1978,1,1), end = pd.datetime(2018,12,31), freq = 'D'))
    for col in result.columns:
        print(col)
        ts_rain, years_removed = keep_full_years(sel = result[col].copy(), threshold = threshold)
        ts_rain = ts_rain.fillna(0)
        rainfall_years = pd.merge(rainfall_years,ts_rain,how = 'outer', left_index = True, right_index = True, sort = True)
    res_isna = rainfall_years.isna().sum(axis=1)
    average = rainfall_years.where(res_isna<=3).mean(axis=1)
    max_values = rainfall_years.where(res_isna<=3).max(axis=1)
    min_values = rainfall_years.where(res_isna<=3).min(axis=1)
    
    rainfall_years['average'] = average
    rainfall_years['maximum'] = max_values
    rainfall_years['minimum'] = min_values
    return rainfall_years

def import_monthly_rain(fn2):
    allfiles = glob.glob(os.path.join(fn2, 'NewRain\TRENDS\MONTH_CORRECTED', 'Thiessen_*.csv'))
    all_rain = pd.DataFrame(data=None)
    for file in allfiles:
        month = pd.read_csv(file, index_col = 'Year', parse_dates=True)  
        month.rename(columns={month.columns[0]:'Thiessen'}, inplace = True)
        all_rain = pd.concat([all_rain, month], axis = 0)
    return all_rain

def collect_swl(fn, figure_plotting = False):
    all_files = glob.glob(os.path.join(fn,'hourly_*_swl_cleaned.csv'))
    
    result = pd.DataFrame(data = None, index = pd.date_range(start = pd.datetime(1980,1,1), end = pd.datetime(2018,12,31), freq = 'H'))
    for file in all_files:
        print(file)
        rain = pd.read_csv(file, index_col = 'date', dtype={'value':np.float32}, parse_dates = True)
        name = file.split('_')[1]
        rain.rename(columns={rain.columns[0]:name}, inplace = True)
        if figure_plotting == True:
            plt.figure()
            plt.plot(rain.index, rain[name])
            plt.show()
            plt.title(name)
#            plt.ylim(0, 250)

        result = pd.merge(result, rain, how = 'outer', left_index = True, right_index = True, sort = True)
    
    result = result.loc[result.index.isin(pd.date_range(start = pd.datetime(1978,1,1), end = pd.datetime(2018,12,31), freq = 'H')),:].copy()
    
    if figure_plotting == True:
        result.plot()
    
    return result

def keep_full_years(sel, threshold):  ####
    """
    
    > sel: is a time-series of the rainfall with a datetime as index
    > threshold: is the minimum number of days to consider a year valid. Here this is somewhat
    ambiguous what a good threshold is as there might be a lot of 0 if it doesn't rain in a year
    
    """      
    
    check = sel.groupby(sel.index.map(lambda x: x.year)).count()
    years_to_remove = check.where(check<threshold).dropna().index.values
    ts = pd.DataFrame(data = sel.copy())
    ts.index.rename('date', inplace = True)
    ts.reset_index(drop = False, inplace = True)
    ts['year'] = ts.date.dt.year    
    ts = ts.set_index('year').drop(labels = years_to_remove).set_index('date')
    return ts, years_to_remove

#%%
def median_detrend_wNa(data, tot_periods, min_periods, figure_plotting = False):
    """Removes trends and SL variation by substracting the moving median
    tot_periods is the number of steps considered to calculate the median
    min_periods is the minimum number of periods considered to calculate the median"""
    
    inland_day_median = data.rolling(tot_periods, min_periods=min_periods, center=False).median()
    inland_day_median = inland_day_median.fillna(method='ffill').copy()
    inland_day_median = inland_day_median.fillna(method='bfill').copy()
    inland_day_detrend = data - inland_day_median
    inland = inland_day_detrend.copy()
    
    if figure_plotting == True:
        plt.figure()
        inland_day_median.plot()
        plt.show()
        
        f, ax = plt.subplots(nrows=len(data.columns), ncols=2, sharex=True)
        ax = ax.reshape(-1)
        for i in np.arange(len(data.columns)):
            print(i)
            ax[int(i*2)].plot(data.index, data.iloc[:,i], '-k', inland_day_median.index, inland_day_median.iloc[:,i], '-r')
            ax[int((i*2)+1)].plot(inland.index, inland.iloc[:,i], '-b')
        plt.show()
        
        plt.figure()
        inland.plot()
        plt.show()

    return inland
   

def lin_detrend_wNa(data, ref_date, remove_means = True, figure_plotting = False):
    """arguments:
        data is a pd.Series with date as index 
        ref_date: if a date is mentioned, remove trend taking the swl on this date as ref
        remove_means: if True, centers the detrended ts around 0
        figure_plotting: if True returns a figure of both ts
    returns:
        the linearly detrended data with time as index"""
     
    y = np.array(data)
    x = np.arange(0,len(y),1)
    not_nan_ind = ~np.isnan(y)
    m, b, r_val, p_val, std_err = sp.linregress(x[not_nan_ind],y[not_nan_ind])
    if remove_means == True:
        detrend_y = y - (m*x + b)
    elif ref_date is not None:
        x_0 = np.flatnonzero(data.index == ref_date)
        detrend_y = y - (m*x + b) + (m * x_0 + b)
    else: 
        detrend_y = y - (m*x)
    
    print('Linear trend is: ', m)
    print('p-value is: ', p_val)
    
    if figure_plotting == True:
        plt.figure()
        plt.plot(x, y, label = 'original')
        plt.plot(x, detrend_y, label = 'detrended')
        plt.legend()
    
    result = pd.DataFrame(data =  detrend_y, index = data.index, columns = [data.name])
    return result


#%% TOP EVENTS
def top_n_events_per_year_tide(x, n_top, label_value = 'tide', time_frequency = 'AS'):
   
    x=pd.DataFrame(x, columns=[label_value])
    x.rename(columns={x.columns.values[0]:label_value}, inplace = True)
    x.index.rename('index', inplace = True)
    
    y= x.groupby(pd.Grouper(freq=time_frequency)).apply(lambda g: g.nlargest(n = n_top, columns = label_value))
  
    res = pd.DataFrame(y)
    res['year'] = [i[0].year for i in res.index]
    res['date'] = [i[1] for i in res.index]
#    res.reset_index(inplace=True, drop = True)  
    return res

def top_n_events_per_year_rainfall(x, n_top, label_value = 'tide', time_frequency = 'AS'):
    x.rename(columns={x.columns.values[0]:label_value}, inplace = True)
    x.index.rename('index', inplace = True)
    
    y= x.groupby(pd.Grouper(freq=time_frequency)).apply(lambda g: g.nlargest(n = n_top, columns = label_value))
  
    res = pd.DataFrame(y)
    res['year'] = [i[0].year for i in res.index]
    res['date'] = [i[1] for i in res.index]
    res.reset_index(inplace=True, drop = True)  
    return res

#%% FFT SKEW

def detrend_fft(daily_skew, fillnavalue=0, frequency = 1. / 365, figure_plotting = 0):
    """Takes a ts with no Nan and continuous time series
    frequency is the corresponding frequency of the index in year (daily --> 1/365)"""
    import scipy.fftpack     
    
    skew_day = daily_skew.fillna(fillnavalue)

    skew_values = skew_day.iloc[:,0].copy()
    skew_fft = scipy.fftpack.fft(np.array(skew_values))
    
    skew_psd = np.abs(skew_fft) ** 2 #Taking the power spectral density
    fftfreq = scipy.fftpack.fftfreq(len(skew_psd), frequency)
    i = fftfreq > 0 #only taking positive frequencies

    temp_fft_bis = skew_fft.copy()
    temp_fft_bis[np.abs(fftfreq) > 1.0] = 0 #    temp_fft_bis[np.abs(fftfreq) > 1.1] = 0
    
    skew_slow = np.real(scipy.fftpack.ifft(temp_fft_bis))
    daily_skew = pd.DataFrame(daily_skew.iloc[:,0] - skew_slow)
    #skew_slow = pd.DataFrame(index=daily_skew.index, data=skew_slow)
    
    #daily_skew_runmean = skew_day - skew_day.rolling(365, min_periods=150, center=True).mean()
    
    if figure_plotting == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(fftfreq[i], 10 * np.log10(skew_psd[i]))
        ax.set_xlim(0, 5)
        ax.set_xlabel('Frequency (1/year)')
        ax.set_ylabel('PSD (dB)')
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.plot(skew_day.index, skew_day, '-b', lw = 0.5)#skew_day.plot(ax=ax, lw=.5)
        ax.plot(skew_day.index, skew_slow, '-r', lw = 2)
        ax.plot(skew_day.index, skew_day.rolling(365, min_periods=150, center=True).mean(), '-g', lw = 1.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Skew surge')
        plt.show()
    
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.plot(daily_skew.index, daily_skew.iloc[:,0],'-b', lw = 0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Skew surge')
        plt.show()
    
    return daily_skew

def remove_NaN_skew(skew):
    isna_skew = skew[skew[skew.columns[0]].isnull()].index
    for na_date in isna_skew:
        # print(na_date)
        i_ind = np.flatnonzero(skew.index == na_date)
        bef = i_ind - 1
        aft = i_ind + 1
        if bef>0:
            skew.iloc[bef,0] = np.nan
        if aft < len(skew):
            skew.iloc[aft,0] = np.nan
    return skew.copy()

def import_monthly_skew(fn):
    date_parser = lambda x: pd.datetime.strptime(x, "%d-%m-%Y %H:%M:%S")
    fn_skew  = os.path.join(fn,'skew_WACC_VungTau_Cleaned_Detrended_Strict_sel_const.csv')
    skew = pd.read_csv(fn_skew, parse_dates = True, date_parser= date_parser, index_col = 'Date')
    skew.rename(columns = {skew.columns[0]:'skew'}, inplace = True)
    skew = remove_NaN_skew(skew)
    skew_day = skew.resample('D').max()
    skew_detrend = detrend_fft(skew_day, fillnavalue=0, frequency = 1./(2*365), figure_plotting =0)
    skew_detrend_day = skew_detrend.resample('D').max()
    skew_month = skew_detrend_day.resample('M').max()
    return skew_month

#%%
def extract_MM(tide, freq='MS', label='sealevel'):
    sel_sel = pd.concat([tide,tide], axis = 1).dropna()
    dates_tide = select_epoch_pairs(sel_sel, epoch = freq, nbofdays = 5, nbofrepet = 500, test_ind = False)
    dates_MM_tide = dates_tide[0].reset_index(drop=True).iloc[:,[0,-1]]
    dates_MM_tide['index'] = [pd.to_datetime(date(d.year, d.month, calendar.monthrange(d.year, d.month)[-1])) for d in dates_MM_tide[f'{label}_date']]
    dates_MM_tide[f'{label}_date'] = [pd.to_datetime(date(d.year, d.month, d.day)) for d in dates_MM_tide[f'{label}_date']]
    dates_MM_tide.set_index('index',inplace = True)
    return dates_MM_tide

def make_cmap_month():
    # COLORSCALE
    # get discrete colormap
    n_clusters = 15
    cmap = plt.get_cmap('hsv', n_clusters)
    colors = cmap(np.linspace(0.05, 0.90, 13))
    
    cmap2 = mpl.colors.ListedColormap(colors)
    
    bounds = np.arange(1,14,1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap2.N)
    
    bounds_day = np.arange(1,366,1)
    norm_day = mpl.colors.BoundaryNorm(bounds_day, cmap2.N)
    return cmap2, norm

def ax_joint_mm(var1, var2, ax, label='_date', lag_joint=0, ls=7, formatting = True, plotting=True):
    var1_name = var1.columns[~var1.columns.str.endswith(label)][0]
    var2_name = var2.columns[~var2.columns.str.endswith(label)][0]    
    var1_date = var1_name+label
    var2_date = var2_name+label
    
    both = pd.concat([var1, var2], axis = 1).dropna()  
    both.reset_index(inplace = True, drop = True)
    
    Joint_MM = both[[var1_date,var2_date]].copy()
    Joint_MM['diff_days'] = Joint_MM.loc[:,var1_date]-Joint_MM.loc[:,var2_date]
    Joint_MM['abs_days'] = np.abs(Joint_MM['diff_days'].dt.days)
#    Joint_MM.reset_index(drop=True, inplace = True)
    
    Joint_MM  = pd.concat([both, Joint_MM[['diff_days','abs_days']]], axis = 1)    
    
    joint_points_MM = Joint_MM.where(Joint_MM.abs_days < lag_joint+1).dropna()
    if len(joint_points_MM)>0:
        time_of_year = day_to_month_rad_year(data = joint_points_MM.loc[:,var1_date])
        time_of_year.rename(columns={time_of_year.columns[0]:'date'}, inplace = True)
        time_of_year = time_of_year.set_index('date').reset_index()
    
    cmap2, norm = make_cmap_month()
    
    if plotting == True:
        ax.scatter(both.loc[:,var1_name], both.loc[:,var2_name], marker = 'o', c = 'white', edgecolors='k', linewidths=0.3, alpha = 0.5, s=6)
        if len(joint_points_MM)>0:
            ax.scatter(joint_points_MM.loc[:,var1_name], joint_points_MM.loc[:,var2_name], marker = 'o', edgecolors ='k', linewidths=0.3, c = time_of_year['month_of_yr'], cmap=cmap2, alpha = 1, s=15, norm=norm)
        
        if formatting == True:
            ax.set_xlabel(var1_name,fontsize=ls)
            ax.set_ylabel(var2_name,fontsize=ls)
            ax.tick_params(axis='both', labelsize=ls)
        
    return Joint_MM

def joint_mm_all_cooc(Joint_MM, max_lag = 7, label = '_date'):
    var1_name = Joint_MM.columns[~Joint_MM.columns.str.endswith(label)][0]
    var2_name = Joint_MM.columns[~Joint_MM.columns.str.endswith(label)][1]
    var1_date = var1_name+label
    var2_date = var2_name+label
    
    var1_result ={}
    month = np.arange(1,13,1)
    dates_month = day_to_month_rad_year(data = Joint_MM.loc[:,var1_date])
    for m in month:
        print(m)
        var1_result[m] ={}
        sel = Joint_MM.where(dates_month.month_of_yr == m).dropna().copy()
        var1_result[m]['data'] = sel      

        corr_sel_MM = sp.kendalltau(sel.loc[:,var2_name].values, sel.loc[:,var1_name].values, nan_policy='omit')  
        var1_result[m]['data_corr'] = corr_sel_MM
        
        co_occur_n_samples = pd.DataFrame(data = None, index = ['N'], columns = np.arange(0,max_lag+1))
        for lag_joint in np.arange(0,max_lag+1):
            joint_points_sel = sel.where(sel.abs_days < lag_joint+1).dropna()    
            co_occur_n_samples.loc['N',lag_joint] = len(joint_points_sel)
        
        var1_result[m]['co_occur_n_samples'] = co_occur_n_samples    
    return var1_result

def joint_mm_permonth(Joint_MM, lag_joint=0, label = '_date'):
    var1_name = Joint_MM.columns[~Joint_MM.columns.str.endswith(label)][0]
    var2_name = Joint_MM.columns[~Joint_MM.columns.str.endswith(label)][1]
    var1_date = var1_name+label
    var2_date = var2_name+label
    
    var1_result ={}
    month = np.arange(1,13,1)
    dates_month = day_to_month_rad_year(data = Joint_MM.loc[:,var1_date])
    for m in month:
        print(m)
        var1_result[m] ={}
        sel = Joint_MM.where(dates_month.month_of_yr == m).dropna().copy()
        var1_result[m]['data'] = sel      

        corr_sel_MM = sp.kendalltau(sel.loc[:,var2_name].values, sel.loc[:,var1_name].values, nan_policy='omit')  
        var1_result[m]['data_corr'] = corr_sel_MM
        joint_points_sel = sel.where(sel.abs_days < lag_joint+1).dropna()    
        if len(joint_points_sel)>0:
            time_of_year = day_to_month_rad_year(data = joint_points_sel.loc[:,var1_date])
            joint_points_sel = pd.concat([joint_points_sel, time_of_year['month_of_yr']], axis = 1)
        try:
            corr_joint_points_sel = sp.kendalltau(joint_points_sel.loc[:,var2_name].values, joint_points_sel.loc[:,var1_name].values, nan_policy='omit')    
    
        except:
            corr_joint_points_sel = np.nan

        var1_result[m]['co_occur_data'] = joint_points_sel
        var1_result[m]['co_occur_corr'] = corr_joint_points_sel
        var1_result[m]['co_occur_n_samples'] = len(joint_points_sel.dropna())
    return var1_result

def plot_cooc_CI(result_pair, ax, lag_joint =0, c = 'r', size = 5, label = None, background = True):
    fm = os.path.join(r'E:\surfdrive\Documents\Master2019\Thomas\data\Binomial')
    #month_label = ['Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
    month_label = ['J', 'F', 'M', 'A', 'M','J','J','A','S','O','N','D'] 

    len_month = [31,28,31,30,31,30,31,31,30,31,30,31]    
    all_exp = pd.DataFrame()
    q50=pd.DataFrame()
    q2=pd.DataFrame()
    q97=pd.DataFrame()
    nb_years = pd.DataFrame(index = np.arange(1,13,1), columns=['nb'])
    obs_cooc = pd.DataFrame(index = np.arange(1,13,1), columns = np.arange(0,8,1))
    for i in np.arange(1,13,1):
        obs_cooc.loc[i,:] = result_pair[i]['co_occur_n_samples'].loc['N',:]
        nb_years.loc[i,'nb'] = 37#len(result_pair[i]['data'])
    
    #We read the table 
    for i in nb_years.index:
        print(i)
        case = os.path.join(str(len_month[i-1])+'days',str(nb_years.loc[i, 'nb'])+'years')
        data = pd.read_csv(os.path.join(fm,case,'Independent_Binomial_Expectation.csv'), index_col = 'index')
        data.rename(index={'expectation':i}, inplace = True)
        all_exp = pd.concat([all_exp, data], axis = 0) 
        
        ci_data = pd.read_csv(os.path.join(fm,case,'Independent_Binomial_Expectation_CI.csv'), index_col = 'quantile')
        ci_data.rename(index={'quantile':i}, inplace = True)
        q2 = pd.concat([q2,pd.DataFrame(ci_data.loc['q2.5',:]).transpose().rename(index={'q2.5':i})], axis = 0)
        q50 = pd.concat([q50,pd.DataFrame(ci_data.loc['q50',:]).transpose().rename(index={'q50':i})], axis = 0)
        q97 = pd.concat([q97,pd.DataFrame(ci_data.loc['q97.5',:]).transpose().rename(index={'q97.5':i})], axis = 0)
    
    # f,ax = plt.subplots(nrows=1, ncols = 1, figsize=(8,3))
    #ax = ax.reshape(-1)
    if background:
        lw=1.3
#        ax.fill_between(np.arange(1,13,1),q2.loc[:,str(lag_joint)].values, q97.loc[:,str(lag_joint)].values, color = 'k', alpha = 0.3)
        ax.plot(all_exp.index, all_exp.loc[:,str(lag_joint)], '--', color = 'k', linewidth = lw) #'*', mec = 'k', mfc = 'k', markersize = size/1.5)  
        length = size
        space = size
        # if c == 'y':
        #     c='orange'
        #     lw = 1
        #     length = 5
        #     space = 10
            
        ax.plot(q2.index, q2.loc[:,str(lag_joint)], ':', color = 'k', linewidth = lw)#, dashes=(size/2, size/2)) #length of 5, space of 1
        ax.plot(q97.index, q97.loc[:,str(lag_joint)], ':', color = 'k', linewidth = lw)#, dashes=(length/2, space/2)) #length of 5, space of 1)
        ax.grid(lw=0.5)
    ax.plot(obs_cooc.index, obs_cooc.loc[:,lag_joint], 'o', markersize = size, mfc = c, mec='k', mew=0.5)
    ax.annotate(label, (0.05,0.90), xycoords='axes fraction', fontsize=8, weight="bold")
    ax.set_xlim(0.7,12.3)
    ax.set_ylim(-0.2,6)
    ax.set_yticks(np.arange(0,7,1))
    ax.set_xticks(np.arange(1,13,1))
    ax.set_xticklabels(month_label, fontsize = 7)

def kendall_CI(bs_data, var1_name = 'Thiessen', var2_name='skew', label='_date', iterations = 500):
    #Calculate kendall CI
    kend_bs = pd.Series(index = np.arange(iterations))
    for x in np.arange(iterations):
        rand1 = bs_data[var1_name].sample(n=bs_data.shape[0], replace=True, axis=0)
        kend_bs[x] = sp.kendalltau(rand1.values, bs_data.loc[:,var2_name].values, nan_policy='omit')[0] 
    kend_025 = kend_bs.quantile(q=0.025, interpolation='linear')
    kend_975 = kend_bs.quantile(q=0.975, interpolation='linear')
    return kend_025, kend_975

def kendall_CI_allmonth(result_pair, var1_name = 'Thiessen', var2_name='skew', label='_date', iterations = 500):
    kend_025 = pd.DataFrame(index=np.arange(1,13,1), columns=['q2.5'])
    kend_975 = pd.DataFrame(index=np.arange(1,13,1), columns=['q97.5'])
    for i in np.arange(1,13,1):
        bs_data = result_pair[i]['data']
        kend_025.loc[i,'q2.5'], kend_975.loc[i,'q97.5']= kendall_CI(bs_data, var1_name = var1_name, var2_name=var2_name, label=label, iterations = iterations)
    return kend_025, kend_975


def ax_kendall_mm(result_pair, ax, var1_name = 'Thiessen', var2_name='skew', label='_date', iterations = 500, c = 'k', size = 7, background=True):
    month_label = ['Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
    #SKEW-RAIN
    kendall = pd.DataFrame(index = np.arange(1,13,1), columns=['kendall', 'p_value'])
    for i in np.arange(1,13,1):
        kendall.loc[i,'kendall'] = sp.kendalltau(result_pair[i]['data'].loc[:,var1_name].values, result_pair[i]['data'].loc[:,var2_name].values, nan_policy='omit')[0] 
        kendall.loc[i,'p_value'] = sp.kendalltau(result_pair[i]['data'].loc[:,var1_name].values, result_pair[i]['data'].loc[:,var2_name].values, nan_policy='omit')[1] 
    
    if background == True:
        lw=1.3
        length = size
        space = size
        if c == 'y':
            c='orange'
            lw = 1
            length = 5
            space = 10
        
        k25, k975 = kendall_CI_allmonth(result_pair, var1_name = var1_name, var2_name = var2_name, label = label, iterations = iterations)
        ax.plot(k25.index, k25.loc[:,'q2.5'], '--', color = c, linewidth = lw, dashes=(size, size)) 
        ax.plot(k975.index, k975.loc[:,'q97.5'], '--', color = c, linewidth = lw, dashes=(length, space))
    ax.axhline(0, color = 'black', lw=1)# , xmin=1, xmax=12, color = 'k', lw=0.5, ls='-')
    ax.plot(kendall.index, kendall.loc[:,'kendall'], 'o', markersize = size, color = c, mfc = c, mec='k', mew=0.5)
    ax.set_xlim(0.9,12.1)
    ax.set_xticks(np.arange(1,13,1))
    ax.set_xticklabels(month_label, fontsize = 7)
#%%
def get_samples_from_dist(n, params, dist_type):
    dist = getattr(sp, dist_type)
    if len(params) == 3: #shape, loc and scale
        data = dist.rvs(params[0], params[1], params[2], n)
    elif len(params) == 2:#loc and scale
        data = dist.rvs(params[0], params[1], n)
    elif len(params) == 1:
        data = dist.rvs(params[0], n)
    else:
        print('Something is wrong!')    
    return data

def get_ICDF_from_dist(q, params, dist_type):
    dist = getattr(sp, dist_type)
    if len(params) == 3: #shape, loc and scale
        data = dist.ppf(q, params[0], params[1], params[2])
    elif len(params) == 2:#loc and scale
        data = dist.ppf(q, params[0], params[1])
    elif len(params) == 1:
        data = dist.ppf(q, params[0])
    else:
        print('Something is wrong!')    
    return data

def get_line_pt_RP_fit(exc_prob_x, params, dist_type):
    dist = getattr(sp, dist_type)
    if len(params) == 3: #shape, loc and scale
        rp_y = dist.isf(exc_prob_x, params[0], params[1], params[2])
    elif len(params) == 2:#loc and scale
        rp_y = dist.isf(exc_prob_x, params[0], params[1])
    elif len(params) == 1:
        rp_y = dist.isf(exc_prob_x, params[0])
    else:
        print('Something is wrong!')
    
    return rp_y

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

def empirical_RP(data):
    #Calculating empirical
    emp_p = pd.DataFrame(data=data)
    emp_p['rank'] = emp_p.iloc[:,0].rank(axis=0, ascending=False)
    emp_p['exc_prob'] = emp_p['rank']/(emp_p['rank'].size+1) #change this line with what Anaïs sends to me, but is already correct
    emp_p['cum_prob'] = 1 - emp_p['exc_prob']
    emp_p['emp_rp'] = 1/emp_p['exc_prob']
    return emp_p


# def get_line_pt_RP_fit(exc_prob_x, data, params, dist_type):
#     dist = getattr(sp, dist_type)
#     if len(params) == 3: #shape, loc and scale
#         #print('Skew param ', f.fitted_param[dist_type][0])
#         print('Check param ', params)
#         inv_cdf_dist = dist.sf(data, params[0], params[1], params[2])
#         rp_y = dist.isf(exc_prob_x, params[0], params[1], params[2])
#     elif len(params) == 2:#loc and scale
#         inv_cdf_dist = dist.sf(data, params[0], params[1])
#         rp_y = dist.isf(exc_prob_x, params[0], params[1])
#     elif len(params) == 1:
#         inv_cdf_dist = dist.sf(data, params[0])
#         rp_y = dist.isf(exc_prob_x, params[0])
#     else:
#         print('Something is wrong!')
    
#     return inv_cdf_dist, rp_y

#%%
def plot_damage_grid(damage_grid, alphas, ax, rstride, ctride, cmap, norm):        
    coords, dam = damage_surface_coord_z(damage_grid)

    RBFi = Rbf(coords[:,0], coords[:,1], dam, function='linear', smooth=0)    
    rain_int = list(np.arange(0,int(damage_grid.index.max())+10, 10))
    sl_int = list(np.arange(0,int(damage_grid.columns.max())+100,100))
    all_S = np.array([ x for x in itertools.product(rain_int,sl_int)])    
    all_dam = RBFi(all_S[:,0], all_S[:,1])    
    X, Y = np.meshgrid(rain_int,  sl_int, indexing='ij')
    damage_grid_plot = damage_surface_df_z(all_S, all_dam)
    Z = damage_grid_plot.to_numpy()
    
    damage_grid_scenario = damage_grid.drop(0,axis=1)
    coords_sce, dam_sce = damage_surface_coord_z(damage_grid_scenario)

    # fig = plt.figure(figsize=[8.5, 4])
    # gs = GridSpec(2, 2, left=0.05, bottom=0.1, right=0.95, top=0.90, width_ratios=[1,1], height_ratios=[1,1], wspace=0.40, hspace=0.50)#, width_ratios=None, height_ratios=[0.9,0.9,0.9,0.9,0.9,0.9])
    # ax = fig.add_subplot(gs[:, 0], projection='3d', azim=-60, elev=25)

    ax.plot_wireframe(X, Y/1000, Z/1e6, color='grey',linewidth=1, antialiased=True, rstride=rstride, cstride=ctride, zorder=1, alpha=0.5) #plot_surface
#    alphas = np.linspace(0.2,1,len(damage_grid_scenario.columns))
    for i in np.arange(0,len(damage_grid_scenario.columns)):
        print(i)
        ax.scatter(damage_grid_scenario.iloc[:,i].index, np.repeat(damage_grid_scenario.columns[i]/1000, len(damage_grid_scenario.iloc[:,i])),damage_grid_scenario.iloc[:,i].values/1e6, c=damage_grid_scenario.iloc[:,i].index,  s = 35, edgecolors='k', linewidths=0, alpha=alphas[i], cmap=cmap, norm=norm, zorder=10, depthshade=False) #alpha=alphas[i], 
        ax.scatter(damage_grid_scenario.iloc[:,i].index.values, np.repeat(damage_grid_scenario.columns[i]/1000, len(damage_grid_scenario.iloc[:,i])),damage_grid_scenario.iloc[:,i].values/1e6, facecolor=(0,0,0,0),  s = 35, edgecolor='k', linewidths=1, depthshade=False, zorder=11)

    #ax.plot_wireframe(xv, yv/1000, Z/1e6, color='black',linewidth=0.2)
    ax.set_xlabel('Rainfall (mm/day)', size = 8)
    ax.set_ylabel('Sea Level (m)', size = 8)
    ax.set_zlabel('Damage (M$)', size = 8)


def damage_surface_coord_z(damage_grid):
    coords = np.zeros((damage_grid.shape[0]*damage_grid.shape[1],2))
    dam = np.zeros((damage_grid.shape[0]*damage_grid.shape[1],1))
    z = 0
    for i in damage_grid.index: #rain
    #    print(i)
        for j in damage_grid.columns: #sea
    #        print(j)
            coords[z,:] = [i, j]
            dam[z] =  damage_grid.loc[i,j]
            z += 1
    return coords, dam

def damage_surface_df_z(coords, dam):
    rain = np.unique(coords[:,0])
    sl = np.unique(coords[:,1])
    Z = pd.DataFrame(index = rain, columns=sl, data=dam.reshape(len(rain), len(sl)))
    return Z

def add_extra_sealevel(i_extra_sealevel, damage_grid, drop_i=[]):
    new_damage_sl = pd.DataFrame(data=None, index=[i_extra_sealevel], columns = damage_grid.index)
    for i_rain in damage_grid.index:
        print(i_rain)
        if len(drop_i)>0:
            sel = damage_grid.drop(drop_i, axis = 1).loc[i_rain,:]
        else:
            sel = damage_grid.loc[i_rain,:]
        X=sel.index.values.reshape(-1, 1) 
        Y =sel.values.reshape(-1,1)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        Y_pred = linear_regressor.predict(np.array(i_extra_sealevel).reshape(1,-1))  # make predictions
        new_damage_sl.loc[i_extra_sealevel, i_rain] = Y_pred
    new_damage_sl = new_damage_sl.astype(float)
    return new_damage_sl

def add_extra_rain(i_extra_rain, damage_grid, drop_i=[]):
    new_damage_rain = pd.DataFrame(data=None, index=[i_extra_rain], columns = damage_grid.columns)
    for i_sl in damage_grid.columns:
        print(i_sl)
        if len(drop_i)>0:
            sel = damage_grid.drop(drop_i, axis = 0).loc[:,i_sl]
        else:
            sel = damage_grid.loc[:,i_sl]        
        X=sel.index.values.reshape(-1, 1) 
        Y =sel.values.reshape(-1,1)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        Y_pred = linear_regressor.predict(np.array(i_extra_rain).reshape(1,-1))  # make predictions
        new_damage_rain.loc[i_extra_rain, i_sl] = float(Y_pred) #f(i_extra_rain)
        
        #sel = new_damage_rain.drop([0,60,120,max_rain], axis=1).loc[i_sl, :]      
    new_damage_rain = new_damage_rain.astype(float)
    return new_damage_rain

def load_damage(fn_trunk, fn_files, max_rain, max_sl, thr_rain, thr_sl):
    damage = pd.read_csv(os.path.join(fn_trunk, fn_files,'summary_damage_cases.csv'), index_col = 'landuse')
    damage_tot = damage.sum(axis = 0)
    
    rain = [np.int(col.split('_')[1].strip('R')) for col in damage.columns]
    sea = [np.int(col.split('_')[2].strip('H')) for col in damage.columns]
    
    damage_grid = pd.DataFrame(index=np.unique(rain), columns = np.unique(sea), data=None)
    for value in damage.columns:
    #    print(value)
        i_rain = np.int(value.split('_')[1].strip('R'))
        i_sea = np.int(value.split('_')[2].strip('H'))
        damage_grid.loc[i_rain,i_sea] = damage_tot[value]
    damage_grid = damage_grid.astype(float)
  
    #Extrapolation
    new_damage_sl_high = add_extra_sealevel(max_sl, damage_grid, drop_i=[610,860,1110])
    new_damage_sl_low = add_extra_sealevel(0, damage_grid, drop_i=[1110, 1360,1610,1860])
    damage_grid = pd.concat([damage_grid, new_damage_sl_high.transpose(), new_damage_sl_low.transpose()], axis = 1)
    damage_grid.sort_index(axis = 1, inplace = True)
    
    new_damage_rain = add_extra_rain(max_rain, damage_grid, drop_i=[0,60,120])
    damage_grid = pd.concat([damage_grid, new_damage_rain], axis = 0)
    new_damage_rain_0 = add_extra_rain(180, damage_grid, drop_i=[0,60,180,300])
    damage_grid.loc[180,0] = new_damage_rain_0.loc[180,0]
    damage_grid.sort_index(inplace = True)
    del new_damage_rain, new_damage_sl_high, new_damage_sl_low, new_damage_rain_0
    damage_grid = damage_grid.astype(float)  
    
    #Setting threshold
    coords, dam = damage_surface_coord_z(damage_grid)
    
    new_damage_rain = [float(scipy.interpolate.griddata(coords, dam, (thr_rain,sl), method = 'linear')) for sl in damage_grid.columns]
    new_line = pd.DataFrame(data=np.array(new_damage_rain), index = damage_grid.columns, columns=[thr_rain]) 
    damage_grid = pd.concat([damage_grid, new_line.transpose()], axis = 0)
    
    coords, dam = damage_surface_coord_z(damage_grid)
    new_damage_sl = [float(scipy.interpolate.griddata(coords, dam, (i_rain,thr_sl), method = 'linear')) for i_rain in damage_grid.index]
    new_line = pd.DataFrame(data=np.array(new_damage_sl), index = damage_grid.index, columns=[thr_sl]) 
    damage_grid = pd.concat([damage_grid, new_line], axis = 1)

    # damage_grid[0] = damage_grid.loc[:,610]    
    damage_grid.sort_index(inplace = True)
    damage_grid.sort_index(axis = 1, inplace = True)
    damage_grid = damage_grid.astype(float)       
    return damage_grid

def simulate_rain(rain_simcdf, params, dist_type):
    rain_rvs = get_ICDF_from_dist(rain_simcdf, params, dist_type)
    rain_rvs = np.reshape(rain_rvs, rain_rvs.shape[0])
    rain_rvs[rain_rvs<0]=0
    return rain_rvs

def simulate_skew(cdf_swl_rvs, params_skew, dist_type_skew):
    skew_rvs = get_ICDF_from_dist(cdf_swl_rvs, params_skew, dist_type_skew)
    skew_rvs = np.reshape(skew_rvs, skew_rvs.shape[0]) * 1000
    return skew_rvs

def sample_tide(month, fn_tide, n):
    tide_sim = pd.read_csv(os.path.join(fn_tide, 'samples_tide_month_{}.csv'.format(str(month))), usecols=['tide'])
    #tide_sim.hist(bins=100)
    # # #################################################################################################################################    
    # #Selected mean = 0.86
    # mean = 0.86
    # std = 0.02
    # tide_sim = np.random.normal(loc=mean, scale=std, size = 50000)
    # tide_sim = pd.DataFrame(tide_sim)
    # #tide_sim.hist(bins=100)
    # ## #tide_sim.hist(bins=100)
    # ## ref = tide_sim/tide_sim.max()
    # ## tide_sim = tide_sim*np.exp(-ref)
    # ## #tide_sim.hist(bins=100)    
    # ## ###################################################################################################################################    
    tide_rvs = tide_sim.sample(n, replace = True).values
    tide_rvs = np.reshape(tide_rvs, tide_rvs.shape[0]) * 1000   
    return tide_rvs

def get_swl(skew_rvs, tide_rvs):
    swl_rvs = skew_rvs + tide_rvs
    swl_rvs = np.reshape(swl_rvs, swl_rvs.shape[0])   
    return swl_rvs

def pairs_cooc(rain_rvs, skew_rvs, tide_rvs):
    cooc_events = pd.concat([pd.DataFrame(rain_rvs, columns = ['rain']), pd.DataFrame(skew_rvs + tide_rvs, columns = ['sealevel'])], axis = 1)
    return cooc_events

def pairs_rain(rain_rvs, tide_rvs, skew_month_avg, month):
    rain_events = pd.concat([pd.DataFrame(rain_rvs, columns = ['rain']), pd.DataFrame(tide_rvs, columns = ['sealevel']) + (skew_month_avg.loc[month,'skew']*1000)], axis = 1)    
    return rain_events

def pairs_sl(skew_rvs, tide_rvs, rainfall_month_avg, month):
    sealevel_events = pd.concat([ pd.DataFrame(np.zeros(tide_rvs.shape) + rainfall_month_avg.loc[month,'Thiessen_sum'], columns = ['rain']), pd.DataFrame(skew_rvs + tide_rvs, columns = ['sealevel'])], axis = 1)
    return sealevel_events
#%%
def calculate_monthly_damage(best_fit_rain, param_rain, best_fit_skew, param_skew, n, monthly_data, coords, dam, skew_month_avg, rainfall_month_avg,
                             p_month, month_duration, cooc, lag_joint, selected_copulas, fn_tide, fn_copula, fn_trunk, varname1='Thiessen', varname2='skew', 
                             dep_type='copula', figure_joint=True):
    #Storing results
    damage_mod = pd.DataFrame(data = None, index = np.arange(1,13,1), columns = ['simulated_highest', 'full_dep', 'ind_highest', 'exclusive_highest'])
    all_events_sampled = pd.DataFrame(data=None, columns=['rain','sealevel','month'])
    all_events_sampled_dep = pd.DataFrame(data=None, columns=['rain','sealevel', 'month'])
    all_events_sampled_ind = pd.DataFrame(data=None, columns=['rain','sealevel', 'month'])
    all_events_sampled_excl = pd.DataFrame(data=None, columns=['rain','sealevel', 'month'])
    
    if figure_joint==True:
        #Preparing figure
        f, axs = plt.subplots(nrows=2, ncols=6, linewidth = 0, facecolor='w', edgecolor='w', sharex=True, sharey=True, figsize=(8, 4)) # , sharex=True, sharey=True gridspec_kw={'height_ratios': [1,1]},  #sharex=True, sharey=True, 
        axs = axs.reshape(-1)
        month_label = ['Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
    
    for month in np.arange(1,13,1):
        # print(month)
        #Select marginal distribution for rain
        dist_type = best_fit_rain.loc[month, 'distribution']
    #    dist_type = 'gumbel_r'
#        print('Rainfall distype: ', dist_type)    
        list_params = param_rain.loc[dist_type,str(month)].replace('(', '').replace(')','').split(',')
        params = [float(e) for e in list_params]
        
        #Select marginal distribution for skew
        dist_type_skew = best_fit_skew.loc[month, 'distribution']
    #        dist_type_skew = 'gumbel_r'
#        print('Skew distype: ', dist_type_skew)    
        list_params_skew = param_skew.loc[dist_type_skew,str(month)].replace('(', '').replace(')','').split(',')
        params_skew = [float(e) for e in list_params_skew]        
            
        if dep_type == 'copula':
            rain_simcdf = pd.read_csv(os.path.join(fn_copula, 'New{}_New{}_data_month_{}.csv'.format(str(varname1),str(varname2),str(month))), usecols=['V1'])
            rain_rvs = simulate_rain(rain_simcdf, params, dist_type)
    
            cdf_swl_rvs = pd.read_csv(os.path.join(fn_copula, 'New{}_New{}_data_month_{}.csv'.format(str(varname1),str(varname2),str(month))), usecols=['V2']).values 
            if varname2 == 'skew':
                skew_rvs = simulate_skew(cdf_swl_rvs, params_skew, dist_type_skew)
                tide_rvs = sample_tide(month, fn_tide, n)
            
            if figure_joint==True:  
                kend = pd.read_csv(os.path.join(fn_trunk, 'Master2019/Thomas/data/NewBivariate/Simulated', 'New{}_New{}_copulatype_month_{}.csv'.format(varname1, varname2, str(month))), index_col = 0)
                pseudo_rain, pseudo_skew =  make_pseudo_obs(monthly_data['Thiessen'].where(monthly_data['month']==month).dropna(), monthly_data['skew'].where(monthly_data['month']==month).dropna())
                axs[month-1].annotate("{}".format(month_label[month-1]),  xy = (0.05, 0.95), xycoords = 'axes fraction', size=7) 
                axs[month-1].annotate(r"$\tau$ ={0:5.3f}".format(float(kend.iloc[-1,0])),  xy = (0.50, 0.95), xycoords = 'axes fraction', size=7) 
        
                # axs[month-1].scatter(cdf_swl_rvs, rain_simcdf,  linestyle = 'None', marker = 'o', c = 'grey', edgecolors='none', alpha = 0.8, s=0.2, zorder=1) #markeredgewidth=0.5,     
                # axs[month-1].scatter(pseudo_skew, pseudo_rain,  marker = 'o', edgecolors ='k', linewidths=0.3, c = 'k', alpha = 1, s=10)
                
                if selected_copulas[month]=='Independence':
                    skew_rvs_shuffled = skew_rvs.copy()
                    rain_rvs_shuffled = rain_rvs.copy()
                    np.random.shuffle(skew_rvs_shuffled)
                    np.random.shuffle(rain_rvs_shuffled)
                    axs[month-1].scatter(skew_rvs_shuffled, rain_rvs_shuffled,  linestyle = 'None', marker = 'o', c = 'blue', edgecolors='none', alpha = 0.8, s=0.2, zorder=1) #markeredgewidth=0.5,     
                else:
                    axs[month-1].scatter(skew_rvs, rain_rvs,  linestyle = 'None', marker = 'o', c = 'blue', edgecolors='none', alpha = 0.8, s=0.2, zorder=1) #markeredgewidth=0.5,     

                # axs[month-1].scatter(monthly_data['skew'].where(monthly_data['month']==month).dropna()*1000, monthly_data['Thiessen'].where(monthly_data['month']==month).dropna(), marker = 'o', edgecolors ='k', linewidths=0.3, c = 'k', alpha = 1, s=10) #markeredgewidth=0.5,     

                # axs[month-1].xaxis.set_major_locator(MultipleLocator(0.5))
                # axs[month-1].yaxis.set_major_locator(MultipleLocator(0.5))
                # axs[month-1].xaxis.set_minor_locator(MultipleLocator(0.1))
                # axs[month-1].yaxis.set_minor_locator(MultipleLocator(0.1))            
                axs[month-1].tick_params(axis='both', labelsize=7, direction='out')                
        
                # axs[month-1].scatter(skew_rvs, rain_rvs,  linestyle = 'None', marker = 'o', c = 'grey', edgecolors='none', alpha = 0.8, s=0.1, zorder=1) #markeredgewidth=0.5,     
                # axs[month-1].scatter(swl_rvs[:n_cooc_month], rain_rvs[:n_cooc_month],  marker = 'o', edgecolors ='k', linewidths=0.3, c = np.repeat(month, len(swl_rvs[:n_cooc_month])), cmap=cmap2, alpha = 1, s=18, norm=norm)
            del cdf_swl_rvs, rain_simcdf
    
    
        if dep_type == 'full corr':
            quantiles = np.random.random(n)
#            print('Quantiles shape:', quantiles.shape)
            rain_rvs = simulate_rain(quantiles, params, dist_type) 
            if varname2 == 'skew':
                skew_rvs = simulate_skew(quantiles, params_skew, dist_type_skew)
            del quantiles
            if figure_joint==True:
                axs[month-1].scatter(monthly_data['skew'].where(monthly_data['month']==month).dropna()*1000, monthly_data['Thiessen'].where(monthly_data['month']==month).dropna(),  marker = 'o', edgecolors ='k', linewidths=0.3, c = 'k', alpha = 1, s=10)
                axs[month-1].scatter(skew_rvs, rain_rvs,  linestyle = 'None', marker = 'o', c = 'blue', edgecolors='none', alpha = 0.8, s=0.2, zorder=1) #markeredgewidth=0.5,     
                axs[month-1].tick_params(axis='both', labelsize=7, direction='out')
        
        if len(rain_rvs) != n:
#            print('Performing analysis on less samples')
            i_random = np.random.choice(np.arange(0, len(rain_rvs)), n, replace = False)
            rain_rvs = rain_rvs[i_random]
            skew_rvs = skew_rvs[i_random]
    
        ##### FULL DEPENDENCE ######
        tide_rvs = sample_tide(month, fn_tide, n) 
        cooc_events = pairs_cooc(rain_rvs, skew_rvs, tide_rvs)

        if figure_joint==True:  
            axs[month-1].scatter(cooc_events.loc[:,'sealevel'], cooc_events.loc[:,'rain'],  linestyle = 'None', marker = 'o', c = 'grey', edgecolors='none', alpha = 0.8, s=0.2, zorder=1) #markeredgewidth=0.5,     
            axs[month-1].scatter(monthly_data['skew'].where(monthly_data['month']==month).dropna()*1000, monthly_data['Thiessen'].where(monthly_data['month']==month).dropna(), marker = 'o', edgecolors ='k', linewidths=0.3, c = 'k', alpha = 1, s=10) #markeredgewidth=0.5,     
        
        sampled_month_dep = pd.DataFrame(data=cooc_events, columns=['rain', 'sealevel'])
        sampled_month_dep['month'] = month    
        
        dam_full_dep = scipy.interpolate.griddata(coords, dam, cooc_events.values, method = 'linear')
        dam_full = np.sum(dam_full_dep)
        damage_mod.loc[month, 'full_dep'] = dam_full/n 
        sampled_month_dep['cooc_damage'] = dam_full_dep
        
        all_events_sampled_dep = pd.concat([all_events_sampled_dep, sampled_month_dep], axis = 0, ignore_index=True)
        del dam_full_dep, dam_full, sampled_month_dep, tide_rvs, cooc_events
        
        ##### EXCLUSIVE ######
        tide_rvs = sample_tide(month, fn_tide, n)
        rain_events = pairs_rain(rain_rvs, tide_rvs, skew_month_avg, month)
        del tide_rvs    
        tide_rvs = sample_tide(month, fn_tide, n)  
        sealevel_events = pairs_sl(skew_rvs, tide_rvs, rainfall_month_avg, month)
    
        dam_excl_rain = scipy.interpolate.griddata(coords, dam, (rain_events.values), method = 'linear') #
        dam_excl_sl = scipy.interpolate.griddata(coords, dam, (sealevel_events.values), method = 'linear') #np.zeros(events_month[:,1].shape)
        
        dam_excl_highest = pd.DataFrame(data=np.concatenate((dam_excl_rain, dam_excl_sl), axis=1), columns = ['rain_damage', 'sealevel_damage'])
        dam_highest = dam_excl_highest.max(axis=1)
        damage_mod.loc[month, 'exclusive_highest'] = (np.sum(dam_highest))/n      
        
        dam_highest_type = dam_excl_highest.idxmax(axis=1)
        sampled_month_excl = pd.concat([pd.concat([rain_events[dam_highest_type=='rain_damage'], dam_excl_highest[dam_highest_type=='rain_damage']['rain_damage']], axis = 1),
                                        pd.concat([sealevel_events[dam_highest_type=='sealevel_damage'], dam_excl_highest[dam_highest_type=='sealevel_damage']['sealevel_damage']], axis = 1)
            ], axis = 0, ignore_index=True)
        sampled_month_excl['month'] = month 
        all_events_sampled_excl = pd.concat([all_events_sampled_excl, sampled_month_excl], axis = 0, ignore_index=True)    
        del rain_events, sealevel_events, tide_rvs, dam_highest, dam_highest_type, dam_excl_rain, dam_excl_sl, sampled_month_excl, dam_excl_highest
        
        #### INDEPENDENCE ####
        n_cooc_ind = int(p_month.loc[month_duration.loc[month,'length'],str(lag_joint)] * n) 
        i_cooc_ind = np.random.choice(np.arange(0, n), n_cooc_ind, replace = False)
        i_ind = np.delete(np.arange(0, n), i_cooc_ind)   
        
        tide_rvs = sample_tide(month, fn_tide, len(i_cooc_ind))
        cooc_events = pairs_cooc(rain_rvs[i_cooc_ind], skew_rvs[i_cooc_ind], tide_rvs)
        
        tide_rvs = sample_tide(month, fn_tide, len(i_ind))
        rain_events = pairs_rain(rain_rvs[i_ind], tide_rvs, skew_month_avg, month)
        
        tide_rvs = sample_tide(month, fn_tide, len(i_ind))
        sealevel_events = pairs_sl(skew_rvs[i_ind], tide_rvs, rainfall_month_avg, month)
    
        dam_excl_rain = scipy.interpolate.griddata(coords, dam, (rain_events.values), method = 'linear') #
        dam_excl_sl = scipy.interpolate.griddata(coords, dam, (sealevel_events.values), method = 'linear') #np.zeros(events_month[:,1].shape)
        dam_cooc = scipy.interpolate.griddata(coords, dam, (cooc_events.values), method = 'linear') #np.zeros(events_month[:,1].shape)
        
        dam_excl_highest = pd.DataFrame(data=np.concatenate((dam_excl_rain, dam_excl_sl), axis=1), columns = ['rain_damage', 'sealevel_damage'])
        dam_highest = dam_excl_highest.max(axis=1)
        dam_highest_type = dam_excl_highest.idxmax(axis=1)
        
        damage_mod.loc[month, 'ind_highest'] = (np.sum(dam_highest) + np.sum(dam_cooc))/n  
        
        sampled_month_ind = pd.concat([pd.concat([rain_events[dam_highest_type=='rain_damage'], dam_excl_highest[dam_highest_type=='rain_damage']['rain_damage']], axis = 1),
                                        pd.concat([sealevel_events[dam_highest_type=='sealevel_damage'], dam_excl_highest[dam_highest_type=='sealevel_damage']['sealevel_damage']], axis = 1),
                                        pd.concat([cooc_events, pd.DataFrame(dam_cooc, columns = ['cooc_damage'])], axis = 1)
            ], axis = 0, ignore_index=True)
        sampled_month_ind['month'] = month 
        all_events_sampled_ind = pd.concat([all_events_sampled_ind, sampled_month_ind], axis = 0, ignore_index=True)    
        del n_cooc_ind, i_cooc_ind, i_ind, cooc_events, rain_events, sealevel_events, tide_rvs, dam_highest, dam_highest_type, dam_excl_rain, dam_excl_sl, sampled_month_ind, dam_cooc, dam_excl_highest
        
        #### MEASURED ####
        rate_month = cooc.loc[month, 'rate']
#        print('Measured rate: ', rate_month)  
        n_cooc_ind = int(rate_month * n)   
        if n_cooc_ind < int(p_month.loc[month_duration.loc[month,'length'],str(lag_joint)] * n):
            n_cooc_ind = int(p_month.loc[month_duration.loc[month,'length'],str(lag_joint)] * n)
        
        i_cooc_ind = np.random.choice(np.arange(0, n), n_cooc_ind, replace = False)
        i_ind = np.delete(np.arange(0, n), i_cooc_ind)    
        
        tide_rvs = sample_tide(month, fn_tide, len(i_cooc_ind))
        cooc_events = pairs_cooc(rain_rvs[i_cooc_ind], skew_rvs[i_cooc_ind], tide_rvs)
        
        tide_rvs = sample_tide(month, fn_tide, len(i_ind))
        rain_events = pairs_rain(rain_rvs[i_ind], tide_rvs, skew_month_avg, month)
        
        tide_rvs = sample_tide(month, fn_tide, len(i_ind))
        sealevel_events = pairs_sl(skew_rvs[i_ind], tide_rvs, rainfall_month_avg, month)
        
        dam_excl_rain = scipy.interpolate.griddata(coords, dam, (rain_events.values), method = 'linear', fill_value = 0) #
        dam_excl_sl = scipy.interpolate.griddata(coords, dam, (sealevel_events.values), method = 'linear', fill_value = 0) #np.zeros(events_month[:,1].shape)
        dam_cooc = scipy.interpolate.griddata(coords, dam, (cooc_events.values), method = 'linear', fill_value = 0) #np.zeros(events_month[:,1].shape)
        
        dam_excl_highest = pd.DataFrame(data=np.concatenate((dam_excl_rain, dam_excl_sl), axis=1), columns = ['rain_damage', 'sealevel_damage'])
        dam_highest = dam_excl_highest.max(axis=1)
        dam_highest_type = dam_excl_highest.idxmax(axis=1)
        
        damage_mod.loc[month, 'simulated_highest'] = (np.sum(dam_highest) + np.sum(dam_cooc))/n  
        
        sampled_month = pd.concat([pd.concat([rain_events[dam_highest_type=='rain_damage'], dam_excl_highest[dam_highest_type=='rain_damage']['rain_damage']], axis = 1),
                                        pd.concat([sealevel_events[dam_highest_type=='sealevel_damage'], dam_excl_highest[dam_highest_type=='sealevel_damage']['sealevel_damage']], axis = 1),
                                        pd.concat([cooc_events, pd.DataFrame(dam_cooc, columns = ['cooc_damage'])], axis = 1)
            ], axis = 0, ignore_index=True)
        sampled_month['month'] = month 
        all_events_sampled = pd.concat([all_events_sampled, sampled_month], axis = 0, ignore_index=True)    
        del n_cooc_ind, i_cooc_ind, i_ind, cooc_events, rain_events, sealevel_events, tide_rvs, dam_highest, dam_highest_type, dam_excl_rain, dam_excl_sl, sampled_month, dam_cooc, dam_excl_highest
        del rate_month, rain_rvs, params, params_skew, month, skew_rvs

    if figure_joint==True:
        f.text(0.5, 0.015, 'CDF - Storm surge', ha='center', size = 9)
        f.text(0.02, 0.5, 'CDF - Rainfall', va='center', rotation='vertical', size = 9)
        plt.subplots_adjust(left = 0.07, right = 0.98, top = 0.98, bottom= 0.1, wspace = 0.2, hspace = 0.1)
        plt.show()
        
        fn_out = os.path.join(fn_trunk, 'Paper\Paper5\FIGURES', 'Bivariate', '{}_{}_monthly_copula.png'.format(varname1, varname2))
        f.savefig(fn_out, frameon=False, dpi = 300,papertype= 'a4', orientation='portrait') #transparent=True, 
    
    return damage_mod, all_events_sampled, all_events_sampled_dep, all_events_sampled_excl, all_events_sampled_ind

#%%
def construct_final_set(selected_copulas, all_events_sampled, all_events_sampled_ind):
    month_ind = selected_copulas[selected_copulas=='Independence'].index.values
    other_month = selected_copulas[selected_copulas!='Independence'].index.values
    
    final_events = all_events_sampled_ind.copy()
    final_events = final_events.loc[final_events['month'].isin(month_ind)]
    final_events = pd.concat([final_events, all_events_sampled.loc[all_events_sampled['month'].isin(other_month)]])
    final_events.reset_index(inplace = True)
    return final_events


def numfmt(x, pos): # your custom formatter function: divide by 100.0
    s = '{}'.format(x / 1000.0)
    return s

def calc_density(all_events_sampled, coords, dam, n, nbins, fn_trunk, fig_plot = True, title = 'damage_rainfall_skew_grid.png'):
    import matplotlib.ticker as tkr 
    import matplotlib.colors as colors
    
    yfmt = tkr.FuncFormatter(numfmt)

    x = all_events_sampled.loc[:, 'rain'].values #RAIN
    y = all_events_sampled.loc[:, 'sealevel'].values #SEALEVEL
    
    #Calculating the 2d histogram of the points
    xedges = np.linspace(0,800,nbins)
    yedges = np.linspace(100,2500,nbins)

    H, xedges, yedges = np.histogram2d(x, y, bins=[xedges, yedges]) 
    
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
    Z = H
    if fig_plot == True:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(X, Y, Z*(10**4), color='black', alpha = 0.5)
        plt.xlabel('Rainfall (mm/day)', fontsize = 16)
        plt.ylabel('Sea Level (m)', fontsize = 16)
        ax.set_zlabel(r'x$10^{-4}$ Density', fontsize = 16)
        plt.show()
    
        plt.figure()
        myextent  = [xedges[0],xedges[-1],yedges[0],yedges[-1]]
        plt.imshow(H.T/H.sum(),origin='low',extent=myextent,interpolation=None,aspect='auto',vmin=1e-6,norm=colors.LogNorm(vmin=1e-6, vmax=H.max()/H.sum()), cmap='Reds') #vmin=1e-19, vmax=None, #interpolation='nearest'
        plt.xlabel('Rainfall (mm/day)')
        plt.ylabel('SeaLevel (m)')
        cbar = plt.colorbar(extend='max')
        cbar.set_label('Density', rotation=270)
        plt.show()
    
    #We multiply this location by the damage
    H_damage = np.zeros(((xedges.shape[0]-1),(yedges.shape[0]-1)))
    Hx = (xedges[1:] + xedges[:-1]) / 2  #At the middle of the bin
    Hy = (yedges[1:] + yedges[:-1]) / 2
    
    ii = 0
    jj = 0
    for i in Hx: #rain
    #    print(i)
        jj = 0
        for j in Hy: #sea
    #        print(j)
            H_damage[ii,jj] =  scipy.interpolate.griddata(coords, dam, (i,j), method = 'linear', fill_value = 0) #rainfall, sealevel
            if H_damage[ii,jj] < 0:
                  print(i, j)
                  print(H_damage[ii,jj])
                # print(ii,jj)
            jj += 1
        ii += 1
    
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
    Z = H_damage

    if fig_plot == True:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(X, Y, Z/1e6, color='black', rstride=1, cstride=1)
        plt.xlabel('Rainfall (mm/day)', fontsize = 16)
        plt.ylabel('Sea Level (m)', fontsize = 16)
        # plt.xlim(0,300)
        # plt.ylim(0.61, 1.86)
        ax.set_zlabel('x 10$^{6}$ Damage', fontsize = 16)
        plt.show()
    
    Damage_reldensity = H_damage * H
    print('Calculated damage is: {0:5.4f}e+09 $'.format(Damage_reldensity.sum()/1e9 /n))

    if fig_plot == True:        
        fig = plt.figure()
        ax = plt.gca()
        myextent  = [xedges[0],xedges[-1],yedges[0],yedges[-1]]
        plt.imshow(Damage_reldensity.T/n ,origin='low', extent=myextent, interpolation=None, aspect='auto', norm=colors.LogNorm(vmin=None, vmax=Damage_reldensity.max()/n),cmap='Reds') #, extent=[80,120,32,0]
        plt.xlabel('Rainfall (mm/day)', fontsize = 8)
        plt.ylabel('Sea Level (m)', fontsize = 8)
        ax.yaxis.set_major_formatter(yfmt)
        cbar = plt.colorbar(extend='max')
        cbar.set_label('Damage ($)', rotation=270, labelpad=10)
        plt.xlim(0,500)  
        plt.ylim(0,2000)
        plt.show()        
        fn_out = os.path.join(fn_trunk, 'Paper\Paper5\FIGURES', 'Bivariate', title)
        fig.savefig(fn_out, frameon=False, dpi = 300,papertype= 'a4', orientation='portrait') #transparent=True, 
        plt.close()
    return xedges, yedges, Damage_reldensity/n

