# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 08:27:11 2020

@author: acn980
"""

import os
import numpy as np
import pandas as pd
import glob
import rasterio 
from damagescanner.core import RasterScanner
import damagescanner.plot

#%%
fn_trunk = 'E:/surfdrive/Documents'
fn_files = 'Paper/Paper5/Hydrodynamic_runs/RISK_maskTelemac'
fn = os.path.join(fn_trunk,fn_files)
#os.chdir(fn)
#%% 1- We select the final land use map
fn_inun = 'Paper/Paper5/Hydrodynamic_runs/TELEMAC3d/new_runs_max_telemac'
list_inun = glob.glob(os.path.join(fn_trunk,fn_inun, '*_correct.tiff'))

#Downloading the base map - we take the max of both
inun_map_base1 = os.path.join(fn_trunk,fn_inun, '1_R000_H0610_maxdepth_linear_masked_correct.tiff')
inun_map_base2 = os.path.join(fn_trunk,fn_inun, '2_R000_H0860_maxdepth_linear_masked_correct.tiff')
with rasterio.open(inun_map_base1) as src:
        base_inun = src.read()[0, :, :]
        transform = src.transform
        crs = src.crs
with rasterio.open(inun_map_base2) as src:
        base_inun2 = src.read()[0, :, :]
base_inun[base_inun<0] = 0
base_inun2[base_inun2<0] = 0
base_inun_tide = np.maximum(base_inun, base_inun2)
        
fn_landuse = 'Paper\Paper5\Hydrodynamic_runs\LAND USE'
landuse_map = os.path.join(fn_trunk,fn_landuse, 'LU2010_mask_rivers_R000_H0610_city_center_area.tif')
curve_path = os.path.join(fn_trunk,fn_landuse, 'curves_15cm_adjres_damagescanner.csv')
maxdam_path = os.path.join(fn_trunk,fn_landuse, 'maxdam_damagescanner.csv')

damage_per_class_all_cases = pd.DataFrame(data=None)
#%%
fn_calib = 'Paper/Paper5/Hydrodynamic_runs/TELEMAC3d/new_runs_max_calibrated'
rst_opts = {
    'driver': 'GTiff',
    'height': base_inun_tide.shape[0],
    'width': base_inun_tide.shape[1],
    'count': 1,
    'dtype': np.float64,
    'crs': crs,
    'transform': transform,
    'compress': "LZW"
}

with rasterio.open(landuse_map) as src:
        landuse = src.read()[0, :, :]
    
for inun_map in list_inun:  
    print(inun_map)
    case = inun_map.split('telemac\\')[-1].split('_linear')[0]
    
    with rasterio.open(inun_map) as src:
        inundation = src.read()[0, :, :]
    
    inundation[inundation<0] = 0
    
    corrected_inun = inundation - base_inun_tide    
    corrected_inun[corrected_inun <= 0] = 0.0
    
    #Saving the corrected inundation maps
    fn_inun_cal = os.path.join(fn_trunk,fn_calib, case+'_calibrated.tiff')
    with rasterio.open(fn_inun_cal, 'w', **rst_opts) as dst:
        corrected_inun[landuse==0] = 0.0
        dst.write(corrected_inun, 1)    
    
    #Calculate the damage and output the raster and damage summary table using the function 
    loss_df_case, _, _, _ = RasterScanner(landuse_map, corrected_inun, curve_path, maxdam_path, save=True, scenario_name=case, output_path=fn, dtype = np.int32)    
    loss_df_case.rename(columns = {'losses':case}, inplace = True) #We change the name of the column with the RP
    damage_per_class_all_cases = damage_per_class_all_cases.join(loss_df_case, how = 'outer') #We append the column to the overall summary table

# Write the total damage per land use class to an Excel file...
damage_per_class_all_cases.to_csv(os.path.join(fn,'summary_damage_cases.csv'), index_label = 'landuse') #Export as a csv
