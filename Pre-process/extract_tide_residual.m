% Created by: Anaïs Couasnon
% Date: 2019
%
% DESCRIPTION:
% This script reads the seawater level from HCMC (raw and detrended) and extract the trend in
% the data and return the tidal signal and non-tidal residuals

clc
clear all 
close all 

addpath('E:\github\HoChiMinh\UTideCurrentVersion\')
addpath(genpath('E:\github\HoChiMinh\'))

lat_HCMC = 10.34;

first = datetime(1978, 1, 1,0,0,0);
last = datetime(2018, 12, 31,23,0,0);
x = first:hours(1):last;
y = datenum(x);
%% SAVING THE ORIGINAL DATA AND DETRENDED DATA
%ORIGINAL -
%E:\surfdrive\Documents\Master2019\Thomas\data\hourly_VungTau_swl_cleaned.csv
%--> WACC_VungTau_Cleaned.mat

%DETRENDED - 
%E:\surfdrive\Documents\Master2019\Thomas\data\hourly_VungTau_swl_cleaned_detrended_strict.csv
%--> WACC_VungTau_Cleaned_Detrendred_Strict.mat

%% ORIGINAL DATA - REMOVING LINEAR TREND

% %Loading the swl data
% root_folder = ('E:\surfdrive\Documents\Master2019\Thomas\data\WACC_VungTau_Cleaned.mat');
% load(root_folder)
% swl(:,1)= datenum(date); %Converting the dates to datenum format
% swl(:,2) = value;
% 
% %Decomposition
% coef = ut_solv(swl(:,1), swl(:,2), [], lat_HCMC, 'auto'); 
% [sl_fit_wTrend, ~ ] = ut_reconstr(y, coef);
% sl_fit_wTrend = sl_fit_wTrend.' ;
% 
% [C, ia, ib] = intersect(swl(:,1), y);
% 
% surge = swl(ia,2) - sl_fit_wTrend(ib);
% max_surge = calc_max_day([swl(ia,1), surge]);
% 
% skew = calc_skew(sl_fit_wTrend(ib), [swl(ia,1), swl(ia,2)]); %astr_tide_trend,totalwatlev
% skew_day = calc_max_day([skew(:,1), skew(:,2)]);
% 
% all_tide = [y' sl_fit_wTrend];
% surge = [swl(ia,1) surge];
% 
% %Comparing the new surge with the old surge
% figure()
% ax1 = subplot(3,1,1);
% plot(swl(:,1),swl(:,2), '-k')
% hold on
% plot(all_tide(:,1), all_tide(:,2), '-r')
% hold off
% datetick('x', 'dd-mm-yy')
% title('Measured SWL')
% 
% ax2 = subplot(3,1,2);
% %plot(swl(:,1),sl_fit_wTrend, '-k')
% plot(surge(:,1),surge(:,2), '-b')
% datetick('x', 'dd-mm-yy')
% title('Residual')
% 
% ax3 = subplot(3,1,3);
% plot(skew(:,1),skew(:,2), '-b')
% datetickzoom('x', 'dd-mm-yy')
% title('Skew Day')
% datetick('x', 'dd-mm-yy')
% linkaxes([ax1 ax2 ax3],'xy')
% 
% save('output_recomposition_WACC_VungTau_Cleaned_Linear.mat','all_tide','skew','surge','max_surge')

%% DETRENDED DATA - SELECTING CONST.

% clear all

%Loading the swl data
root_folder = ('E:\surfdrive\Documents\Master2019\Thomas\data\WACC_VungTau_Cleaned.mat');
load(root_folder)
swl(:,1)= datenum(date); %Converting the dates to datenum format
swl(:,2) = value;

%Decomposition
coef = ut_solv(swl(:,1), swl(:,2), [], lat_HCMC, {'M2  ', 'K1  ', 'O1  ', 'S2  ', 'SA  ', 'P1  ','N2  ', 'K2  ', 'Q1  ', 'SSA ', 'MK3', 'NU2 ', 'NO1 '}, 'NoTrend'); 
[sl_fit_wTrend, ~ ] = ut_reconstr(y, coef);
sl_fit_wTrend = sl_fit_wTrend.' ;

[C, ia, ib] = intersect(swl(:,1), y);

surge = swl(ia,2) - sl_fit_wTrend(ib);
max_surge = calc_max_day([swl(ia,1), surge]);

skew = calc_skew(sl_fit_wTrend(ib), [swl(ia,1), swl(ia,2)]); %astr_tide_trend,totalwatlev
skew_day = calc_max_day([skew(:,1), skew(:,2)]);

all_tide = [y' sl_fit_wTrend];
surge = [swl(ia,1) surge];

%Comparing the new surge with the old surge
figure()
ax1 = subplot(3,1,1);
plot(swl(:,1),swl(:,2), '-k')
hold on
plot(all_tide(:,1), all_tide(:,2), '-r')
hold off
datetick('x', 'dd-mm-yy')
title('Measured SWL')

ax2 = subplot(3,1,2);
%plot(swl(:,1),sl_fit_wTrend, '-k')
plot(surge(:,1),surge(:,2), '-b')
datetick('x', 'dd-mm-yy')
title('Residual')

ax3 = subplot(3,1,3);
plot(skew(:,1),skew(:,2), '-b')
datetickzoom('x', 'dd-mm-yy')
title('Skew Day')
datetick('x', 'dd-mm-yy')
linkaxes([ax1 ax2 ax3],'xy')

save('output_recomposition_WACC_VungTau_Cleaned_Detrended_Strict_sel_const.mat','all_tide','skew','surge','max_surge')

