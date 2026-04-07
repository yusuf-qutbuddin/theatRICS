# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 18:53:36 2022

@author: yusuf
"""

import numpy as np 
import scipy as sp
# from sympy import *
# import sympy as sym
from scipy.optimize import fsolve
from statsmodels.sandbox.stats.runs import runstest_1samp 
import matplotlib.pyplot as plt
#%%
# goodness of fit
weighted_r_glob = 0
instant_correlation_glob = 0
# Bayesian Information Criterion is being used to score the goodness between different models for fitting the FCS curves
# the function penalizes increase in parameters and thus penalizes overfitting, while rewarding the minimization of chi-square
# lower BIC indicates a more efficient parametrized model to predict the data 
# the BIC formula used here comes from the implementation of scipy minimize 
def BIC_func(n, k, RSS):
    # print(n)
    # print(k)
    # print(RSS)
    BIC = n*np.log(RSS/n) + k*np.log(n)
    return BIC

def iMSD_calc(tau, aR, N_FIT, cc_FIT, offset_FIT):
    reIMSD_list = list()
    def func(x):
        # equation = list()
        # for i in range(len(tau)):
            # equation.append(cc_FIT[i] - offset_FIT - 1/(sqrt(8)*N_FIT*(1 + x[i])*sqrt(1 + x[i]/aR**2)))
        # return equation
        return cc_FIT[i_tau] - offset_FIT - 1/(np.sqrt(8)*N_FIT*(1 + x[0])*np.sqrt(1 + x[0]/aR**2))
    for i_tau in range(len(tau)):
        # temp_value = list(sym.solveset(cc_FIT[i_tau] - offset_FIT - 1/(sqrt(8)*N_FIT*(1 + reIMSD)*sqrt(1 + reIMSD/aR**2)),reIMSD, domain = S.Reals))
        temp_value = list(fsolve(func,[0]))
    # temp_value = list(fsolve(func,np.zeros(len(tau))))
        reIMSD_list.append(temp_value[0])
    # return temp_value
    return reIMSD_list
   
    
    # return temp_value
def runs_test_criterion_func(weighted_r, goodness_of_fit_criterion):
    weighted_r_1 = weighted_r[1:]
    weighted_r_2 = weighted_r[:-1]
    global weighted_r_glob
    weighted_r_glob = weighted_r
    instant_correlation = np.multiply(weighted_r_2,weighted_r_1)
    global instant_correlation_glob 
    instant_correlation_glob = instant_correlation
    p_ttest, p_wilcoxon, p_runstest, p_runstest_residuals = None, None, None, None
    if 'instant_correlation_ttest' in goodness_of_fit_criterion:
        stat_ttest,p_ttest = sp.stats.ttest_1samp(instant_correlation, popmean = 0, alternative = 'greater')
    if 'instant_correlation_wilcoxon' in goodness_of_fit_criterion:
        stat_wilcoxon, p_wilcoxon = sp.stats.wilcoxon(instant_correlation, alternative = 'greater')
    if 'instant_correlation_runsstest' in goodness_of_fit_criterion:
        stat_runstest, p_runstest = runstest_1samp(instant_correlation, cutoff = 0)
    if 'weighted_residual_runsstest' in goodness_of_fit_criterion:
        stat_runstest_residuals, p_runstest_residuals = runstest_1samp(weighted_r, cutoff = 0)
    
    # fig, ax = plt.subplots(nrows = 2, ncols = 1)
    # ax[0].hist(weighted_r, density = True, bins = 20)
    # ax[1].hist(instant_correlation, density = True, bins = 20)
    # fig.show()
    return p_ttest, p_wilcoxon, p_runstest, p_runstest_residuals

#%%
# Calculations from Fits
def calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction, k, parameters, param_cov,given_params, fitting_model):
    if len(param_cov) != 0: 
        param_err = np.sqrt(np.diag(param_cov))
    r = G - ccPrediction
    weighted_r = r/sigma_G
    chi_squared = np.sum((r/sigma_G)**2)
    r_chi_squared = chi_squared/(len(tau)-k)
    BIC = BIC_func(len(tau), k, chi_squared)
    p_ttest, p_wilcoxon, p_runstest, p_runstest_residuals = runs_test_criterion_func(weighted_r, goodness_of_fit_criterion)
    if fitting_model == 'g3diffCal':
        N_fitted, tau_D_fitted, PSFaspectratio_fitted = parameters
        dN_fitted, dtau_D_fitted, dPSFaspectratio_fitted= param_err
        PSF_radius = np.sqrt(4*corrected_D*tau_D_fitted)
        CPP_peak = count_rate/N_fitted
        CPP_avg = CPP_peak/(2*np.sqrt(2))
        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSFaspectratio_fitted,
                'sigma PSF aspect ratio': dPSFaspectratio_fitted,'N': N_fitted,'sigma N': dN_fitted,
                'D':corrected_D, 'Tau diffusion': tau_D_fitted,   'sigma tau diffusion': dtau_D_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r':r, 'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}
    
    elif fitting_model == 'g3diff':
        N_fitted, tau_D_fitted = parameters
        dN_fitted, dtau_D_fitted= param_err
        PSF_radius = given_params['PSF_radius']
        D_fitted = (PSF_radius**2)/(4*tau_D_fitted)
        try: 
            dD_fitted = D_fitted * dtau_D_fitted/tau_D_fitted
        except:
            dD_fitted = np.nan
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        CPP_peak = count_rate/N_fitted
        CPP_avg = CPP_peak/(2*np.sqrt(2))
        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio,'N': N_fitted,'sigma N': dN_fitted,
                'D':D_fitted, 'dD':dD_fitted, 'Tau diffusion': tau_D_fitted,   'sigma tau diffusion': dtau_D_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r':r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}


    elif fitting_model == 'g3diffOffset':
        N_fitted, tau_D_fitted, offset_fitted = parameters
        dN_fitted, dtau_D_fitted, doffset_fitted = param_err
        PSF_radius = given_params['PSF_radius']
        D_fitted = (PSF_radius**2)/(4*tau_D_fitted)
        try: 
            dD_fitted = D_fitted * dtau_D_fitted/tau_D_fitted
        except:
            dD_fitted = np.nan
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        CPP_peak = count_rate/N_fitted
        CPP_avg = CPP_peak/(2*np.sqrt(2))
        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio,'N': N_fitted,'sigma N': dN_fitted,
                'D':D_fitted, 'dD':dD_fitted, 'Tau diffusion': tau_D_fitted,   'sigma tau diffusion': dtau_D_fitted,
                'offset':offset_fitted, 'sigma offset':doffset_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r':r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}

    
    elif fitting_model == 'g3diffLargeParticles':
        N_fitted, tau_D_fitted = parameters
        dN_fitted, dtau_D_fitted= param_err
        PSF_radius = given_params['PSF_radius']
        Radius = given_params['Radius of the particle']
        PSF_apparent_radius = np.sqrt(PSF_radius**2 + Radius**2)
        D_fitted = (PSF_apparent_radius**2)/(4*tau_D_fitted)
        try: 
            dD_fitted = D_fitted * dtau_D_fitted/tau_D_fitted
        except:
            dD_fitted = np.nan
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        PSF_apparent_aspect_ratio = PSF_aspect_ratio * PSF_radius/PSF_apparent_radius
        N = N_fitted*(1 + (Radius/PSF_radius)**2)
        try:
            dN = dN_fitted*(1 + (Radius/PSF_radius)**2)
        except:
            dN = np.nan
        CPP_peak = count_rate/N_fitted
        CPP_avg = CPP_peak/(2*np.sqrt(2))
        return {'PSF radius': PSF_apparent_radius, 'PSF aspect ratio': PSF_apparent_aspect_ratio,'N': N,'sigma N': dN,
                'D':D_fitted, 'dD':dD_fitted, 'Tau diffusion': tau_D_fitted,   'sigma tau diffusion': dtau_D_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r':r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}
    
    elif fitting_model == 'g2diff':
        PSF_radius = given_params['PSF_radius']
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        N_fitted, tau_D_fitted = parameters
        dN_fitted, dtau_D_fitted= param_err
        D_fitted = (PSF_radius**2)/(4*tau_D_fitted)
        try: 
            dD_fitted = D_fitted * dtau_D_fitted/tau_D_fitted
        except:
            dD_fitted = np.nan
        CPP_peak = count_rate/N_fitted
        CPP_avg = CPP_peak/(2*np.sqrt(2))
        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio,'N': N_fitted,'sigma N': dN_fitted,
                'D':D_fitted, 'dD': dD_fitted, 'Tau diffusion': tau_D_fitted,   'sigma tau diffusion': dtau_D_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r':r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}
    
    elif fitting_model == 'g2diffSFCS':
        PSF_radius = given_params['PSF_radius']
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        N_fitted, tau_D_fitted = parameters
        dN_fitted, dtau_D_fitted = param_err
        D_fitted = (PSF_radius ** 2) / (4 * tau_D_fitted)
        try:
            dD_fitted = D_fitted * dtau_D_fitted / tau_D_fitted
        except:
            dD_fitted = np.nan
        CPP_peak = count_rate / N_fitted
        CPP_avg = CPP_peak / (2 * np.sqrt(2))
        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio, 'N': N_fitted, 'sigma N': dN_fitted,
                'D': D_fitted, 'dD': dD_fitted, 'Tau diffusion': tau_D_fitted, 'sigma tau diffusion': dtau_D_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r': r,
                'weighted_r': weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest': p_ttest, 'p_wilcoxon': p_wilcoxon,
                'p_runstest': p_runstest,
                'p_runstest_residuals': p_runstest_residuals, 'BIC': BIC}

    elif fitting_model == 'g2diffOffset':
        PSF_radius = given_params['PSF_radius']
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        N_fitted, tau_D_fitted, offset_fitted = parameters
        dN_fitted, dtau_D_fitted, doffset_fitted= param_err
        D_fitted = (PSF_radius**2)/(4*tau_D_fitted)
        try: 
            dD_fitted = D_fitted * dtau_D_fitted/tau_D_fitted
        except:
            dD_fitted = np.nan
        CPP_peak = count_rate/N_fitted
        CPP_avg = CPP_peak/(2*np.sqrt(2))

        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio,'N': N_fitted,'sigma N': dN_fitted,
                'D':D_fitted, 'dD': dD_fitted, 'Tau diffusion': tau_D_fitted, 'dTauD': dtau_D_fitted,  'sigma tau diffusion': dtau_D_fitted,
                'offset':offset_fitted, 'sigma offset':doffset_fitted,'CPP average': CPP_avg, 'CPP peak': CPP_peak, 
                'Chi squared': r_chi_squared, 'r':r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 
                'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}
    
    elif fitting_model == 'g2diffBlink':
        fitted_params = parameters
        PSF_radius = given_params['PSF_radius']
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        N_fitted = fitted_params['N'].value
        tau_D_fitted = fitted_params['tau_D'].value
        dtau_D_fitted =fitted_params['tau_D'].stderr
        tau_Blink_fitted = fitted_params['tau_Blink'].value
        F_Blink_fitted = fitted_params['F_Blink'].value
        D_fitted = (PSF_radius ** 2) / (4 * tau_D_fitted)
        try: 
            dD_fitted = D_fitted * dtau_D_fitted/tau_D_fitted
        except:
            dD_fitted = np.nan
        CPP_peak = count_rate / N_fitted
        CPP_avg = CPP_peak / (2 * np.sqrt(2))

        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio, 'N': N_fitted, 'D': D_fitted, 'dD': dD_fitted,
                'Tau diffusion': tau_D_fitted, 'Tau Blink': tau_Blink_fitted, 'F Blink': F_Blink_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r': r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}
    
    elif fitting_model == 'g3diffBlink':
        fitted_params = parameters
        PSF_radius = given_params['PSF_radius']
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        N_fitted = fitted_params['N'].value
        tau_D_fitted = fitted_params['tau_D'].value
        dtau_D_fitted =fitted_params['tau_D'].stderr
        tau_Blink_fitted = fitted_params['tau_Blink'].value
        F_Blink_fitted = fitted_params['F_Blink'].value
        D_fitted = (PSF_radius ** 2) / (4 * tau_D_fitted)
        try: 
            dD_fitted = D_fitted * dtau_D_fitted/tau_D_fitted
        except:
            dD_fitted = np.nan
        CPP_peak = count_rate / N_fitted
        CPP_avg = CPP_peak / (2 * np.sqrt(2))

        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio, 'N': N_fitted, 'D': D_fitted,'dD':dD_fitted, 
                'Tau diffusion': tau_D_fitted, 'Tau Blink': tau_Blink_fitted, 'F Blink': F_Blink_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r': r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}

    elif fitting_model == 'g3diffBlinkOffset':
        fitted_params = parameters
        PSF_radius = given_params['PSF_radius']
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        N_fitted = fitted_params['N'].value
        tau_D_fitted = fitted_params['tau_D'].value
        dtau_D_fitted =fitted_params['tau_D'].stderr
        tau_Blink_fitted = fitted_params['tau_Blink'].value
        F_Blink_fitted = fitted_params['F_Blink'].value
        offset_fitted = fitted_params['offset'].value
        D_fitted = (PSF_radius ** 2) / (4 * tau_D_fitted)
        try: 
            dD_fitted = D_fitted * dtau_D_fitted/tau_D_fitted
        except:
            dD_fitted = np.nan
        CPP_peak = count_rate / N_fitted
        CPP_avg = CPP_peak / (2 * np.sqrt(2))

        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio, 'N': N_fitted, 'D': D_fitted, 'dD':dD_fitted, 
                'Tau diffusion': tau_D_fitted, 'Tau Blink': tau_Blink_fitted, 'F Blink': F_Blink_fitted, 'offset':offset_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r': r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}

    elif fitting_model == 'g3diffBlinkCal':
        fitted_params = parameters
        N_fitted = fitted_params['N'].value
        tau_D_fitted = fitted_params['tau_D'].value
        dtau_D_fitted =fitted_params['tau_D'].stderr
        tau_Blink_fitted = fitted_params['tau_Blink'].value
        F_Blink_fitted = fitted_params['F_Blink'].value
        PSF_aspect_ratio = fitted_params['PSF_aspect_ratio'].value
        D_fitted = corrected_D
        try: 
            dD_fitted = D_fitted * dtau_D_fitted/tau_D_fitted
        except:
            dD_fitted = np.nan
        PSF_radius = np.sqrt(4 * D_fitted * tau_D_fitted)
        CPP_peak = count_rate / N_fitted
        CPP_avg = CPP_peak / (2 * np.sqrt(2))

        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio, 'N': N_fitted, 'D': D_fitted, 'dD':dD_fitted, 
                'Tau diffusion': tau_D_fitted, 'Tau Blink': tau_Blink_fitted, 'F Blink': F_Blink_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r': r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}
    
    elif fitting_model ==  'g3diffDoubleBlink':
        fitted_params = parameters
        PSF_radius = given_params['PSF_radius']
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        N_fitted = fitted_params['N'].value
        tau_D_fitted = fitted_params['tau_D'].value
        dtau_D_fitted =fitted_params['tau_D'].stderr
        tau_Blink1_fitted = fitted_params['tau_Blink1'].value
        F_Blink1_fitted = fitted_params['F_Blink1'].value
        tau_Blink2_fitted = fitted_params['tau_Blink2'].value
        F_Blink2_fitted = fitted_params['F_Blink2'].value

        D_fitted = (PSF_radius**2)/(4*tau_D_fitted)
        try: 
            dD_fitted = D_fitted * dtau_D_fitted/tau_D_fitted
        except:
            dD_fitted = np.nan
        CPP_peak = count_rate/N_fitted
        CPP_avg = CPP_peak/(2*np.sqrt(2))

        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio,'N': N_fitted,'D':D_fitted, 'dD':dD_fitted, 
                'Tau diffusion': tau_D_fitted,'Tau Blink 1': tau_Blink1_fitted, 'F Blink 1': F_Blink1_fitted,
                'Tau Blink 2': tau_Blink2_fitted, 'F Blink 2': F_Blink2_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r':r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}
    
    elif fitting_model == 'g3lorentzianZ':
        PSF_radius = given_params['PSF_radius']
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        N_fitted, tau_D_fitted = parameters
        dN_fitted, dtau_D_fitted= param_err
        D_fitted = (PSF_radius**2)/(4*tau_D_fitted)
        try: 
            dD_fitted = D_fitted * dtau_D_fitted/tau_D_fitted
        except:
            dD_fitted = np.nan
        CPP_peak = count_rate/N_fitted
        CPP_avg = CPP_peak/(2*np.sqrt(2))

        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio,'N': N_fitted,'sigma N': dN_fitted,
                'D':D_fitted, 'dD':dD_fitted, 'Tau diffusion': tau_D_fitted,   'sigma tau diffusion': dtau_D_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r':r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}
    
    elif fitting_model == 'g3lorentzianZCal':
        N_fitted, tau_D_fitted, PSF_aspect_ratio_fitted = parameters
        dN_fitted, dtau_D_fitted, dPSF_aspect_ratio_fitted= param_err
        PSF_radius = np.sqrt(4*corrected_D*tau_D_fitted)
        CPP_peak = count_rate/N_fitted
        CPP_avg = CPP_peak/(2*np.sqrt(2))

        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio_fitted,'N': N_fitted,'sigma N': dN_fitted,
                'D':corrected_D, 'Tau diffusion': tau_D_fitted,   'sigma tau diffusion': dtau_D_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r':r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}
    
    elif fitting_model == 'g3anomalousDiff':
        PSF_radius = given_params['PSF_radius']
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        N_fitted, gamma_fitted, alpha_fitted = parameters
        dN_fitted, dgamma_fitted, dalpha_fitted= param_err
        CPP_peak = count_rate/N_fitted
        CPP_avg = CPP_peak/(2*np.sqrt(2))
        tau_D_fitted = (PSF_radius**2 / 4 / gamma_fitted)** (1/alpha_fitted)
        D_app_fitted = gamma_fitted**(1/alpha_fitted) * (PSF_radius**2 / 4)**(1-1/alpha_fitted)
        
        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio,'N': N_fitted,
                'sigma N': dN_fitted,'Gamma':gamma_fitted, 'Alpha': alpha_fitted, 'Tau diffusion': tau_D_fitted,'D':D_app_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r':r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}
    
    elif fitting_model == 'g3anomalousDiffBlink':
        fitted_params = parameters
        PSF_radius = given_params['PSF_radius']
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        N_fitted = fitted_params['N'].value
        dN_fitted = fitted_params['N'].stderr
        gamma_fitted = fitted_params['gamma'].value
        alpha_fitted = fitted_params['alpha'].value
        tau_D_fitted = (PSF_radius**2 / 4 / gamma_fitted)** (1/alpha_fitted)
        D_app_fitted = gamma_fitted**(1/alpha_fitted) * (PSF_radius**2 / 4)**(1-1/alpha_fitted)
        tau_Blink_fitted = fitted_params['tau_Blink'].value
        F_Blink_fitted = fitted_params['F_Blink'].value
        CPP_peak = count_rate / N_fitted
        CPP_avg = CPP_peak / (2 * np.sqrt(2))

        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio, 'N': N_fitted, 'dN':dN_fitted, 
                'Gamma':gamma_fitted, 'Alpha': alpha_fitted, 'D': D_app_fitted,
                'Tau diffusion': tau_D_fitted, 'Tau Blink': tau_Blink_fitted, 'F Blink': F_Blink_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r': r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}

    elif fitting_model == 'g3diffTwoComponents':
        fitted_params = parameters
        PSF_radius = given_params['PSF_radius']
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        N_fitted = fitted_params['N'].value
        tau_D1_fitted = fitted_params['tau_D1'].value
        dtau_D1_fitted =fitted_params['tau_D1'].stderr
        f1_fitted = fitted_params['f1'].value
        tau_D2_fitted = fitted_params['tau_D2'].value
        dtau_D2_fitted =fitted_params['tau_D2'].stderr
        f2_fitted = fitted_params['f2'].value
        D1_fitted = (PSF_radius ** 2) / (4 * tau_D1_fitted)
        try: 
            dD1_fitted = D1_fitted * dtau_D1_fitted/tau_D1_fitted
        except:
            dD1_fitted = np.nan
        D2_fitted = (PSF_radius ** 2) / (4 * tau_D2_fitted)
        try: 
            dD2_fitted = D2_fitted * dtau_D2_fitted/tau_D2_fitted
        except:
            dD2_fitted = np.nan
        CPP_peak = count_rate / N_fitted
        CPP_avg = CPP_peak / (2 * np.sqrt(2))

        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio, 'N': N_fitted,
                'D1': D1_fitted, 'dD1': dD1_fitted, 'Tau diffusion 1': tau_D1_fitted, 'f1': f1_fitted,
                'D2': D2_fitted, 'dD2': dD2_fitted, 'Tau diffusion 2': tau_D2_fitted, 'f2': f2_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r': r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}
    elif fitting_model == 'g2diffTwoComponents':
        fitted_params = parameters
        PSF_radius = given_params['PSF_radius']
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        N_fitted = fitted_params['N'].value
        tau_D1_fitted = fitted_params['tau_D1'].value
        dtau_D1_fitted =fitted_params['tau_D1'].stderr
        f1_fitted = fitted_params['f1'].value
        tau_D2_fitted = fitted_params['tau_D2'].value
        dtau_D2_fitted =fitted_params['tau_D2'].stderr
        f2_fitted = fitted_params['f2'].value
        D1_fitted = (PSF_radius ** 2) / (4 * tau_D1_fitted)
        try: 
            dD1_fitted = D1_fitted * dtau_D1_fitted/tau_D1_fitted
        except:
            dD1_fitted = np.nan
        D2_fitted = (PSF_radius ** 2) / (4 * tau_D2_fitted)
        try: 
            dD2_fitted = D2_fitted * dtau_D2_fitted/tau_D2_fitted
        except:
            dD2_fitted = np.nan
        CPP_peak = count_rate / N_fitted
        CPP_avg = CPP_peak / (2 * np.sqrt(2))

        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio, 'N': N_fitted,
                'D1': D1_fitted, 'dD1': dD1_fitted, 'Tau diffusion 1': tau_D1_fitted, 'f1': f1_fitted,
                'D2': D2_fitted, 'dD2': dD2_fitted, 'Tau diffusion 2': tau_D2_fitted, 'f2': f2_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r': r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}
    elif fitting_model == 'g3diffTwoComponentsBlink':
        fitted_params = parameters
        PSF_radius = given_params['PSF_radius']
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        N_fitted = fitted_params['N'].value
        tau_D1_fitted = fitted_params['tau_D1'].value
        dtau_D1_fitted =fitted_params['tau_D1'].stderr
        f1_fitted = fitted_params['f1'].value
        tau_D2_fitted = fitted_params['tau_D2'].value
        dtau_D2_fitted =fitted_params['tau_D2'].stderr
        f2_fitted = fitted_params['f2'].value
        tau_Blink_fitted = fitted_params['tau_Blink'].value
        F_Blink_fitted = fitted_params['F_Blink'].value
        D1_fitted = (PSF_radius ** 2) / (4 * tau_D1_fitted)
        try: 
            dD1_fitted = D1_fitted * dtau_D1_fitted/tau_D1_fitted
        except:
            dD1_fitted = np.nan
        D2_fitted = (PSF_radius ** 2) / (4 * tau_D2_fitted)
        try: 
            dD2_fitted = D2_fitted * dtau_D2_fitted/tau_D2_fitted
        except:
            dD2_fitted = np.nan
        CPP_peak = count_rate / N_fitted
        CPP_avg = CPP_peak / (2 * np.sqrt(2))

        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio, 'N': N_fitted,
                'D1': D1_fitted, 'dD1': dD1_fitted, 'Tau diffusion 1': tau_D1_fitted, 'f1' : f1_fitted,
                'D2': D2_fitted, 'dD2': dD2_fitted, 'Tau diffusion 2': tau_D2_fitted, 'f2': f2_fitted,
                'Tau Blink': tau_Blink_fitted, 'F Blink': F_Blink_fitted,
                'CPP average': CPP_avg, 'CPP peak': CPP_peak, 'Chi squared': r_chi_squared, 'r': r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}
    
    elif fitting_model == 'siFCS':
        G0_fitted, tau_c_fitted = parameters
        dG0_fitted, dtau_c_fitted = param_err
        
        return {'G0': G0_fitted,'sigma G0': dG0_fitted, 'N': 1/G0_fitted,
                'Tau characteristic decay': tau_c_fitted,   'sigma tau characteristic decay': dtau_c_fitted,
                'Chi squared': r_chi_squared, 'r':r, 'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}
    
    elif fitting_model == 'siFCSTwoComponents':
        G0_1_fitted, G0_2_fitted, tau_c1_fitted, tau_c2_fitted = parameters
        dG0_1_fitted, dG0_2_fitted, dtau_c1_fitted, dtau_c2_fitted  = param_err
        
        return {'G0_1': G0_1_fitted, 'G0_2': G0_2_fitted,'sigma G0_1': dG0_1_fitted,'sigma G0_2': dG0_2_fitted, 'N': 1/(G0_1_fitted+G0_2_fitted),
                'Tau characteristic decay short': tau_c1_fitted,   'sigma tau characteristic decay short': dtau_c1_fitted,
                'Tau characteristic decay long': tau_c2_fitted,   'sigma tau characteristic decay long': dtau_c2_fitted,
                'Chi squared': r_chi_squared, 'r':r, 'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC}
    
    elif fitting_model == 'g3diffMEMFCS':
        PSF_radius = given_params['PSF_radius']
        PSF_aspect_ratio = given_params['PSF_aspect_ratio']
        tau_D = parameters['tau D']
        D = (PSF_radius ** 2) / (4 * tau_D)
        a_fit_normalized = parameters['Amplitudes']
        max_index = np.argmax(a_fit_normalized)
        max_freq_tau_D = tau_D[max_index]
        max_freq_D = D[max_index]
        mean_tau_D = np.sum(tau_D*a_fit_normalized)/np.sum(a_fit_normalized)
        G_fit_normalized = G
        
        # tau_D_plot = plt.figure()
        # ax = tau_D_plot.add_subplot(1,1,1)
        # ax.set_xlabel("Diffusion time")
        # ax.set_ylabel("Amplitude")
        # ax.semilogx(tau_D, a_fit_normalized)
        # ax.axvline(x=max_freq_tau_D)

        # ax.set_title("I like $\pi$")
       
        D_fitted = (PSF_radius ** 2) / (4 * mean_tau_D)
        D_plot = plt.figure()
        ax = D_plot.add_subplot(1,1,1)
        ax.set_xlabel("Diffusion Coefficient")
        ax.set_ylabel("Amplitude")
        ax.semilogx(D, a_fit_normalized)
        ax.axvline(x=max_freq_D)
        
        return {'PSF radius': PSF_radius, 'PSF aspect ratio': PSF_aspect_ratio,'Chi squared': r_chi_squared, 'r':r,'weighted_r':weighted_r,
                'ccPrediction': ccPrediction, 'Count Rate': count_rate, 'p_ttest':p_ttest, 'p_wilcoxon':p_wilcoxon,'p_runstest':p_runstest,
                'p_runstest_residuals':p_runstest_residuals,'BIC': BIC, 'D Plot': D_plot, 'tau D': tau_D, 'Amplitudes': a_fit_normalized, 'mean tau diffusion': mean_tau_D, 'D': D_fitted}
        
        
    

        
    
    

   