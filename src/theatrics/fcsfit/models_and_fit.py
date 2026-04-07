#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:09:53 2022

@author: yusufqq
"""
################################################################################################################################################################################
#%%
import traceback
import os
import math
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import special
from lmfit import minimize, Parameters, fit_report
import matplotlib.pyplot as plt
from theatrics.fcsfit import calculations as calculate
import matplotlib
matplotlib.use("Agg")
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.figsize'] = (15,10)

# Fitting Models defined as classes
# suffix Cal refers to calibration 
# Lorentzian Z referes to a lorentzian profile for the PSF in the z direction and a Gaussian profile in the xy
class g3diffMEMFCS:
    def __init__(self):
        pass
    def g3diffMEMFCS_fun(self, tau, tau_D):
        return (1/((1 + tau/tau_D) * np.sqrt(1 + tau/(tau_D*self.PSFaspectratio**2))))

class g3diffCal:
    def __init__(self):
        pass
    def g3diffCal_fun(self, tau, N, tau_D, PSFaspectratio):
        return (((self.count_rate/(self.count_rate+self.BG))**2)/# BG correction
                   (2*np.sqrt(2)*N)/ # Amplitude
                   ((1 + tau/tau_D) * np.sqrt(1+tau/(PSFaspectratio**2*tau_D))) # Diffusion
                   )
    
class g3diff:
    def __init__(self):
        pass
    def g3diff_fun(self, tau, N, tau_D):
        return (((self.count_rate/(self.count_rate+self.BG))**2)/ # BG correction
                   (2*np.sqrt(2)*N)/ # Amplitude
                   ((1 + tau/tau_D) * np.sqrt(1+tau/(self.PSFaspectratio**2*tau_D))) # Diffusion
                   )

class g3diffOffset:
    def __init__(self):
        pass
    def g3diffOffset_fun(self, tau, N, tau_D, offset):
        return (((self.count_rate/(self.count_rate+self.BG))**2)/ # BG correction
                   (2*np.sqrt(2)*N)/ # Amplitude
                   ((1 + tau/tau_D) * np.sqrt(1+tau/(self.PSFaspectratio**2*tau_D))) # Diffusion
                   + offset) # y-offset

class g3diffBlink:
    def __init__(self):
        pass

    def g3diffBlink_residual(self, params, tau, G):
        N = params['N'].value
        tau_D = params['tau_D'].value
        tau_Blink = params['tau_Blink'].value
        F_Blink = params['F_Blink'].value
        residual = (G -  # Difference data-model
                    ((self.count_rate / (self.count_rate + self.BG)) ** 2) /  # Model BG correction
                    (2 * np.sqrt(2) * N) /  # Model amplitude
                    ((1 + tau / tau_D) * np.sqrt(1 + tau / (self.PSFaspectratio ** 2 * tau_D))) *  # Model diffusion
                    (1 - F_Blink + F_Blink * np.exp(-tau / tau_Blink)) / (1 - F_Blink)  # Model blinking
                    )
        returner = residual / self.sigma_G
        return returner

    def g3diffBlink_fun(self, params, tau, G):
        N = params['N'].value
        tau_D = params['tau_D'].value
        tau_Blink = params['tau_Blink'].value
        F_Blink = params['F_Blink'].value
        return (((self.count_rate / (self.count_rate + self.BG)) ** 2) /  # BG correction
                (2 * np.sqrt(2) * N) /  # Amplitude
                ((1 + tau / tau_D) * np.sqrt(1 + tau / (self.PSFaspectratio ** 2 * tau_D))) *  # Diffusion
                (1 - F_Blink + F_Blink * np.exp(-tau / tau_Blink)) / (1 - F_Blink)  # Blinking
                )

class g3diffBlinkOffset:
    def __init__(self):
        pass

    def g3diffBlinkOffset_residual(self, params, tau, G):
        N = params['N'].value
        tau_D = params['tau_D'].value
        tau_Blink = params['tau_Blink'].value
        F_Blink = params['F_Blink'].value
        offset = params['offset'].value
        residual = (G -  # Difference data-model
                    (((self.count_rate / (self.count_rate + self.BG)) ** 2) /  # Model BG correction
                    (2 * np.sqrt(2) * N) /  # Model amplitude
                    ((1 + tau / tau_D) * np.sqrt(1 + tau / (self.PSFaspectratio ** 2 * tau_D))) *  # Model diffusion
                    (1 - F_Blink + F_Blink * np.exp(-tau / tau_Blink)) / (1 - F_Blink)  # Model blinking
                    + offset) # Model y-offset
                    )
        returner = residual / self.sigma_G
        return returner

    def g3diffBlinkOffset_fun(self, params, tau, G):
        N = params['N'].value
        tau_D = params['tau_D'].value
        tau_Blink = params['tau_Blink'].value
        F_Blink = params['F_Blink'].value
        offset = params['offset'].value
        return (((self.count_rate / (self.count_rate + self.BG)) ** 2) /  # BG correction
                (2 * np.sqrt(2) * N) /  # Amplitude
                ((1 + tau / tau_D) * np.sqrt(1 + tau / (self.PSFaspectratio ** 2 * tau_D))) *  # Diffusion
                (1 - F_Blink + F_Blink * np.exp(-tau / tau_Blink)) / (1 - F_Blink)  # Blinking
                + offset) # Model y-offset


class g3diffBlinkCal:
    def __init__(self):
        pass

    def g3diffBlinkCal_formula(self, tau, tau_B, tau_D, g0, F, aspectRatio):
        return g0 * (1 - F) ** (-1) * (1 - F + F * np.exp(-tau / tau_B)) * (1 + tau / tau_D) ** (-1) / np.sqrt(
            1 + aspectRatio ** 2 * tau / tau_D)

    def g3diffBlinkCal_residual(self, params, tau, G):
        residual = G - self.g3diffBlinkCal_fun(params, tau, G)
        returner = residual / self.sigma_G
        return returner

    def g3diffBlinkCal_fun(self, params, tau, G):
        N = params['N'].value
        tau_D = params['tau_D'].value
        tau_Blink = params['tau_Blink'].value
        F_Blink = params['F_Blink'].value
        PSFaspectratio = params['PSF_aspect_ratio'].value
        g0 = ((self.count_rate / (self.count_rate + self.BG)) ** 2) / (2 * np.sqrt(2) * N)
        return self.g3diffBlinkCal_formula(tau, tau_Blink, tau_D, g0, F_Blink, PSFaspectratio)


class g3diffDoubleBlink:
    def __init__(self):
        pass

    def g3diffDoubleBlink_residual(self, params, tau, G):
        N = params['N'].value
        tau_D = params['tau_D'].value
        tau_Blink1 = params['tau_Blink1'].value
        F_Blink1 = params['F_Blink1'].value
        tau_Blink2 = params['tau_Blink2'].value
        F_Blink2 = params['F_Blink2'].value
        residual = (G -  # Difference data-model
                    ((self.count_rate / (self.count_rate + self.BG)) ** 2) /  # Model BG correction
                    (2 * np.sqrt(2) * N) /  # Model amplitude
                    ((1 + tau / tau_D) * np.sqrt(1 + tau / (self.PSFaspectratio ** 2 * tau_D))) *  # Model diffusion
                    ((1 - F_Blink1 + F_Blink1 * np.exp(-tau / tau_Blink1)) / (1 - F_Blink1)) *  # Model blinking term 1
                    ((1 - F_Blink2 + F_Blink2 * np.exp(-tau / tau_Blink2)) / (1 - F_Blink2))  # Model blinking term 2
                    )
        returner = residual / self.sigma_G
        return returner

    def g3diffDoubleBlink_fun(self, params, tau, G):
        N = params['N'].value
        tau_D = params['tau_D'].value
        tau_Blink1 = params['tau_Blink1'].value
        F_Blink1 = params['F_Blink1'].value
        tau_Blink2 = params['tau_Blink2'].value
        F_Blink2 = params['F_Blink2'].value
        return (((self.count_rate / (self.count_rate + self.BG)) ** 2) /  # BG correction
                (2 * np.sqrt(2) * N) /  # Amplitude
                ((1 + tau / tau_D) * np.sqrt(1 + tau / (self.PSFaspectratio ** 2 * tau_D))) *  # Diffusion
                ((1 - F_Blink1 + F_Blink1 * np.exp(-tau / tau_Blink1)) / (1 - F_Blink1)) *  # Blinking term 1
                ((1 - F_Blink2 + F_Blink2 * np.exp(-tau / tau_Blink2)) / (1 - F_Blink2))  # Blinking term 2
                )

class g2diff:
    def __init__(self):
        pass
    def g2diff_fun(self, tau, N, tau_D):
        
        return (((self.count_rate/(self.count_rate+self.BG))**2)/ # BG correction
                   (2*N)/ # Amplitude
                   (1 + tau/tau_D) # Diffusion
                   )


class g2diffSFCS:
    def __init__(self):
        pass

    def g2diffSFCS_fun(self, tau, N, tau_D):
        return (((self.count_rate / (self.count_rate + self.BG)) ** 2) /  # BG correction
                (2 * N) /  # Amplitude
                np.sqrt(1 + tau / tau_D)/
                np.sqrt(1 + tau / (self.PSFaspectratio ** 2 * tau_D)) # Diffusion
                )

class g2diffOffset:
    def __init__(self):
        pass
    def g2diffOffset_fun(self, tau, N, tau_D, offset):
        
        return (((self.count_rate/(self.count_rate+self.BG))**2)/ # BG correction
                   (2*N)/ # Amplitude
                   (1 + tau/tau_D) # Diffusion
                   + offset) # model offset
    
    
class g2diffBlink:
    def __init__(self):
        pass
    def g2diffBlink_residual(self, params, tau, G):
        N = params['N'].value
        tau_D = params['tau_D'].value
        tau_Blink = params['tau_Blink'].value
        F_Blink = params['F_Blink'].value
        residual = (G -  # Difference data-model
                    (((self.count_rate / (self.count_rate + self.BG)) ** 2) *# BG correction
                     (1 - F_Blink + F_Blink * np.exp(-tau / tau_Blink)) / (1 - F_Blink) /  # Blinking
                        (2 * N) /  # Amplitude
                        ((1 + tau / tau_D)) # Diffusion    
                        ))
        returner = residual / self.sigma_G
        return returner
    
    def g2diffBlink_fun(self, params, tau, G):
        N = params['N'].value
        tau_D = params['tau_D'].value
        tau_Blink = params['tau_Blink'].value
        F_Blink = params['F_Blink'].value
        return (((self.count_rate / (self.count_rate + self.BG)) ** 2) /  # BG correction
                (2 * N) /  # Amplitude
                ((1 + tau / tau_D)) *  # Diffusion
                (1 - F_Blink + F_Blink * np.exp(-tau / tau_Blink)) / (1 - F_Blink)  # Blinking
                )
    
class g3lorentzianZ:
    def __init__(self):
        pass
    def g3lorentzianZ_fun(self, tau, N, tau_D):
        conv_factor = np.sqrt(0.5*np.log(2))
        return (((self.count_rate/(self.count_rate+self.BG))**2)/ # BG correction
                   (2*N)/ # Amplitude
                   (1 + tau/tau_D) * # xy diffusion (Gaussian)
                   (2*conv_factor*self.PSF_aspect_ratio*np.sqrt(np.pi)/np.sqrt(tau/tau_D))*(special.erfcx(2*conv_factor*self.PSF_aspect_ratio/np.sqrt(tau/tau_D))) # Z diffusion (Lorentzian)
                   )

class g3lorentzianZCal:
    def __init__(self):
        pass
    def g3lorentzianZCal_fun(self, tau, N, tau_D, PSF_aspect_ratio):
        conv_factor = np.sqrt(0.5*np.log(2))
        return (((self.count_rate/(self.count_rate+self.BG))**2)/ # BG correction
                   (2*N)/ # Amplitude
                   (1 + tau/tau_D) * # xy diffusion (Gaussian)
                   (2*conv_factor*PSF_aspect_ratio*np.sqrt(np.pi)/np.sqrt(tau/tau_D))*(special.erfcx(2*conv_factor*PSF_aspect_ratio/np.sqrt(tau/tau_D))) # Z diffusion (Lorentzian)
                   )
    
class g3anomalousDiff:
    def __init__(self):
        pass
    def g3anomalousDiff_fun(self, tau, N, gamma, alpha):
        return (((self.count_rate/(self.count_rate+self.BG))**2)/ # BG correction
                   (2*np.sqrt(2)*N)/ # Amplitude
                   ((1 + 4*gamma*(tau**alpha)/(self.PSF_radius**2)) * np.sqrt(1 + 4*gamma*(tau**alpha)/((self.PSFaspectratio*self.PSF_radius)**2))) # Motion
                   )

class g3anomalousDiffBlink:
    def __init__(self):
        pass

    def g3anomalousDiffBlink_residual(self, params, tau, G):
        N = params['N'].value
        gamma = params['gamma'].value
        alpha = params['alpha'].value
        tau_Blink = params['tau_Blink'].value
        F_Blink = params['F_Blink'].value
        residual = (G -  # Difference data-model
                    ((self.count_rate / (self.count_rate + self.BG)) ** 2) /  # Model BG correction
                    (2 * np.sqrt(2) * N) /  # Model amplitude
                    ((1 + 4*gamma*(tau**alpha)/(self.PSF_radius**2)) * np.sqrt(1 + 4*gamma*(tau**alpha)/((self.PSFaspectratio*self.PSF_radius)**2)))* # Motion
                    (1 - F_Blink + F_Blink * np.exp(-tau / tau_Blink)) / (1 - F_Blink)  # Model blinking
                    )
        returner = residual / self.sigma_G
        return returner

    def g3anomalousDiffBlink_fun(self, params, tau, G):
        N = params['N'].value
        gamma = params['gamma'].value
        alpha = params['alpha'].value
        tau_Blink = params['tau_Blink'].value
        F_Blink = params['F_Blink'].value
        return (((self.count_rate / (self.count_rate + self.BG)) ** 2) /  # BG correction
                (2 * np.sqrt(2) * N) /  # Amplitude
                ((1 + 4*gamma*(tau**alpha)/(self.PSF_radius**2)) * np.sqrt(1 + 4*gamma*(tau**alpha)/((self.PSFaspectratio*self.PSF_radius)**2)))* # Motion
                (1 - F_Blink + F_Blink * np.exp(-tau / tau_Blink)) / (1 - F_Blink)  # Blinking
                )
    
class g3diffTwoComponents:
    def __init__(self):
        pass

    def g3diffTwoComponents_formula(self, tau, g0, f1, tau_D1, f2, tau_D2, aspect_ratio):
        G_Diff1 = f1  * (1+tau/tau_D1)**(-1) / (np.sqrt(1 + aspect_ratio**(-2) * tau/tau_D1))
        G_Diff2 = f2  * (1 + tau / tau_D2) ** (-1) / (np.sqrt(1 + aspect_ratio ** (-2) * tau / tau_D2))
        return g0 * (G_Diff1 + G_Diff2)

    def g3diffTwoComponents_fun(self, params, tau):
        N = params['N'].value
        tau_D1 = params['tau_D1'].value
        tau_D2 = params['tau_D2'].value
        f1 = params['f1'].value
        f2 = params['f2'].value

        BG_correction = ((self.count_rate / (self.count_rate + self.BG)) ** 2)
        g0 = BG_correction/(2 * np.sqrt(2) * N)
        return self.g3diffTwoComponents_formula(tau, g0, f1, tau_D1, f2, tau_D2, self.PSFaspectratio)

    def g3diffTwoComponents_residual(self, params, tau, G):
        residual = G - self.g3diffTwoComponents_fun(params, tau)
        return residual / self.sigma_G

class g2diffTwoComponents:
    def __init__(self):
        pass

    def g2diffTwoComponents_formula(self, tau, g0, f1, tau_D1, f2, tau_D2):
        G_Diff1 = f1  / (1 + tau / tau_D1)
        G_Diff2 = f2  / (1 + tau / tau_D2) 
        return g0 * (G_Diff1 + G_Diff2)

    def g2diffTwoComponents_fun(self, params, tau):
        N = params['N'].value
        tau_D1 = params['tau_D1'].value
        tau_D2 = params['tau_D2'].value
        f1 = params['f1'].value
        f2 = params['f2'].value

        BG_correction = ((self.count_rate / (self.count_rate + self.BG)) ** 2)
        g0 = BG_correction/(2 * N)
        return self.g2diffTwoComponents_formula(tau, g0, f1, tau_D1, f2, tau_D2)

    def g2diffTwoComponents_residual(self, params, tau, G):
        residual = G - self.g2diffTwoComponents_fun(params, tau)
        return residual / self.sigma_G

class g3diffTwoComponentsBlink:
    def __init__(self):
        pass

    def g3diffTwoComponentsBlink_formula(self, tau, g0, f1, tau_D1, f2, tau_D2, F_B, tau_B, aspect_ratio):
        G_Blink = (1 - F_B + F_B * np.exp(-tau/tau_B))/(1 - F_B)
        G_Diff1 = f1  * (1+tau/tau_D1)**(-1) / (np.sqrt(1 + aspect_ratio**(-2) * tau/tau_D1))
        G_Diff2 = f2  * (1 + tau / tau_D2) ** (-1) / (np.sqrt(1 + aspect_ratio**(-2) * tau / tau_D2))
        return g0 * G_Blink * (G_Diff1 + G_Diff2)

    def g3diffTwoComponentsBlink_fun(self, params, tau):
        N = params['N'].value
        tau_D1 = params['tau_D1'].value
        tau_D2 = params['tau_D2'].value
        f1 = params['f1'].value
        f2 = params['f2'].value
        tau_Blink = params['tau_Blink'].value
        F_Blink = params['F_Blink'].value

        BG_correction = ((self.count_rate / (self.count_rate + self.BG)) ** 2)
        g0 = BG_correction/(2 * np.sqrt(2) * N)
        return self.g3diffTwoComponentsBlink_formula(tau, g0, f1, tau_D1, f2, tau_D2, F_Blink, tau_Blink, self.PSFaspectratio)

    def g3diffTwoComponentsBlink_residual(self, params, tau, G):
        residual = G - self.g3diffTwoComponentsBlink_fun(params, tau)
        return residual / self.sigma_G

class siFCS:
    def __init__(self):
        pass
    
    def siFCS_fun(self, tau, G0, tau_c):
        return (G0*(np.exp(-tau/tau_c)))
    
    def siFCSTwoComponents_fun(self, tau, G01, G02, tau_c1, tau_c2):
        return (G01*np.exp(-tau/tau_c1) + G02*np.exp(-tau/tau_c2))
################################################################################################################################################################################

#%%


# fitting fuction definitions

def g3diffCal_fit(tau, G, sigma_G, count_rate, corrected_D, BG, initial_params, goodness_of_fit_criterion):
    # object definition and initialization/construction
    fit_object = g3diffCal()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    
    # initial parameter definitions
    
    N_0 = initial_params['N']
    tau_D_0 = initial_params['tau diffusion']
    PSF_aspect_ratio_0 = initial_params['PSF aspect ratio']
    # fitting using scipy curve_fit for best fitting parameters and parameter covariance, all parameters are bound for positivity
    parameters, param_cov = curve_fit(fit_object.g3diffCal_fun, tau, G, p0 = [N_0, tau_D_0, PSF_aspect_ratio_0], bounds = (0,np.inf), sigma = sigma_G, absolute_sigma = True, method = 'dogbox')
    ccPrediction = fit_object.g3diffCal_fun(tau, *parameters)
    given_params = dict()
    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,len(parameters), parameters, param_cov,given_params ,'g3diffCal')
 

def g3diff_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):
    # object definition and initialization/construction

    fit_object = g3diff()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSFaspectratio = PSF_aspect_ratio
    
    # initial parameter definitions

    N_0 = initial_params['N']
    tau_D_0 = initial_params['tau diffusion']
    
    # fitting using scipy curve_fit for best fitting parameters and parameter covariance, all parameters are bound for positivity

    parameters, param_cov = curve_fit(fit_object.g3diff_fun, tau, G, p0 = [N_0, tau_D_0], bounds = (0,np.inf), sigma = sigma_G, absolute_sigma = True, method = 'dogbox')
    ccPrediction = fit_object.g3diff_fun(tau, *parameters)
    given_params = {'PSF_aspect_ratio':PSF_aspect_ratio, 'PSF_radius':PSF_radius}

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,len(parameters), parameters, param_cov,given_params ,'g3diff')


def g3diffOffset_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):
    # object definition and initialization/construction

    fit_object = g3diffOffset()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSFaspectratio = PSF_aspect_ratio
    
    # initial parameter definitions

    N_0 = initial_params['N']
    tau_D_0 = initial_params['tau diffusion']
    offset_0 = initial_params['offset']
    
    # fitting using scipy curve_fit for best fitting parameters and parameter covariance, all parameters are bound for positivity

    parameters, param_cov = curve_fit(fit_object.g3diffOffset_fun, 
                                      tau, 
                                      G, 
                                      p0 = [N_0, tau_D_0, offset_0], 
                                      bounds = ([0, 0, -np.inf] ,[np.inf, np.inf, np.inf]), 
                                      sigma = sigma_G, 
                                      absolute_sigma = True, 
                                      method = 'dogbox')
    ccPrediction = fit_object.g3diffOffset_fun(tau, *parameters)
    given_params = {'PSF_aspect_ratio':PSF_aspect_ratio, 'PSF_radius':PSF_radius}

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,len(parameters), parameters, param_cov,given_params ,'g3diffOffset')


def g3diffLargeParticles_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):
    # object definition and initialization/construction

    fit_object = g3diff()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSFaspectratio = PSF_aspect_ratio
    
    # initial parameter definitions

    N_0 = initial_params['N']
    tau_D_0 = initial_params['tau diffusion']
    Radius = initial_params['Radius of the particle']
    # fitting using scipy curve_fit for best fitting parameters and parameter covariance, all parameters are bound for positivity

    parameters, param_cov = curve_fit(fit_object.g3diff_fun, tau, G, p0 = [N_0, tau_D_0], bounds = (0,np.inf), sigma = sigma_G, absolute_sigma = True, method = 'dogbox')
    ccPrediction = fit_object.g3diff_fun(tau, *parameters)
    given_params = {'PSF_aspect_ratio':PSF_aspect_ratio, 'PSF_radius':PSF_radius, 'Radius of the particle': Radius}

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,len(parameters), parameters, param_cov,given_params ,'g3diffLargeParticles')

def g2diff_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):
    
    # object definition and initialization/construction

    fit_object = g2diff()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSFaspectratio = PSF_aspect_ratio
   
    # initial parameter definitions

    N_0 = initial_params['N']
    tau_D_0 = initial_params['tau diffusion']

    # fitting using scipy curve_fit for best fitting parameters and parameter covariance, all parameters are bound for positivity

    parameters, param_cov = curve_fit(fit_object.g2diff_fun, 
                                      tau, 
                                      G, 
                                      p0 = [N_0, tau_D_0], 
                                      bounds = (0,np.inf), 
                                      sigma = sigma_G, 
                                      absolute_sigma = True, 
                                      method = 'dogbox')
    ccPrediction = fit_object.g2diff_fun(tau, *parameters)
    given_params = {'PSF_aspect_ratio':PSF_aspect_ratio, 
                    'PSF_radius':PSF_radius}

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,len(parameters), parameters, param_cov,given_params ,'g2diff')

def g2diffSFCS(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):
    # object definition and initialization/construction

    fit_object = g2diffSFCS()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSFaspectratio = PSF_aspect_ratio

    # initial parameter definitions

    N_0 = initial_params['N']
    tau_D_0 = initial_params['tau diffusion']

    # fitting using scipy curve_fit for best fitting parameters and parameter covariance, all parameters are bound for positivity

    parameters, param_cov = curve_fit(fit_object.g2diffSFCS_fun,
                                      tau,
                                      G,
                                      p0=[N_0, tau_D_0],
                                      bounds=(0, np.inf),
                                      sigma=sigma_G,
                                      absolute_sigma=True,
                                      method='dogbox')
    ccPrediction = fit_object.g2diffSFCS_fun(tau, *parameters)
    given_params = {'PSF_aspect_ratio': PSF_aspect_ratio,
                    'PSF_radius': PSF_radius}

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG, tau, G, sigma_G,
                                        ccPrediction, len(parameters), parameters, param_cov, given_params, 'g2diffSFCS')


def g2diffOffset_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):
    
    # object definition and initialization/construction

    fit_object = g2diffOffset()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSFaspectratio = PSF_aspect_ratio
   
    # initial parameter definitions

    N_0 = initial_params['N']
    tau_D_0 = initial_params['tau diffusion']
    offset_0 = initial_params['offset']

    # fitting using scipy curve_fit for best fitting parameters and parameter covariance, all parameters are bound for positivity

    parameters, param_cov = curve_fit(fit_object.g2diffOffset_fun, 
                                      tau, 
                                      G, 
                                      p0 = [N_0, tau_D_0, offset_0], 
                                      bounds = ([0, 0, -np.inf] ,[np.inf, np.inf, np.inf]), 
                                      sigma = sigma_G, 
                                      absolute_sigma = True, 
                                      method = 'dogbox')
    ccPrediction = fit_object.g2diffOffset_fun(tau, *parameters)
    given_params = {'PSF_aspect_ratio':PSF_aspect_ratio, 
                    'PSF_radius':PSF_radius}

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,len(parameters), parameters, param_cov,given_params ,'g2diffOffset')



def g3diffBlink_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):
    # object definition and initialization/construction

    fit_object = g3diffBlink()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSFaspectratio = PSF_aspect_ratio
    fit_object.G = G
    fit_object.sigma_G = sigma_G

    # initial parameter definitions

    fit_params = Parameters()
    fit_params.add('N', value=initial_params['N'], min=0, vary=True)
    fit_params.add('tau_D', value=initial_params['tau diffusion'], min = 0, vary=True)
    fit_params.add('delta', value=initial_params['delta'], min=1, vary=True)
    fit_params.add('tau_Blink', expr='tau_D/delta',
                   vary=False)  # tau_Blink < tau_D or tau_Blink = tau_D/delta where delta > 1
    fit_params.add('F_Blink', value=initial_params['F_Blink'], min=0, max=1, vary=True)

    # fitting using lmfit minimize for best fitting parameters

    result = minimize(fit_object.g3diffBlink_residual, fit_params, args=(tau, G), method='nelder')
    print(fit_report(result), "\n")
    fitted_params = result.params
    ccPrediction = fit_object.g3diffBlink_fun(fitted_params, tau, G)
    given_params = {'PSF_aspect_ratio':PSF_aspect_ratio, 'PSF_radius':PSF_radius}

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,result.nvarys, fitted_params, np.zeros((2,2)),given_params ,'g3diffBlink')

def g2diffBlink_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):
    # object definition and initialization/construction

    fit_object = g2diffBlink()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSFaspectratio = PSF_aspect_ratio
    fit_object.G = G
    fit_object.sigma_G = sigma_G

    # initial parameter definitions

    fit_params = Parameters()
    fit_params.add('N', value=initial_params['N'], min=0, vary=True)
    fit_params.add('tau_D', value=initial_params['tau diffusion'],min = 0, vary=True)
    fit_params.add('delta', value=initial_params['delta'], min=1, vary=True)
    fit_params.add('tau_Blink', expr='tau_D/delta',
                   vary=False)  # tau_Blink < tau_D or tau_Blink = tau_D/delta where delta > 1
    fit_params.add('F_Blink', value=initial_params['F_Blink'], min=0, max=1, vary=True)

    # fitting using lmfit minimize for best fitting parameters

    result = minimize(fit_object.g2diffBlink_residual, fit_params, args=(tau, G), method='nelder')
    print(fit_report(result), "\n")
    fitted_params = result.params
    ccPrediction = fit_object.g2diffBlink_fun(fitted_params, tau, G)
    given_params = {'PSF_aspect_ratio':PSF_aspect_ratio, 'PSF_radius':PSF_radius}

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,result.nvarys, fitted_params, np.zeros((2,2)),given_params ,'g2diffBlink')

        
def g3diffBlinkOffset_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):
    # object definition and initialization/construction

    fit_object = g3diffBlinkOffset()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSFaspectratio = PSF_aspect_ratio
    fit_object.G = G
    fit_object.sigma_G = sigma_G

    # initial parameter definitions

    fit_params = Parameters()
    fit_params.add('N', value=initial_params['N'], min=0, vary=True)
    fit_params.add('tau_D', value=initial_params['tau diffusion'], min=0, vary=True)
    fit_params.add('delta', value=initial_params['delta'], min=1, vary=True)
    fit_params.add('tau_Blink', expr='tau_D/delta',
                   vary=False)  # tau_Blink < tau_D or tau_Blink = tau_D/delta where delta > 1
    fit_params.add('F_Blink', value=initial_params['F_B'], min=0, max=1, vary=True)
    fit_params.add('offset', value=initial_params['offset'], vary=True)

    # fitting using lmfit minimize for best fitting parameters

    result = minimize(fit_object.g3diffBlinkOffset_residual, fit_params, args=(tau, G), method='nelder')
    print(fit_report(result), "\n")
    fitted_params = result.params
    ccPrediction = fit_object.g3diffBlinkOffset_fun(fitted_params, tau, G)
    given_params = {'PSF_aspect_ratio':PSF_aspect_ratio, 'PSF_radius':PSF_radius}

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,result.nvarys, fitted_params, np.zeros((2,2)),given_params ,'g3diffBlinkOffset')

    

def g3diffBlinkCal_fit(tau, G, sigma_G, count_rate, corrected_D, BG, initial_params, goodness_of_fit_criterion):
    # object definition and initialization/construction

    fit_object = g3diffBlinkCal()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.G = G
    fit_object.sigma_G = sigma_G

    # initial parameter definitions

    fit_params = Parameters()
    fit_params.add('N', value=initial_params['N'], min=0, vary=True)
    fit_params.add('tau_D', value=initial_params['tau diffusion'], vary=True)
    fit_params.add('delta', value=initial_params['delta'], min=1, vary=True)
    fit_params.add('tau_Blink', expr='tau_D/delta',
                   vary=False)  # tau_Blink < tau_D or tau_Blink = tau_D/delta where delta > 1
    fit_params.add('F_Blink', value=initial_params['F_B'], min=0, max=1, vary=True)
    fit_params.add('PSF_aspect_ratio', value=initial_params['PSF aspect ratio'], min=0, vary=True)

    # fitting using lmfit minimize for best fitting parameters

    result = minimize(fit_object.g3diffBlinkCal_residual, fit_params, args=(tau, G), method='nelder')
    print(fit_report(result), "\n")
    fitted_params = result.params
    ccPrediction = fit_object.g3diffBlinkCal_fun(fitted_params, tau, G)
    given_params = dict()
    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,result.nvarys, fitted_params, np.zeros((2,2)),given_params ,'g3diffBlinkCal')

    

def g3diffDoubleBlink_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion): #~# to be completed
    fit_object = g3diffDoubleBlink()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSFaspectratio = PSF_aspect_ratio
    fit_object.G = G
    fit_object.sigma_G = sigma_G
    fit_params = Parameters()
    fit_params.add('N', value=initial_params['N'], min=0, vary=True)
    fit_params.add('tau_D', value=initial_params['tau diffusion'], vary=True)
    fit_params.add('delta2', value=initial_params['delta2'], min=1, vary=True)
    fit_params.add('tau_Blink2', expr='tau_D/delta2',
                   vary=False)  # tau_Blink2 < tau_D or tau_Blink2 = tau_D/delta2 where delta2 > 1
    fit_params.add('delta1', value=initial_params['delta1'], min=1, vary=True)
    fit_params.add('tau_Blink1', expr='tau_Blink2/delta1',
                   vary=False)  # tau_Blink1 < tau_Blink2 or tau_Blink1 = tau_Blink2/delta1 where delta1 > 1
    fit_params.add('F_Blink1', value=initial_params['F_B1'], min=0, max=1, vary=True)
    fit_params.add('Sigma_F', value=initial_params['Sigma_F'], min=0, max=1, vary=True)
    fit_params.add("F_Blink2", expr='Sigma_F - F_Blink1', min=0,
                   vary=False)  # ensures F_B1 + F_B2 < 1, as Sigma_F = F_B1 + F_B2 where Sigma_F < 1
    
    # fitting using lmfit minimize for best fitting parameters

    result = minimize(fit_object.g3diffDoubleBlink_residual, fit_params, args = (tau,G), method = 'nelder') #~# 'leastsq'

    fitted_params = result.params
    print(fit_report(result), "\n")
    ccPrediction = fit_object.g3diffDoubleBlink_fun(fitted_params, tau, G)
    given_params = {'PSF_aspect_ratio':PSF_aspect_ratio, 'PSF_radius':PSF_radius}

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,result.nvarys, fitted_params, np.zeros((2,2)),given_params ,'g3diffDoubleBlink')

    

def g3lorentzianZ_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):
    
    # object definition and initialization/construction

    fit_object = g3lorentzianZ()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSF_aspect_ratio = PSF_aspect_ratio

    # initial parameter definitions

    N_0 = initial_params['N']
    tau_D_0 = initial_params['tau diffusion']
    
    # fitting using scipy curve_fit for best fitting parameters and parameter covariance, all parameters are bound for positivity

    parameters, param_cov = curve_fit(fit_object.g3lorentzianZ_fun, tau, G, p0 = [N_0, tau_D_0], bounds = (0,np.inf), sigma = sigma_G, absolute_sigma = True, method = 'dogbox')
    ccPrediction = fit_object.g3lorentzianZ_fun(tau, *parameters)
    given_params = {'PSF_aspect_ratio':PSF_aspect_ratio, 'PSF_radius':PSF_radius}

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,len(parameters), parameters, param_cov,given_params ,'g3lorentzianZ')

   
   
def g3lorentzianZCal_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):
    
    # object definition and initialization/construction

    fit_object = g3lorentzianZCal()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    
    # initial parameter definitions

    N_0 = initial_params['N']
    tau_D_0 = initial_params['tau diffusion']
    PSF_aspect_ratio_0 = initial_params['PSF aspect ratio']

    # fitting using scipy curve_fit for best fitting parameters and parameter covariance, all parameters are bound for positivity

    parameters, param_cov = curve_fit(fit_object.g3lorentzianZCal_fun, tau, G, p0 = [N_0, tau_D_0, PSF_aspect_ratio_0], bounds = (0,np.inf), sigma = sigma_G, absolute_sigma = True, method = 'dogbox')
    ccPrediction = fit_object.g3lorentzianZCal_fun(tau, *parameters)
    given_params = dict()

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,len(parameters), parameters, param_cov,given_params ,'g3lorentzianZCal')

   
   
def g3anomalousDiff_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):
    
    # object definition and initialization/construction

    fit_object = g3anomalousDiff()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSFaspectratio = PSF_aspect_ratio
    fit_object.PSF_radius = PSF_radius
    
    # initial parameter definitions

    N_0 = initial_params['N']
    tau_D_0 = initial_params['tau diffusion']
    Gamma_0 = initial_params['Gamma']
    Alpha_0 = initial_params['Alpha']
    
    # fitting using scipy curve_fit for best fitting parameters and parameter covariance, all parameters are bound for positivity

    parameters, param_cov = curve_fit(fit_object.g3anomalousDiff_fun, tau, G, p0 = [N_0, Gamma_0, Alpha_0], bounds = (0,np.inf), sigma = sigma_G, absolute_sigma = True, method = 'dogbox')
    ccPrediction = fit_object.g3anomalousDiff_fun(tau, *parameters)
    given_params = {'PSF_aspect_ratio':PSF_aspect_ratio, 'PSF_radius':PSF_radius}

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,len(parameters), parameters, param_cov,given_params ,'g3anomalousDiff')

def g3anomalousDiffBlink_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):
    # object definition and initialization/construction

    fit_object = g3anomalousDiffBlink()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSFaspectratio = PSF_aspect_ratio
    fit_object.PSF_radius = PSF_radius
    fit_object.G = G
    fit_object.sigma_G = sigma_G

    # initial parameter definitions

    fit_params = Parameters()
    fit_params.add('N', value=initial_params['N'], min=0, vary=True)
    fit_params.add('gamma', value=initial_params['Gamma'], min = 0, vary=True)
    fit_params.add('alpha', value=initial_params['Alpha'], min = 0, vary=True)
    fit_params.add('delta', value=initial_params['delta'], min=1, vary=True)
    fit_params.add('PSF_radius', value=PSF_radius, vary=False)
    fit_params.add('tau_Blink', expr='(PSF_radius**2 / 4 / gamma)**(1/alpha)/delta',
                   vary=False)  # tau_Blink < tau_D or tau_Blink = tau_D/delta where delta > 1
    fit_params.add('F_Blink', value=initial_params['F_B'], min=0, max=1, vary=True)

    # fitting using lmfit minimize for best fitting parameters

    result = minimize(fit_object.g3anomalousDiffBlink_residual, fit_params, args=(tau, G), method='nelder')
    print(fit_report(result), "\n")
    fitted_params = result.params
    ccPrediction = fit_object.g3anomalousDiffBlink_fun(fitted_params, tau, G)
    given_params = {'PSF_aspect_ratio':PSF_aspect_ratio, 'PSF_radius':PSF_radius}

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,result.nvarys, fitted_params, np.zeros((2,2)),given_params ,'g3anomalousDiffBlink')

   
def g3diffTwoComponents_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):
    # object definition and initialization/construction

    fit_object = g3diffTwoComponents()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSFaspectratio = PSF_aspect_ratio
    fit_object.G = G
    fit_object.sigma_G = sigma_G

    fit_params = Parameters()
    fit_params.add('N', value=initial_params['N'], min=0, vary=True)
    fit_params.add('tau_D1', value=initial_params['tau diffusion'], vary=True)
    fit_params.add('f1', value=initial_params['f1'], min=0, max=1, vary=True)

    fit_params.add('rho_D', value=initial_params['rho_D'], min=1, vary=True)  # rho_D = tau_D1/tau_D2
    fit_params.add('tau_D2', expr='tau_D1/rho_D', vary=False)  # with rho_D > 1, ensures tau_D1 > tau_D2
    fit_params.add('f2', expr='1-f1', vary=False)
    
    # fitting using lmfit minimize for best fitting parameters

    result = minimize(fit_object.g3diffTwoComponents_residual, fit_params, args=(tau, G),
                      method='nelder')
    

    fitted_params = result.params
    print(fit_report(result), "\n")
    ccPrediction = fit_object.g3diffTwoComponents_fun(fitted_params, tau)
    given_params = {'PSF_aspect_ratio':PSF_aspect_ratio, 'PSF_radius':PSF_radius}
    
    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,result.nvarys, fitted_params, np.zeros((2,2)),given_params ,'g3diffTwoComponents')

def g2diffTwoComponents_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):
    # object definition and initialization/construction

    fit_object = g2diffTwoComponents()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSFaspectratio = PSF_aspect_ratio
    fit_object.G = G
    fit_object.sigma_G = sigma_G

    fit_params = Parameters()
    fit_params.add('N', value=initial_params['N'], min=0, vary=True)
    fit_params.add('tau_D1', value=initial_params['tau diffusion'], vary=True)
    fit_params.add('f1', value=initial_params['f1'], min=0, max=1, vary=True)

    fit_params.add('rho_D', value=initial_params['rho_D'], min=1, vary=True)  # rho_D = tau_D1/tau_D2
    fit_params.add('tau_D2', expr='tau_D1/rho_D', vary=False)  # with rho_D > 1, ensures tau_D1 > tau_D2
    fit_params.add('f2', expr='1-f1', vary=False)
    
    # fitting using lmfit minimize for best fitting parameters

    result = minimize(fit_object.g2diffTwoComponents_residual, fit_params, args=(tau, G),
                      method='nelder')
    

    fitted_params = result.params
    print(fit_report(result), "\n")
    ccPrediction = fit_object.g2diffTwoComponents_fun(fitted_params, tau)
    given_params = {'PSF_aspect_ratio':PSF_aspect_ratio, 'PSF_radius':PSF_radius}
    
    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,result.nvarys, fitted_params, np.zeros((2,2)),given_params ,'g2diffTwoComponents')

     

def g3diffTwoComponentsBlink_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):

    fit_object = g3diffTwoComponentsBlink()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSFaspectratio = PSF_aspect_ratio
    fit_object.G = G
    fit_object.sigma_G = sigma_G

    fit_params = Parameters()
    fit_params.add('N', value=initial_params['N'], min=0, vary=True)
    fit_params.add('tau_D1', value=initial_params['tau diffusion'], vary=True)
    fit_params.add('f1', value=initial_params['f1'], min=0, max=1, vary=True)

    fit_params.add('rho_D', value=initial_params['rho_D'], min=1, vary=True)  # rho_D = tau_D1/tau_D2
    fit_params.add('tau_D2', expr='tau_D1/rho_D', vary=False)  # with rho_D > 1, ensures tau_D1 > tau_D2
    fit_params.add('f2', expr='1-f1', vary=False)

    fit_params.add('rho_B', value=initial_params['rho_B'], min=1, vary=True)  # rho_B = tau_D2/tau_Blink
    fit_params.add('tau_Blink', expr='tau_D2/rho_B', vary=False)  # with rho_B > 1, ensures tau_D2 > tau_Blink
    fit_params.add('F_Blink', value=initial_params['F_B'], min=0, max=1, vary=True)
    
    # fitting using lmfit minimize for best fitting parameters

    result = minimize(fit_object.g3diffTwoComponentsBlink_residual, fit_params, args=(tau, G), method='nelder')  # ~# 'leastsq'
    fitted_params = result.params
    print(fit_report(result), "\n")
    ccPrediction = fit_object.g3diffTwoComponentsBlink_fun(fitted_params, tau)
    given_params = {'PSF_aspect_ratio':PSF_aspect_ratio, 'PSF_radius':PSF_radius}

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, ccPrediction,result.nvarys, fitted_params, np.zeros((2,2)),given_params ,'g3diffTwoComponentsBlink')

def siFCS_fit(tau, G, sigma_G, count_rate, initial_params, goodness_of_fit_criterion):
    # object definition and initialization/construction

    fit_object = siFCS()
    fit_object.count_rate = count_rate
    
    # initial parameter definitions

    G0_0 = initial_params['G0']
    tau_c_0 = initial_params['tau characteristic decay']
    
    # fitting using scipy curve_fit for best fitting parameters and parameter covariance, all parameters are bound for positivity

    parameters, param_cov = curve_fit(fit_object.siFCS_fun, tau, G, p0 = [G0_0, tau_c_0], bounds = (0,np.inf), sigma = sigma_G, absolute_sigma = True, method = 'dogbox')
    ccPrediction = fit_object.siFCS_fun(tau, *parameters)
    given_params = {}

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, 0, 0,tau, G, sigma_G, ccPrediction,len(parameters), parameters, param_cov,given_params ,'siFCS')

def siFCSTwoComponents_fit(tau, G, sigma_G, count_rate, initial_params, goodness_of_fit_criterion):
    # object definition and initialization/construction

    fit_object = siFCS()
    fit_object.count_rate = count_rate
    
    # initial parameter definitions

    G0_1_0 = initial_params['G0_1']
    G0_2_0 = initial_params['G0_2']
    tau_c1_0 = initial_params['tau characteristic decay short']
    tau_c2_0 = initial_params['tau characteristic decay long']
    
    # fitting using scipy curve_fit for best fitting parameters and parameter covariance, all parameters are bound for positivity

    parameters, param_cov = curve_fit(fit_object.siFCSTwoComponents_fun, tau, G, p0 = [G0_1_0, G0_2_0, tau_c1_0, tau_c2_0], bounds = (0,np.inf), sigma = sigma_G, absolute_sigma = True, method = 'dogbox')
    ccPrediction = fit_object.siFCSTwoComponents_fun(tau, *parameters)
    given_params = {}

    # Calculations from fitting parameters
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, 0, 0,tau, G, sigma_G, ccPrediction,len(parameters), parameters, param_cov,given_params ,'siFCSTwoComponents')

def g3diffMEMFCS_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion):
    # object definition and initialization/construction

    fit_object = g3diffMEMFCS()
    fit_object.count_rate = count_rate
    fit_object.BG = BG
    fit_object.PSFaspectratio = PSF_aspect_ratio
    
    # initital parameter definitions
    n_tau_D = initial_params['number of diffusion components']
    tau_D_lims = initial_params['tau_D limits']  
    n_iterations = initial_params['number of iterations']
    
    # initialization with initial diffusion constant having a flat distribituion
    n_tau = len(tau)
    # n_tau_D = 200
    
    tau_D = np.logspace(tau_D_lims[0],tau_D_lims[1],n_tau_D) # evenly spaced tau_D in the log scale base = 10
    dtau_D = np.diff(tau_D)
    dtau_D = np.append(dtau_D, dtau_D[n_tau_D-2])
    a_avg = 1/n_tau_D # setting up the flat distribution
    a_fit = np.zeros(n_tau_D)
    G_fit = np.zeros(n_tau)
    for i in range(n_tau_D):
        a_fit[i] = a_avg
    # Now, each row will correspond to a different tau_D value in the evenly distributed log space, and the columns
    # are different values of lag time 
    fun_fit = np.zeros((n_tau_D, n_tau))
    for i in range(n_tau_D):
        fun_fit[i] = fit_object.g3diffMEMFCS_fun(tau, tau_D[i])# this generates initial curves with flat distribution 
    # fun_fit = np.transpose(fun_fit)
    # iterations
    stop_criterion = 5e-6
    iterator = 0
    x = 2e-4
    r_chi_squared = list()
    S = list()
    alpha_f = list()
    # fig, ax = plt.subplots(nrows = 1, ncols =2)
    # ax[0].semilogx(tau, G, 'r', label = 'G observed')

    while iterator<n_iterations:
        sum_a_fit = np.sum(a_fit)
        a_fit_normalized = a_fit/sum_a_fit
        fun_fit = np.transpose(fun_fit)
        G_fit = np.dot(fun_fit,a_fit_normalized) # ndarray wrt lag time (sum over all diffusion time components)
        
        G_fit_normalized = G_fit/G_fit[0] # Normalize G_fit
        G_fit_normalized = G_fit_normalized*np.mean(G[0:10]) # scaled to the average amplitude of the first two points from the data
        r = G_fit_normalized - G
        weighted_r = r/sigma_G
        r_chi_squared.append(np.sum((r/sigma_G)**2)/n_tau) # storing least squares
        S.append(-np.sum(a_fit_normalized*np.log(a_fit_normalized))) # storing entropies
        # ax[0].semilogx(tau, G_fit_normalized, 'g', label = 'G fit')
        
        # iteration termination criterion
        if iterator % 2000 == 0 and iterator > 1:
            r_chi_sq = np.asarray(r_chi_squared)
            mask1 = np.full(len(r_chi_squared), False)
            mask1[iterator-200:iterator-100] = True
            mask2 = np.full(len(r_chi_squared), False)
            mask2[iterator-100:iterator] = True
            A1 = np.sum(r_chi_sq, where = mask1)
            A2 = np.sum(r_chi_sq, where = mask2)
            # print ((A1-A2)/A2)
            if abs((A1 - A2)/A2) < stop_criterion:
                print('stopping criterion reached')
                break
       
        fun_fit = np.transpose(fun_fit)
        # First order derivative of least-squares
        D_r_chi_squared = np.sum(2*r*fun_fit/sigma_G**2, axis = 1)/n_tau
        
        # first order derivative of entropy
        D_S = -1 - np.log(a_fit_normalized)
        
        # Scaling factor
        alpha_f.append(abs(D_r_chi_squared)/(20*abs(D_S)))
        # search direction construct
        e_G = a_fit_normalized*(alpha_f[iterator]*D_S - D_r_chi_squared/2) # del Q * a_fit
        # e_G = e_G/abs(e_G)
        # update the a_fit using the search direction
        a_fit_normalized = a_fit_normalized + e_G*x
        for i in range(n_tau_D):
            if a_fit_normalized[i] <= 0 : 
                a_fit_normalized[i] = 0.0001
        a_fit = a_fit_normalized
        iterator +=1 
        
    parameters = {'tau D': tau_D, 'Amplitudes': a_fit_normalized}
    param_cov = list()
    print('Total number of iterations: ', iterator)
    given_params = {'PSF_aspect_ratio':PSF_aspect_ratio, 'PSF_radius':PSF_radius}
    # print('Hello')
    # plt.plot(r_chi_squared, 'r')
    # plt.plot(S, 'g')
    # ax[1].semilogx(tau_D, a_fit, 'r')
    return calculate.calculate_from_fit(goodness_of_fit_criterion, count_rate, corrected_D, BG,tau, G, sigma_G, G_fit_normalized,0, parameters, param_cov,given_params ,'g3diffMEMFCS')

    # display figures for 2 seconds
    # fig.show()
    # Saving figures 
    # plt.savefig('\\samba-pool-schwille-spt\pool-schwille-spt\P9_CPegoraro_CL\Analysis\20220523_polymer_conjugates.sptw\GroupMeas_2\Results.png', dpi = 400)
    
    
   



################################################################################################################################################################################
#%%
# Main function loop 

def main(path, fitting_model, result_name, corrected_D, save_path, BG, PSF_radius, PSF_aspect_ratio,
         user_initial_params, initial_params,figure_display_delay, user_tau_domain=False, tau_domain=(1e-6, 1), goodness_of_fit_criterion = ['instant_correlation_wilcoxon']):
    # parsing correlation csv files with Pandas DataFrame
    data = pd.read_csv(path+'.csv', header = None)

    # parsing columns to numpy arrays for further processing
    tau = data.iloc[:,0].to_numpy()
    G = data.iloc[:, 1].to_numpy()
    sigma_G = data.iloc[:, 3].to_numpy()
    count_rate = data.iloc[0,2]
    
    if user_tau_domain:
    
        lim_inf_tau, lim_sup_tau = np.min(tau_domain), np.max(tau_domain)
        # mask = (tau <= lim_sup_tau) & (tau >= lim_inf_tau)
        # [print(type(value)) for value in tau]
        mask = np.logical_and(tau <= lim_sup_tau, tau >= lim_inf_tau)
        tau = tau[mask]
        G = G[mask]
        sigma_G = sigma_G[mask]

    # return_dict is the dictionary of parameters returned from fitting functions, initialization of the dictionary
    
    return_dict = dict()
    # Initial parameters pre-processing, selection of default parameter if the user_initial_params flag is False, otherwise using user defined initial params for fitting

    if user_initial_params is True:
        if 'N' not in initial_params.keys():
            initial_params['N'] = 0.5
        if 'tau diffusion' not in initial_params.keys():
            initial_params['tau diffusion'] = 1E-3
        if 'Gamma' not in initial_params.keys():
            initial_params['Gamma'] = 100
        if 'Alpha' not in initial_params.keys():
            initial_params['Alpha'] = 1
        if 'PSF aspect ratio' not in initial_params.keys():
            initial_params['PSF aspect ratio'] = 4
        if 'f1' not in initial_params.keys():
            initial_params['f1'] = 0.5
        if 'delta' not in initial_params.keys():
            initial_params['delta'] = 1E+2
        if 'F_B' not in initial_params.keys():
            initial_params['F_B'] = 0.1
        if 'offset' not in initial_params.keys():
            initial_params['offset'] = 0
        if 'delta1' not in initial_params.keys():
            initial_params['delta1'] = 1E+1
        if 'delta2' not in initial_params.keys():
            initial_params['delta2'] = 1E+1
        if 'F_B1' not in initial_params.keys():
            initial_params['F_B1'] = 0.1
        if 'Sigma_F' not in initial_params.keys():
            initial_params['Sigma_F'] = 0.2
        if 'rho_D' not in initial_params.keys():
            initial_params['rho_D'] = 1E+2
        if 'rho_B' not in initial_params.keys():
            initial_params['rho_B'] = 5E+3
        if 'G0' not in initial_params.keys():
            initial_params['G0'] = 1E-4
        if 'tau characteristic decay' not in initial_params.keys():
            initial_params['tau characteristic decay'] = 30
        if 'tau_D limits' not in initial_params.keys():
            initial_params['tau_D limits'] = [-7,-2]
        if 'number of diffusion components' not in initial_params.keys():
            initial_params['number of diffusion components'] = 200
        if 'tau characteristic decay' not in initial_params.keys():
            initial_params['number of iterations'] = 10000
        if 'Radius' not in initial_params.keys():
            initial_params['Radius of the particle'] = 0.1

    else:
        initial_params['tau diffusion'] = 1E-3
        initial_params['Gamma'] = 100
        initial_params['Alpha'] = 1
        initial_params['PSF aspect ratio'] = 4
        initial_params['f1'] = 0.5
        initial_params['delta'] = 1E+2
        initial_params['F_B'] = 0.1
        initial_params['offset'] = 0
        initial_params['delta1'] = 1E+1
        initial_params['delta2'] = 1E+1
        initial_params['F_B1'] = 0.1
        initial_params['Sigma_F'] = 0.2
        initial_params['rho_D'] = 1E+2
        initial_params['rho_B'] = 5E+3
        initial_params['G0'] = 1E-4
        initial_params['tau characteristic decay'] = 30
        initial_params['tau_D limits'] = [-7,-2] # logspace between 1E-7 to 1E-2
        initial_params['number of diffusion components'] = 200
        initial_params['number of iterations'] =  10000
        initial_params['Radius of the particle'] =  0.1
    # selection of fitting model
    if fitting_model == 'g3diffCal':
       return_dict = g3diffCal_fit(tau, G, sigma_G, count_rate, corrected_D, BG, initial_params, goodness_of_fit_criterion)
       # saving the fitted data to csv file
       estimate_data = {'PSF radius': [return_dict['PSF radius']],
                        'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                        'Count Rate': [return_dict['Count Rate']], 'D': [return_dict['D']], 'N': [return_dict['N']],
                        'Chi squared': [return_dict['Chi squared']], 'Tau diffusion': [return_dict['Tau diffusion']],
                        'CPP peak': [return_dict['CPP peak']],
                        'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                        'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames

    elif fitting_model == 'g3diff':
        return_dict = g3diff_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio,
                                 initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'PSF radius': [return_dict['PSF radius']],
                         'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'D': [return_dict['D']], 'dD': [return_dict['dD']],
                         'N': [return_dict['N']], 'Chi squared': [return_dict['Chi squared']], 'Tau diffusion': [return_dict['Tau diffusion']],
                         'CPP peak': [return_dict['CPP peak']],
                         'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames

    elif fitting_model == 'g3diffOffset':
        return_dict = g3diffOffset_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio,
                                 initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'PSF radius': [return_dict['PSF radius']],
                         'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'D': [return_dict['D']], 'dD': [return_dict['dD']],'N': [return_dict['N']],
                         'Chi squared': [return_dict['Chi squared']], 'Tau diffusion': [return_dict['Tau diffusion']], 'offset': [return_dict['offset']],
                         'CPP peak': [return_dict['CPP peak']],
                         'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames

        
    elif fitting_model == 'g3diffLargeParticles':
        return_dict = g3diffLargeParticles_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio,
                                 initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'PSF radius': [return_dict['PSF radius']],
                         'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'D': [return_dict['D']], 'dD': [return_dict['dD']],'N': [return_dict['N']],
                         'Chi squared': [return_dict['Chi squared']], 'Tau diffusion': [return_dict['Tau diffusion']],
                         'CPP peak': [return_dict['CPP peak']],
                         'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames
        
    elif fitting_model == 'g3diffBlink':
        return_dict = g3diffBlink_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio,
                                      initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'PSF radius': [return_dict['PSF radius']],
                         'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'D': [return_dict['D']], 'dD': [return_dict['dD']],'N': [return_dict['N']],
                         'Chi squared': [return_dict['Chi squared']], 'Tau diffusion': [return_dict['Tau diffusion']],
                         'Tau Blink': [return_dict['Tau Blink']], 'F Blink': [return_dict['F Blink']],
                         'CPP peak': [return_dict['CPP peak']],
                         'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames

    elif fitting_model == 'g3diffBlinkOffset':
        return_dict = g3diffBlinkOffset_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio,
                                      initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'PSF radius': [return_dict['PSF radius']],
                         'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'D': [return_dict['D']], 'dD': [return_dict['dD']],'N': [return_dict['N']],
                         'Chi squared': [return_dict['Chi squared']], 'Tau diffusion': [return_dict['Tau diffusion']],
                         'Tau Blink': [return_dict['Tau Blink']], 'F Blink': [return_dict['F Blink']], 'Offset': [return_dict['offset']],
                         'CPP peak': [return_dict['CPP peak']],
                         'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames

    
    elif fitting_model == 'g3diffDoubleBlink':
        return_dict = g3diffDoubleBlink_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion)
    
        estimate_data = {'PSF radius': [return_dict['PSF radius']],
                         'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'D': [return_dict['D']], 'dD': [return_dict['dD']],'N': [return_dict['N']],
                         'Chi squared': [return_dict['Chi squared']], 'Tau diffusion': [return_dict['Tau diffusion']],
                         'Tau Blink 1': [return_dict['Tau Blink 1']], 'F Blink 1': [return_dict['F Blink 1']],
                         'Tau Blink 2': [return_dict['Tau Blink 2']], 'F Blink 2': [return_dict['F Blink 2']],
                         'CPP peak': [return_dict['CPP peak']],
                         'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}
    
    elif fitting_model == 'g3diffBlinkCal':
        return_dict = g3diffBlinkCal_fit(tau, G, sigma_G, count_rate, corrected_D, BG, initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'PSF radius': [return_dict['PSF radius']],
                         'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'D': [return_dict['D']], 'N': [return_dict['N']],
                         'Chi squared': [return_dict['Chi squared']], 'Tau diffusion': [return_dict['Tau diffusion']],
                         'Tau Blink': [return_dict['Tau Blink']], 'F Blink': [return_dict['F Blink']],
                         'CPP peak': [return_dict['CPP peak']],
                         'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames
    
    elif fitting_model == 'g2diff':
        return_dict = g2diff_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio,
                                 initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'PSF radius': [return_dict['PSF radius']],
                         'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'D': [return_dict['D']], 'dD': [return_dict['dD']],'N': [return_dict['N']],
                         'Chi squared': [return_dict['Chi squared']], 'Tau diffusion': [return_dict['Tau diffusion']],
                         'CPP peak': [return_dict['CPP peak']],
                         'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames

    elif fitting_model == 'g2diffSFCS':
        return_dict = g2diff_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio,
                                 initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'PSF radius': [return_dict['PSF radius']],
                         'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'D': [return_dict['D']], 'dD': [return_dict['dD']],
                         'N': [return_dict['N']],
                         'Chi squared': [return_dict['Chi squared']], 'Tau diffusion': [return_dict['Tau diffusion']],
                         'CPP peak': [return_dict['CPP peak']],
                         'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']],
                         'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']],
                         'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames

    elif fitting_model == 'g2diffOffset':
        return_dict = g2diffOffset_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio,
                                 initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'PSF radius': [return_dict['PSF radius']],
                         'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'D': [return_dict['D']], 'dD': [return_dict['dD']],'N': [return_dict['N']], 'dN': [return_dict['sigma N']],
                         'Chi squared': [return_dict['Chi squared']], 'Tau diffusion': [return_dict['Tau diffusion']], 'dTau_D': [return_dict['dTauD']],
                         'CPP peak': [return_dict['CPP peak']],
                         'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames
    
    elif fitting_model == 'g2diffBlink':
        return_dict = g2diffBlink_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio,
                                      initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'PSF radius': [return_dict['PSF radius']],
                         'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'D': [return_dict['D']], 'dD': [return_dict['dD']], 'N': [return_dict['N']],
                         'Chi squared': [return_dict['Chi squared']], 'Tau diffusion': [return_dict['Tau diffusion']],
                         'Tau Blink': [return_dict['Tau Blink']], 'F Blink': [return_dict['F Blink']],
                         'CPP peak': [return_dict['CPP peak']],
                         'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames
    
    elif fitting_model == 'g3lorentzianZ':
        return_dict = g3lorentzianZ_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio,
                                        initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'PSF radius': [return_dict['PSF radius']],
                         'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'D': [return_dict['D']], 'dD': [return_dict['dD']],'N': [return_dict['N']],
                         'Chi squared': [return_dict['Chi squared']], 'Tau diffusion': [return_dict['Tau diffusion']],
                         'CPP peak': [return_dict['CPP peak']],
                         'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames
    
    elif fitting_model == 'g3lorentzianZCal':
        return_dict = g3lorentzianZCal_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio,
                                           initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'PSF radius': [return_dict['PSF radius']],
                         'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'D': [return_dict['D']], 'N': [return_dict['N']],
                         'Chi squared': [return_dict['Chi squared']], 'Tau diffusion': [return_dict['Tau diffusion']],
                         'CPP peak': [return_dict['CPP peak']],
                         'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames
    
    elif fitting_model == 'g3anomalousDiff':
        return_dict = g3anomalousDiff_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio,
                                          initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'PSF radius': [return_dict['PSF radius']],
                         'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'Gamma': [return_dict['Gamma']],
                         'Alpha': [return_dict['Alpha']], 'Tau diffusion': [return_dict['Tau diffusion']], 'D_app': [return_dict['D']],'N': [return_dict['N']],
                         'Chi squared': [return_dict['Chi squared']], 'CPP peak': [return_dict['CPP peak']],
                         'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames
    
    elif fitting_model == 'g3anomalousDiffBlink':
        return_dict = g3anomalousDiffBlink_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio,
                                      initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'PSF radius': [return_dict['PSF radius']],
                         'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'Gamma': [return_dict['Gamma']], 
                         'Alpha': [return_dict['Alpha']], 'D_app': [return_dict['D']], 'N': [return_dict['N']],
                         'Chi squared': [return_dict['Chi squared']], 'Tau diffusion': [return_dict['Tau diffusion']],
                         'Tau Blink': [return_dict['Tau Blink']], 'F Blink': [return_dict['F Blink']],
                         'CPP peak': [return_dict['CPP peak']],
                         'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames

    elif fitting_model == 'g3diffTwoComponents':
    
        return_dict = g3diffTwoComponents_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius,
                                              PSF_aspect_ratio, initial_params, goodness_of_fit_criterion)
    
        # saving the fitted data to csv file
        estimate_data = {'PSF radius': [return_dict['PSF radius']], 'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'D1': [return_dict['D1']],'dD1': [return_dict['dD1']],
                         'D2': [return_dict['D2']], 'dD2': [return_dict['dD2']],'N': [return_dict['N']],'Chi squared': [return_dict['Chi squared']],
                         'Tau diffusion 1': [return_dict['Tau diffusion 1']], 'f1': [return_dict['f1']],
                         'Tau diffusion 2': [return_dict['Tau diffusion 2']], 'f2': [1 - return_dict['f1']],
                         'CPP peak': [return_dict['CPP peak']], 'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames
    elif fitting_model == 'g2diffTwoComponents':
    
        return_dict = g2diffTwoComponents_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius,
                                              PSF_aspect_ratio, initial_params, goodness_of_fit_criterion)
    
        # saving the fitted data to csv file
        estimate_data = {'PSF radius': [return_dict['PSF radius']], 'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'D1': [return_dict['D1']],'dD1': [return_dict['dD1']],
                         'D2': [return_dict['D2']], 'dD2': [return_dict['dD2']],'N': [return_dict['N']],'Chi squared': [return_dict['Chi squared']],
                         'Tau diffusion 1': [return_dict['Tau diffusion 1']], 'f1': [return_dict['f1']],
                         'Tau diffusion 2': [return_dict['Tau diffusion 2']], 'f2': [1 - return_dict['f1']],
                         'CPP peak': [return_dict['CPP peak']], 'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames
    
    elif fitting_model == 'g3diffTwoComponentsBlink':
        return_dict = g3diffTwoComponentsBlink_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio,
                                           initial_params, goodness_of_fit_criterion)
    
        estimate_data = {'PSF radius': [return_dict['PSF radius']],
                         'PSF aspect ratio': [return_dict['PSF aspect ratio']],
                         'Count Rate': [return_dict['Count Rate']], 'D1': [return_dict['D1']], 'dD1': [return_dict['dD1']],'D2': [return_dict['D2']], 'dD2': [return_dict['dD2']],'N': [return_dict['N']],
                         'Chi squared': [return_dict['Chi squared']], 'Tau diffusion 1': [return_dict['Tau diffusion 1']], 'f1': [return_dict['f1']],
                         'Tau diffusion 2': [return_dict['Tau diffusion 2']], 'f2': [return_dict['f2']],
                         'Tau Blink': [return_dict['Tau Blink']], 'F Blink': [return_dict['F Blink']],
                         'CPP peak': [return_dict['CPP peak']],
                         'CPP average': [return_dict['CPP average']], 'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}
    
    elif fitting_model == 'siFCS':
        return_dict = siFCS_fit(tau, G, sigma_G, count_rate, initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'Count Rate': [return_dict['Count Rate']], 'G0': [return_dict['G0']],
                         'Chi squared': [return_dict['Chi squared']], 'Tau characteristic decay': [return_dict['Tau characteristic decay']],
                         'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames
    
    elif fitting_model == 'siFCSTwoComponents':
        return_dict = siFCSTwoComponents_fit(tau, G, sigma_G, count_rate, initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'Count Rate': [return_dict['Count Rate']], 'G0_1': [return_dict['G0_1']],'G0_2': [return_dict['G0_2']],
                         'Chi squared': [return_dict['Chi squared']], 'Tau characteristic decay short': [return_dict['Tau characteristic decay short']],
                         'Tau characteristic decay long': [return_dict['Tau characteristic decay long']],'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                         'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames
    
    elif fitting_model == 'g3diffMEMFCS':
        return_dict = g3diffMEMFCS_fit(tau, G, sigma_G, count_rate, corrected_D, BG, PSF_radius, PSF_aspect_ratio, initial_params, goodness_of_fit_criterion)
        # saving the fitted data to csv file
        estimate_data = {'Count Rate': [return_dict['Count Rate']],'Chi squared': [return_dict['Chi squared']],'Mean Tau D': [return_dict['mean tau diffusion']],'D': [return_dict['D']],'BIC': [return_dict['BIC']], 'p_ttest': [return_dict['p_ttest']],
                          'p_wilcoxon': [return_dict['p_wilcoxon']], 'p_runstest': [return_dict['p_runstest']], 'p_runstest_residuals': [return_dict['p_runstest_residuals']]}  # dictionary for DataFrames
    
    else:
        print('Fitting model does not match with available options')

   
    
   

     # ---- compute-only return for GUI ----
    # NOTE: `path` is the base path WITHOUT ".csv" (your code reads path + ".csv")
    out = {
        "base_path": str(path),
        "fitting_model": str(fitting_model),

        "tau": tau,
        "G": G,
        "sigma_G": sigma_G,

        "ccPrediction": return_dict.get("ccPrediction"),
        "weighted_r": return_dict.get("weighted_r"),

        # used for iMSD calculation in GUI (if applicable)
        "N": return_dict.get("N"),
        "PSF_aspect_ratio": return_dict.get("PSF aspect ratio"),
        "offset": return_dict.get("offset", 0.0),

        # keep both for flexibility in GUI exports
        "estimate_data": estimate_data,   # dict of lists
        "return_dict": return_dict,       # full fit output dict
    }
    return out
   
################################################################################################################################################################################
#%%