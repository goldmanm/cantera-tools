# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:40:12 2016

@author: mark

This file contains methods for calculating kinetics values that don't
necessarily require cantera or another outside less known software package. 

all rates are currently only in kcal/mol (except arrhenius)
"""

# -*- coding: utf-8 -*-

import numpy as np


def calculate_arr_rate(a,n,ea,T):
    Rkin = 8.314#J/molK
    return a*T**n*np.exp(-ea/T/Rkin)
       

def calculate_troe_rate_constant(troe_parameters, high_parameters, low_parameters, temperature, molar_density,efficiencies={}, concentrations={}):
    """parameter order is based off of cantera's code for troe formulas"""
    # ignoring efficienies for now    
    corrected_density = calculate_falloff_efficienies(molar_density,efficiencies,concentrations)    
    
    reduced_pressure = calculate_reduced_pressure(high_parameters,low_parameters,corrected_density,temperature)
    
    F_cent  =  (1-troe_parameters[0])*np.exp(-temperature/troe_parameters[1])  + troe_parameters[0]*np.exp(-temperature/troe_parameters[2])  
    # The lack of 4th parameter makes the whole expression last term zero
    if len(troe_parameters) > 3:
        F_cent +=  np.exp(-troe_parameters[3]/temperature)
    log_F_numerator = np.log10(F_cent)    
    C = -0.4 - 0.67*log_F_numerator
    N = 0.75 - 1.27*log_F_numerator
    log_pressure = np.log10(reduced_pressure)
    log_F_denominator = 1.+ ((log_pressure + C)/(N-.14*(log_pressure + C)))**2
    F = 10.**(log_F_numerator/log_F_denominator)
    
    # get lindemann falloff form now
    k_lind = calculate_falloff_lindemann(high_parameters,low_parameters,corrected_density,temperature)
    
    return F*k_lind
    
def calculate_falloff_efficienies(molar_denisty,efficiencies,concentration):
    # currently no efficiency calculation. this is hard-coded.   
    return molar_denisty*1.45
    
def calculate_falloff_lindemann(high_parameters,low_parameters,density_molar,temperature):
    high_rate_constant = calculate_arr_rate(high_parameters[0],high_parameters[1],high_parameters[2],temperature)
    low_rate_constant =  calculate_arr_rate(low_parameters[0],low_parameters[1],low_parameters[2],temperature)
    return high_rate_constant/(1+high_rate_constant/(low_rate_constant*density_molar))
    
def calculate_reduced_pressure(high_parameters,low_parameters,corrected_density, temperature):
    high_rate_constant = calculate_arr_rate(high_parameters[0],high_parameters[1],high_parameters[2],temperature)
    low_rate_constant =  calculate_arr_rate(low_parameters[0],low_parameters[1],low_parameters[2],temperature)
    return low_rate_constant*corrected_density/high_rate_constant
    