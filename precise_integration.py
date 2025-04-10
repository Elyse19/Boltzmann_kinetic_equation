#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 16:08:18 2025

@author: egliott
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import fft
from joblib import Parallel, delayed
from scipy.special import iv, gamma, factorial
import os.path
from os import path
from scipy.special import roots_legendre
from scipy.special import roots_sh_legendre

##Weights and coordinates for sh th gaussian quadrature

order = 4 #Precision order

N_large = 10*(2**order)
k_list = np.arange(-N_large, N_large + 1, 1)
#Infinite sum is reduced to the interval [-N_large,N_large]

h_sth = 2**(-order) #Parameter h_sth that depends on the precision order

w_list_sth = []
x_list_sth = []

for kk in k_list:
    
    w_sth = 0.5*np.pi*h_sth*np.cosh(kk*h_sth)/(np.cosh((np.pi/2.0)*np.sinh(kk*h_sth))**2) #Generates warning overflow for now
    x_sth = np.tanh((np.pi/2.0)*np.sinh(kk*h_sth)) #Generates warning overflow for now
    
    if np.abs(w_sth) >= 10**-15 and np.abs(x_sth) < 1.0:
        w_list_sth.append(w_sth)
        x_list_sth.append(x_sth)
    #Take only non negligeable values and non duplicates 
    
w_arr_sth = np.array(w_list_sth) #List of weights (sh/th gaussian quadrature)
x_arr_sth = np.array(x_list_sth) #List of coordinates (sh/th gaussian quadrature)


N_integr = len(x_arr_sth) #Number of points used to calculate the integrals

ind_mesh = np.arange(0,N_integr,1)
i_1,i_2 = np.meshgrid(ind_mesh,ind_mesh, indexing = 'ij')
#2D meshgrid of indices (size N_integr*N_integr)

x_list, w_list = x_arr_sth, w_arr_sth

xi, xj = x_list[i_1], x_list[i_2]
wi, wj = w_list[i_1], w_list[i_2]
#2D meshgrids of the coordinates and weights (for integrals I2 and I3)


def occupation(n_func_loc, eps_loc, e1_loc, e2_loc,eps_min_loc,eps_max_loc):
    '''
    Occupation number term in integrand for a given

    Parameters
    ----------
    n_func_loc (func): previous n_\epsilon before propagation in time
    eps_loc (float): external energy value \epsilon in n_\epsilon(t_loc)
    e1_loc (2D array): integration variable
    e2_loc (2D array): integration variable
    eps_min_loc (float): eps_min
    eps_max_loc (float): eps_max

    Returns
    -------
    sum of occupation numbers (2D array)

    '''
    e3_loc = e1_loc + e2_loc - eps_loc
    
    n_3 = np.where((e3_loc <= eps_max_loc)*(e3_loc >= eps_min_loc),n_func_loc(e3_loc),0)
    #Neglect n(e3) outside of interval [eps_min,eps_max]
    
    ##Classical version
    # return n_func_loc(e1_loc)*n_func_loc(e2_loc)*(n_func_loc(eps_loc) +  n_3) - n_func_loc(eps_loc)*n_3*(n_func_loc(e1_loc) + n_func_loc(e2_loc))
    ##Quantum version
    return n_func_loc(e1_loc)*n_func_loc(e2_loc)*(n_func_loc(eps_loc) + 1)*(1 + n_3) - n_func_loc(eps_loc)*n_3*(n_func_loc(e1_loc) + 1)*(1 + n_func_loc(e2_loc))

##Integral calculations on different intervals

def integral_I0(n_func_loc,eps_loc, eps_min_loc, eps_max_loc):
    '''
    Double integral for which the boundaries are :
    e1 \in [0,eps_loc], e2 \in [eps_loc - e1,eps_loc]

    Parameters
    ----------
    n_func_loc (func): previous n_\epsilon before propagation in time
    eps_loc (float): external energy value \epsilon in n_\epsilon(t_loc)
    eps_min_loc (float): eps_min
    eps_max_loc (float): eps_max

    Returns
    -------
    Result of the double integral I0 for eps_loc

    '''

    emin_1, emax_1 = eps_loc/2, eps_loc
    #Using diagonal symmetry
 
    x_list_1 = (emax_1 - emin_1)*(x_arr_sth + 1)/2.0 + emin_1
    #Change of variable to [-1,1]

    integral_interm = []
    
    for xx1 in x_list_1:
        
        emin_2, emax_2 = eps_loc - xx1, xx1
        #Edge of triangle condition e2 > e - e1
        x_list_2 = (emax_2-emin_2)*(x_arr_sth + 1)/2.0 + emin_2
        #Change of variable to [-1,1]

        W_I = np.sqrt(xx1 + x_list_2 - eps_loc)/np.sqrt(eps_loc)
        integrande1d = occupation(n_func_loc,eps_loc,xx1,x_list_2, eps_min_loc, eps_max_loc)*W_I
        
        integral_interm.append( ((emax_2 - emin_2)/2.0) *np.sum(w_arr_sth*integrande1d))
        #First integral
    
    integral = ((emax_1 - emin_1)/2.0)* np.sum(w_arr_sth*np.array(integral_interm))
    #Second integral
    return integral*2

def integral_I2(n_func_loc,eps_loc,eps_min_loc,eps_max_loc):
    '''
    Double integral for which the boundaries are :
    e1,e2 \in [eps_loc,eps_max]

    Parameters
    ----------
    n_func_loc (func): previous n_\epsilon before propagation in time
    eps_loc (float): external energy value \epsilon in n_\epsilon(t_loc)
    eps_min_loc (float): eps_min
    eps_max_loc (float): eps_max

    Returns
    -------
    Result of the double integral I2 for eps_loc

    '''
  
    emin, emax = eps_loc,eps_max_loc
    
    xx = (emax - emin)*(xi + 1)/2.0 + emin
    yy = (emax - emin)*(xj+ 1)/2.0 + emin
    #Change of variable to [-1,1]
      
    interande2d = occupation(n_func_loc,eps_loc,xx,yy, eps_min_loc, eps_max_loc)
    integral = (((emax - emin)/2.0)**2) * np.sum(np.sum(wi*wj*interande2d, axis=-1), axis = -1)
    #Double integral on e1 and e2

    return integral


def integral_I3(n_func_loc,eps_loc,eps_min_loc,eps_max_loc):
    '''
    Double integral for which the boundaries are :
    e1 \in [eps_loc,eps_max], e2 \in [eps_min,eps_loc]

    Parameters
    ----------
    n_func_loc (func): previous n_\epsilon before propagation in time
    eps_loc (float): external energy value \epsilon in n_\epsilon(t_loc)
    eps_min_loc (float): eps_min
    eps_max_loc (float): eps_max

    Returns
    -------
    Result of the double integral I3 for eps_loc

    '''
    
    emin_1, emax_1 = eps_loc, eps_max_loc
    emin_2, emax_2 = eps_min_loc, eps_loc
    
    xx = (emax_1 - emin_1)*(xi + 1)/2.0 + emin_1
    yy = (emax_2 - emin_2)*(xj+ 1)/2.0 + emin_2
    #Change of variable to [-1,1]
    
    W_III = np.sqrt(yy)/np.sqrt(eps_loc)
    interande2d = occupation(n_func_loc,eps_loc,xx,yy, eps_min_loc, eps_max_loc)*W_III
    
    integral = ((emax_1 - emin_1)/2.0)*((emax_2 - emin_2)/2.0) * np.sum(np.sum(wi*wj*interande2d, axis=-1), axis = -1)
    #Double integral on e1 and e2

    return integral
    

def integral_coll(n_func_loc,eps_loc,eps_tab,e1,e2):
    '''
    Total result of the double integral of the interacting term on the rhs of the ode

    Parameters
    ----------
    n_func_loc (func): previous n_\epsilon before propagation in time
    eps_loc (float): external energy value \epsilon in n_\epsilon(t_loc)
    eps_min_loc (float): eps_min
    eps_max_loc (float): eps_max
    
    Returns
    -------
    I (float): Double integral result for eps_loc

    '''
    e_max = eps_tab[-1]
    e_min = eps_tab[0]

    I = integral_I0(n_func_loc,eps_loc, e_min, e_max) + integral_I2(n_func_loc,eps_loc,e_min,e_max) + 2*integral_I3(n_func_loc, eps_loc, e_min, e_max)
    
    return I
