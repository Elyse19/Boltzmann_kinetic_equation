#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 16:01:19 2025

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

def W(eps_loc,e1_loc,e2_loc):
    '''
    Parameters
    ----------
    eps_loc : float (energy)
    e1_loc : 2D array (local integral variable)
    e2_loc : 2D array (local integral variable)

    Returns
    -------
    2D array
        Interaction kernel for a 3D interacting Bose gas

    '''
    eps3_loc = e1_loc + e2_loc - eps_loc
    
    ind = (eps3_loc >= 0)
    e3_loc = np.where(ind,eps3_loc,0)
    #Insures that e3 only takes positive values
    
    b = np.minimum(np.sqrt(eps_loc),np.sqrt(e1_loc))
    c = np.minimum(np.sqrt(e2_loc),np.sqrt(e3_loc))
    L = np.minimum(b,c)
    
    return L/np.sqrt(eps_loc)

def occupation(n_func_loc, eps_loc, e1_loc, e2_loc,e_min_loc,e_max_loc):
    '''
    Parameters
    ----------
    z : function
        corresponds to interpolated function n(\epsilon,t)
    eps_loc : float 
    e1_loc : 2D array
    e2_loc : 2D array

    Returns
    -------
    2D array
        Occupation number term in the integrand

    '''
    e3_loc = e1_loc + e2_loc - eps_loc
    
    n_3 = np.where((e3_loc <= e_max_loc)*(e3_loc >= e_min_loc),n_func_loc(e3_loc),10**-200)
    #n(e3) is set to 0 outside of interval [eps_min,eps_max]
    
    # return n_func_loc(e1_loc)*n_func_loc(e2_loc)*(n_func_loc(eps_loc) + n_3) - n_func_loc(eps_loc)*n_3*(n_func_loc(e1_loc) + n_func_loc(e2_loc))
    ##Classical version n_eps>>1
    
    return n_func_loc(e1_loc)*n_func_loc(e2_loc)*(n_func_loc(eps_loc) + 1)*(1 + n_3) - n_func_loc(eps_loc)*n_3*(n_func_loc(e1_loc) + 1)*(1 + n_func_loc(e2_loc))
    ##Quantum version with the 1s



def integral_coll(n_func_loc,eps_loc,eps_tab,e1_tab,e2_tab):
    '''
    Parameters
    ----------
    z : function
        corresponds to interpolated function n(\epsilon,t)
    ie : int
        index of epsilon_i

    Returns
    -------
    integral : float
        Corresponds to dn_{epsilon_i}/dt (collision integral for a single energy value)

    '''
    
    I = occupation(n_func_loc,eps_loc,e1_tab,e2_tab,eps_tab[0],eps_tab[-1])*W(eps_loc,e1_tab,e2_tab)
    #integrand (2D)

    integral = integrate.simpson(integrate.simpson(I,eps_tab,axis=0),eps_tab,axis=0)
    #Double integral on e1 and e2

    return integral