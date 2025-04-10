#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:08:29 2023

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
plt.rcParams ['figure.dpi'] = 300
import sys
import warnings

import time

warnings.filterwarnings("ignore")

sys.path.append('/users/jussieu/egliott/Documents/git_boltz/Boltzmann_kinetic_equation/')


A = 1 #Interaction prefactor

N = 256 #Number of points in energy grid 

eps_max = 10**2
eps_min = 10**-4
#Upper and lower energy bounds

# eps0 = 1/np.sqrt(5) #Example for BEC
eps0 = 6  #Example for normal gas
##eps0 is a quench parameter describing the width of initial gaussian post-quench

pref = 2/((eps0**(3/2))*scipy.special.gamma(3/4)) #Normalization prefactor

method_int = 'precise' #Integration method ('naive' or 'precise')
#'naive' uses a standard Simpson integration method with numpy arrays
#'precise' uses sinh tanh integration method on different integration domains

nc = 8 #Number of cores used in parallelization (should not exceed number of cores present on node)

h = 0.01 #dt time step
#Usually dt=0.01 gives sufficiently accurate results (with RK4 method) at long times

T = 10 #Final time for integration

t_save_neps = 2 #Save n_eps every t_save_neps
save_neps = int(1/h)*t_save_neps #index corresponding to t_save_neps

def n0(eps_loc):
    '''
    Parameters
    ----------
    eps_loc : 1D array (energy grid)

    Returns
    -------
    Initial condition n_0(\epsilon): 1D array 

    '''
    return pref*np.exp(-(eps_loc**2/eps0**2))

epsilon = np.logspace(np.log10(eps_min),np.log10(eps_max),N) #1D energy grid in logspace
ind_epsilon = np.arange(0,len(epsilon),1) #Indices of 1D energy grid

ie1,ie2 = np.meshgrid(ind_epsilon,ind_epsilon) #2D meshgrid of indices corresponding to epsilon_1 and epsilon_2 
e1,e2 = epsilon[ie1],epsilon[ie2] #Meshgrids corresponding to the energies epsilon_1 and epsilon_2

if method_int == 'naive':
    from naive_integration import integral_coll
elif method_int == 'precise':
    from precise_integration import integral_coll
else:
    raise ValueError("Invalid method name")

 
def tot(n_loc):
    '''
    Main function solving ODE
    External loop over epsilon is parallelized using joblib

    Parameters
    ----------
    n_loc : 1D array
        Corresponds to n_\epsilon(t-dt)

    Returns
    -------
    1D array
        Corresponds to n_\epsilon(t)

    '''
    
    n_func = interpolate.interp1d(epsilon,n_loc,kind='quadratic',fill_value='extrapolate')
    #Standard quadratic interpolation of n_eps at time t -> gives function
    
    y_loc = np.array(Parallel(n_jobs=nc)(delayed (integral_coll)(n_func,epsilon[i],epsilon,e1,e2) for i in range(len(epsilon))))
    ##Parallelization of epsilon loop with joblib

    # y_loc = []
    # for i in range(len(epsilon)):
    #     y_loc.append(integral_coll(n_func,epsilon[i],epsilon,e1,e2))
    # y_loc = np.array(y_loc)
    
    ##Version without parallelization (for testing purposes)
        
    return y_loc*A

t0 = time.time()

path = '/users/jussieu/egliott/Documents/git_boltz/Boltzmann_kinetic_equation/'

name = '3D_Boltz_normal_'
#name = '3D_Boltz_BEC_'

file_name = f"dt={h}_N={N}_Tmax={T}_emax={eps_max}_emin={eps_min}_A={A}_epsc={eps0}_tsave={t_save_neps}_"
path_save = path + name + file_name + method_int + ".npz"

n_0 = [] #Central density : n_0(t)
tot_E = [] #Total energy E(t) (should be conserved)
Cons_N = [] #Total number of particles N(t) (should be conserved)
n_eps = [] #Full 2D array : n_\epsilon(t) (saved regularly every t_save_neps)
t_data = [] #time

t_i = 0
i = 0

y = n0(epsilon) #Initialize ode
Norm0 = integrate.simpson(n0(epsilon)*np.sqrt(epsilon),epsilon) #Number of particles at t=0
Norm = Norm0

while t_i <= T:

    t_data.append(t_i) #list of times
    
    if i%10 == 0 and i>0:
        print('t_i = ' + str(round(t_i,2)))
        print('Norm conservation = ' + str(Norm))
    
    
    ##Euler method (used for testing purposes)   
    # y = y + h*tot1(np.copy(y))
    
    ##Runge Kutta order 4 method 
    k1 = tot(np.copy(y))
    k2 = tot(np.copy(y) + k1*h/2)
    k3 = tot(np.copy(y) + k2*h/2)
    k4 = tot(np.copy(y) + k3*h)
    
    y = y + (k1/6 + k2/3 + k3/3 + k4/6)*h
    
        
    t_i += h
    i += 1
    
    Norm = np.abs(integrate.simpson(np.copy(y)*np.sqrt(epsilon),epsilon)-Norm0)/Norm0 #Deviation from initial number of particles
    
    if i%save_neps == 0: 
        n_eps.append(np.copy(y)) 
        
    Cons_N.append(Norm)     
    tot_E.append(integrate.simpson(np.copy(y)*np.sqrt(epsilon)*epsilon,epsilon))
    n_0.append(np.copy(y)[0])
    #Values we want to save (n_eps(t),norm deviation, total energy, central density n_0(t))

    np.savez_compressed(path_save,t_data,n_eps,n_0,tot_E,Cons_N)
    #Save a npz file (compressed) under path_save with these values
    #Use np.load(path_save) and extract files
    
    if Norm > 0.1 :
        n_eps.append(y)    
        print('Divergence')
        print('t_div = '+ str(round(t_i,2)))
        break
    #Break loop when norm is not conserved and starts to diverge
    #If this quantity diverges, either there is a numerical issue or the kinetic equation is no longer valid
        

t1 = time.time()
print('Time = ' + str(t1-t0))

