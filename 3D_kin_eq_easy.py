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
from scipy.special import roots_legendre
from scipy.special import roots_sh_legendre
plt.rcParams ['figure.dpi'] = 300


import time

fig1, axs1 = plt.subplots(1,1)
fig2, axs2 = plt.subplots(1,1)

A = 1
#Interaction prefactor

N = 200
#Number of points in energy grid 

eps_max = 10**4
eps_min = 10**-12
#Upper and lower energy bounds

eps0 = 1/np.sqrt(5)
#Width of the initial gaussian (can be varied)

pref = 2/((eps0**(3/2))*scipy.special.gamma(3/4))
#Normalization prefactor

def n0(eps_loc):
    return pref*np.exp(-(eps_loc**2/eps0**2))
#Initial condition

nc = 8
#Number of cores used in parallelization
#Should not exceed number of cores present on computer

 
def W(eps_loc,e1_loc,e2_loc):
    eps3_loc = e1_loc + e2_loc - eps_loc
    
    ind = (eps3_loc >= 0)
    e3_loc = np.where(ind,eps3_loc,0)
    #Insures that e3 only takes positive values
    
    b = np.minimum(np.sqrt(eps_loc),np.sqrt(e1_loc))
    c = np.minimum(np.sqrt(e2_loc),np.sqrt(e3_loc))
    L = np.minimum(b,c)
    
    return L/np.sqrt(eps_loc)
#Interaction kernel for a 3D interacting Bose gas (2D array)


def occupation(z, eps_loc, e1_loc, e2_loc):
    e3_loc = e1_loc + e2_loc - eps_loc
    
    n_3 = np.where((e3_loc <= eps_max)*(e3_loc >= eps_min),z(e3_loc),10**-200)
    #n(e3) is set to 0 outside of interval [eps_min,eps_max]
    
    # return z(e1_loc)*z(e2_loc)*(z(eps_loc) + n_3) - z(eps_loc)*n_3*(z(e1_loc) + z(e2_loc))
    ##Classical version n_eps>>1
    
    return z(e1_loc)*z(e2_loc)*(z(eps_loc) + 1)*(1 + n_3) - z(eps_loc)*n_3*(z(e1_loc) + 1)*(1 + z(e2_loc))
    ##Quantum version with the 1s
#Occupation number term (2D array)

epsilon = np.logspace(np.log10(eps_min),np.log10(eps_max),N)
#1D energy grid in logspace
ind_epsilon = np.arange(0,len(epsilon),1)
#Indices of 1D energy grid

ie1,ie2 = np.meshgrid(ind_epsilon,ind_epsilon)
#2D meshgrid of indices corresponding to epsilon_1 and epsilon_2 
e1,e2 = epsilon[ie1],epsilon[ie2]
#Meshgrids corresponding to the energies epsilon_1 and epsilon_2

   

def integral_coll(z,ie):
    
    I = occupation(z,epsilon[ie],e1,e2)*W(epsilon[ie],e1,e2)
    #integrand (2D)

    integral = integrate.simpson(integrate.simpson(I,epsilon,axis=0),epsilon,axis=0)
    #Double integral on e1 and e2

    return integral
#This term corresponds to dn_{epsilon_i}/dt (collision integral) -> returns a single value

 
def tot(n_loc):
    
    n_func = interpolate.interp1d(epsilon,n_loc,kind='quadratic',fill_value='extrapolate')
    #Standard quadratic interpolation of n_eps at time t -> gives function
    
    
    y_loc = np.array(Parallel(n_jobs=nc)(delayed (integral_coll)(n_func,i) for i in range(len(epsilon))))
    ##Parallelization of epsilon loop with joblib

    # y_loc = []
    
    # for i in range(len(epsilon)):
    #     y_loc.append(integration(n_func,i))
    # y_loc = np.array(y_loc)
    
    ##Version without parallelization
        
    return y_loc*A
#Gives n_eps 1D array at a given time

Norm0 = integrate.simpson(n0(epsilon)*np.sqrt(epsilon),epsilon)
#Number of particles at t=0

t0 = time.time()

h = 0.1
#dt time step
#Usually dt=0.01 gives sufficiently accurate results

T = 20
#Final time for integration

# method = 'Euler'
method = 'RK'


path = ''

name = '3D_Boltz_'
file_name = f"dt={h}_N={N}_Tmax={T}_emax={eps_max}_emin={eps_min}_A={A}_epsc={eps0}_tsave=every_"
path_save = path + name + file_name + method + ".npz"

n_0 = []
tot_E = []
Cons_N = []
n_eps = []
t_data = []

t_i = 0
i = 0

y = n0(epsilon)
Norm = Norm0

while t_i <= T:

    t_data.append(t_i)
    ##list of times
    
    if i%10 == 0 and i>0:
        print('t_i = ' + str(round(t_i,2)))
        print('Norm conservation = ' + str(Norm))
    
    
    ##Euler method
    
    # y = y + h*tot1(np.copy(y))
    
    ##Runge Kutta order 4 method 

    k1 = tot(np.copy(y))
    k2 = tot(np.copy(y) + k1*h/2)
    k3 = tot(np.copy(y) + k2*h/2)
    k4 = tot(np.copy(y) + k3*h)
    
    y = y + (k1/6 + k2/3 + k3/3 + k4/6)*h
    
        
    t_i += h
    i += 1
    
    Norm = np.abs(integrate.simpson(np.copy(y)*np.sqrt(epsilon),epsilon)-Norm0)/Norm0
    #Deviation from initial number of particles
        
    n_eps.append(y) 
    Cons_N.append(Norm)     
    tot_E.append(integrate.simpson(np.copy(y)*np.sqrt(epsilon)*epsilon,epsilon))
    n_0.append(np.copy(y)[0])
    #Values we want to save (n_eps(t),norm deviation, total energy, central density n_0(t))

    np.savez_compressed(path_save,t_data,n_eps,n_0,tot_E,Cons_N)
    #Save a npz file (compressed) under path_save with these values
    #To use, simply use np.load(path_save) and extract files
    
    if Norm > 0.1 :
        n_eps.append(y)    
        print('Div')
        print('t_div = '+ str(round(t_i,2)))
        break
    #Break loop when norm is not conserved
        

t1 = time.time()
print('Time = ' + str(t1-t0))

##Examples of plots

##Central density n_0(t)
axs1.plot(t_data[:-1],n_0[:-1])

axs1.set_xscale('log')
axs1.set_yscale('log')
axs1.set_xlabel('t')
axs1.set_ylabel(r'$n_0(t)$')

##n_eps(t)
for j in range(len(n_eps)):
    if j%10 == 0:
        axs2.plot(epsilon,n_eps[j],label = 't = ' + str(round(t_data[j],2)))
        
axs2.legend()        
axs2.set_xscale('log')
axs2.set_yscale('log')
axs2.set_xlabel(r'$\epsilon$')
axs2.set_ylabel(r'$n_\epsilon(t)$')
axs2.set_ylim(10**-5,10**10)
