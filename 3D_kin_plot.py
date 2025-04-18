#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:27:02 2025

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


fig1, axs1 = plt.subplots(1,1) #n0(t)
fig2, axs2 = plt.subplots(1,1) #n_eps(t)
fig3, axs3 = plt.subplots(1,1) #1/n_eps(t)

A = 1 #Interaction prefactor

N = 256 #Number of points in energy grid 

eps_max = 10**2
eps_min = 10**-4
#Upper and lower energy bounds

# eps0 = 1/np.sqrt(5) #Width of the initial gaussian 
eps0 = 6

h = 0.1 #dt time step
T = 10 #Final time for integration
epsilon = np.logspace(np.log10(eps_min),np.log10(eps_max),N) #1D energy grid in logspace

def f_affine(e_loc,T,mu):
    return (e_loc - mu)/T

cmap = plt.cm.Reds_r(np.linspace(0, 1, 15))
cmap = cmap[:-1]

method_int = 'precise'
t_save_neps = 2 #Save n_eps every t_save_neps
save_neps = int(1/h)*t_save_neps #index corresponding to t_save_neps

path = '/users/jussieu/egliott/Documents/git_boltz/Boltzmann_kinetic_equation/'
path_im = '/users/jussieu/egliott/Documents/Schema_these/' 

name = '3D_Boltz_normal_'
file_name = f"dt={h}_N={N}_Tmax={T}_emax={eps_max}_emin={eps_min}_A={A}_epsc={eps0}_tsave={t_save_neps}_"
path_save = path + name + file_name + method_int + ".npz"

files = np.load(path_save,allow_pickle='True')
t_data, n_eps, n_0, E_diff, Cons_N = files['arr_0'], files['arr_1'], files['arr_2'], files['arr_3'], files['arr_4']  

t_data = np.array(t_data)
n_eps = np.array(n_eps)
n_0 = np.array(n_0)
E_diff = np.array(E_diff)
Cons_N = np.array(Cons_N)

##Central density n_0(t)
axs1.plot(t_data[:-1],n_0[:-1],color = 'r')

##n_eps(t)
k = 0
axs2.plot(epsilon,n_eps[0],label = 't = ' + str(round(t_data[0],2)), color = 'k')

for j in range(0,len(n_eps)):
    if j%1 == 0:
        axs2.plot(epsilon,n_eps[j],label = 't = ' + str(round(j*t_save_neps,2)), color =cmap[k])
        k += 1
        
##Fit of n_eps at long times with BE distrib (valid only for quench Normal -> Normal)

inv_log = np.log(1 + 1/n_eps[-1])
popt,pcov = curve_fit(f_affine,epsilon,inv_log)
        
axs3.plot(epsilon,np.log(1 + 1/n_eps[-1]), color = 'r', label = 'Numerical solution')
axs3.plot(epsilon,f_affine(epsilon,popt[0],popt[1]), color = 'k', ls = '--', label = r'Fit with $(\epsilon - \mu)/T$' + ' \n' + rf'for $T$ = {popt[0]:.2f}, $\mu$ = {popt[1]:.2f}')


axs1.set_xscale('log')
axs1.set_yscale('log')
axs1.set_xlabel('t', fontsize = 'large')
axs1.set_ylabel(r'$n_0(t)$', fontsize = 'large')
# plt.savefig(path_im + 'n_0_BE_distrib.pdf')
        
axs2.legend(loc = 'lower left')        
axs2.set_xscale('log')
axs2.set_yscale('log')
axs2.set_xlabel(r'$\epsilon$', fontsize = 'large')
axs2.set_ylabel(r'$n_\epsilon(t)$', fontsize = 'large')
axs2.set_xlim(10**-4,25)
axs2.set_ylim(10**-3,0.5)
# plt.show()
# plt.savefig(path_im + 'n_eps_BE_distribution_diff_times.pdf')

axs3.legend()
axs3.set_xlabel(r'$\epsilon$', fontsize = 'large')
axs3.set_ylabel(r'$\log(1 + 1/n_\epsilon(t=25))$', fontsize = 'large')
# plt.savefig(path_im + 'BE_distrib_fit.pdf')
