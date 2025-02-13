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
import time

nc = 16 #Number of cores for parallelization

A = 1 #Interaction prefactor
B = 0 #Diffusion prefactor

eps_max = 10**5
eps_min = 10**-12
#Upper and lower energy bounds 
#Should be large in both UV and IR to have precise results in different regimes

N = 2000 #Number of points in external energy array

epsilon = np.logspace(np.log10(eps_min),np.log10(eps_max),N) #Energy array used for n_eps (external energy)

eps_c = 1/np.sqrt(5) #Width of the initial gaussian (can be varied)
pref_IC = 2/((eps_c**(3/2))*scipy.special.gamma(3/4)) #Normalization prefactor (such that N = 1)

def n0(e):
    '''
    Initial condition

    Parameters
    ----------
    e : float or array (energy)

    Returns
    -------
    float or array
        Initial gaussian post-quench

    '''
    return pref_IC*np.exp(-(e**2/eps_c**2))


###Interaction term

##Weights and coordinates for sh th gaussian quadrature

order = 4 #Precision order

N_large = 10*(2**order)
k_list = np.arange(-N_large, N_large + 1, 1)
#Infinite sum is reduced to the interval [-N_large,N_large]

h_sth = 2**(-order) #Parameter h_sth that depends on the precision order

w_list_sth = []
x_list_sth = []

for kk in k_list:
    
    w_sth = 0.5*np.pi*h_sth*np.cosh(kk*h_sth)/(np.cosh((np.pi/2.0)*np.sinh(kk*h_sth))**2)
    x_sth = np.tanh((np.pi/2.0)*np.sinh(kk*h_sth))
    
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


def occupation(n_func_loc, eps_loc, e1_loc, e2_loc):
    '''
    Occupation number term in integrand for a given

    Parameters
    ----------
    n_func_loc (func): previous n_\epsilon before propagation in time
    eps_loc (float): external energy value \epsilon in n_\epsilon(t_loc)
    e1_loc (2D array): integration variable
    e2_loc (2D array): integration variable

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    e3_loc = e1_loc + e2_loc - eps_loc
    
    n_3 = np.where((e3_loc <= eps_max)*(e3_loc >= eps_min),n_func_loc(e3_loc),0)
    #Neglect n(e3) outside of interval [eps_min,eps_max]
    
    ##Classical version
    # return n_func_loc(e1_loc)*n_func_loc(e2_loc)*(n_func_loc(eps_loc) +  n_3) - n_func_loc(eps_loc)*n_3*(n_func_loc(e1_loc) + n_func_loc(e2_loc))
    ##Quantum version
    return n_func_loc(e1_loc)*n_func_loc(e2_loc)*(n_func_loc(eps_loc) + 1)*(1 + n_3) - n_func_loc(eps_loc)*n_3*(n_func_loc(e1_loc) + 1)*(1 + n_func_loc(e2_loc))

##Integral calculations on different intervals

def integral_I0(n_func_loc,eps_loc):
    '''
    Double integral for which the boundaries are :
    e1 \in [0,eps_loc], e2 \in [eps_loc - e1,eps_loc]

    Parameters
    ----------
    n_func_loc (func): previous n_\epsilon before propagation in time
    eps_loc (float): external energy value \epsilon in n_\epsilon(t_loc)

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
        integrande1d = occupation(n_func_loc,eps_loc,xx1,x_list_2)*W_I
        
        integral_interm.append( ((emax_2 - emin_2)/2.0) *np.sum(w_arr_sth*integrande1d))
        #First integral
    
    integral = ((emax_1 - emin_1)/2.0)* np.sum(w_arr_sth*np.array(integral_interm))
    #Second integral
    return integral*2

def integral_I2(n_func_loc,eps_loc):
    '''
    Double integral for which the boundaries are :
    e1,e2 \in [eps_loc,eps_max]

    Parameters
    ----------
    n_func_loc (func): previous n_\epsilon before propagation in time
    eps_loc (float): external energy value \epsilon in n_\epsilon(t_loc)

    Returns
    -------
    Result of the double integral I2 for eps_loc

    '''
  
    emin, emax = eps_loc,eps_max
    
    xx = (emax - emin)*(xi + 1)/2.0 + emin
    yy = (emax - emin)*(xj+ 1)/2.0 + emin
    #Change of variable to [-1,1]
      
    interande2d = occupation(n_func_loc,eps_loc,xx,yy)
    integral = (((emax - emin)/2.0)**2) * np.sum(np.sum(wi*wj*interande2d, axis=-1), axis = -1)
    #Double integral on e1 and e2

    return integral


def integral_I3(n_func_loc,eps_loc):
    '''
    Double integral for which the boundaries are :
    e1 \in [eps_loc,eps_max], e2 \in [eps_min,eps_loc]

    Parameters
    ----------
    n_func_loc (func): previous n_\epsilon before propagation in time
    eps_loc (float): external energy value \epsilon in n_\epsilon(t_loc)

    Returns
    -------
    Result of the double integral I3 for eps_loc

    '''
    
    emin_1, emax_1 = eps_loc, eps_max
    emin_2, emax_2 = eps_min, eps_loc
    
    xx = (emax_1 - emin_1)*(xi + 1)/2.0 + emin_1
    yy = (emax_2 - emin_2)*(xj+ 1)/2.0 + emin_2
    #Change of variable to [-1,1]
    
    W_III = np.sqrt(yy)/np.sqrt(eps_loc)
    interande2d = occupation(n_func_loc,eps_loc,xx,yy)*W_III
    
    integral = ((emax_1 - emin_1)/2.0)*((emax_2 - emin_2)/2.0) * np.sum(np.sum(wi*wj*interande2d, axis=-1), axis = -1)
    #Double integral on e1 and e2

    return integral
    

def integral_coll(n_func_loc,eps_loc):
    '''
    Total result of the double integral of the interacting term on the rhs of the ode

    Parameters
    ----------
    n_func_loc (func): previous n_\epsilon before propagation in time
    eps_loc (float): external energy value \epsilon in n_\epsilon(t_loc)
    
    Returns
    -------
    I (float): Double integral result for eps_loc

    '''

    I = integral_I0(n_func_loc,eps_loc) + integral_I2(n_func_loc,eps_loc) + 2*integral_I3(n_func_loc, eps_loc)
    
    return I


def tot1(n_loc):
    '''
    Rhs of the ode (interaction term)

    Parameters
    ----------
    n_loc (1D array) : Previous n_\epsilon before propagation in time

    Returns
    -------
    1D array : n_\epsilon(t) after propagation

    '''
    
    n_loco = np.copy(n_loc)
    n_loco[n_loco < 10**-200] = 10**-200
    #Avoid error accumulation over time by fixing small values
    
    n_func = interpolate.PchipInterpolator(epsilon,n_loco,extrapolate = True)
    #Monotonic interpolator (to avoid error accumulation at high energies - rapidly decreasing function)
    #Extrapolation useful to access n(e3)
    
    y_loc = np.array(Parallel(n_jobs=nc)(delayed (integral_coll)(n_func,epsilon[i]) for i in range(len(epsilon))))
    #Parallelize over external epsilon loop to obtain the 1D array n_\epsilon(t_loc)
        
    return y_loc*A



###Diffusion term


z = np.logspace(np.log10(10**-30), np.log10(float(10**30)), 2**17)


def f_i(z_loc):
    '''
    Modified Bessel function I(z)*exp(-z) for z<30 
    and series expansion of order 20 for z>30

    Parameters
    ----------
    z_loc (1D array) 

    Returns
    -------
    1D array

    '''
    if z_loc > 30:
        m = np.arange(0, 20)
        pref_nu_m = (-1/(2*z_loc))**m * gamma(-1/4 + m + 1/2) / (factorial(m) * gamma(-1/4 - m + 1/2) )
        return np.sum(pref_nu_m) / (2*np.pi*z_loc)**0.5
    else:
        return iv(-1/4, z_loc) * np.exp(-z_loc)

res = np.array([f_i(z_loc) for z_loc in z])
Func_res = interpolate.interp1d(np.log(z), np.log(res), kind='cubic')

def fi_R_plus(z_loc):
    return np.exp(Func_res(np.log(z_loc)))


def integrande_convol(ep_loc, eps_target_loc, t_loc, n_fonc_loc):
    '''
    Integrand of integral_convol

    Parameters
    ----------
    ep_loc (1D array) : integration variable
    eps_target_loc (float) : external energy value \epsilon in n_\epsilon(t_loc)
    t_loc (float) : dimensionless time
    n_fonc_loc (func) : previous n_\epsilon before propagation

    Returns
    -------
    Integrand (1D array)

    '''    
    pref = (eps_target_loc**(1/4))/(2*B*t_loc)
    integr = (ep_loc**(3/4))*n_fonc_loc(ep_loc)*np.exp(-(eps_target_loc - ep_loc)**2/(4*B*t_loc))*fi_R_plus(eps_target_loc*ep_loc/(2*B*t_loc))
    
    return integr*pref

def integral_convol(eps_target_loc, t_loc, n_fonc_loc):
    '''
    Analytical solution of the diffusion term used to propagate the solution

    Parameters
    ----------
    eps_target_loc (float): external energy value \epsilon in n_\epsilon(t_loc)
    t_loc (float): dimensionless time
    n_fonc_loc (func): previous n_\epsilon before propagation

    Returns
    -------
    integral (float) : Corresponds to n(eps_target_loc,t_loc)

    '''
    qte = np.sqrt(4*B*t_loc*(-np.log(10**-25)))
    ep_min, ep_max = eps_target_loc - qte , eps_target_loc + qte
    #Integration interval that is close to gaussian peak such that exp(-(eps - eps_prime)^2/(4Bt)) > 10**-25
    
    if ep_min <=  0:
        ep_min = eps_min
    #If min boundary is negative, default to eps_min
    
    x_list, w_list = x_arr_sth, w_arr_sth
    
    emin, emax = ep_min, ep_max
    xx = (emax - emin)*(x_list + 1)/2.0 + emin
    #Change of integration variable from [ep_min,ep_max] to [-1,1]
    
    integrande = integrande_convol(xx,eps_target_loc,t_loc,n_fonc_loc)
    integral = (((emax - emin)/2.0)) * np.sum(w_list*integrande) 
    #Calculation of the integral using the shth method
    
    return integral


def tot2(t_loc, n_loc):
    '''
    Propagation of the solution (diffusion term)

    Parameters
    ----------
    t_loc (float) :  dimensionless time
    n_loc (1D array): solution of the ode n_\epsilon at previous time step

    Returns
    -------
    y_loc (1D array) : n_\epsilon after propagation in time with the diffusion term

    '''
    
    n_loco = np.copy(n_loc)
    n_loco[n_loco < 10**-200] = 10**-200
    #Avoid error accumulation over time by fixing small values
    
    n_func = interpolate.PchipInterpolator(epsilon,n_loco,extrapolate = True)
    #Monotonic interpolator (to avoid error accumulation at high energies - rapidly decreasing function)
    
    y_loc = np.array(Parallel(n_jobs=nc)(delayed (integral_convol)(epsilon[i],t_loc,n_func) for i in range(len(epsilon))))
    #Parallelize over external epsilon loop to obtain the 1D array n_\epsilon(t_loc)
        
    return y_loc



###Split-step method

Norm0 = integrate.simpson(n0(epsilon)*np.sqrt(epsilon),epsilon)

t0 = time.time()

dico_step = {"begin":0.01,"interm":0.05,"long":0.1} #Variable time step
# h = dico_step["begin"]
h = 0.001 #dt time step

T = 100 #Final time for integration

# method = 'Euler'
method = 'RK'

time_save = 1.00

print('B = ' + str(B))

# path = ''
path = '/users/jussieu/egliott/Documents/Eq_cinetique/Split_step/short_time_data/'

name = 'tps_courts_'
file_name = f"dt={h}_N={N}_Nint={N_integr}_Tmax={T}_emax={eps_max}_emin={eps_min}_A={A}_B={B}_epsc={eps_c}_tsave=every_"
path_save = path + name + file_name + method + ".npz"

n_0 = [] #Central density
E_diff = [] #Total energy
Cons_N = [] #Particle conservation
n_eps = [] #n_\epsilon(t)
t_data = [] #Time values 

t_i = 0
i = 0

    
y = n0(epsilon)


while t_i <= T:

    ##Euler (not sufficient for precise results)
    # y = tot2(h/4, np.copy(y)) #Propagate using the diffusion term for dt/2
    # y = y + (h/2)*tot1(np.copy(y)) #Propagate using the interaction term for dt
    # y = tot2(h/4, np.copy(y)) #Propagate using the diffusion term for dt/2
    
    ##Runge Kutta
    
    t_data.append(t_i)

    y = tot2(h/2, np.copy(y)) #Propagate using the diffusion term for dt/2
    
    k1 = tot1(np.copy(y))
    k2 = tot1(np.copy(y) + k1*h/2)
    k3 = tot1(np.copy(y) + k2*h/2)
    k4 = tot1(np.copy(y) + k3*h)
    #Propagate using RK34 with the interaction term for dt
    
    y = y + (k1/6 + k2/3 + k3/3 + k4/6)*(h)
    
    y = tot2(h/2, np.copy(y)) #Propagate using the diffusion term for dt/2

    
    ## A = 0 case
    # y = tot2(t_i + h, n0(epsilon))
    
    ## B = 0 case (Euler)
    # y1 = y1 + h*tot1( np.copy(y1))

    ##Variable time step
    # if t_i >= (5 - h) :
    #     h = dico_step['interm']
    # if t_i >= (10 - h) :
    #     h = dico_step['long']
        
    t_i += h
    i += 1
    
    
    Norm = np.abs(integrate.simpson(np.copy(y)*np.sqrt(epsilon),epsilon)-Norm0)/Norm0
    #Deviation from initial number of particles
    
    # if t_i < (5 - h) and i%100 == 0 :
    #     n_eps.append(y)    
    #     print(Norm)
    #     print(t_i)
    # if t_i < (10 - h) and t_i >= (5-h) and i%20 == 0 :
    #     n_eps.append(y)    
    #     print(Norm)
    #     print(t_i)
    # if t_i >= (10 - h) and i%10 == 0 :
    #     n_eps.append(y)    
    #     print(Norm)
    #     print(t_i)
        
    # if i%10 == 0 :
    #      n_eps.append(y)    
    #      print(Norm)
    #      print(t_i)
        
    n_eps.append(y) 
    Cons_N.append(Norm)     
    E_diff.append(integrate.simpson(np.copy(y)*np.sqrt(epsilon)*epsilon,epsilon))
    n_0.append(np.copy(y)[0])
    #Values we want to study (n_eps(t), Norm deviation, energy, n_0(t))

    np.savez_compressed(path_save,t_data,n_eps,n_0,E_diff,Cons_N)
    #Save a compressed npz file under path_save with these values
    
    if Norm > 0.1 :
        n_eps.append(y)    
        print(Norm)
        print(t_i)
        print('Norm conservation failed')
        break
    #Break loop when norm is not conserved
        

t1 = time.time()
print('Time = ' + str(t1-t0))


    
