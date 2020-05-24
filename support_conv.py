#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 00:59:09 2020

@author: zoran
"""

import numpy as np
from scipy.special import factorial

class Support:

    def __init__(self, tstr_min=0.1, tstr_max=2, buff_len=10, k=8, dt=0.001, g=0, Nt=1000): 
        
        
        self.tstr_min = tstr_min
        self.tstr_max = tstr_max
        self.buff_len = buff_len
        self.k = k
        self.dt = dt
        self.g = g
        self.Nt = Nt
        
        # Create power-law growing Taustarlist and compute corresponding s
        a = (self.tstr_max/self.tstr_min)**(1./(buff_len-1))
        pow_vec = np.arange(0,buff_len)
        self.Taustarlist = self.tstr_min * a**pow_vec
        s = self.k/self.Taustarlist
        
        time_vec = np.arange(0,self.Nt*dt,dt)
        self.til_f = np.zeros((self.buff_len,len(time_vec)))
        self.s = s
        
        for tstr_ind in range(self.buff_len):
            c = (1/self.Taustarlist[tstr_ind])*(k**(k+1)/factorial(k))*(self.Taustarlist[tstr_ind]**self.g)
            self.til_f[tstr_ind,:] = c*((time_vec/self.Taustarlist[tstr_ind])**(k+1))*np.exp(k*(-time_vec/self.Taustarlist[tstr_ind]))
        