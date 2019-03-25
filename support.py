#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 11:04:50 2018

@author: zoran
"""

import numpy as np
from scipy.special import factorial

class Support:

    def __init__(self, tstr_min=0.1, tstr_max=2, buff_len=10, k=8, dt=0.001, g=0, Nt=1000): 

        """
        Construct a supported dimension (til_f) spanned with buff_len tuning curves from tstr_min to tstr_max.


        Keyword arguments:

        tstr_min -- smallest taustar (lowest value that til_f can encode)
        tstr_max -- largest taustar (highest value that til_f can encode)
        buff_len -- number of tuning curves that consitute til_f
        k -- value of k (determines sharpnes of the tuning curves)
        dt -- time step
        g -- parameter g (determines scaling of til_f - g=1->equal amplitude, g=0->power-law decay of amplitude)
        Nt -- number of (time) points
        """

        self.tstr_min = tstr_min
        self.tstr_max = tstr_max
        self.buff_len = buff_len
        self.k = k
        self.dt = dt
        self.g = g
        self.Nt = Nt
        
        # Total number of leaky integrators (2k explansion needed to compute the derivate)
        self.N = self.buff_len+2*self.k 

        # Create power-law growing Taustarlist and compute corresponding s
        a = (self.tstr_max/self.tstr_min)**(1./buff_len)-1
        pow_vec = np.arange(-self.k,buff_len + self.k) #-1
        self.Taustarlist = self.tstr_min * (1 + a)**pow_vec
        s = self.k/self.Taustarlist

        # Create a matrix that implements k-th order derivative
        self._DerivMatrix = np.zeros((self.N,self.N))
        for i in range(1,self.N-1):
            self._DerivMatrix[i, i-1] = -(s[i+1]-s[i])/(s[i]-s[i-1])/(s[i+1] - s[i-1])
            self._DerivMatrix[i, i] = ((s[i+1]-s[i])/(s[i]- s[i-1])-(s[i]-s[i-1])/(s[i+1]-s[i]))/(s[i+1] - s[i-1])
            self._DerivMatrix[i, i+1] = (s[i]-s[i-1])/(s[i+1]-s[i])/(s[i+1] - s[i-1])
            
        self.F = np.zeros((self.N,self.Nt))
        self.til_f = np.zeros((self.N,self.Nt))
        self.s = s

    # Compute aproximation of the inverse Laplace transfrom
    def invert(self, time_index):
        F_diff = np.dot(np.linalg.matrix_power(self._DerivMatrix, self.k), self.F[:,time_index])
        L1 = (-1)**self.k*self.s**(self.k+1) # this can be taken out
        L2 = (F_diff/factorial(self.k))*(self.Taustarlist**self.g)
        self.til_f[:,time_index] = L1.T*L2.T
        self.til_f[self.til_f[:,time_index]<0,time_index] = 0 # in case of numerical errors

    # Update the memory representation at the current time step
    def update(self, time_index, f=0, alpha=1):
        self.F[:,time_index] = self.F[:,time_index-1]+(alpha*(-self.s.T*self.F[:,time_index-1]+f)*self.dt)
        # to avoid numerical errors if small t grows too high
        if np.max(self.F[:,time_index]) > 0.01:
            self.F[:,time_index] = np.zeros(self.N)
        self.invert(time_index)

    # Translate the memory representation
    def jump(self, time_index, delta):
        R = np.diag(np.exp(delta*self.s))
        self.F[:,time_index] = np.dot(R,self.F[:,time_index])

    # Encode a scalar as a distribution
    def set_input(self, time_index, f=0):
        self.F[:,time_index] = np.exp(-self.s*f)
        self.invert(time_index)

    # Set the memory representation to zero
    def reset(self):
        self.F = np.zeros((self.N,self.Nt))
        self.til_f = np.zeros((self.N,self.Nt))
