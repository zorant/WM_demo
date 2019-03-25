#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 20:52:42 2018

@author: zoran
"""

import numpy as np

# Translation by multiplication of leaky integrators by exp(-delta).
# Since leaky integrators have exponentially decaying impulse reposense, this
# shifts the response by deta time. 
def translate(a_support, delta, time_index):
    a_support.jump(time_index, -delta)
    a_support.invert(time_index)
    return a_support

# Adding two distribution (a_support, b_support) by multipying their Laplace 
# transform and computing the inverse.
def add_distributions(a_support, b_support, c_support, time_index):
    c_support.F[:,time_index] = np.multiply(a_support.F[:,time_index], b_support.F[:,time_index])
    c_support.invert(time_index)
    return c_support

# Subtracting two distribution (a_support - b_support) by first reflecting 
# b around tstr_max/2, then adding tstr_max/2 to it and finallying computing a+b.
# All the operations are done in the Laplace domain and the outout is inverted 
# at the end. Notice that 0 of the result will be at tstr_max/2, left of which 
# will be negative values and right of which will be positive values. 
def subtract_distributions(a_support, b_support, c_support, time_index):
    b_support.F[:,time_index] = 1/b_support.F[:,time_index] # reflect b around tstr_max/2
    b_support.jump(time_index, delta=-b_support.tstr_max/2) # add tstr_max/2 to b
    c_support.F[:,time_index] = np.multiply(a_support.F[:,time_index], b_support.F[:,time_index])
    c_support.invert(time_index)
    return c_support