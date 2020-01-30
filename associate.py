#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 17:31:41 2018

@author: zoran
"""

import numpy as np
class Associate:
    def __init__(self, buff_len=10, assoc_len=100):
        self.buff_len = buff_len
        self.assoc_len = assoc_len
        self.M = np.zeros((self.buff_len, self.assoc_len, self.assoc_len))
        self.M_bar = np.zeros((self.buff_len, self.assoc_len, self.assoc_len))
        
    def update(self, f, til_f, learn_rate=1):
        for tstr in range(self.buff_len):
            self.M[tstr, :, :] = learn_rate*self.M[tstr, :, :] + np.outer(f, til_f[:, tstr])
            for input_no1 in range(self.assoc_len):
                norm = np.sum(self.M[tstr, input_no1, :])
                for input_no2 in range(self.assoc_len):
                    self.M_bar[tstr, input_no1, input_no2] = self.M[tstr, input_no1, input_no2]/norm
                    
                    
class JBIT:
    def __init__(self, buff_len=10, assoc_len=100):
        self.buff_len = buff_len
        self.assoc_len = assoc_len
        self.J = np.zeros((self.buff_len, self.buff_len, self.assoc_len, self.assoc_len))
        
    def update(self, til_f, learn_rate=1):
        for tstr1 in range(self.buff_len):
            for tstr2 in range(self.buff_len):
                outer_prod = np.outer(til_f[:, tstr1], til_f[:, tstr2])
                self.J[tstr1, tstr2, :, :] = learn_rate*self.J[tstr1, tstr2, :, :] + outer_prod
    
    def probe(self, f, til_f, tstr1, tstr2):
        self.result = np.inner(self.J[tstr1, tstr2, :, :])

                    