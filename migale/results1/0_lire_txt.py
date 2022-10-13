#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 12:23:35 2022

@author: lena
"""


import numpy as np
import matplotlib.pyplot as plt

title_size = 15
label_size = 17
legend_size = 12
line_size = 3

step_min = 0.1
step_max = 3 
nb_step = 200



sp3_list = np.ones(100)
sp5_list = np.ones(100)
for i in range(0,100):
    j = np.arange(8016,8216,2)[i]
    fichier3 = open(f"sp3_{i}.o157{j}", "r")
    txt3 = fichier3.read()
    index3 = txt3.find("speed")+8
    sp3_list[i] = txt3[index3:-1]
    fichier5 = open(f"sp5_{i}.o157{j+1}", "r")
    txt5 = fichier5.read() 
    index5 = txt5.find("speed")+8
    sp5_list[i] = txt5[index5:-1]
    fichier3.close()
    fichier5.close()




fig, ax = plt.subplots()
ax.plot(step_record, speed_record) 
ax.set(xlabel='Spatial step size', ylabel='Speed') 
ax.xaxis.label.set_size(label_size); ax.yaxis.label.set_size(label_size)   
plt.grid()  
plt.show() 