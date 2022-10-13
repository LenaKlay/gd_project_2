#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 12:23:35 2022

@author: lena
"""


import numpy as np
import matplotlib.pyplot as plt

# Colors used
col_pink = ['indigo', 'purple', 'darkmagenta', 'm', 'mediumvioletred', 'crimson', 'deeppink', 'hotpink', 'lightpink', 'pink' ]    
col_blue = ['navy', 'blue','royalblue', 'cornflowerblue', 'lightskyblue']    


title_size = 15
label_size = 17
legend_size = 12
line_size = 3

step_min = 0.1
step_max = 3 
nb_step = 200

step_record = np.linspace(step_min, step_max, nb_step)
    
speed_sp2 = np.zeros(200)
for i in range(0,200):
    fichier2 = open(f"2_{i}_speed.txt", "r")
    speed_sp2[i] = fichier2.read()
    fichier2.close()

np.savetxt(f"0_speed_sp2.txt", speed_sp2)  

speed_sp1 = np.loadtxt(f'0_speed_sp1.txt')
speed_sp3 = np.loadtxt(f'0_speed_sp3.txt')
speed_sp5 = np.loadtxt(f'0_speed_sp5.txt')


fig, ax = plt.subplots()
#ax.axhline(y=0, color='dimgray')
ax.plot(step_record, speed_sp1, label = 'sp = 0.1', color = 'cornflowerblue') 
ax.plot(step_record, speed_sp2, label = 'sp = 0.2', color = 'forestgreen') 
ax.plot(step_record, speed_sp3, label = 'sp = 0.3', color = col_pink[0]) 
ax.plot(step_record, speed_sp5, label = 'sp = 0.5', color = col_pink[4]) 
ax.set(xlabel='Spatial step size', ylabel='Speed') 
ax.xaxis.label.set_size(label_size); ax.yaxis.label.set_size(label_size)  
plt.legend() 
plt.grid()  
fig.savefig(f"figure.png", format='png')  
fig.savefig(f"figure.pdf", format='pdf')  
fig.savefig(f"figure.svg", format='svg')
plt.show() 