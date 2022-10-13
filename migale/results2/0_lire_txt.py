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

list_sp3 = np.ones(100)
list_sp5 = np.ones(100)
for i in range(0,100):
    j = np.arange(8016,8216,2)[i]
    fichier3 = open(f"sp3_{i}.o157{j}", "r")
    txt3 = fichier3.read()
    index3 = txt3.find("speed")+8
    list_sp3[i] = txt3[index3:-1]
    fichier5 = open(f"sp5_{i}.o157{j+1}", "r")
    txt5 = fichier5.read() 
    index5 = txt5.find("speed")+8
    list_sp5[i] = txt5[index5:-1]
    fichier3.close()
    fichier5.close()
    
list_sp3_bis = np.ones(200)
for i in range(1,200):
    fichier3 = open(f"3_{i}_speed.txt", "r")
    list_sp3_bis[i-1] = fichier3.read()
    fichier3.close()
    
speed_sp5[-1] = 0

np.savetxt(f"0_speed_sp3.txt", speed_sp3)   
np.savetxt(f"0_speed_sp5.txt", speed_sp5)   
np.savetxt(f"0_speed_sp1.txt", speed_sp1)  

#speed_sp1 = np.loadtxt(f'0_list_sp1.txt')
list_sp3 = np.loadtxt(f'0_list_sp3.txt')
#list_sp5 = np.loadtxt(f'0_list_sp5.txt')

speed_sp3 = np.zeros(nb_step)
speed_sp3[:len(list_sp3)] = list_sp3
#speed_sp5 = np.zeros(nb_step)
#speed_sp5[:len(list_sp3)] = list_sp5

fig, ax = plt.subplots()
#ax.axhline(y=0, color='dimgray')
ax.plot(step_record, speed_sp1, label = 'sp = 0.1', color = 'cornflowerblue') 
ax.plot(step_record, speed_sp3, label = 'sp = 0.3', color = col_pink[0]) 
ax.plot(step_record, speed_sp1, color = 'cornflowerblue') 
ax.plot(step_record, speed_sp5, label = 'sp = 0.5', color = col_pink[4]) 
ax.set(xlabel='Spatial step size', ylabel='Speed') 
ax.xaxis.label.set_size(label_size); ax.yaxis.label.set_size(label_size)  
plt.legend() 
plt.grid()  
fig.savefig(f"figure.png", format='png')  
fig.savefig(f"figure.pdf", format='pdf')  
fig.savefig(f"figure.svg", format='svg')
plt.show() 