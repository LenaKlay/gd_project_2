#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 09:49:47 2021



lena.klay@sorbonne-universite.fr
"""

############################## Libraries ############################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
import scipy.sparse.linalg  as la
import os

########################## External functions #######################################

# Fitness, conversion, recombinaison...
from steps import fitness
from steps import f
from steps import coef

# To control the program through locus ab and loccus cd sums
from control import continuous_evolution_ab
from control import continuous_evolution_cd


############################## Functions ############################################


# List of possible gametes and position of modified alleles
locusABCD = []
indexABCD = np.zeros((4,16))
index=0;
for locusA in ['a','A'] : 
    for locusB in ['b','B'] : 
        for locusC in ['c','C'] : 
            for locusD in ['d','D'] : 
                locusABCD.append(locusA+locusB+locusC+locusD)
                if locusA == 'A' : indexABCD[0,index]=1
                if locusB == 'B' : indexABCD[1,index]=1
                if locusC == 'C' : indexABCD[2,index]=1
                if locusD == 'D' : indexABCD[3,index]=1
                index += 1
                
 
# Print the value of the vector next to the corresponding allele (NB : Need a vector of size 16)
# Simple gamete
def print_vect(vect) :
    new_list = []
    for i in range(len(locusABCD)) : 
        new_list.append([locusABCD[i],vect[i]])
    return(new_list)
  
# Couple of gametes
def print_vect_long(vect) : 
    new_list = []
    l = 0
    for i in range(16) : 
        for j in range(i,16) : 
            new_list.append([locusABCD[i], locusABCD[j], vect[l]])
            l = l+1
    return(new_list)
    
# Fitness for couple of gametes
def print_fitness() : 
    new_list = []
    for i in range(16) : 
        for j in range(i,16) : 
            new_list.append([locusABCD[i], locusABCD[j], fitness([locusABCD[i],locusABCD[j]],sd,sp,st)])
    return(new_list)


def continuous_evolution(r,sd,st,sp,dif,gamma,T,L,M,N,theta,mod_a):   
   
    # Steps
    dt = T/M    # time
    dx = L/N    # space
    
    # Spatial domain (1D)
    X = np.linspace(0,N,N+1)*dx   
    
    # Initialization (frequency vector : abcd  abcD  abCd  abCD  aBcd  aBcD  aBCd  aBCD  Abcd  AbcD  AbCd  AbCD  ABcd  ABcD  ABCd  ABCD)   
    prop_gametes = np.zeros((16,N+1))   # prop_gametes : each row represents a gamete, each column represents a site in space  
    if CI == "equal" :                              
        prop_gametes = np.ones((16,N+1))*(1/16)     
    if CI == "ABCD_global" : 
        prop_gametes[15,:] = CI_prop_drive         
    if CI == "ABCD_left" : 
        prop_gametes[15,0:N//2+1] = CI_prop_drive  
    if CI == "ABCD_center" : 
        prop_gametes[15,N//2-CI_lenght//2:N//2+CI_lenght//2+1] = CI_prop_drive  
    prop_gametes[0,:] = 1 - prop_gametes[15,:]
    
    # Speed of the wave, function of time
    position = np.array([])             # list containing the first position where the proportion of wild alleles is higher than 0.5.
    time = np.array([])                 # list containing the time at which we calculate the speed.
    speed_fct_of_time = np.array([])    # list containing the speed corresponding to the time list.
         
    # Spatial graph 
    nb_graph = 1
    if show_graph_ini :
        graph_x(X, 0, prop_gametes)
      
    # Time graph
    nb_point = 1
    if show_graph_t : 
        points = np.zeros((5,int(T/mod_t)+1))
        points = graph_t(X, 0, prop_gametes, coef_gametes_couple, points, 0)
    
    # Matrix
    C0 = -2*np.ones(N+1); C0[0]=C0[0]+1; C0[-1]=C0[-1]+1               
    C1 = np.ones(N+1) 
    A = spa.spdiags([C1,C0,C1],[-1,0,1], N+1, N+1)          # 1D discrete Laplacian with Neumann boundary conditions (derivative=0)  
    
    B = spa.identity(N+1)+((1-theta)*dif*dt/dx**2)*A            # Matrix for the explicit side of the Crank Nicholson scheme  
    B_ = spa.identity(N+1)-(theta*dif*dt/dx**2)*A               # Matrix for the implicit side of the Crank Nicholson scheme  

    # Evolution
    for t in np.linspace(dt,T,M) : 
        
        t = round(t,2)
        
        reaction_term = f(prop_gametes, coef_gametes_couple)[0]
  
        for i in range(16) : 
            prop_gametes[i,:] = la.spsolve(B_, B.dot(prop_gametes[i,:]) + dt*reaction_term[i,:])
            
        # speed of the C or D wave
        C_or_D = np.dot(indexABCD[2,:],prop_gametes)
        if np.isin(True, C_or_D < 0.5) and np.isin(True, C_or_D > 0.99) :  
            position = np.append(position,np.where(C_or_D < 0.5)[0][0])   # first position where the wave is over 0.5
          
        # spatial graph  
        if t>=mod_a*nb_graph :  
            if show_graph_x :
                graph_x(X, t, prop_gametes)
            nb_graph += 1
            # We calculate the speed each time we make a graph and store this speed.
            if t >= T/10 :
                time = np.append(time, t)
                speed_fct_of_time = np.append(speed_fct_of_time, np.mean(np.diff(position))*dx/dt)
                # first line : time, second line : speed of the wave at the corresponding time
            
        # time graph
        if t>=mod_t*nb_point and show_graph_t :  
            points = graph_t(X, t, prop_gametes, coef_gametes_couple, points, nb_point)
            nb_point += 1
    
    # last graph
    if show_graph_fin :   
        graph_x(X, T, prop_gametes)
   
    # speed function of time (from 2*T/5 to T)
    if np.shape(position)[0] != 0 :        
        fig, ax = plt.subplots()
        ax.plot(time, speed_fct_of_time) 
        ax.set(xlabel='Time', ylabel='Speed', title = f'Speed function of time with f0={CI_prop_drive}')   
        if save_fig :
            fig.savefig(f"../outputs/graph_space_r_{r}_gamma_{gamma}_sd_{sd}_st_{st}_sp_{sp}_dif_{dif}_f0_{CI_prop_drive}_{CI}/speed_fct_time.pdf")   
        plt.show() 
    else :
        print('No wave')
        
    file = open(f"../outputs/graph_space_r_{r}_gamma_{gamma}_sd_{sd}_st_{st}_sp_{sp}_dif_{dif}_f0_{CI_prop_drive}_{CI}/parametres.txt", "w") 
    file.write(f"Parametres : \nr = {r} \nsd = {sd} \nst = {st} \nsp = {sp} \ndif = {dif} \ngamma = {gamma} \nCI = {CI} \nT = {T} \nL = {L} \nM = {M} \nN = {N} \ntheta = {theta} \nf0 = {CI_prop_drive}") 
    file.close()
        
    return(prop_gametes)  



############################### Graph and saving figures ######################################
    
# Proportion of allele in space at time t
def graph_x(X, t, prop_gametes):
        fig, ax = plt.subplots()
        if WT :
            ax.plot(X, prop_gametes[0,:], color='green', label='WT', linewidth=3)
        for i in range(3) :
            if [alleleA,alleleB,alleleCD][i] :
                lab = ['A','B','C or D'][i]
                col = ['yellowgreen','orange','deeppink'][i]
                ax.plot(X, np.dot(indexABCD[i,:],prop_gametes), color=col, label=lab, linewidth=3)
            if [cd,CdcD,CD][i] :
                lab = ['cd','cD+Cd','CD'][i]
                col = ['yellowgreen','skyblue','blue'][i]
                vect = [(1-indexABCD)[2,:]*(1-indexABCD)[3,:], (1-indexABCD)[2,:]*indexABCD[3,:]+indexABCD[2,:]*(1-indexABCD)[3,:], indexABCD[2,:]*indexABCD[3,:]][i]
                ax.plot(X, np.dot(vect,prop_gametes), color=col, label=lab, linewidth=3)
            if [ab,AbaB,AB][i] :
                lab = ['ab','aB+Ab','AB'][i]
                col = ['yellowgreen','skyblue','blue'][i]
                vect = [(1-indexABCD)[0,:]*(1-indexABCD)[1,:], (1-indexABCD)[0,:]*indexABCD[1,:]+indexABCD[0,:]*(1-indexABCD)[1,:], indexABCD[0,:]*indexABCD[1,:]][i]
                ax.plot(X, np.dot(vect,prop_gametes), color=col, label=lab, linewidth=3)
        ax.grid()      
        ax.set(xlabel='Space', ylabel='Frequency', ylim=[-0.02,1.02], title = f'Evolution : f0={CI_prop_drive}, time = {int(t)}')   
        ax.legend()  
        if save_fig :
            save_figure(t, "graph_space", r, gamma, sd, st, sp, dif, CI, CI_prop_drive, fig) 
        plt.show() 
        
# Proportion of allele in time at spatial site 'focus x'
def graph_t(X, t, prop_gametes, coef_gametes_couple, values, nb_point):
    sumABCD = np.dot(indexABCD, prop_gametes)
    mean_fitness = f(prop_gametes, coef_gametes_couple)[1]   
    values[0,nb_point] = t 
      
    if graph_t_type == 'ABCD' : 
        lab = ['A','B','C','D']
        col = ['red','orange','yellowgreen','cornflowerblue']
        for i in range(1,5) :
            values[i,nb_point]=sumABCD[i-1,N//2+focus_x]   
        
    if graph_t_type == 'fig4' : 
        lab = ['B','C or D','1-W']
        col = ['orange','cornflowerblue','black']
        values[1,nb_point] = sumABCD[1,N//2+focus_x]      
        values[2,nb_point] = sumABCD[2,N//2+focus_x]
        values[3,nb_point] = 1-mean_fitness[N//2+focus_x]  
    
    if nb_point != np.shape(values)[1]-1 : 
        return(values)
    else : 
        fig, ax = plt.subplots()
        for i in range(len(lab)) : 
            ax.plot(values[0,:], values[i+1,:], color=col[i], label=lab[i], linewidth=3)
        ax.grid()      
        ax.set(xlabel='Time', ylabel='Frequency', ylim=[-0.02,1.02], title = f'Evolution : f0={CI_prop_drive}, position = {focus_x}, time = {int(t)}')   
        ax.legend()  
        if save_fig :
            save_figure(t, "graph_time", r, gamma, sd, st, sp, dif, CI, CI_prop_drive, fig) 
        plt.show() 
        
        
def save_figure(t, graph_type, r, gamma, sd, st, sp, dif, CI, CI_prop_drive, fig)   :           
            if t == 0 : 
                actual_dir = os.getcwd()
                print ("The current working directory is %s" % actual_dir)
                new_dir = f"../outputs/{graph_type}_r_{r}_gamma_{gamma}_sd_{sd}_st_{st}_sp_{sp}_dif_{dif}_f0_{CI_prop_drive}_{CI}"
                try:
                    os.mkdir(new_dir)
                except OSError:
                    print ("Creation of the directory %s failed" % new_dir)
                else:
                    print ("Successfully created the directory %s " % new_dir)
                    
            fig.savefig(f"../outputs/{graph_type}_r_{r}_gamma_{gamma}_sd_{sd}_st_{st}_sp_{sp}_dif_{dif}_f0_{CI_prop_drive}_{CI}/t_{t}.pdf")   
         
       


############################### Parameters ######################################

# Recombination rate 
r = 0.5
# Conversion rate (probability of a successfull gene drive conversion)
gamma = 0.9

# Fitness disadvantage
sd = 0.02
st = 0.9
sp = 0.1

# Coefficents for the reaction term
coef_gametes_couple = coef(sd,sp,st,gamma,r)

# Diffusion rate
dif = 0.2

# Initial repartition
CI = "ABCD_left" #"ABCD_center"     # "equal"  "ABCD_global"  "ABCD_left"  "ABCD_center" 
CI_prop_drive = 1   # Drive initial proportion in "ABCD_global"  "ABCD_left"  "ABCD_center" 
CI_lenght = 20         # for "ABCD_center", lenght of the initial drive condition in the center (CI_lenght divisible by N and 2) 

# Numerical parameters
T = 8*8000         # final time
L = 8*2800         # length of the spatial domain
M = T*4          # number of time steps
N = L          # number of spatial steps
theta = 0.5      # discretization in space : theta = 0.5 for Crank Nicholson
                 # theta = 0 for Euler Explicit, theta = 1 for Euler Implicit  

# Graphics
show_graph_x = True       # whether to show the graph in space or not
show_graph_ini = True     # whether to show the allele graph or not at time t=0
show_graph_fin = True     # whether to show the allele graph or not at time t=T
show_graph_t = False      # whether to show the graph in time or not
mod_a = T/50              # time at which to draw allele graphics
mod_t = T/50              # time points used to draw the graph in time
save_fig = True           # save the figures (.pdf)

# Which alleles to show in the graph
WT = False             
alleleA = True; alleleB = alleleA; alleleCD = alleleA
checkab = False; ab = checkab; AbaB = ab; AB = ab 
checkcd = False; cd = checkcd; CdcD = cd; CD = cd

# What kind of graph in time
graph_t_type = "ABCD"         # "fig4" or "ABCD"
focus_x = 20               # where to look, on the x-axis (0 = center)


############################### Evolution ########################################
   
prop = continuous_evolution(r,sd,st,sp,dif,gamma,T,L,M,N,theta,mod_a) 


############################### Control ########################################

# Check ab
if checkab : 
    ab_,aB_,Ab_,AB_ = continuous_evolution_ab(r,sd,dif,gamma,T,L,M,N,theta,CI,show_graph_ini,show_graph_fin,mod_a,show_graph_x,ab,AbaB,AB)
    print('ab check :',(abs(ab_ - np.sum(prop[0:4], axis=0)) < 0.001)[0])
    print('aB check :',(abs(aB_ - np.sum(prop[4:8], axis=0)) < 0.001)[0])
    print('Ab check :',(abs(Ab_ - np.sum(prop[8:12], axis=0)) < 0.001)[0])
    print('AB check :',(abs(AB_ - np.sum(prop[12:16], axis=0)) < 0.001)[0])

# Check cd
if checkcd :    
    cd_,cD_,Cd_,CD_ = continuous_evolution_cd(r,sp,st,dif,gamma,T,L,M,N,theta,CI,show_graph_ini,show_graph_fin,mod_a,show_graph_x,cd,CdcD,CD)
    print('cd check :',(abs(cd_ - np.sum(prop[(0,4,8,12),:], axis=0)) < 0.001)[0])
    print('cD check :',(abs(cD_ - np.sum(prop[(1,5,9,13),:], axis=0)) < 0.001)[0])
    print('Cd check :',(abs(Cd_ - np.sum(prop[(2,6,10,14),:], axis=0)) < 0.001)[0])
    print('Cd check :',(abs(CD_ - np.sum(prop[(3,7,11,15),:], axis=0)) < 0.001)[0])

############################### Print parameters ########################################

print('\nr = ',r,' sd =', sd,' dif =',dif,' gamma =',gamma, ' CI =', CI)
print('T =',T,' L =',L,' M =',M,' N =',N,' theta =',theta, ' f0 =', CI_prop_drive)


