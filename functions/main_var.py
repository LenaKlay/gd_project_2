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


def continuous_evolution(r,sd,st,sp,dif,gamma,T,M,X,theta,mod_x):   
   
    # Time step
    dt = T/M    
    
    # Space step
    N = len(X)-1
    dx = X[1:]-X[:-1]
    denomx = dx[1:]*dx[:-1]*(dx[1:]+dx[:-1])/2
      
    # Initialization (frequency vector : abcd  abcD  abCd  abCD  aBcd  aBcD  aBCd  aBCD  Abcd  AbcD  AbCd  AbCD  ABcd  ABcD  ABCd  ABCD)   
    prop_gametes = np.zeros((16,N+1))   # prop_gametes : each row represents a gamete, each column represents a site in space  
    #if CI == "equal" :                              
    #    prop_gametes = np.ones((16,N+1))*(1/16)     
    if CI == "equal" :                              
        prop_gametes[0:4,:] = np.ones((4,N+1))*(1/4)   
    if CI == "left" : 
        prop_gametes[15,0:N//2+1] = CI_prop_drive  
    if CI == "left_cd" : 
        prop_gametes[3,0:N//2+1] = CI_prop_drive
    if CI == "center" : 
        prop_gametes[15,N//2-CI_lenght//2:N//2+CI_lenght//2+1] = CI_prop_drive  
    if CI == "center_cd" : 
        prop_gametes[3,N//2-CI_lenght//2:N//2+CI_lenght//2+1] = CI_prop_drive  
    prop_gametes[0,:] = 1 - np.sum(prop_gametes[1:16,:], axis=0)
    
    # Speed of the wave, function of time
    position = np.array([])             # list containing the first position where the proportion of wild alleles is higher than the treshold value.
    time = np.array([])                 # list containing the time at which we calculate the speed (wave the most on the left).
    speed_fct_of_time = np.array([])    # list containing the speed of the left wave corresponding to the time list.
    treshold = 0.5                      # indicates which position of the wave we follow to compute the speed (first position where the C-D wave come under the threshold)    
    
    # Spatial graph 
    nb_graph = 1
    if show_graph_ini :
        print('blabla')
        graph_x(X, 0, prop_gametes)
      
    # Time graph
    nb_point = 1
    if show_graph_t : 
        points = np.zeros((5,int(T/mod_t)+1))
        points = graph_t(X, 0, prop_gametes, coef_gametes_couple, points, 0)
    
    # Matrix    
    C0 = -np.ones(N-1)*(dx[1:]+dx[:-1]); C0[0]=-dx[0]; C0[-1]=-dx[-1]; C0 = C0/denomx
    C1 = np.zeros(N-1); C1[1:] = dx[:-2]/denomx[:-1]
    C2 = np.zeros(N-1); C2[:-1] = dx[2:]/denomx[1:]
    A = spa.spdiags([C1,C0,C2],[1,0,-1], N-1, N-1)       # 1D discrete Laplacian with Neumann boundary conditions (derivative=0)  
    
    # Example for spdiags...
    #data = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    #diags = np.array([0, -1, 2])
    #spa.spdiags(data, diags, 4, 4).toarray()
        
    B = spa.identity(N-1)+((1-theta)*dif*dt)*A            # Matrix for the explicit side of the Crank Nicholson scheme  
    B_ = spa.identity(N-1)-(theta*dif*dt)*A               # Matrix for the implicit side of the Crank Nicholson scheme  

    # Evolution
    for t in np.linspace(dt,T,M) : 
        
        t = round(t,2)
        
        reaction_term = f(prop_gametes, coef_gametes_couple)[0]
  
        for i in range(16) : 
            #prop_gametes[i,:] = la.spsolve(B_, B.dot(prop_gametes[i,:]) + dt*reaction_term[i,:])
            prop_gametes[i,1:-1] = la.spsolve(B_, B.dot(prop_gametes[i,1:-1]) + dt*reaction_term[i,1:-1])
            prop_gametes[i,0] = prop_gametes[i,1]   # alpha=0
            prop_gametes[i,-1] = prop_gametes[i,-2]  # beta=0
        
        if CI != "equal" :
            # Position of the wave cd
            wave_cd = np.dot((1-indexABCD)[2,:]*(1-indexABCD)[3,:],prop_gametes)
            # we recorde the position only if the cd wave is still in the environment. We do not recorde the 0 position since the treshold value of the wave might be outside the window.            
            if np.isin(True, wave_cd > treshold) and np.isin(True, wave_cd < 0.99) and np.where(wave_cd > treshold)[0][0] != 0  :  
                # first position where the cd wave (Wild-type) is over the treshold value
                position = np.append(position, X[np.where(wave_cd > treshold)[0][0]]) 
            elif np.isin(True, wave_cd < treshold) and np.isin(True, wave_cd > 0.01) and np.where(wave_cd < treshold)[0][0] != 0  :  
                # first position where the cd wave (Wild-type) is under the treshold value
                position = np.append(position, X[np.where(wave_cd < treshold)[0][0]])
            # compute the speed
            if len(position) > 20 : 
                time = np.append(time, t)
                speed_fct_of_time = np.append(speed_fct_of_time, np.mean(np.diff(position[int(4*len(position)/5):len(position)]))/dt)
            # if the treshold value of the wave is outside the window, stop the simulation  
            if not(np.isin(False, wave_cd>treshold) and np.isin(False, wave_cd<treshold) ) :
                print("t =",t)
                break 
            
        # spatial graph  
        if t>=mod_x*nb_graph :  
            if show_graph_x :
                graph_x(X, t, prop_gametes)
            nb_graph += 1
            
        # time graph
        if t>=mod_t*nb_point and show_graph_t :  
            points = graph_t(X, t, prop_gametes, coef_gametes_couple, points, nb_point)
            nb_point += 1
    
    # last graph
    if show_graph_fin :   
        graph_x(X, T, prop_gametes)
   
    # speed function of time
    if CI != "equal" :
        if len(speed_fct_of_time) != 0 :        
            fig, ax = plt.subplots()
            ax.plot(time, speed_fct_of_time) 
            ax.set(xlabel='Time', ylabel='Speed', title = f'Speed function of time with f0={CI_prop_drive}')   
            if save_fig :
                fig.savefig(f"../outputs/r_{r}_gamma_{gamma}_sd_{sd}_st_{st}_sp_{sp}_dif_{dif}_f0_{CI_prop_drive}_{CI}/speed_fct_time.pdf")   
            plt.show() 
        else :
            print('No wave')
        
    file = open(f"../outputs/r_{r}_gamma_{gamma}_sd_{sd}_st_{st}_sp_{sp}_dif_{dif}_f0_{CI_prop_drive}_{CI}/parameters.txt", "w") 
    file.write(f"Parameters : \nr = {r} \nsd = {sd} \nst = {st} \nsp = {sp} \ndif = {dif} \ngamma = {gamma} \nCI = {CI} \nT = {T} \nM = {M} \ntheta = {theta} \nf0 = {CI_prop_drive}") 
    file.close()
    
    return(prop_gametes, time, speed_fct_of_time)  



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
        if ticks : 
            ax.set_xticks(X)   
        ax.axes.xaxis.set_ticklabels([])
        ax.set(xlabel='Space', ylabel='Frequency', ylim=[-0.02,1.02], title = f'Evolution : f0={CI_prop_drive}, time = {int(t)}')   
        ax.legend()  
        if save_fig :
            save_figure(t, "graph_space", r, gamma, sd, st, sp, dif, CI, CI_prop_drive, fig) 
        plt.show() 
        
# Proportion of allele in time at spatial site 'focus x'
def graph_t(X, t, prop_gametes, coef_gametes_couple, values, nb_point):
    N = len(X)-1
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
        ax.set(xlabel='Time', ylabel='Frequency', ylim=[-0.02,1.02], title = f'Evolution : f0={CI_prop_drive}, position = {N//2+focus_x}, time = {int(t)}')   
        ax.legend()  
        if save_fig :
            fig.savefig(f"../outputs/r_{r}_gamma_{gamma}_sd_{sd}_st_{st}_sp_{sp}_dif_{dif}_f0_{CI_prop_drive}_{CI}/focus_on_one_site.pdf")   
        plt.show() 
        
        
def save_figure(t, graph_type, r, gamma, sd, st, sp, dif, CI, CI_prop_drive, fig)   :           
            if t == 0 : 
                actual_dir = os.getcwd()
                print ("The current working directory is %s" % actual_dir)
                new_dir = f"../outputs/r_{r}_gamma_{gamma}_sd_{sd}_st_{st}_sp_{sp}_dif_{dif}_f0_{CI_prop_drive}_{CI}"
                try:
                    os.mkdir(new_dir)
                except OSError:
                    print ("Creation of the directory %s failed" % new_dir)
                else:
                    print ("Successfully created the directory %s " % new_dir)
            print("t=",t)
            fig.savefig(f"../outputs/r_{r}_gamma_{gamma}_sd_{sd}_st_{st}_sp_{sp}_dif_{dif}_f0_{CI_prop_drive}_{CI}/t_{t}.pdf")  

       


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
CI = "left"      # "equal"  "left"  "center"  "left_cd"  "center_cd" 
CI_prop_drive = 1   # Drive initial proportion in "ABCD_global"  "ABCD_left"  "ABCD_center" 
CI_lenght = 20      # for "ABCD_center", lenght of the initial drive condition in the center (CI_lenght divisible by N and 2) 

# Numerical parameters
T = 2000         # final time
M = T*6          # number of time steps
L = 200
#X = np.linspace(0,L,101)
X = np.concatenate((np.arange(0,L//2,1), np.arange(100, L+1,4))) # spatial domain
#X = np.sort(np.random.random_sample(L)*L)
theta = 0.5      # discretization in space : theta = 0.5 for Crank Nicholson
                 # theta = 0 for Euler Explicit, theta = 1 for Euler Implicit  

# Graphics
show_graph_x = True       # whether to show the graph in space or not
show_graph_ini = True     # whether to show the allele graph or not at time t=0
show_graph_fin = False    # whether to show the allele graph or not at time t=T
ticks = True

show_graph_t = False      # whether to show the graph in time or not
graph_t_type = "ABCD"     # "fig4" or "ABCD"
focus_x = 20              # where to look, on the x-axis (0 = center)

mod_x = T/4               # time at which to draw allele graphics
mod_t = T/50              # time points used to draw the graph in time
save_fig = True           # save the figures (.pdf)

# Which alleles to show in the graph
WT = False             
alleleA = True; alleleB = alleleA; alleleCD = alleleA
ab = False; AbaB = ab; AB = ab 
cd = False; CdcD = cd; CD = cd


############################### Evolution ########################################
 
prop, time, speed = continuous_evolution(r,sd,st,sp,dif,gamma,T,M,X,theta,mod_x) 


