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
# Change font to serif
plt.rcParams.update({'font.family':'serif'})

########################## External functions #######################################

# Fitness, conversion, recombinaison...
from steps import fitness
from steps import f
from steps import coef

########################## Graph parameters #######################################

title_size = 15
label_size = 17
legend_size = 12
line_size = 3

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


def continuous_evolution(r,sd,st,sp,cst_value,gamma,T,M,X,theta,mod_x):   
   
    # Time step
    dt = T/M    
    
    # Space step
    N = len(X)-1
    dx = X[1:]-X[:-1]
    denomx = dx[1:]*dx[:-1]*(dx[1:]+dx[:-1])/2
    
    # Diffusion rate
    if diffusion == 'cst dif': dif = cst_value
    if diffusion == 'cst m': m = cst_value; dif = (m*dx[1:]*dx[:-1])/(2*dt)
    
      
    # Initialization (frequency vector : abcd  abcD  abCd  abCD  aBcd  aBcD  aBCd  aBCD  Abcd  AbcD  AbCd  AbCD  ABcd  ABcD  ABCd  ABCD)   
    prop_gametes = np.zeros((16,N+1))   # prop_gametes : each row represents a gamete, each column represents a site in space  
    # Introduction of  ABCD or CD ?
    if CI[-3] == '_': CI_nb_allele = 3
    if CI[-3] == 'b': CI_nb_allele = 15
    # Where to introduce the drive ? (left, center, square)
    if CI[0] == 'l': CI_where = np.arange(0,(N//2+1))
    if CI[0] == 'c': 
        if CI_lenght >= N : print('CI_lenght bigger than the domain lenght')
        else: CI_where = np.arange(((N-CI_lenght)//2+1), ((N+CI_lenght)//2+1))
    # Special case: equal CI means that all genotypes are present at the beginning, in the same proportion.
    if CI == "equal": prop_gametes = np.ones((16,N+1))*(1/16)   
    # Drive introduction
    else : prop_gametes[CI_nb_allele, CI_where] = CI_prop_drive 
    # Complete the domain with wild-type individuals
    prop_gametes[0,:] = 1 - np.sum(prop_gametes[1:16,:], axis=0)   
      
    
    # Speed of the wave, function of time
    position = np.array([])             # list containing the first position where the proportion of wild alleles is higher than the treshold value.
    time = np.array([])                 # list containing the time at which we calculate the speed (wave the most on the left).
    speed_fct_of_time = np.array([])    # list containing the speed of the left wave corresponding to the time list.
    treshold = 0.5                      # indicates which position of the wave we follow to compute the speed (first position where the C-D wave come under the threshold)    
    
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
    C0 = -np.ones(N-1)*(dx[1:]+dx[:-1]); C0[0]=-dx[0]; C0[-1]=-dx[-1]; C0 = C0/denomx
    C1 = np.zeros(N-1); C1[1:] = dx[:-2]/denomx[:-1]
    C2 = np.zeros(N-1); C2[:-1] = dx[2:]/denomx[1:]
    # Diffusion    
    if diffusion == 'cst dif': C0 = C0*dif; C1 = C1*dif; C2 = C2*dif
    if diffusion == 'cst m': C0 = C0*dif; C1[1:] = C1[1:]*dif[:-1]; C2[:-1] = C2[:-1]*dif[1:]
    # Laplacian with diffusion and dx
    A = spa.spdiags([C1,C0,C2],[1,0,-1], N-1, N-1)       # 1D discrete Laplacian with Neumann boundary conditions (derivative=0)  
    
    # Example for spdiags...
    #data = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    #diags = np.array([0, -1, 2])
    #spa.spdiags(data, diags, 4, 4).toarray()
        
    B = spa.identity(N-1)+((1-theta)*dt*A)           # Matrix for the explicit side of the Crank Nicholson scheme  
    B_ = spa.identity(N-1)-(theta*dt*A)               # Matrix for the implicit side of the Crank Nicholson scheme  
    
    # Evolution
    for t in np.linspace(dt,T,M) : 
        
        t = np.round(t,2)
        
        reaction_term = f(prop_gametes, coef_gametes_couple)[0]
  
        for i in range(16) : 
            #prop_gametes[i,:] = la.spsolve(B_, B.dot(prop_gametes[i,:]) + dt*reaction_term[i,:])
            prop_gametes[i,1:-1] = la.spsolve(B_, B.dot(prop_gametes[i,1:-1]) + dt*reaction_term[i,1:-1])
            prop_gametes[i,0] = prop_gametes[i,1]   # Neumann condition alpha=0
            prop_gametes[i,-1] = prop_gametes[i,-2]  # Neumann condition beta=0
        
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
    if show_graph_end :   
        graph_x(X, t, prop_gametes)
   
    # speed function of time
    if CI != "equal" :
        if len(speed_fct_of_time) != 0 and show_graph_x :   
            fig, ax = plt.subplots() 
            if CI[0] == 'l':
                ax.plot(time, speed_fct_of_time, color = "cornflowerblue") 
            if CI[0] == 'c':
                ax.plot(time, -speed_fct_of_time, color = "cornflowerblue") 
            plt.hlines(y=0, color='dimgray', xmin=time[0], xmax=time[-1])
            ax.set(xlabel='Time', ylabel='Speed', ylim = [ymin_speed,ymax_speed])   
            #ax.set_title("Speed of the wave C or D function of time", fontsize = title_size)  
            ax.grid()
            if save_fig :
                save_fig_or_data(out_dir, fig, speed_fct_of_time, "speed_fct_time")   
                save_fig_or_data(out_dir, [], time, "time")    
            plt.show() 
        if np.shape(position)[0] == 0 :
            print('No wave')
        
    file = open(f"../outputs/{out_dir}/parameters.txt", "w") 
    file.write(f"Parameters : \nr = {r} \nsd = {sd} \nst = {st} \nsp = {sp} \n{diffusion} = {cst_value} \ngamma = {gamma} \nCI = {CI} \nT = {T} \nM = {M} \ntheta = {theta} \nf0 = {CI_prop_drive}") 
    file.close()
    
    return(prop_gametes, time, speed_fct_of_time)  



############################### Graph and saving figures ######################################
    
# Proportion of allele in space at time t
def graph_x(X, t, prop_gametes):
        fig, ax = plt.subplots()
        if WT :
            ax.plot(X, prop_gametes[0,:], color='green', label='WT', linewidth=line_size)
        for i in range(3) :
            if [alleleA,alleleB,alleleCD][i] :
                lab = [r'$X_A$',r'$X_B$',r'$X_C$ or $X_D$'][i]
                col = ['yellowgreen','orange','deeppink'][i]
                ax.plot(X, np.dot(indexABCD[i,:],prop_gametes), color=col, label=lab, linewidth=line_size)
            if [cd,CdcD,CD][i] :
                lab = ['cd','cD+Cd','CD'][i]
                col = ['yellowgreen','skyblue','blue'][i]
                vect = [(1-indexABCD)[2,:]*(1-indexABCD)[3,:], (1-indexABCD)[2,:]*indexABCD[3,:]+indexABCD[2,:]*(1-indexABCD)[3,:], indexABCD[2,:]*indexABCD[3,:]][i]
                ax.plot(X, np.dot(vect,prop_gametes), color=col, label=lab, linewidth=line_size)
            if [ab,AbaB,AB][i] :
                lab = ['ab','aB+Ab','AB'][i]
                col = ['yellowgreen','skyblue','blue'][i]
                vect = [(1-indexABCD)[0,:]*(1-indexABCD)[1,:], (1-indexABCD)[0,:]*indexABCD[1,:]+indexABCD[0,:]*(1-indexABCD)[1,:], indexABCD[0,:]*indexABCD[1,:]][i]
                ax.plot(X, np.dot(vect,prop_gametes), color=col, label=lab, linewidth=line_size)
        ax.grid() 
        if ticks : 
            ax.set_xticks(X)   
        ax.axes.xaxis.set_ticklabels([])
        ax.set(xlabel='Space', ylabel='Frequency', ylim=[-0.02,1.02])   
        ax.xaxis.label.set_size(label_size); ax.yaxis.label.set_size(label_size)   
        ax.set_title(f"t = {int(t)}", fontsize = title_size, loc='right')
        plt.rc('legend', fontsize=legend_size)
        ax.legend()  
        if save_fig :
            save_fig_or_data(out_dir, fig, [], f"t_{int(t)}")   
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
            ax.plot(values[0,:], values[i+1,:], color=col[i], label=lab[i], linewidth=line_size)
        ax.grid()      
        ax.set(xlabel='Time', ylabel='Frequency', ylim=[-0.02,1.02]) 
        ax.xaxis.label.set_size(label_size); ax.yaxis.label.set_size(label_size)   
        ax.set_title(f"position = {N//2+focus_x}, t = {t}", fontsize = title_size, loc='right')
        plt.rc('legend', fontsize=legend_size)
        ax.legend()  
        if save_fig :
            save_fig_or_data(out_dir, fig, [], f"focus_on_site_{focus_x}") 
        plt.show() 
        
   

def save_fig_or_data(new_dir, fig, data, title):
    new_dir = f"../outputs/{new_dir}"
    # ... if the directory doesn't already exist
    if not os.path.exists(new_dir): 
        try:
            os.mkdir(new_dir)
        except OSError:
            print ("Fail : %s " % new_dir)
    # Save figure
    if fig != [] :
        fig.savefig(f"../outputs/{new_dir}/{title}.png", format='png')  
        fig.savefig(f"../outputs/{new_dir}/{title}.pdf", format='pdf')  
        fig.savefig(f"../outputs/{new_dir}/{title}.svg", format='svg')
    # Save datas
    if data != [] :
        np.savetxt(f"../outputs/{new_dir}/{title}.txt", data)   


############################### Parameters ######################################

# Recombination rate 
r = 0.5
# Conversion rate (probability of a successfull gene drive conversion)
gamma = 0.9

# Fitness disadvantage
sd = 0.02
st = 0.9
sp = 0.35

# Coefficents for the reaction term
coef_gametes_couple = coef(sd,sp,st,gamma,r)

 
# Initial repartition
CI = "center_abcd"    # "equal"  "left_abcd" "left_cd" "center_abcd" "center_cd" 
CI_prop_drive = 1   # Drive initial proportion in "ABCD_global"  "ABCD_left"  "ABCD_center" 
CI_lenght = 20     # /!\ should be < N. For "center_abcd" and "center_cd", lenght of the initial drive condition in the center, in number of spatial steps.

# Numerical parameters
T = 1000          # final time
L = 200          # length of the spatial domain
M = T*10         # number of time steps
theta = 0.5      # discretization in space : theta = 0.5 for Crank Nicholson
                 # theta = 0 for Euler Explicit, theta = 1 for Euler Implicit                  
            
# Spatial domain
#X = np.linspace(0,L,201)   # homogeneous
#X = np.concatenate((np.arange(0,L//2,1), np.arange(L//2, L+1,2)))   # heterogeneous half half
#X = np.concatenate((np.arange(0,L//4,2),  np.arange(L//4,3*L//4,1), np.arange(3*L//4, L+1,2)))   # heterogeneous center step 1, outside step 2
X = np.sort(np.random.random_sample(2*L)*L)     # heterogeneous randomized
            
# Diffusion rate: constant or depending on m, dx and dt
diffusion = 'cst dif'     # cst dif or cst m
cst_value = 0.2           # value of the constant diffusion rate or value of the constant migration rate, depending on the line before 

# Graphics
show_graph_x = True       # whether to show the graph in space or not
show_graph_ini = True     # whether to show the allele graph or not at time t=0
show_graph_end = False    # whether to show the allele graph or not at time t=T
ticks = True

show_graph_t = False      # whether to show the graph in time or not
graph_t_type = "ABCD"     # "fig4" or "ABCD"
focus_x = 20              # where to look, on the x-axis (0 = center)

mod_x = T//20             # time at which to draw allele graphics
mod_t = T//50             # time points used to draw the graph in time
save_fig = True           # save the figures (.pdf)

ymin_speed = -0.4         # final graph (speed fct of time) : min y value
ymax_speed = 0.4          # final graph (speed fct of time) : max y value

# Which alleles to show in the graph
WT = False             
alleleA = True; alleleB = alleleA; alleleCD = alleleA
ab = False; AbaB = ab; AB = ab 
cd = False; CdcD = cd; CD = cd

# Where to store the outputs
out_dir = f"var_r_{r}_gam_{gamma}_sd_{sd}_st_{st}_sp_{sp}_{diffusion}_{cst_value}_{CI}"

############################### Evolution ########################################
 
prop, time, speed_fct_of_time = continuous_evolution(r,sd,st,sp,cst_value,gamma,T,M,X,theta,mod_x) 


print('\nr = ',r,' sd =', sd, diffusion, cst_value, 'gamma =', gamma, ' CI =', CI)
print('T =',T,' L =',L,' M =',M, 'theta =',theta, ' f0 =', CI_prop_drive)
