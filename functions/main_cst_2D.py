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

########################## Graph parameters #######################################

title_size = 15
label_size = 17
legend_size = 12
line_size = 3

# Colors used
col_pink = ['indigo', 'purple', 'darkmagenta', 'm', 'mediumvioletred', 'crimson', 'deeppink', 'hotpink', 'lightpink', 'pink' ]    
col_blue = ['navy', 'blue','royalblue', 'cornflowerblue', 'lightskyblue']    

########################## External functions #######################################

# Fitness, conversion, recombinaison...
from terms import fitness
from terms import f
from terms import coef

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


def continuous_evolution(r,sd,st,sp,cst_value,gamma,T,L,M,N,theta,mod_x):   
   
    # Steps
    dt = T/M    # time
    dx = L/N    # space
    # One dimension space or one edge of the square in two dimensions
    X = np.linspace(0,N,N+1)*dx  
    
    # Diffusion rate
    if diffusion == 'cst dif': dif = cst_value
    if diffusion == 'cst m': m = cst_value; dif = (m*dx**2)/(2*dt) 
       
    # Dimensions
    # prop_gametes : each row represents a gamete, each column represents a site in space    
    if dim == 1:
        prop_gametes = np.zeros((16,N+1))  
        tot_lenght = N+1
        factor = 1
    if dim == 2:
        prop_gametes = np.zeros((16,(N+1)**2))
        tot_lenght = (N+1)**2
        factor = (N+1)
         
    # Initial conditions
    # Introduction of  ABCD or CD ?
    if CI[-3] == '_': CI_nb_allele = 3
    if CI[-3] == 'b': CI_nb_allele = 15
    # Where to introduce the drive ? (left, center, square)
    if CI[0] == 'l': CI_where = np.arange(0,(N//2+1)*factor)
    if CI[0] == 'c': 
        if CI_lenght >= N : print('CI_lenght bigger than the domain lenght')
        else: CI_where = np.arange(((N-CI_lenght)//2+1)*factor, ((N+CI_lenght)//2+1)*factor)
    if CI[0] == 's':
        if CI_lenght >= N : print('CI_lenght bigger than the domain lenght')
        else : edge = np.arange((N-CI_lenght)//2+1, (N+CI_lenght)//2+1); CI_where = np.tile(edge, len(edge))+np.repeat(edge, len(edge))*(N+1)
    
    # Special case: equal CI means that all genotypes are present at the beginning, in the same proportion.
    if CI == "equal": prop_gametes = np.ones((16,tot_lenght))*(1/16)   
    # Drive introduction
    else : prop_gametes[CI_nb_allele, CI_where] = CI_prop_drive 
    # Complete the domain with wild-type individuals
    prop_gametes[0,:] = 1 - np.sum(prop_gametes[1:16,:], axis=0)   
      
          
    # Speed of the wave, function of time
    position = np.array([])             # list containing the first position where the proportion of wild alleles is higher than the treshold value.
    time = np.array([])                 # list containing the time at which we calculate the speed.
    speed_fct_of_time = np.array([])    # list containing the speed corresponding to the time list.
    treshold = 0.5                      # indicates which position of the wave we follow to compute the speed (first position where the C-D wave come under the treshold)    
       
    # Spatial graph 
    nb_graph = 1; Z_list = np.zeros((T//mod_x+2,1000,1000))
    if show_graph_ini :
         if dim == 1: graph_x(0, prop_gametes, X)
         if dim == 2: graph_xy(0, prop_gametes); Z_list = graph_xy_contour(0, prop_gametes, Z_list, nb_graph)
      
    # Time graph
    nb_point = 1
    if show_graph_t : 
        points = np.zeros((5,int(T/mod_t)+1))
        points = graph_t(X, 0, prop_gametes, coef_gametes_couple, points, 0)
    
     # Matrix and index
    if dim == 1:
        # Build the Laplacian matrix 
        C0 = -2*np.ones(N-1); C0[0]=C0[0]+1; C0[-1]=C0[-1]+1               
        C1 = np.ones(N-1) 
        A = spa.spdiags([C1,C0,C1],[-1,0,1], N-1, N-1)        # 1D discrete Laplacian with Neumann boundary conditions (derivative=0)
        B = spa.identity(N-1)+((1-theta)*dif*dt/dx**2)*A      # Matrix for the explicit side of the Crank Nicholson scheme  
        B_ = spa.identity(N-1)-(theta*dif*dt/dx**2)*A         # Matrix for the implicit side of the Crank Nicholson scheme  
        
    if dim == 2:
        # Build the Laplacian matrix 
        index_diag_mat = np.arange(0,(N-1)**2+1,N-1)                  # (0, N-1, 2*(N-1), 3*(N-1), ....)
        C0 = -4*np.ones((N-1)**2); C0[np.array([0,N-2,(N-1)**2-(N-1),(N-1)**2-1])]=-2   # place -2
        C0[1:N-2] = -3; C0[(N-1)**2-(N-1)+1:(N-1)**2-1]=-3                              # place -3 in between -2      
        C0[index_diag_mat[1:-2]] = -3; C0[index_diag_mat[2:-1]-1] = -3                                  # place the others -3
        C1 = np.ones((N-1)**2+1); C1[index_diag_mat]=0
        C2 = np.ones((N-1)**2)
        A = spa.spdiags([C2,C1[1:],C0,C1[:-1],C2],[-N+1,-1,0,1,N-1], (N-1)**2, (N-1)**2) # 1D discrete Laplacian with Neumann boundary conditions (derivative=0)
        B = spa.identity((N-1)**2)+((1-theta)*dif*dt/dx**2)*A      # Matrix for the explicit side of the Crank Nicholson scheme  
        B_ = spa.identity((N-1)**2)-(theta*dif*dt/dx**2)*A         # Matrix for the implicit side of the Crank Nicholson scheme  
        # N,S,W,E the four cardinal points (Exterior index)
        index_N = np.arange(N+1,(N+1)*N,N+1); index_S = np.arange(N+1,(N+1)*N,N+1)+N; index_W = np.arange(1,N,1); index_E = np.arange(N*(N+1)+1,(N+1)**2-1,1) 
        index_NW = 0; index_NE = (N+1)**2-(N+1); index_SW = N; index_SE = ((N+1)**2-1)
        index_exterior = np.sort(np.concatenate((index_N, index_S, index_E, index_W, np.array([index_NW, index_NE, index_SW, index_SE]))))
        index_interior = np.array(list(set(np.arange(0,(N+1)**2,1)).difference(set(index_exterior))))
          
    
    # Example for spdiags...
    #data = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    #diags = np.array([0, -1, 2])
    #spa.spdiags(data, diags, 4, 4).toarray()
    
    
    # Evolution
    for t in np.linspace(dt,T,M) : 
        
        t = round(t,2)
        
        reaction_term = f(prop_gametes, coef_gametes_couple)[0]
  
        for i in range(16) : 
            if dim == 1: 
                # Interior
                prop_gametes[i,1:-1] = la.spsolve(B_, B.dot(prop_gametes[i,1:-1]) + dt*reaction_term[i,1:-1])
                # Boundaries conditions : Neumann (null derivative)
                prop_gametes[i,0] = prop_gametes[i,1]  
                prop_gametes[i,-1] = prop_gametes[i,-2]
            if dim == 2:
                # Interior
                prop_gametes[i,index_interior] = la.spsolve(B_, B.dot(prop_gametes[i,index_interior]) + dt*reaction_term[i,index_interior])
                # Boundaries conditions : Neumann (null derivative)
                prop_gametes[i,index_N] = prop_gametes[i,index_N+1] 
                prop_gametes[i,index_S] = prop_gametes[i,index_S-1]
                prop_gametes[i,index_W] = prop_gametes[i,index_W+N+1] 
                prop_gametes[i,index_E] = prop_gametes[i,index_E-(N+1)] 
                prop_gametes[i,index_NW] = prop_gametes[i,index_NW+1] 
                prop_gametes[i,index_NE] = prop_gametes[i,index_NE+1]
                prop_gametes[i,index_SW] = prop_gametes[i,index_SW-1] 
                prop_gametes[i,index_SE] = prop_gametes[i,index_SE-1] 
        
        if CI != "equal" :
            # Position of the wave cd
            if dim == 1:
                wave_cd = np.dot((1-indexABCD)[2,:]*(1-indexABCD)[3,:],prop_gametes)
            if dim == 2:
                # we use the "middle" line of the spatial matrix to compute the speed.
                wave_cd = np.dot((1-indexABCD)[2,:]*(1-indexABCD)[3,:],prop_gametes[:,np.arange(index_W[(N-1)//2],index_E[(N-1)//2]+N+1,N+1)])
            # we recorde the position only if the cd wave is still in the environment. We do not recorde the 0 position since the treshold value of the wave might be outside the window.            
            if np.isin(True, wave_cd > treshold) and np.isin(True, wave_cd < 0.99) and np.where(wave_cd > treshold)[0][0] != 0 :  
                # first position where the wave is over the treshold value
                position = np.append(position, X[np.where(wave_cd > treshold)[0][0]])   
            elif np.isin(True, wave_cd < treshold) and np.isin(True, wave_cd > 0.01) and np.where(wave_cd < treshold)[0][0] != 0 :  
                # first position where the wave is under the treshold value
                position = np.append(position, X[np.where(wave_cd < treshold)[0][0]])
            # Compute the speed
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
                    nb_graph += 1
                    if dim == 1: graph_x(t, prop_gametes, X)
                    if dim == 2: graph_xy(t, prop_gametes); Z_list = graph_xy_contour(t, prop_gametes, Z_list, nb_graph)
                # time graph
            if t>=mod_t*nb_point and show_graph_t and dim :  
                nb_point += 1
                points = graph_t(X, t, prop_gametes, coef_gametes_couple, points, nb_point)
                
    
    # last graph
    if show_graph_end :   
        nb_graph += 1
        if dim == 1: graph_x(t, prop_gametes, X)
        if dim == 2: graph_xy(t, prop_gametes); Z_list = graph_xy_contour(t, prop_gametes, Z_list, nb_graph)
   
    # speed function of time 
    if CI != "equal" :
        if np.shape(position)[0] != 0 and show_graph_x :        
            fig, ax = plt.subplots()
            ax.plot(time, speed_fct_of_time)  
            ax.set(xlabel='Time', ylabel='Speed', ylim = [-0.25,0.5])             
            plt.hlines(y=0, color='dimgray', xmin=time[0], xmax=time[-1])
            ax.xaxis.label.set_size(label_size); ax.yaxis.label.set_size(label_size)   
            #ax.set_title("Speed of the wave C or D function of time", fontsize = title_size)
            plt.rc('legend', fontsize=legend_size)
            plt.grid() 
            if save_fig :
                save_fig_or_data(out_dir, fig, speed_fct_of_time, "speed_fct_time")  
                save_fig_or_data(out_dir, [], time, "time")  
                file = open(f"../outputs/{out_dir}/parameters.txt", "w") 
                file.write(f"Parameters : \nr = {r} \nsd = {sd} \nst = {st} \nsp = {sp} \n{diffusion} = {cst_value} \ngamma = {gamma} \nCI = {CI} \nT = {T} \nL = {L} \nM = {M} \nN = {N} \ntheta = {theta} \nf0 = {CI_prop_drive} \ndim = {dim}") 
                file.close()                          
            plt.show() 
        if np.shape(position)[0] == 0 :
            print('No wave')

    
    # speed_fct_of_time = np.loadtxt(f'../outputs/save2/wave/neg_abcd_2/speed_fct_time.txt')
    

    return(prop_gametes, time, speed_fct_of_time)  


############################### Graph and saving figures ######################################

# Proportion of allele in space at time t
def graph_x(t, prop_gametes, X):
        fig, ax = plt.subplots()
        if WT :
            ax.plot(X, prop_gametes[0,:], color='green', label='WT', linewidth=line_size)
        for i in range(3) :
            if [alleleA,alleleB,alleleCD][i] :
                lab = [r'$X_A$',r'$X_B$',r'$X_C/X_D$'][i]
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
        ax.set(xlabel='Space', ylabel='Frequency', ylim=[-0.02,1.02])   
        ax.xaxis.label.set_size(label_size); ax.yaxis.label.set_size(label_size)   
        ax.set_title(f't = {int(t)}', fontsize = title_size, loc='right')
        plt.rc('legend', fontsize=legend_size)
        ax.legend()  
        if save_fig :
            #num = str(int(t)//mod_x)
            #if len(num)==1: num = '0'+'0'+num
            #if len(num)==2: num = '0'+num
            #save_fig_or_data(out_dir, fig, [], f"{num}")
            save_fig_or_data(out_dir, fig, [], f"t_{t}")  
        plt.show() 
        
        

def graph_xy(t, prop_gametes):
    for allele_nb in np.arange(0,3):
        if [alleleA,alleleB,alleleCD][allele_nb] :
            allele_letter = ["A","B","C"][allele_nb]
            heatmap_values = np.resize(np.dot(indexABCD[allele_nb,:],prop_gametes),(N+1,N+1)).transpose()
            fig, ax = plt.subplots() 
            im = ax.imshow(heatmap_values,cmap='Blues', aspect='auto', vmin=0, vmax=1)  
            ax.figure.colorbar(im, ax=ax)   
            fig.suptitle(f"Allele {allele_letter} at time {np.round(t,2)}", fontsize=14)
            if save_fig :
                save_fig_or_data(out_dir, fig, [], f"{allele_letter}_t_{t}")
            plt.show() 
    
def graph_xy_contour(t, prop_gametes, Z_list, nb_graph):
    contour_threshold = 0.2
    for allele_nb in np.arange(0,3):
        if [alleleA,alleleB,alleleCD][allele_nb] :
            allele_letter = ["A","B","C"][allele_nb]
            heatmap_values = np.resize(np.dot(indexABCD[allele_nb,:],prop_gametes),(N+1,N+1)).transpose()
            fig, ax = plt.subplots() 
            g1 = lambda x,y: heatmap_values[int(y),int(x)] 
            g2 = np.vectorize(g1)
            x = np.linspace(0,heatmap_values.shape[1], 1001)[:-1]
            y = np.linspace(0,heatmap_values.shape[0], 1001)[:-1]
            X, Y= np.meshgrid(x,y)
            Z = g2(X,Y)  
            Z_list[nb_graph-1] = Z
            #x = np.linspace(0,heatmap_values.shape[1], heatmap_values.shape[1]*100)
            #y = np.linspace(0,heatmap_values.shape[0], heatmap_values.shape[0]*100)
            #X, Y= np.meshgrid(x[:-1],y[:-1])
            #Z = g2(X[:-1],Y[:-1])  
            #Z_list[nb_graph-1] = Z[:,1:]
            ax.set_aspect('equal', adjustable='box')
            #im = ax.imshow(heatmap_values,cmap='Blues', aspect='auto', vmin=0, vmax=1)  
            #ax.figure.colorbar(im, ax=ax)   
            #ax.contour(np.arange(N+1), np.arange(N+1), heatmap_values, levels=[0.4]) 
            for i in range(nb_graph) : 
                label = f'{int(mod_x*i)}'
                if i == nb_graph - 1 : label = f'{int(t)}'
                contour = ax.contour(Z_list[i], [contour_threshold], colors=col_pink[i], linewidths=[line_size], extent=[0-0.5, x[:-1].max()-0.5,0-0.5, y[:-1].max()-0.5])
                fmt = {}; fmt[contour_threshold] = label
                ax.clabel(contour, np.ones(1)*contour_threshold, inline=True, fmt=fmt, fontsize=10) 
            if save_fig :
                save_fig_or_data(out_dir, fig, [], f"contour_{allele_letter}_t_{t}")
            plt.show()   
    return(Z_list)
             

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
        
    
def speed_fct_of_spatial_step(step_min, step_max, nb_step, log_scale) :  
    step_record = np.array([])  
    speed_record = np.array([]) 
    if log_scale : step_record = np.logspace(-1, 1, num=nb_step)
    else : step_record = np.linspace(step_min, step_max, nb_step)
    for step in step_record :
        N = int(L/step)
        prop, time, speed = continuous_evolution(r,sd,st,sp,cst_value,gamma,T,L,M,N,theta,mod_x) 
        speed_record = np.append(speed_record, speed[-1]) 
        print("step :", step, "and speed :", speed[-1]) 
    fig, ax = plt.subplots()
    ax.plot(step_record, speed_record) 
    if log_scale : ax.set_xscale('log')
    ax.set(xlabel='Spatial step size', ylabel='Speed') 
    ax.xaxis.label.set_size(label_size); ax.yaxis.label.set_size(label_size)   
    #ax.set_title('Speed of the wave C or D function of the spatial steps', fontsize = title_size, loc='right')
    plt.grid()   
    if save_fig :
        if log_scale : end_title = "_log"
        else : end_title = ""
        save_fig_or_data(out_dir, fig, [], f"towards_discretization{end_title}")
        save_fig_or_data(out_dir, [], step_record, f"step_record{end_title}")
        save_fig_or_data(out_dir, [], speed_record, f"speed_record{end_title}")
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
sp = 0.1   # 0.1 pos, 0.5 neg

# Coefficents for the reaction term
coef_gametes_couple = coef(sd,sp,st,gamma,r)

# Numerical parameters
dim = 2         # number of spatial dimensions (1 or 2)
T = 400         # final time
L = 80          # length of the spatial domain
M = T*6        # number of time steps
N = int(L*(1/1.5))         # number of spatial steps

theta = 0.5      # discretization in space : theta = 0.5 for Crank Nicholson
                 # theta = 0 for Euler Explicit, theta = 1 for Euler Implicit   
                 
# Initial repartition
CI = "square_abcd"           # "equal"  "left_abcd" "left_cd" "center_abcd" "center_cd"   and 2D only : " square_abcd" "square_cd"
CI_prop_drive = 1            # Drive initial proportion in "ABCD_global"  "ABCD_left"  "ABCD_center" 
CI_lenght = int((L/10)*(N/L))      # /!\ should be < N. For "center_abcd" and "center_cd", lenght of the initial drive condition in the center, in number of spatial steps.

            
# Diffusion rate: constant or depending on m, dx and dt
diffusion = 'cst dif'     # cst dif or cst m
cst_value = 0.2           # value of the constant diffusion rate or value of the constant migration rate, depending on the line before 

# Graphics
show_graph_x = True      # whether to show the graph in space or not
show_graph_ini = True   # whether to show the allele graph or not at time t=0
show_graph_end = True    # whether to show the allele graph or not at time t=T

show_graph_t = False      # whether to show the graph in time or not
graph_t_type = "ABCD"     # "fig4" or "ABCD"
focus_x = 20              # where to look, on the x-axis (0 = center)

mod_x = T//4             # time at which to draw allele graphics
mod_t = T//50             # time points used to draw the graph in time
save_fig = True       # save the figures

# Which alleles to show in the graph
WT = False             
alleleA = False; alleleB = alleleA; alleleCD = True
checkab = False; ab = checkab; AbaB = ab; AB = ab 
checkcd = False; cd = checkcd; CdcD = cd; CD = cd

# To compute the speed function of spatial step size
show_speed_fct_of_spatial_step = False
step_min = 0.1
step_max = 3 
nb_step = 200
log_scale = False

# Where to store the outputs
out_dir = f"cst_dim_{dim}_r_{r}_gam_{gamma}_sd_{sd}_st_{st}_sp_{sp}_{diffusion}_{cst_value}_{CI}"


############################### Evolution ########################################
 
if show_speed_fct_of_spatial_step :
    show_graph_x = False; show_graph_ini = False; show_graph_end = False
    speed_fct_of_spatial_step(step_min, step_max, nb_step, log_scale)
else : 
    prop, time, speed_fct_of_time = continuous_evolution(r,sd,st,sp,cst_value,gamma,T,L,M,N,theta,mod_x) 
    print("Speed :",speed_fct_of_time[-1])
    

############################### Control ########################################
  
# Diffusion rate
if diffusion == 'cst dif': dif = cst_value
if diffusion == 'cst m':  dt = T/M; dx = L/N; m = cst_value; dif = (m*dx**2)/(2*dt) 

# Check ab
if checkab : 
    ab_,aB_,Ab_,AB_ = continuous_evolution_ab(r,sd,dif,gamma,T,L,M,N,theta,CI,show_graph_ini,show_graph_end,mod_x,show_graph_x,ab,AbaB,AB)
    print('ab check :',(abs(ab_ - np.sum(prop[0:4], axis=0)) < 0.001)[0])
    print('aB check :',(abs(aB_ - np.sum(prop[4:8], axis=0)) < 0.001)[0])
    print('Ab check :',(abs(Ab_ - np.sum(prop[8:12], axis=0)) < 0.001)[0])
    print('AB check :',(abs(AB_ - np.sum(prop[12:16], axis=0)) < 0.001)[0])

# Check cd
if checkcd :    
    cd_,cD_,Cd_,CD_ = continuous_evolution_cd(r,sp,st,dif,gamma,T,L,M,N,theta,CI,show_graph_ini,show_graph_end,mod_x,show_graph_x,cd,CdcD,CD)
    print('cd check :',(abs(cd_ - np.sum(prop[(0,4,8,12),:], axis=0)) < 0.001)[0])
    print('cD check :',(abs(cD_ - np.sum(prop[(1,5,9,13),:], axis=0)) < 0.001)[0])
    print('Cd check :',(abs(Cd_ - np.sum(prop[(2,6,10,14),:], axis=0)) < 0.001)[0])
    print('Cd check :',(abs(CD_ - np.sum(prop[(3,7,11,15),:], axis=0)) < 0.001)[0])

############################### Print parameters ########################################

print('\nr = ',r,' sd =', sd, diffusion, cst_value,' gamma =',gamma, ' CI =', CI)
print('T =',T,' L =',L,' M =',M,' N =',N,' theta =',theta, ' f0 =', CI_prop_drive) 

