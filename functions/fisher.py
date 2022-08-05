#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 17:11:11 2022

@author: lena
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 09:49:47 2021
lena.klay@sorbonne-universite.fr
"""

############################## Libraries ############################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg  as la
import os
# Change font to serif
plt.rcParams.update({'font.family':'serif'})

########################## Graph parameters #######################################

title_size = 15
label_size = 17
legend_size = 12
line_size = 3



def graph(X, t, P):
        fig, ax = plt.subplots()
        ax.plot(X, P, color='green', label='WT', linewidth=line_size)
        ax.grid()      
        ax.set(xlabel='Space', ylabel='Frequency', ylim=[-0.02,1.02])   
        ax.xaxis.label.set_size(label_size); ax.yaxis.label.set_size(label_size)   
        ax.set_title(f't = {int(t)}', fontsize = title_size, loc='right')
        plt.rc('legend', fontsize=legend_size)
        ax.legend()  
        plt.show() 


def evolution(r,dif,T,L,M,N,theta,mod_x):   
   
    # Steps
    dt = T/M    # time
    dx = L/N    # space

    # Spatial domain (1D)
    X = np.linspace(0,N,N+1)*dx  
        
    # Initialization             
    P = np.zeros(N+1); P[0:N//2] = 1
    
    if show_graph :
        graph(X, 0, P)
    nb_graph = 1
        
    # Matrix
    C0 = -2*np.ones(N+1); C0[0]=C0[0]+1; C0[-1]=C0[-1]+1               
    C1 = np.ones(N+1) 
    A = sp.spdiags([C1,C0,C1],[-1,0,1], N+1, N+1)              # 1D discrete Laplacian with Neumann boundary conditions (derivative=0)    
    B = sp.identity(N+1)+((1-theta)*dif*dt/dx**2)*A            # Matrix for the explicit side of the Crank Nicholson scheme  
    B_ = sp.identity(N+1)-(theta*dif*dt/dx**2)*A               # Matrix for the implicit side of the Crank Nicholson scheme  
       

    # Speed of the wave, function of time
    position = np.array([])             # list containing the first position where the proportion of wild alleles is higher than the treshold value.
    time = np.array([])                 # list containing the time at which we calculate the speed.
    speed_fct_of_time = np.array([])    # list containing the speed corresponding to the time list.
    treshold = 0.5                      # indicates which position of the wave we follow to compute the speed (first position where the C-D wave come under the threshold)    
      
    # Example for spdiags...
    #data = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    #diags = np.array([0, -1, 2])
    #spa.spdiags(data, diags, 4, 4).toarray()
   
    # Evolution
    for t in np.linspace(dt,T,M) :        
        t = round(t,2)
        f = r*P*(1-P)
        P = la.spsolve(B_, B.dot(P) + dt*f)    
                                  
        # we recorde the position only if the wave is still in the environment. We do not recorde the 0 position since the treshold value of the wave might be outside the window.            
        if np.isin(True, P > treshold) and np.isin(True, P < 0.99) and np.where(P > treshold)[0][0] != 0 :  
            # first position where the wave is over the treshold value
            position = np.append(position, X[np.where(P > treshold)[0][0]])   
        elif np.isin(True, P < treshold) and np.isin(True, P > 0.01) and np.where(P < treshold)[0][0] != 0 :  
            # first position where the wave is under the treshold value
            position = np.append(position, X[np.where(P < treshold)[0][0]])
        # Compute the speed
        if len(position) > 20 : 
            time = np.append(time, t)
            speed_fct_of_time = np.append(speed_fct_of_time, np.mean(np.diff(position[int(4*len(position)/5):len(position)]))/dt)
        # if the treshold value of the wave is outside the window, stop the simulation  
        if not(np.isin(False, P>treshold) and np.isin(False, P<treshold) ) :
            print("t =",t)
            break 
            
        # spatial graph  
        if t>=mod_x*nb_graph :  
            if show_graph :
                graph(X, t, P)
            nb_graph += 1

   
    # speed function of time 
    if np.shape(position)[0] != 0 :        
        fig, ax = plt.subplots()
        ax.plot(time, speed_fct_of_time) 
        ax.set(xlabel='Time', ylabel='Speed')             
        ax.xaxis.label.set_size(label_size); ax.yaxis.label.set_size(label_size)   
        #ax.set_title("Speed of the wave C or D function of time", fontsize = title_size)
        plt.rc('legend', fontsize=legend_size)
        plt.grid() 
        if save_fig :
            save_fig_or_data(out_dir, fig, speed_fct_of_time, "speed_fct_time")  
            save_fig_or_data(out_dir, [], time, "time")                         
        plt.show() 
    if np.shape(position)[0] == 0 :
        print('No wave')

    return(P, time, speed_fct_of_time)  

 
def speed_fct_of_spatial_step(step_min, step_max, nb_step, log_scale) :  
    step_record = np.array([])  
    speed_record = np.array([]) 
    if log_scale : step_range = np.logspace(-1, 1, num=nb_step)
    else : step_range = np.linspace(step_min, step_max, nb_step)
    for step in step_range :
        N = int(L/step)
        step_record = np.append(step_record, step) 
        prop, time, speed = evolution(r,dif,T,L,M,N,theta,mod_x)
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

# Growth rate
r = 1

# Diffusion          
dif = 1

# Numerical parameters
T = 100         # final time
L = 400          # length of the spatial domain
M = T*40         # number of time steps
N = L*10         # number of spatial steps
theta = 0.5      # discretization in space : theta = 0.5 for Crank Nicholson
                 # theta = 0 for Euler Explicit, theta = 1 for Euler Implicit           

# Show and save
show_graph = True  
mod_x = T//4              # time at which to draw allele graphics
save_fig = True       # save the figures

# To compute the speed function of spatial step size
show_speed_fct_of_spatial_step = True
step_min = 0.01
step_max = 10 
nb_step = 20
log_scale = False

# Where to store the outputs
out_dir = f"fisher_test"


############################### Evolution ########################################
 
if show_speed_fct_of_spatial_step :
    show_graph = False
    speed_fct_of_spatial_step(step_min, step_max, nb_step, log_scale)
else : 
    prop, time, speed_fct_of_time = evolution(r,dif,T,L,M,N,theta,mod_x)
    print("Speed :",speed_fct_of_time[-1])
    
