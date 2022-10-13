#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 09:49:47 2021
lena.klay@sorbonne-universite.fr
"""


############################## Libraries ############################################

import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg  as la


num = numero
########################## Graph parameters #######################################

title_size = 15
label_size = 17
legend_size = 12
line_size = 3

# Colors used
col_pink = ['indigo', 'purple', 'darkmagenta', 'm', 'mediumvioletred', 'crimson', 'deeppink', 'hotpink', 'lightpink', 'pink' ]    
col_blue = ['navy', 'blue','royalblue', 'cornflowerblue', 'lightskyblue']    

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
   
# Dictionnary to find the index of the allele in locusABCD   
dico_locusABCD = {}
for i in range(16) : 
    dico_locusABCD[locusABCD[i]] = i 

# Fitness : produce a scalair that indicate the fitness of the gametes
def fitness(gametes_produced,sd,sp,st) : 
    genome = gametes_produced[0]+gametes_produced[1]
    nb_sd = genome.count('A') + genome.count('B')
    nb_sp = int(('C' in genome)|('D' in genome))
    nb_st = int(('C' in genome)^('D' in genome))
    return(np.round((1-sd)**nb_sd*(1-sp)**nb_sp*(1-st)**nb_st,10))    


def recombinaison_and_homing(gametes_produced,r,gamma) : 
# Recombinaison : indicates the gametes produced thanks to the recombinaison and their proportions regarding r.
    if gametes_produced[0] == gametes_produced[1] : 
        gametes_produced = [gametes_produced[0]]
        path = np.ones(1)
    else :
        path = np.ones(2)
        for cut in range(1,4) :
            for g in range(0,len(gametes_produced),2) :                
                 if gametes_produced[g][0:cut] != gametes_produced[g+1][0:cut] and gametes_produced[g][cut:4] != gametes_produced[g+1][cut:4] :
                    # new1, new2 : New haplotypes after one event of recombination (one cut)
                    new1 = gametes_produced[g][0:cut]+gametes_produced[g+1][cut:4]
                    new2 = gametes_produced[g+1][0:cut]+gametes_produced[g][cut:4]
                    # Append the new haplotypes at the list
                    gametes_produced.append(new1); gametes_produced.append(new2)
                    # Ajust the probability
                    path = np.append(path, path[g:g+2]*r)
                    path[g:g+2] = path[g:g+2]*(1-r)
                        
        
# Homing : indicates the gametes produced thanks to the drive conversion and their proportions regarding gamma. 
# The initial gametes are taken from the previous step : the recombination.
        for g in range(0,len(gametes_produced),2) : 
            genome = gametes_produced[g]+gametes_produced[g+1]
            if ('A' in genome) and ('B' in genome) and ('b' in genome):
                new = genome.replace('b','B')
                gametes_produced.append(new[0:4]); gametes_produced.append(new[4:8])
                path = np.append(path, path[g:g+2]*gamma)
                path[g:g+2] = path[g:g+2]*(1-gamma)
        for g in range(0,len(gametes_produced),2) :
            genome = gametes_produced[g]+gametes_produced[g+1]
            if ('B' in genome) and ('C' in genome) and ('c' in genome):
                new = genome.replace('c','C')
                gametes_produced.append(new[0:4]); gametes_produced.append(new[4:8])
                path = np.append(path, path[g:g+2]*gamma)
                path[g:g+2] = path[g:g+2]*(1-gamma)
            if ('B' in genome) and ('D' in genome) and ('d' in genome):
                new = genome.replace('d','D')
                gametes_produced.append(new[0:4]); gametes_produced.append(new[4:8])
                path = np.append(path, path[g:g+2]*gamma)
                path[g:g+2] = path[g:g+2]*(1-gamma)
                
# Place items in the same order than locusABCD
    tidy_path = np.zeros(16)
    for l in range(len(gametes_produced)):
        index_dico = dico_locusABCD[gametes_produced[l]]
        tidy_path[index_dico] += path[l]            
    return(tidy_path) 
    
    

# Set the coefficients regarding the parameters
def coef(sd,sp,st,gamma,r):
    coef_gametes_couple = np.zeros((16,np.sum(np.arange(17))))
    l = 0
    # [i,j] defines the indexes for the couple of gametes. There are 16+15+...+1 couples [i,j].
    for i in range(16) :
        for j in range(i,16) :
            coef_gametes_couple[:,l] = fitness([locusABCD[i],locusABCD[j]],sd,sp,st)*recombinaison_and_homing([locusABCD[i],locusABCD[j]],r,gamma)
            l += 1
    return(coef_gametes_couple)

            
# Set the equations regarding the coefficients and the previous proportion of gametes
def f(prop_gametes, coef_gametes_couple) : 
    # proportion of each couple of gametes (we erase the inverse copy of each couple, so the lenght of the vector is 16+15+...+1 )
    prop_gametes_couple = np.zeros((np.sum(np.arange(17)), len(prop_gametes[0,:])))
    l = 0
    for i in range(16) : 
        for j in range(i,16) : 
            prop_gametes_couple[l,:] = prop_gametes[i,:]*prop_gametes[j,:]
            l += 1
    # Mean fitness
    mean_fitness = np.sum(np.dot(coef_gametes_couple, prop_gametes_couple),axis=0) 
    # Reaction term (it is a vector, each coordinate correspond to a gamete, in the 'locusABCD order') 
    reaction_term = np.dot(coef_gametes_couple, prop_gametes_couple)/np.sum(np.dot(coef_gametes_couple, prop_gametes_couple),axis=0) - prop_gametes
    # Minus prop gametes because we are now in continuous space
    return(reaction_term, mean_fitness)
    
    

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

    return(prop_gametes, time, speed_fct_of_time)  


############################### Graph and saving figures ######################################
        
    
def speed_fct_of_spatial_step(step_min, step_max, nb_step, log_scale) : 
    if log_scale : step_record = np.logspace(-1, 1, num=nb_step)
    else : step_record = np.linspace(step_min, step_max, nb_step)
    step = step_record[num]
    N = int(L/step)
    prop, time, speed = continuous_evolution(r,sd,st,sp,cst_value,gamma,T,L,M,N,theta,mod_x) 
    print("step :", step, "and speed :", speed[-1]) 
    #np.savetxt(f"5_{num}_step.txt", np.ones(1)*step)   
    np.savetxt(f"5_{num}_speed.txt", np.ones(1)*speed[-1])  


############################### Parameters ######################################

# Recombination rate 
r = 0.5
# Conversion rate (probability of a successfull gene drive conversion)
gamma = 0.9

# Fitness disadvantage
sd = 0.02
st = 0.9
sp = 0.5   # 0.1 pos, 0.5 neg

# Coefficents for the reaction term
coef_gametes_couple = coef(sd,sp,st,gamma,r)

# Numerical parameters
dim = 1         # number of spatial dimensions (1 or 2)
T = 4000         # final time
L = 800          # length of the spatial domain
M = T*10        # number of time steps

theta = 0.5      # discretization in space : theta = 0.5 for Crank Nicholson
                 # theta = 0 for Euler Explicit, theta = 1 for Euler Implicit   
                 
# Initial repartition
CI = "left_cd"           # "equal"  "left_abcd" "left_cd" "center_abcd" "center_cd"   and 2D only : " square_abcd" "square_cd"
CI_prop_drive = 1            # Drive initial proportion in "ABCD_global"  "ABCD_left"  "ABCD_center" 

            
# Diffusion rate: constant or depending on m, dx and dt
diffusion = 'cst dif'     # cst dif or cst m
cst_value = 0.2           # value of the constant diffusion rate or value of the constant migration rate, depending on the line before 

# Graphics
show_graph_x = False      # whether to show the graph in space or not
show_graph_ini = False   # whether to show the allele graph or not at time t=0
show_graph_end = False    # whether to show the allele graph or not at time t=T

show_graph_t = False      # whether to show the graph in time or not
graph_t_type = "ABCD"     # "fig4" or "ABCD"
focus_x = 20              # where to look, on the x-axis (0 = center)

mod_x = T//4             # time at which to draw allele graphics
mod_t = T//50             # time points used to draw the graph in time

# Which alleles to show in the graph
WT = False             
alleleA = False; alleleB = alleleA; alleleCD = True
checkab = False; ab = checkab; AbaB = ab; AB = ab 
checkcd = False; cd = checkcd; CdcD = cd; CD = cd

# To compute the speed function of spatial step size
step_min = 0.1
step_max = 3
nb_step = 200
log_scale = False


############################### Evolution ########################################
 
show_graph_x = False; show_graph_ini = False; show_graph_end = False
speed_fct_of_spatial_step(step_min, step_max, nb_step, log_scale)

############################### Print parameters ########################################

print('\n num = ', num, 'r = ',r,' sd =', sd, diffusion, cst_value,' gamma =',gamma)
print(' CI =', CI, 'T =',T,' L =',L,' M =',M,' theta =',theta, ' f0 =', CI_prop_drive) 

