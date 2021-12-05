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

# Fitness : produce a scalair that indicate the fitness of the gamete(s)
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
    prop_gametes_couple = np.zeros((np.sum(np.arange(17)),N+1))
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
       
    nb_graph = 1
    if show_graph_a and show_graph_ini :
        graph_a(X, 0, prop_gametes)
        
    nb_point = 1
    if show_graph_t : 
        points = np.zeros((5,int(T/mod_t)+1))
        points = graph_t(X, 0, prop_gametes, coef_gametes_couple, points, 0)
        
        
    # Matrix
    C0 = -2*np.ones(N+1); C0[0]=C0[0]+1; C0[-1]=C0[-1]+1               
    C1 = np.ones(N+1) 
    A = spa.spdiags([C1,C0,C1],[-1,0,1], N+1, N+1)          # 1D discrete Laplacian with Neumann boundary conditions (derivative=0)  
    
    B = spa.identity(N+1)+((1-theta)*dt/dx**2)*A            # Matrix for the explicit side of the Crank Nicholson scheme  
    B_ = spa.identity(N+1)-(theta*dt/dx**2)*A               # Matrix for the implicit side of the Crank Nicholson scheme  

    # Evolution
    for t in np.linspace(dt,T,M) : 
        
        t = round(t,2)
        
        reaction_term = f(prop_gametes, coef_gametes_couple)[0]
  
        for i in range(16) : 
            prop_gametes[i,:] = la.spsolve(B_, B.dot(prop_gametes[i,:]) + dt*reaction_term[i,:])

        if t>=mod_a*nb_graph and show_graph_a :  
            graph_a(X, t, prop_gametes)
            nb_graph += 1
            
        if t>=mod_t*nb_point and show_graph_t :  
            points = graph_t(X, t, prop_gametes, coef_gametes_couple, points, nb_point)
            nb_point += 1
                  
    return(prop_gametes)  



############################### Graph and saving figures ######################################
    

def graph_a(X, t, prop_gametes):
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
        ax.set(xlabel='Time', ylabel='Frequency', ylim=[-0.02,1.02], title = f'Evolution : f0={CI_prop_drive}, time = {int(t)}')   
        ax.legend()  
        if save_fig :
            save_figure(t, "graph_allele", r, gamma, sd, st, sp, dif, CI, CI_prop_drive) 
        plt.show() 
        
        
def graph_t(X, t, prop_gametes, coef_gametes_couple, values, nb_point):
    sumABCD = np.dot(indexABCD, prop_gametes)
    mean_fitness = f(prop_gametes, coef_gametes_couple)[1]   
    values[0,nb_point] = t 
      
    if graph_t_type == 'ABCD' : 
        lab = ['A','B','C','D']
        col = ['red','orange','yellowgreen','cornflowerblue']
        for i in range(1,5) :
            values[i,nb_point]=sumABCD[i-1,N//2+position_t]   
        
    if graph_t_type == 'fig4' : 
        lab = ['B','C or D','1-W']
        col = ['orange','cornflowerblue','black']
        values[1,nb_point] = sumABCD[1,N//2+position_t]      
        values[2,nb_point] = sumABCD[2,N//2+position_t]
        values[3,nb_point] = 1-mean_fitness[N//2+position_t]  
    
    if nb_point != np.shape(values)[1]-1 : 
        return(values)
    else : 
        fig, ax = plt.subplots()
        for i in range(len(lab)) : 
            ax.plot(values[0,:], values[i+1,:], color=col[i], label=lab[i], linewidth=3)
        ax.grid()      
        ax.set(xlabel='Time', ylabel='Frequency', ylim=[-0.02,1.02], title = f'Evolution : f0={CI_prop_drive}, position = {position_t}, time = {int(t)}')   
        ax.legend()  
        if save_fig :
            save_figure(t, "graph_time", r, gamma, sd, st, sp, dif, CI, CI_prop_drive) 
        plt.show() 
        
        
def save_figure(t, graph_type, r, gamma, sd, st, sp, dif, CI, CI_prop_drive)   :           
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
         
       
        
############################### CONTROLE AB ########################################

def graph_ab(X,t,ab_,aB_,Ab_,AB_):
    
    fig, ax = plt.subplots()
    if ab :
        ax.plot(X, ab_, label='ab', linewidth=3, color = 'yellowgreen')
    if AbaB : 
        ax.plot(X, aB_+Ab_, label='aB+Ab', linewidth=3, color = 'skyblue')
    if AB : 
        ax.plot(X, AB_, label='AB', linewidth=3, color = 'blue')
    ax.grid()      
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    ax.set(xlabel='Space', ylabel='Frequency', ylim=[-0.1,1.1], title = f'Contrôle ab : sd = {sd}, time = {int(t)}')   
    ax.legend()
    plt.show() 
    
    
def continuous_evolution_ab(r,sd,dif,gamma,T,L,M,N,theta):   
   
    # Steps
    dt = T/M    # time
    dx = L/N    # spatial
    
    X = np.linspace(0,N,N+1)*dx
    
    # Initialization (frequency vector : abcd  abcD  abCd  abCD  aBcd  aBcD  aBCd  aBCD  Abcd  AbcD  AbCd  AbCD  ABcd  ABcD  ABCd  ABCD)   
    if CI == "equal" : 
        ab = np.ones(N+1)*(1/4); aB = np.ones(N+1)*(1/4); Ab = np.ones(N+1)*(1/4); AB = np.ones(N+1)*(1/4)  
            
    # Matrix
    C0 = -2*np.ones(N+1); C0[0]=C0[0]+1; C0[-1]=C0[-1]+1               
    C1 = np.ones(N+1) 
    A = spa.spdiags([C1,C0,C1],[-1,0,1], N+1, N+1)          # 1D discrete Laplacian with Neumann boundary conditions (derivative=0)  
    
    B = spa.identity(N+1)+((1-theta)*dt/dx**2)*A            # Matrix for the explicit side of the Crank Nicholson scheme  
    B_ = spa.identity(N+1)-(theta*dt/dx**2)*A               # Matrix for the implicit side of the Crank Nicholson scheme
    
    print("\nCHECK AB BELOW")
    nb_graph = 1
    if show_graph_a and show_graph_ini :
        graph_ab(X,0,ab,aB,Ab,AB)
    
    # Evolution
    for t in np.linspace(dt,T,M) : 
        
        t = round(t,2)
        
        #mean_fitness = ab**2 + 2*ab*aB*(1-sd) + 2*ab*Ab*(1-sd) + 2*ab*AB*(1-sd)**2 + aB**2*(1-sd)**2 + 2*aB*Ab*(1-sd)**2 + 2*aB*AB*(1-sd)**3 + Ab**2*(1-sd)**2 + 2*Ab*AB*(1-sd)**3 + AB**2*(1-sd)**4
        
        fab = ab**2 + aB*Ab*(1-sd)**2*(1-gamma)*r + ab*aB*(1-sd) + ab*Ab*(1-sd) + ab*AB*(1-sd)**2*(1-gamma)*(1-r) 
        faB = aB**2*(1-sd)**2 + aB*Ab*(1-sd)**2*(1-r*(1-gamma)) + aB*AB*(1-sd)**3 + ab*aB*(1-sd) + ab*AB*(1-sd)**2*(1-(1-r)*(1-gamma)) 
        fAb = Ab**2*(1-sd)**2 + aB*Ab*(1-sd)**2*(1-gamma)*(1-r) + Ab*AB*(1-sd)**3*(1-gamma) + ab*Ab*(1-sd) + ab*AB*(1-sd)**2*(1-gamma)*r
        fAB = AB*ab*(1-sd)**2*(1-r*(1-gamma)) + AB*Ab*(1-sd)**3*(1+gamma) + AB**2*(1-sd)**4 + aB*Ab*(1-sd)**2*(r+gamma*(1-r)) + aB*AB*(1-sd)**3
        # Version de l'article
        # faB = (aB**2*(1-sd)**2 + aB*Ab*(1-sd)**2*(1-r*(1-gamma)) + aB*AB*(1-sd)**3 + ab*aB*(1-sd) + ab*AB*(1-sd)**2*(1-r*(1-gamma)) )/mean_fitness -aB
          
        mean_fitness = fab + faB + fAb + fAB
        
        fab =  fab /mean_fitness - ab 
        faB =  faB /mean_fitness - aB  
        fAb =  fAb /mean_fitness - Ab   
        fAB =  fAB /mean_fitness - AB    
        
        #print((fab+ab+fAb+Ab+faB+aB+fAB+AB)[0])
         
        ab = la.spsolve(B_, B.dot(ab) + dt*fab)
        aB = la.spsolve(B_, B.dot(aB) + dt*faB)
        Ab = la.spsolve(B_, B.dot(Ab) + dt*fAb)
        AB = la.spsolve(B_, B.dot(AB) + dt*fAB)
        
        #print((fab+ab+fAb+Ab+faB+aB+fAB+AB)[0])
        
        if t>=mod_a*nb_graph and show_graph_a :  
            graph_ab(X,t,ab,aB,Ab,AB)
            nb_graph += 1
                        
    return(ab,aB,Ab,AB) 



############################### CONTROLE CD ########################################

def graph_cd(X,t,cd_,cD_,Cd_,CD_):

    fig, ax = plt.subplots()
    if cd :
        ax.plot(X, cd_, label='cd', linewidth=3, color = 'yellowgreen')
    if CdcD : 
        ax.plot(X, cD_+Cd_, label='cD+Cd', linewidth=3, color = 'skyblue')
    if CD : 
        ax.plot(X, CD_, label='CD', linewidth=3, color = 'blue')
    ax.grid()      
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    ax.set(xlabel='Space', ylabel='Frequency', ylim=[-0.1,1.1], title = f'Contrôle cd : sp = {sp}, st={st}, time = {int(t)}')   
    ax.legend()
    plt.show() 

def continuous_evolution_cd(r,sp,st,dif,gamma,T,L,M,N,theta):   
   
    # Steps
    dt = T/M    # time
    dx = L/N    # spatial
    
    X = np.linspace(0,N,N+1)*dx
    
    # Initialization (frequency vector : abcd  abcD  abCd  abCD  aBcd  aBcD  aBCd  aBCD  Abcd  AbcD  AbCd  AbCD  ABcd  ABcD  ABCd  ABCD)   
    if CI == "equal" : 
        cd = np.ones(N+1)*(1/4); cD = np.ones(N+1)*(1/4); Cd = np.ones(N+1)*(1/4); CD = np.ones(N+1)*(1/4)  
              
    # Matrix
    C0 = -2*np.ones(N+1); C0[0]=C0[0]+1; C0[-1]=C0[-1]+1               
    C1 = np.ones(N+1) 
    A = spa.spdiags([C1,C0,C1],[-1,0,1], N+1, N+1)          # 1D discrete Laplacian with Neumann boundary conditions (derivative=0)  
    
    B = spa.identity(N+1)+((1-theta)*dt/dx**2)*A            # Matrix for the explicit side of the Crank Nicholson scheme  
    B_ = spa.identity(N+1)-(theta*dt/dx**2)*A               # Matrix for the implicit side of the Crank Nicholson scheme  

    print("\nCHECK CD BELOW")
    nb_graph = 1
    if show_graph_a and show_graph_ini :
        graph_cd(X,0,cd,cD,Cd,CD)
        
    # Evolution
    for t in np.linspace(dt,T,M) : 
        
        t = round(t,2)
        
        #mean_fitness = cd**2 + 2*cd*cD*(1-st)*(1-sp) + 2*cd*Cd*(1-st)*(1-sp) + 2*cd*CD*(1-sp) + cD**2*(1-st)*(1-sp) + 2*cD*Cd*(1-sp) + 2*cD*CD*(1-sp) + Cd**2*(1-st)*(1-sp) + 2*Cd*CD*(1-sp) + CD**2*(1-sp)
        
        fcd = cd**2 + cd*cD*(1-sp)*(1-st) + cd*Cd*(1-sp)*(1-st) + cD*Cd*(1-sp)*r + cd*CD*(1-sp)*(1-r)
        fcD = cd*cD*(1-sp)*(1-st) + cD**2*(1-sp)*(1-st) + cD*Cd*(1-sp)*(1-r) + cd*CD*(1-sp)*r + cD*CD*(1-sp)
        fCd = cd*Cd*(1-sp)*(1-st) + Cd**2*(1-sp)*(1-st) + cD*Cd*(1-sp)*(1-r) + cd*CD*(1-sp)*r + Cd*CD*(1-sp)
        fCD = cd*CD*(1-sp)*(1-r) + cD*CD*(1-sp) + Cd*CD*(1-sp) + CD**2*(1-sp) + cD*Cd*(1-sp)*r      
      
        mean_fitness = fcd + fcD + fCd + fCD
        
        fcd =  fcd /mean_fitness - cd 
        fcD =  fcD /mean_fitness - cD  
        fCd =  fCd /mean_fitness - Cd   
        fCD =  fCD /mean_fitness - CD
        
        cd = la.spsolve(B_, B.dot(cd) + dt*fcd)
        cD = la.spsolve(B_, B.dot(cD) + dt*fcD)
        Cd = la.spsolve(B_, B.dot(Cd) + dt*fCd)
        CD = la.spsolve(B_, B.dot(CD) + dt*fCD)
             
        if t>=mod_a*nb_graph and show_graph_a :  
            graph_cd(X,t,cd,cD,Cd,CD)
            nb_graph += 1
            
    return(cd,cD,Cd,CD)






############################### Parameters ######################################

# Problem for ab with r = 1, gamma = 0, sd = 0.2, st = 0, sp = 0
# Problem for ab with r = 0, gamma = 1, sd = 0.2, st = 0, sp = 0
        
# Egalement pour r=0 et gamma = 0.4

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
dif = 0.1

# Initial repartition
CI = "ABCD_center"     # "equal"  "ABCD_global"  "ABCD_left"  "ABCD_center" 
CI_prop_drive = 0.12   # Drive initial proportion in "ABCD global", "ABCD left" and "ABCD center"  
CI_lenght = 20         # for "ABCD center", lenght of the initial drive condition in the center (CI_lenght divisible by N and 2) 

# Numerical parameters
T = 2000          # final time
L = 1000         # length of the spatial domain
M = T*4         # number of time steps
N = L           # number of spatial steps
theta = 0.5     # discretization in space : theta = 0.5 for Crank Nicholson
                # theta = 0 for Euler Explicit, theta = 1 for Euler Implicit  

# Graphics
show_graph_a = True       # whether to show the allele graph or not
show_graph_ini = True     # whether to show the allele graph or not at time t=0
show_graph_t = False        # whether to show the graph in time or not
mod_a = T/10                # time at which to draw allele graphics
mod_t = T/50               # time points used to draw the graph in time
save_fig = True        # save the figures (.pdf)

# Which alleles to show in the allele graph
WT = False             
alleleA = True; alleleB = alleleA; alleleCD = alleleA
checkab = False; ab = checkab; AbaB = ab; AB = ab 
checkcd = False; cd = checkcd; CdcD = cd; CD = cd

# What kind of graph in time
graph_t_type = "ABCD"         # "fig4" or "ABCD"
position_t = 20               # where to look, on the x-axis (0 = center)

############################### Evolution ########################################
   

prop = continuous_evolution(r,sd,st,sp,dif,gamma,T,L,M,N,theta,mod_a) 

# Check ab
if checkab : 
    ab_,aB_,Ab_,AB_ = continuous_evolution_ab(r,sd,dif,gamma,T,L,M,N,theta)

    print('ab check :',(abs(ab_ - np.sum(prop[0:4], axis=0)) < 0.001)[0])
    print('aB check :',(abs(aB_ - np.sum(prop[4:8], axis=0)) < 0.001)[0])
    print('Ab check :',(abs(Ab_ - np.sum(prop[8:12], axis=0)) < 0.001)[0])
    print('AB check :',(abs(AB_ - np.sum(prop[12:16], axis=0)) < 0.001)[0])


# Check cd
if checkcd :    
    cd_,cD_,Cd_,CD_ = continuous_evolution_cd(r,sp,st,dif,gamma,T,L,M,N,theta)

    print('cd check :',(abs(cd_ - np.sum(prop[(0,4,8,12),:], axis=0)) < 0.001)[0])
    print('cD check :',(abs(cD_ - np.sum(prop[(1,5,9,13),:], axis=0)) < 0.001)[0])
    print('Cd check :',(abs(Cd_ - np.sum(prop[(2,6,10,14),:], axis=0)) < 0.001)[0])
    print('Cd check :',(abs(CD_ - np.sum(prop[(3,7,11,15),:], axis=0)) < 0.001)[0])



############################### Print parameters ########################################

print('\nr = ',r,' sd =', sd,' dif =',dif,' gamma =',gamma, ' CI =', CI)
print('T =',T,' L =',L,' M =',M,' N =',N,' theta =',theta, ' f0 =', CI_prop_drive)


