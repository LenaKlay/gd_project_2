#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:49:37 2022

@author: lena
"""

############################## Libraries ############################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
import scipy.sparse.linalg  as la

        
############################### CONTROLE AB ########################################

def graph_ab(X,t,ab_,aB_,Ab_,AB_,ab,AbaB,AB,sd,r):
    
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
    ax.set(xlabel='Space', ylabel='Frequency', ylim=[-0.1,1.1], title = f'Contrôle ab : sd = {sd}, r={r}, time = {int(t)}')   
    ax.legend()
    plt.show() 
    
 
def continuous_evolution_ab(r,sd,dif,gamma,T,L,M,N,theta,CI,show_graph_ini,show_graph_fin,mod_x,show_graph_x,ab,AbaB,AB):   
   
    print("\nCHECK AB BELOW")
    
    # Steps
    dt = T/M    # time
    dx = L/N    # spatial
    
    # Space abscisse
    X = np.linspace(0,N,N+1)*dx
    
    # Initialization (frequency vector : abcd  abcD  abCd  abCD  aBcd  aBcD  aBCd  aBCD  Abcd  AbcD  AbCd  AbCD  ABcd  ABcD  ABCd  ABCD)   
    if CI == "equal" : 
        ab_ = np.ones(N+1)*(1/4); aB_ = np.ones(N+1)*(1/4); Ab_ = np.ones(N+1)*(1/4); AB_ = np.ones(N+1)*(1/4)  
            
    # Matrix
    C0 = -2*np.ones(N+1); C0[0]=C0[0]+1; C0[-1]=C0[-1]+1               
    C1 = np.ones(N+1) 
    A = spa.spdiags([C1,C0,C1],[-1,0,1], N+1, N+1)          # 1D discrete Laplacian with Neumann boundary conditions (derivative=0)  
     
    B = spa.identity(N+1)+((1-theta)*dif*dt/dx**2)*A            # Matrix for the explicit side of the Crank Nicholson scheme  
    B_ = spa.identity(N+1)-(theta*dif*dt/dx**2)*A               # Matrix for the implicit side of the Crank Nicholson scheme  


    # Plot the initial graph
    nb_graph = 1
    if show_graph_ini :
        graph_ab(X,0,ab_,aB_,Ab_,AB_,ab,AbaB,AB,sd,r)
        
    # Speed of the wave, function of time
    position = np.array([])             # list containing the first position where the proportion of wild alleles is higher than 0.5.
    time = np.array([])                 # list containing the time at which we calculate the speed.
    speed_fct_of_time = np.array([])    # list containing the speed corresponding to the time list.
    treshold = 0.5                      # indicates which position of the wave we follow to compute the speed (first position where the ab_ wave come over the threshold)    
    
    # Evolution
    for t in np.linspace(dt,T,M) : 
        
        # Time
        t = round(t,2)
        
        # Growth terms
        fab = ab_**2 + aB_*Ab_*(1-sd)**2*(1-gamma)*r + ab_*aB_*(1-sd) + ab_*Ab_*(1-sd) + ab_*AB_*(1-sd)**2*(1-gamma)*(1-r) 
        faB = aB_**2*(1-sd)**2 + aB_*Ab_*(1-sd)**2*(1-r*(1-gamma)) + aB_*AB_*(1-sd)**3 + ab_*aB_*(1-sd) + ab_*AB_*(1-sd)**2*(1-(1-r)*(1-gamma)) 
        fAb = Ab_**2*(1-sd)**2 + aB_*Ab_*(1-sd)**2*(1-gamma)*(1-r) + Ab_*AB_*(1-sd)**3*(1-gamma) + ab_*Ab_*(1-sd) + ab_*AB_*(1-sd)**2*(1-gamma)*r
        fAB = AB_*ab_*(1-sd)**2*(1-r*(1-gamma)) + AB_*Ab_*(1-sd)**3*(1+gamma) + AB_**2*(1-sd)**4 + aB_*Ab_*(1-sd)**2*(r+gamma*(1-r)) + aB_*AB_*(1-sd)**3
     
        # Mean fitness
        mean_fitness = fab + faB + fAb + fAB
        # mean_fitness = ab**2 + 2*ab*aB*(1-sd) + 2*ab*Ab*(1-sd) + 2*ab*AB*(1-sd)**2 + aB**2*(1-sd)**2 + 2*aB*Ab*(1-sd)**2 + 2*aB*AB*(1-sd)**3 + Ab**2*(1-sd)**2 + 2*Ab*AB*(1-sd)**3 + AB**2*(1-sd)**4
        
        # Reaction terms
        fab =  fab /mean_fitness - ab_ 
        faB =  faB /mean_fitness - aB_  
        fAb =  fAb /mean_fitness - Ab_   
        fAB =  fAB /mean_fitness - AB_    
         
        # New proportions
        ab_ = la.spsolve(B_, B.dot(ab_) + dt*fab)
        aB_ = la.spsolve(B_, B.dot(aB_) + dt*faB)
        Ab_ = la.spsolve(B_, B.dot(Ab_) + dt*fAb)
        AB_ = la.spsolve(B_, B.dot(AB_) + dt*fAB)
      
        if CI != "equal" :
            # Position of the wave ab_
            if np.isin(True, ab_ > treshold) and np.isin(True, ab_ < 0.99) and np.where(ab_ > treshold)[0][0] != 0 :  
                # first position where the wave is under the treshold value
                position = np.append(position, np.where(ab_ > treshold)[0][0])   
            # if the treshold value of the wave is outside the window, stop the simulation  
            if not(np.isin(False, ab_>treshold) and np.isin(False, ab_<treshold) ) :
                print("t=",t)
                break 
        
        # Graph
        if t>=mod_x*nb_graph :  
            # Show graphs
            if show_graph_x :
                graph_ab(X,t,ab_,aB_,Ab_,AB_,ab,AbaB,AB,sd,r)
                nb_graph += 1
            # We calculate the speed each time we make a graph and store this speed.
            if t >= 2*T/5 and CI != "equal" :
                time = np.append(time, t)
                speed_fct_of_time = np.append(speed_fct_of_time, np.mean(np.diff(position))*dx/dt)
            
    # Plot the last graph
    if show_graph_fin :   
        graph_ab(X,T,ab_,aB_,Ab_,AB_,ab,AbaB,AB,sd,r)
        
    # Plot the speed graph function of time (from 2*T/5 to T)
    if CI != "equal" and np.shape(position)[0] != 0 :        
        fig, ax = plt.subplots()
        ax.plot(time, speed_fct_of_time) 
        ax.set(xlabel='Time', ylabel='Speed', title = f'Speed function of time')   
        plt.show() 
                        
    return(ab_,aB_,Ab_,AB_) 



############################### CONTROLE CD ########################################

def graph_cd(X,t,cd_,cD_,Cd_,CD_,cd,CdcD,CD,sp,st,r):

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
    ax.set(xlabel='Space', ylabel='Frequency', ylim=[-0.1,1.1], title = f'Contrôle cd : sp = {sp}, st={st}, r={r}, time = {int(t)}')   
    ax.legend()
    plt.show() 

def continuous_evolution_cd(r,sp,st,dif,gamma,T,L,M,N,theta,CI,show_graph_ini,show_graph_fin,mod_x,show_graph_x,cd,CdcD,CD): 
    
    print("\nCHECK CD BELOW")
    
    # Steps
    dt = T/M    # time
    dx = L/N    # spatial
    
    # Space abscisse
    X = np.linspace(0,N,N+1)*dx
    
    # Initialization (frequency vector : abcd  abcD  abCd  abCD  aBcd  aBcD  aBCd  aBCD  Abcd  AbcD  AbCd  AbCD  ABcd  ABcD  ABCd  ABCD)   
    if CI == "equal" : 
        cd_ = np.ones(N+1)*(1/4); cD_ = np.ones(N+1)*(1/4); Cd_ = np.ones(N+1)*(1/4); CD_ = np.ones(N+1)*(1/4)  
    if CI == "left_cd" :
        CD_ = np.ones(N+1) ; CD_[N//2:] = 0 ; cd_ = 1 - CD_ ; cD_ = np.zeros(N+1); Cd_ = np.zeros(N+1)

              
    # Matrix
    C0 = -2*np.ones(N+1); C0[0]=C0[0]+1; C0[-1]=C0[-1]+1               
    C1 = np.ones(N+1) 
    A = spa.spdiags([C1,C0,C1],[-1,0,1], N+1, N+1)          # 1D discrete Laplacian with Neumann boundary conditions (derivative=0)  
    
    B = spa.identity(N+1)+((1-theta)*dif*dt/dx**2)*A            # Matrix for the explicit side of the Crank Nicholson scheme  
    B_ = spa.identity(N+1)-(theta*dif*dt/dx**2)*A               # Matrix for the implicit side of the Crank Nicholson scheme  

    # Plot the initial graph 
    nb_graph = 1
    if show_graph_ini :
        graph_cd(X,0,cd_,cD_,Cd_,CD_,cd,CdcD,CD,sp,st,r)
        
    # Speed of the wave, function of time
    position = np.array([])             # list containing the first position where the proportion of wild alleles is higher than 0.5.
    time = np.array([])                 # list containing the time at which we calculate the speed.
    speed_fct_of_time = np.array([])    # list containing the speed corresponding to the time list.
    treshold = 0.5                      # indicates which position of the wave we follow to compute the speed (first position where the cd_ wave come over the threshold)    
 
        
    # Evolution
    for t in np.linspace(dt,T,M) : 
        
        # Time
        t = round(t,2)
        
        # Growth terms
        fcd = cd_**2 + cd_*cD_*(1-sp)*(1-st) + cd_*Cd_*(1-sp)*(1-st) + cD_*Cd_*(1-sp)*r + cd_*CD_*(1-sp)*(1-r)
        fcD = cd_*cD_*(1-sp)*(1-st) + cD_**2*(1-sp)*(1-st) + cD_*Cd_*(1-sp)*(1-r) + cd_*CD_*(1-sp)*r + cD_*CD_*(1-sp)
        fCd = cd_*Cd_*(1-sp)*(1-st) + Cd_**2*(1-sp)*(1-st) + cD_*Cd_*(1-sp)*(1-r) + cd_*CD_*(1-sp)*r + Cd_*CD_*(1-sp)
        fCD = cd_*CD_*(1-sp)*(1-r) + cD_*CD_*(1-sp) + Cd_*CD_*(1-sp) + CD_**2*(1-sp) + cD_*Cd_*(1-sp)*r      
        
        # Mean fitness
        mean_fitness = fcd + fcD + fCd + fCD
        #mean_fitness = cd_**2 + 2*cd_*cD_*(1-st)*(1-sp) + 2*cd_*Cd_*(1-st)*(1-sp) + 2*cd_*CD_*(1-sp) + cD_**2*(1-st)*(1-sp) + 2*cD_*Cd_*(1-sp) + 2*cD_*CD_*(1-sp) + Cd_**2*(1-st)*(1-sp) + 2*Cd_*CD_*(1-sp) + CD_**2*(1-sp)
        
        
        # Reaction terms
        fcd =  fcd /mean_fitness - cd_ 
        fcD =  fcD /mean_fitness - cD_  
        fCd =  fCd /mean_fitness - Cd_   
        fCD =  fCD /mean_fitness - CD_
        
        # New proportions
        cd_ = la.spsolve(B_, B.dot(cd_) + dt*fcd)
        cD_ = la.spsolve(B_, B.dot(cD_) + dt*fcD)
        Cd_ = la.spsolve(B_, B.dot(Cd_) + dt*fCd)
        CD_ = la.spsolve(B_, B.dot(CD_) + dt*fCD)
        
        if CI != "equal" :
            # Position of the wave cd_
            if np.isin(True, cd_ > treshold) and np.isin(True, cd_ < 0.99) and np.where(cd_ > treshold)[0][0] != 0 :  
                # first position where the wave is under the treshold value
                position = np.append(position, np.where(cd_ > treshold)[0][0])   
            # if the treshold value of the wave is outside the window, stop the simulation  
            if not(np.isin(False, cd_>treshold) and np.isin(False, cd_<treshold) ) :
                print("t=",t)
                break 

        # Graph
        if t>=mod_x*nb_graph :  
            # Show graphs
            if show_graph_x :
                graph_cd(X,t,cd_,cD_,Cd_,CD_,cd,CdcD,CD,sp,st,r)
                nb_graph += 1
        # We calculate the speed each time we make a graph and store this speed.
        if CI != "equal" and  t >= T/10 :
            time = np.append(time, t)
            speed_fct_of_time = np.append(speed_fct_of_time, np.mean(np.diff(position))*dx/dt)
       
    # Plot the last graph
    if show_graph_fin :   
        graph_cd(X,T,cd_,cD_,Cd_,CD_,cd,CdcD,CD,sp,st,r)
            
    # Plot the speed graph function of time (from T/5 to T)
    if CI != "equal" and np.shape(position)[0] != 0 :        
        fig, ax = plt.subplots()
        ax.plot(time, speed_fct_of_time) 
        ax.set(xlabel='Time', ylabel='Speed', title = f'Speed function of time')   
        plt.show() 
        
    return(cd_,cD_,Cd_,CD_)




