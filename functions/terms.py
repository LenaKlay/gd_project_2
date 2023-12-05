#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:10:33 2022

@author: lena
"""

############################## Libraries ############################################

import numpy as np


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
    
    
    
       
    
    
    
