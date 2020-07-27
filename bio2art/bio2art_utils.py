#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle 
import numpy as np
import random

# Load the names of the regions of the indicated connectome
def get_names(path_to_connectome_folder, data_name):
    '''
    Load the names of the regions of the indicated dataset
    
    Input
    -----
    path_to_connectome_folder: the path to the folder with connectome files
    
    data_name: string with the name of the connectome file of the connectome you 
        would like to use. Currently available:
        
    Drosophila                     
    Human_Betzel_Normalized        
    Macaque_Normalized             
    Marmoset_Normalized            
    Mouse_Gamanut_Normalized       
    Mouse_Ypma_Oh
    
    Output
    ------
    names: list of strings of len N, with N the number of areas in the 
        connectome, indicating the names of each brain region/node
    
    Note: Pickle is used for loading the names
    
    '''
    data_name = 'Names_' + data_name + '.lst' # Prefix and suffix for the file
    file_to_open = path_to_connectome_folder / data_name
    with open(file_to_open, 'rb') as fp: 
        names = pickle.load(fp)
    
    return names
 
#  Load the neuron density of the regions of the indicated dataset
def get_neuron_density(path_to_connectome_folder, data_name):
    '''
    Load the neuron density of the regions of the indicated dataset
    
    Input
    -----
    path_to_connectome_folder: the path to the folder with connectome files
    
    data_name: string with the name of the connectome file of the connectome you 
        would like to use. Currently available:
              
    Macaque_Normalized             
    Marmoset_Normalized            

    Output
    ------
    neuron_density: list of strings of len N, with N the number of areas in the 
        connectome, indicating the names of each brain region/node
    
    '''
    
    file_conn = 'ND_' + data_name + '.npy' # Prefix and suffix for the file
    file_to_open = path_to_connectome_folder / file_conn
    neuron_density = np.load(file_to_open)
    
    return neuron_density 

#  Construct a scaled neuron_density array based on the seed_neuron
def scale_neuron_density(neuron_density, 
                         seed_neuron=1,
                         scale_type='rank'):
    '''
    Construct a scaled neuron_density array based on the seed_neuron
     
    Input
    -----
    neuron_density: numpy array of shape (N,), with N the number of areas in 
        the connectome with entry i denoting the neuron denisty of region i 
        (returned from function get_neuron_density).
    
    seed_neuron: int, default 1, specifying the number that will be multiplied
        by the scaled neuron_density.
        
    scale_type: string 'ratio' 'rank', default 'rank', specifying how the 
        neuron_density valeus will be scaled. 
        'rank': the values are rank ordered and multiplied by seed_neuron
        'ratio':the values are converted to ratios, 
        neuron_density[i] / min(neuron_density) and multiplied by seed_neuron 
       
    
    Output
    ------
    scaled_neuron_density: numpy array of shape (N,), with N the number  
        of areas in the connectome, indicating the scaled neuron_density of 
        each brain region/node
    
    '''
    # Copy and sort the neuron densities
    neuron_density_srt = neuron_density.copy()
    neuron_density_srt.sort()
    
    # Get the idx so you can go match neuron_density_srt to neuron_density 
    # The y_ind will contain the desired idx such that:
    # neuron_density[y_ind[i]] == nd_sorted[i] for every i=0,1,2...N 
    # with N=neuron_density.shape[0] 
    xy, x_ind, y_ind = np.intersect1d(neuron_density_srt, 
                                      neuron_density, 
                                      return_indices=True)
    
    scaled_neuron_density = neuron_density.copy()
    
    if scale_type == 'rank':
        scaled_neuron_density[y_ind] = np.asarray(range(1, neuron_density.shape[0]+1)
                                                 ) * seed_neuron
    elif scale_type == 'ratio':
        for i, item in enumerate(neuron_density_srt):
            if i == 0: 
                scaled_neuron_density[y_ind[i]] = 1
                denominator = neuron_density_srt[i]
            scaled_neuron_density[y_ind[i]] = round((item / denominator) * seed_neuron)       
        
    scaled_neuron_density = scaled_neuron_density.astype('int64')                                  
    
    return scaled_neuron_density 

# Partition integer i in n random integers that sum to i 
def int_partition(i, n):
    '''
    Partition integer i in n random integers that sum to i 
    
    Input
    -----
    i:  int, a positive integer that needs to be partitioned in n integers
        that sum to i 
    n:  int, a positive integer specifying the number of partitions/integers
        to be generated
         
    Output
    ------   
    partitions, list, list of n integers with the property 
        sum(partitions)==i 
    '''
    partitions=[]
    if n > i:
        print('\nInteger i must be higher or equal to n\n') 
        return
    if i < 0 or n < 0:
        print('\nIntegers i and n must be positive\n') 
        return
             
    spectrum = (i-n)+2
    for k in range(n-1):
        if spectrum > 1:
            new_val = np.random.randint(1, high=spectrum)
            partitions.append(new_val)
            spectrum = spectrum-new_val
        else:
            new_val = np.random.randint(1, high=2)
            partitions.append(new_val)
   
    partitions.append(i-sum(partitions))
    # Shuffle list so that the order of generation does not bias magnitude 
    # of integers as they are generated
    random.shuffle(partitions)
    
    return partitions

# Partition float f in n random floats that sum to f 
def float_partition(f, n):
    '''
    Partition float f in n random floats that sum to f 
    
    Input
    -----
    f:  float, a positive float that needs to be partitioned in n floats
        that sum to i 
    n:  int, a positive integer specifying the number of partitions/integers
        to be generated
         
    Output
    ------   
    partitions, list, list of n float numbers with the property 
        sum(partitions)==f
    '''
    partitions=[]             
    spectrum = f
    for k in range(n-1):
        new_val = random.uniform(0., abs(spectrum))
        partitions.append(new_val)
        spectrum = spectrum-new_val
   
    partitions.append(f-sum(partitions))
    # Shuffle list so that the order of generation does not bias magnitude 
    # of integers as they are generated
    random.shuffle(partitions)
    
    return partitions            